import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import StoppingCriteria
from transformers import StoppingCriteriaList, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader

from .base_model import BaseModel
from .base_model import STOP_SEQUENCES
# from .model_utils import model_example_map

class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False

class QwenModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,
                                                                        torch_dtype=torch.float16,
                                                                        low_cpu_mem_usage=True,
                                                                        load_in_4bit=True,
                                                                        attn_implementation="flash_attention_2",
                                                                        device_map='auto')
        self.tokenizer = self.processor.tokenizer
        if hasattr(self.model.config, "max_sequence_length"):
            context_len = self.model.config.max_sequence_length
        else:
            context_len = 4096

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = context_len
        self.device = 'cuda'

    def process_input(self, prompt, image_path=None):
        if image_path is not None:
            # image = Image.open(image_path).convert('RGB')
            conv = [
                {   
                    'role': 'system',
                    'content': "A chat between a curious human and an artificial intelligence assistant. The assistant gives trustworthy answers to the user's questions."
                },
                {
                    'role':'user',
                    'content': [
                        {
                            'type': 'image',
                            'image': image_path
                        },
                        {
                            'type': 'text', 
                            'text': f"{prompt}"
                        },
                    ]
                }
            ]
        else:
            # image=None
            conv = [
                {   
                    'role': 'system',
                    'content': "A chat between a curious human and an artificial intelligence assistant. The assistant gives trustworthy answers to the user's questions."
                },
                {
                    'role':'user',
                    'content': [
                        {'type': 'text', 'text': f"{prompt}"},
                    ]
                }
            ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = process_vision_info(conv)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )        
        inputs['input_text'] = prompt # add prompt text
        return inputs
    
        
    def predict_prompt_image(self, prompt, image_path, temperature, top_p=1, beam_search=False, num_beams=0):
        input_data = self.process_input(prompt, image_path)
        return self.predict(input_data=input_data, temperature=temperature, top_p=top_p, beam_search=beam_search, num_beams=num_beams)
    
    
    def predict(self, input_data, temperature, top_p, return_full=False, beam_search=False, num_beams=0, layer_idx=-1):
        # Implement prediction.
        del input_data['input_text']
        input_data = input_data.to(self.device)
        input_ids = input_data['input_ids'].to(device=self.device, non_blocking=True)
        input_text = self.tokenizer.batch_decode(input_ids)[0]
        
        # if input_data['pixel_values'] != None:
        #     image_tensor = input_data['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        # else:
        #     image_tensor = None
        # image_sizes = input_data['image_sizes']

        pad_token_id = self.tokenizer.eos_token_id

        # if self.stop_sequences is not None:
        #     stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        #         stops=self.stop_sequences,
        #         initial_length=len(input_ids[0]),
        #         tokenizer=self.tokenizer)])
        # else:
        stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            if beam_search == False:
                outputs = self.model.generate(
                    **input_data,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p = top_p,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                )
            else:
                if num_beams == 0:
                    print("Using beam search but num_beams set to 0, reset to 1, this is greedy search")
                    num_beams = 1
                outputs = self.model.generate(**input_data, max_new_tokens=self.max_new_tokens, temperature=None, do_sample=False, top_p = None, use_cache=True, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, num_beams=num_beams, num_return_sequences=num_beams, num_beam_groups=num_beams, diversity_penalty=0.5)
        
        if len(outputs.sequences[0]) > self.token_limit: # take first generation
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        # full_answer_list = self.tokenizer.batch_decode(
        #     outputs.sequences, skip_special_tokens=True)
        full_answer_list = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False)

        if return_full:
            return full_answer_list
        
        # For some models, we need to remove the input_data from the answer.
        if full_answer_list[0].startswith(input_text): # take first generation
            input_data_offset = len(input_text)
            n_input_token = self.tokenizer(input_text, return_tensors="pt")['input_ids'].shape[1]
        else:
            input_data_offset = 0
            n_input_token = 1 # 1 for tokenizer <s>
            # raise ValueError('Have not tested this in a while.')
        
        # Remove input from answer.
        answer_list = [full_answer[input_data_offset:] for full_answer in full_answer_list]
        sliced_answer_list = []
        last_token_embedding_list = []
        log_likelihoods_list = []
        # pre-compute transition_scores to get the log-likelihood later
        # import pdb; pdb.set_trace()
        if not hasattr(self.model.config, 'vocab_size'):
            self.model.config.vocab_size = self.model.config.text_config.vocab_size
        if beam_search:
            transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True)
        else:
            transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        
        
        for ans_id, answer in enumerate(answer_list):
            # Remove stop_words from answer.
            stop_at = len(answer)
            sliced_answer = answer
            if self.stop_sequences is not None:
                for stop in self.stop_sequences:
                    if stop in answer:
                        stop_at = answer.find(stop)
                        sliced_answer = answer[:stop_at]
                        break
                    # if answer.endswith(stop):
                    #     stop_at = len(answer) - len(stop)
                    #     sliced_answer = answer[:stop_at]
                    #     break
                if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                    error_msg = 'Error: Stop words not removed successfully!'
                    error_msg += f'Response: >{full_answer_list[ans_id]}< \n'
                    error_msg += f'Answer: >{answer}< \n'
                    error_msg += f'Sliced Answer: >{sliced_answer}<' 
                    if 'falcon' not in self.model_name.lower():
                        # raise ValueError(error_msg)
                        print("Bypass the error", error_msg)
                    else:
                        logging.error(error_msg)

            # Remove whitespaces from answer (in particular from beginning.)
            sliced_answer = sliced_answer.strip()
            sliced_answer_list.append(sliced_answer)

            # Get the number of tokens until the stop word comes up.
            # Note: Indexing with `stop_at` already excludes the stop_token.
            # Note: It's important we do this with full answer, since there might be
            # non-trivial interactions between the input_data and generated part
            # in tokenization (particularly around whitespaces.)
            token_stop_index = self.tokenizer(full_answer_list[ans_id][:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
            n_generated = token_stop_index - n_input_token
            # n_generated exclude stop words

            if n_generated == 0:
                logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
                n_generated = 1


            # Get the last hidden state (last layer) and the last token's embedding of the answer.
            # Note: We do not want this to be the stop token.

            # outputs.hidden_state is a tuple of len = n_generated_tokens.
            # The first hidden state is for the input tokens and is of shape
            #     (n_layers) x (batch_size, input_size, hidden_size).
            # (Note this includes the first generated token!)
            # The remaining hidden states are for the remaining generated tokens and is of shape
            #    (n_layers) x (batch_size, 1, hidden_size).

            # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
            # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
            # token_stop_index to arrive at the right output.

            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states
        
            if len(hidden) == 1:
                logging.warning(
                    'Taking first and only generation for hidden! '
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer_list[ans_id],
                    )
                last_input = hidden[0]
            elif ((n_generated - 1) >= len(hidden)):
                # If access idx is larger/equal.
                logging.error(
                    'Taking last state because n_generated is too large'
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s, slice_answer: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer_list[ans_id], sliced_answer
                    )
                last_input = hidden[-1]
            else:
                try:
                    # import pdb; pdb.set_trace()
                    if len(hidden) > n_generated:
                        last_input = hidden[n_generated] # <eos> token
                    else:
                        last_input = hidden[n_generated - 1] # before <eos> token
                except:
                    import pdb; pdb.set_trace()

            # Then access last layer for input
            if layer_idx == 'full':
                last_token_embedding_all_layers = []
                for each_layer in last_input:
                    # Then access last token in input.
                    last_token_embedding = each_layer[ans_id][-1, :].cpu() # tensor with size (hidden_size)
                    last_token_embedding_all_layers.append(last_token_embedding)
                last_token_embedding_all_layers = torch.stack(last_token_embedding_all_layers)
                last_token_embedding_list.append(last_token_embedding_all_layers)
            else:
                last_token_embedding_list.append(last_input[-1][ans_id][-1, :].cpu())
                # last_token_embedding_all_layers

            # Get log_likelihoods.
            # outputs.scores are the logits for the generated tokens.
            # outputs.scores is a tuple of len = n_generated_tokens.
            # Each entry is shape (bs, vocabulary size).
            # outputs.sequences is the sequence of all tokens: input and generated.

            # Transition_scores[ans_id] only contains the scores for the current generated sequence.
            log_likelihoods = [score.item() for score in transition_scores[ans_id]]
            if len(log_likelihoods) == 1:
                logging.warning('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                if len(log_likelihoods) > n_generated:
                    log_likelihoods = log_likelihoods[:n_generated+1] # stop at eos
                else:
                    log_likelihoods = log_likelihoods[:n_generated] # stop at eos
            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')
            if len(log_likelihoods) == 0:
                pass
            
            log_likelihoods_list.append(log_likelihoods) # log_likelihoods: array

        if len(sliced_answer_list) == 1:
            return sliced_answer_list[0], log_likelihoods_list[0], last_token_embedding_list[0]
        else:
            last_token_embedding_list = torch.stack(last_token_embedding_list).permute(1, 0, 2) # reshape to (num_layer, num_gen, emb_length)
            if beam_search:
                beam_sequence_scores = outputs.sequences_scores.to('cpu').tolist()
                return sliced_answer_list, log_likelihoods_list, last_token_embedding_list, beam_sequence_scores
            return sliced_answer_list, log_likelihoods_list, last_token_embedding_list

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""
        input_ids = input_data['input_ids'].to(device=self.device, non_blocking=True)
        image_tensor = input_data['image_tensor'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        image_sizes = input_data['image_sizes']

        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.
        target_ids_true = input_ids.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100
        with torch.no_grad():
            # model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)
            model_output_true = self.model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, labels=target_ids_true)

        loss_true = model_output_true.loss
        return -loss_true.item()
