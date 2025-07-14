import argparse
import os
import pathlib
import pickle
import random
import evaluate
import numpy as np
import torch
from tqdm import tqdm
import math

import sys
sys.path.append(".")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

########## MAIN ###########
parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument("--beam_search", action='store_true')

# llava args
parser.add_argument("--image-folder", type=str, default="")
parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
parser.add_argument("--outdir", type=str, default="/home/daohieu/maplecg_nfs/research/VLM/su_vlm/output/")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--reason", type=str, choices=['cot', 'none'], default=None)
args = parser.parse_args()

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set torch random seed
torch.manual_seed(seed_value)

# wrapper model
if 'cogvlm' in args.model_path.lower():
    from modules.models.cogvlm_models import CogVLMModel
    model = CogVLMModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif 'contactdoctor' in args.model_path.lower():
    from modules.models.biomedllama_models import BioMedLlamaModel
    model = BioMedLlamaModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif 'qwen' in args.model_path.lower():
    from modules.models.qwen_models import QwenModel
    model = QwenModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif 'liuhaotian' in args.model_path.lower():
    from modules.models.llava_models import HuggingfaceModel as LlavaModel
    model = LlavaModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
else:
    from modules.models.vision_models import VisionModel
    model = VisionModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)     

#TODO wrap dataset
import json
questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

if args.reason == 'none' or args.reason is None:
    prefix_prompt = 'Answer this question in only a word or a phrase. '
elif args.reason == 'cot':
    prefix_prompt="Your task is to answer the question provided to you to the best of your abilities.\n" \
"### Format of the answer:\n" \
"<Steps for reasoning out what the answer would be>. ANSWER: <answer>\n" \
"Always state the answer by the end of your reasoning process." \
"End your response with the word 'ANSWER:' followed by the final answer." \
"Now you have understood the format and guidelines. Please answer according to the guidelines:\n"
else:
    raise ValueError(f"Unknown reasoning type: {args.reason}")

# Metric
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

# Generation
sequences = []
number_of_generations = args.num_generations_per_prompt
for line in tqdm(questions, total=len(questions)):
    idx = line["question_id"]
    cur_prompt = prefix_prompt + line["text"]
    image_name = line['image']
    image_path = os.path.join(args.image_folder, image_name)

    # Most likely generation for evaluation
    most_likely_generation_output_text, most_likely_generation_log_likelihood, most_likely_generation_embedding = model.predict_prompt_image(cur_prompt, image_path, temperature=0.1, top_p=0.9)

    # Sampling 
    generation_list = []
    generation_log_likelihood_list = []
    embedding = []
    for i in range(number_of_generations):
        generation, generation_log_likelihood, generation_embedding = model.predict_prompt_image(cur_prompt, image_path, temperature=args.temperature, top_p=args.top_p)
        generation_list.append(generation)
        generation_log_likelihood_list.append(generation_log_likelihood)
        embedding.append(generation_embedding)
    embedding = np.array(torch.stack(embedding).tolist())
    
    # Save dictionary
    sequence_dict = {
        'question_id': idx,
        # 'question_ids': input_ids.to('cpu'),
        'question_text': cur_prompt,
        'image': line['image'],
        # 'generations_ids': generation_list[i].to('cpu')
    }

    # Save generated answers
    sequence_dict['generations_text']= generation_list
    sequence_dict['generations_log_likelihood'] = generation_log_likelihood_list
    sequence_dict['most_likely_generation_text'] = most_likely_generation_output_text
    sequence_dict['most_likely_generation_log_likelihood'] = most_likely_generation_log_likelihood

    # Save grouth truth answers
    reference_answers = line['answers']

    # Evaluate most likely generation against reference answers
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    for rouge_type in rouge_types:
        if rouge_type in line:
            sequence_dict[rouge_type + '_reference_answers'] = line[rouge_type]

        else:
            sequence_dict[rouge_type + '_reference_answers'] = None
        sequence_dict[rouge_type + '_to_target'] = 0.0
    
    sequence_dict['exact_match'] = 0.0
    sequence_dict['answers'] = reference_answers
    for answer in reference_answers:
        predictions = [sequence_dict['most_likely_generation_text'].lstrip()]
        references = [answer]
        results = exact_match_metric.compute(predictions=predictions,
                                            references=references,
                                            ignore_case=True,
                                            ignore_punctuation=True)
        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
        rouge_results = rouge.compute(predictions=predictions, references=references)
        for rouge_type in rouge_types:
            sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type],
                                                            sequence_dict[rouge_type + '_to_target'])
    sequence_dict['internal_embedding'] = embedding
    sequences.append(sequence_dict)     
    # import pdb; pdb.set_trace() 
    # break

pathlib.Path(f'{args.outdir}').mkdir(parents=True, exist_ok=True)

with open(f'{args.outdir}/generations.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)

print("Done!")
