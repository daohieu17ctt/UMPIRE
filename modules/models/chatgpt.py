import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
load_dotenv('.env') # add this line to load environment variables from .env file

import asyncio
from concurrent.futures import ThreadPoolExecutor

model_mapping = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4": "gpt-4-0613",
    "gpt-4-turbo": "gpt-4-1106-preview",
    "gpt-3.5": "gpt-3.5-turbo-1106",
}
class ChatGPTPredictor:
    def __init__(self, api_key=None):
        """
        Initialize the predictor with an API key.
        If no key is provided, it attempts to load it from the environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", False)
        if not self.api_key:
            raise KeyError("Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.")
        self.client = OpenAI(api_key=self.api_key)

    # @retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
    def predict(self, prompt, temperature=1.0, model="gpt-4", system_prompt=None, max_tokens=200, top_p=1, n=1, logprobs=False):
        """
        Predict with GPT models.
        
        Args:
            prompt (str or list): The prompt string or list of messages.
            temperature (float): Sampling temperature.
            model (str): The model name (e.g., 'gpt-4', 'gpt-4-turbo', or 'gpt-3.5').

        Returns:
            str: The response content from the GPT model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Prepare the messages in the required format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages.append(prompt)
        
        if model not in model_mapping:
            raise KeyError(f"Model {model} not found in the mapping. Please use a valid model name.")
        model = model_mapping.get(model, model)  # Use mapping or the given model name

        # Call the OpenAI API
        output = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            logprobs=logprobs,
        )
        if logprobs:
            # Handle logprobs if needed
            logprobs_data = []
            responses = []
            for choice in output.choices:
                choice_logprobs = []
                if hasattr(choice, 'logprobs'):
                    for token_logprob in choice.logprobs.content:
                        if hasattr(token_logprob, 'logprob'):
                            # Process logprobs as needed
                            choice_logprobs.append(token_logprob.logprob)
                logprobs_data.append(choice_logprobs)
                responses.append(choice.message.content)
            # Return logprobs data along with the content
            return responses, logprobs_data
        else:
            return [choice.message.content for choice in output.choices]

# Assume ChatGPTPredictor is the class we defined earlier.

class AsyncChatGPT:
    def __init__(self, api_key=None, use_tqdm=True):
        self.use_tqdm = use_tqdm
        self.predictor = ChatGPTPredictor(api_key)

    def sync_predict(self, prompt, temperature=1, model='gpt-4', system_prompt=None, max_tokens=200, top_p=1, n=1, logprobs=False):
        try:
            return self.predictor.predict(prompt, temperature=temperature, model=model, system_prompt=system_prompt, max_tokens=max_tokens, top_p=top_p, n=n, logprobs=logprobs)
        except Exception as e:
            print(f"Error for prompt {prompt}: {e}")
            return None

    async def async_predict(self, prompts, max_workers=20, temperature=1, model='gpt-4', system_prompt=None, max_tokens=200, top_p=1, n=1, logprobs=False):
        loop = asyncio.get_running_loop()
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self.sync_predict, prompt, temperature, model, system_prompt, max_tokens, top_p, n, logprobs)
                for prompt in prompts
            ]
            if self.use_tqdm:
                responses = await asyncio.gather(*tasks)
                for response in tqdm(responses, total=len(tasks), desc="Predicting"):
                    results.append(response)
            else:
                for response in await asyncio.gather(*tasks):
                    results.append(response)
        return results
    
    def predict(self, prompts, max_workers=20, temperature=1, model='gpt-4o', system_prompt=None, max_tokens=200, top_p=1, n=1, logprobs=False):
        return asyncio.run(self.async_predict(prompts, max_workers=max_workers, temperature=temperature, model=model, system_prompt=system_prompt, max_tokens=max_tokens, top_p=top_p, n=n, logprobs=logprobs))

def build_prompt(question, gt_answer, pred_answer, system_prompt = "You are a creative and helpful assistant."):
    gt_answer_str = ", ".join(gt_answer)
    prompt = f'''
We are assessing the quality of answers to the following question: {question}
Here is the list of {len(gt_answer)} expected answers: {gt_answer_str}
The proposed answer is: {pred_answer}
Within the context of the question, does the proposed answer mean the same as any of the expected answers? Respond only with yes or no.'''
# Within the context of the question and example answer, is the proposed answer correct? Respond only with yes or no.'''
# Within the context of the question, does the proposed answer mean the same as or a part of the expected answer? Respond only with yes or no.'''
    return prompt

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def build_prompt_chatgpt_with_image(text_prompt, image_path, detail='low'):
    image_base64 = encode_image(image_path)
    prompt_message = {
            "role": "user",
            "content": [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{image_base64}",
                        'detail': detail,
                    }
                },
                {
                    'type': 'text',
                    'text': text_prompt
                }
            ]
        }
    return prompt_message


# Example usage:
if __name__ == "__main__":
    import pickle
    file_path = f"/external1/daohieu/save_temp/su_vlm/output/llava_OpenEnded_mscoco_val2014/generation_embedding/llava-v1.5-13b.pkl"
    img_dir = '/home/daohieu/maplecg_nfs/research/VLM/su_vlm/data/vqav2/val2014'
    with open(file_path, 'rb') as r:
        llava_results = pickle.load(r)

    llava_results = llava_results[:100]
    prefix_prompt='Answer this question in only a word or a phrase. '
    dataset = [build_prompt_chatgpt_with_image(prefix_prompt + sample['question_text'], os.path.join(img_dir, sample['image'])) for sample in llava_results]
    
    async_predictor = AsyncChatGPT()
    import time
    start = time.time()
    responses = async_predictor.predict(dataset, max_workers=20, temperature=1, model='gpt-4o-mini', system_prompt="You are a creative and helpful assistant.", top_p=0.9, n=50, logprobs=True)
    print(f"Time taken: {time.time() - start:.2f}s")
    import pdb; pdb.set_trace()
    for i, (sample, response) in enumerate(zip(dataset, responses)):
        print(i)
        print(response)
        print("")
        # time.sleep(1)
    # predictor = ChatGPTPredictor()
    # for sample in dataset:
    #     # prompt = build_prompt_chatgpt_with_image(sample['question_text'], os.path.join(img_dir, sample['image']))
    #     response = predictor.predict(sample, temperature=1, model='gpt-4o', system_prompt="You are a creative and helpful assistant.", top_p=0.9, n=50, logprobs=True)
    #     import pdb; pdb.set_trace()
    #     print(response)
    #     print("")