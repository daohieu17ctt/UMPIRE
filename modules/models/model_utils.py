from transformers import AutoTokenizer


model_example_map = {
    "llava-7B": get_llava_7B,
    "llava-13B": get_llava_13B,
    "llava-next-7B": get_llava_next_7B,
    "llava-next-13B": get_llava_next_13B,
    "blip-2-2B": get_blip2_2B,
    "mllama-11B": get_mllama_11B,
}


def get_llava_7b(question: str, modality: str):
    assert modality == "image"
    prompt = lambda x: f"USER: <image>\n{question}\nASSISTANT:" 
    model_name = "llava-hf/llava-1.5-7b-hf"
    stop_token_ids = None
    return model_name, prompt, stop_token_ids

def get_llava_13b(question: str, modality: str):
    assert modality == "image"
    prompt = lambda x: f"USER: <image>\n{question}\nASSISTANT:"
    model_name = "llava-hf/llava-1.5-13b-hf"
    stop_token_ids = None
    return model_name, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def get_llava_next_mistral_7b(question: str, modality: str):
    assert modality == "image"
    prompt = f"[INST] <image>\n{question} [/INST]"
    model="llava-hf/llava-v1.6-mistral-7b-hf"
    stop_token_ids = None
    return model, prompt, stop_token_ids

# LLaVA-1.6/LLaVA-NeXT
# def get_llava_next_vicuna_7b(question: str, modality: str):
#     assert modality == "image"
#     prompt = f"[INST] <image>\n{question} [/INST]"
#     model="llava-hf/llava-v1.6-vicuna-13b-hf"
#     stop_token_ids = None
#     return model, prompt, stop_token_ids


    # BLIP-2
def get_blip2_2B(question: str, modality: str):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    model="Salesforce/blip2-opt-2.7b"
    stop_token_ids = None
    return model, prompt, stop_token_ids

# LLama 3.2
def get_mllama_11B(modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    # llm = LLM(
    #     model=model_name,
    #     max_model_len=4096,
    #     max_num_seqs=16,
    #     enforce_eager=True,
    # )

    prompt = lambda question: f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return model_name, prompt, stop_token_ids
