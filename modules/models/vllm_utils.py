from transformers import AutoTokenizer

# from vllm import LLM, SamplingParams
# from vllm.assets.image import ImageAsset
# from vllm.assets.video import VideoAsset
# from vllm.utils import FlexibleArgumentParser

def run_llava(question: str, modality: str):
    assert modality == "image"

    prompt = lambda x: f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=4096)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question: str, modality: str):
    assert modality == "image"

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=8192)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(question: str, modality: str):
    assert modality == "video"

    prompt = f"USER: <video>\n{question} ASSISTANT:"
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf", max_model_len=8192)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-OneVision
def run_llava_onevision(question: str, modality: str):

    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Fuyu
def run_fuyu(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}\n"
    llm = LLM(model="adept/fuyu-8b", max_model_len=2048, max_num_seqs=2)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(question: str, modality: str):
    assert modality == "image"

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma
def run_paligemma(question: str, modality: str):
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma-3b-mix-224")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Chameleon
def run_chameleon(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}<image>"
    llm = LLM(model="facebook/chameleon-7b", max_model_len=4096)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# MiniCPM-V
def run_minicpmv(question: str, modality: str):
    assert modality == "image"

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    #2.6
    model_name = "openbmb/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question: str, modality: str):
    assert modality == "image"

    model_name = "OpenGVLab/InternVL2-2B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


# NVLM-D
def run_nvlm_d(question: str, modality: str):
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# BLIP-2
def run_blip2(question: str, modality: str):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model="Salesforce/blip2-opt-2.7b")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen
def run_qwen_vl(question: str, modality: str):
    assert modality == "image"

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
    )

    prompt = f"{question}Picture 1: <img></img>\n"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(question: str, modality: str):
    assert modality == "image"

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=5,
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLama 3.2
def run_mllama(modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
    )

    prompt = lambda question: f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Molmo
def run_molmo(question, modality):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    prompt = question
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# GLM-4v
def run_glm4v(question: str, modality: str):
    assert modality == "image"
    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name,
              max_model_len=2048,
              max_num_seqs=2,
              trust_remote_code=True,
              enforce_eager=True)
    prompt = question
    stop_token_ids = [151329, 151336, 151338]
    return llm, prompt, stop_token_ids


model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "fuyu": run_fuyu,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "minicpmv": run_minicpmv,
    "blip-2": run_blip2,
    "internvl_chat": run_internvl,
    "NVLM_D": run_nvlm_d,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "mllama": run_mllama,
    "molmo": run_molmo,
    "glm4v": run_glm4v,
}
