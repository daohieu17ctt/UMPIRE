import argparse
import torch
import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.utils.extmath import fast_logdet
import numpy as np

from modules.logdet_utils import normalize_embedding

###### UMPIRE implementation ######
def compute_umpire(sample, jitter=1e-8, alpha=1, length_normalize=False):
    raw_llh = sample['generations_log_likelihood'] # [[<token1_prob>, <token2_prob>, ...], ...]
    k = len(raw_llh) # number of generations
    embeddings = normalize_embedding(sample['generations_embedding'])
    if embeddings.shape[0] == k + 1:
        embeddings = embeddings[1:] # shape: (k, embedding_dim)

    # Logdet term
    kernel = np.dot(embeddings, embeddings.T) # shape: (k, k)
    logdet = 1/(2 * k) * fast_logdet(kernel + np.identity(kernel.shape[0])*jitter) # logdet
    
    # Quadratic entropy term
    # Sequence probs - shape: (k,), length normalized if needed.
    if length_normalize: 
        seq_prob = np.array([np.exp(np.sum(i)/len(i)) for i in raw_llh])
    else:
        seq_prob = np.array([np.exp(np.sum(i)) for i in raw_llh]) 
    incoherence_scores = 1 - seq_prob
    quad_entropy = 1/k * np.sum(incoherence_scores)
    
    # UMPIRE
    UMPIRE =  logdet + alpha * quad_entropy
    return UMPIRE

###### Semantic Entropy implementation ######
from modules.semantic_entropy import get_semantic_ids, logsumexp_by_id, predictive_entropy_rao
from modules.semantic_entropy import EntailmentDeberta


####### Main code to compute SE and UMPIRE ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help='The input prompt')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model_name', type=str, default='liuhaotian/llava-v1.5-13b', help='Model name or path')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of new tokens to generate')
    parser.add_argument('--num_generations', type=int, default=10, help='Number of generations per prompt')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for nucleus sampling')
    parser.add_argument('--jitter', type=float, default=1e-8, help='Jitter value for numerical stability in logdet computation')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value to balance logdet and quadratic entropy in UMPIRE')
    parser.add_argument('--length_normalize', action='store_true', help='Whether to length normalize the sequence log-likelihoods when computing UMPIRE')
    parser.add_argument('--entailment_model', type=str, default='deberta', help='Entailment model for Semantic Entropy')
    parser.add_argument('--reason', type=str, default='none', help='Reasoning type: none, cot')
    args = parser.parse_args()
    
    # ===== Load Vision Model =====
    print(f"Loading model: {args.model_name}")
    # wrapper model
    if 'qwen' in args.model_name.lower():
        from modules.models.qwen_models import QwenModel
        model = QwenModel(model_name=args.model_name, stop_sequences=[], max_new_tokens=args.max_new_tokens)
    elif 'llava' in args.model_name.lower():
        from modules.models.llava_models import HuggingfaceModel as LlavaModel
        model = LlavaModel(model_name=args.model_name, stop_sequences=[], max_new_tokens=args.max_new_tokens)     
    else:
        from modules.models.vision_models import VisionModel
        model = VisionModel(model_name=args.model_name, stop_sequences=[], max_new_tokens=args.max_new_tokens)    

    # ===== Load Entailment Model for Semantic Entropy =====
    print(f"Loading entailment model: {args.entailment_model}")
    if args.entailment_model == 'deberta':
        entailment_model = EntailmentDeberta()
    else:
        entailment_model = EntailmentDeberta()  # Default to DeBERTa
    
    # ===== Generate Predictions with Embeddings =====

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
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Image: {args.image_path}")
    print(f"Generating {args.num_generations} samples...")
    
    prompt_with_prefix = prefix_prompt + args.prompt
    generations_text = []
    generations_log_likelihood = []
    generations_embedding = []
    
    # Generate most likely prediction (greedy)
    most_likely_text, most_likely_llh, most_likely_emb = model.predict_prompt_image(
        prompt_with_prefix, args.image_path, temperature=0.1, top_p=0.9
    )
    generations_text.append(most_likely_text)
    generations_log_likelihood.append(most_likely_llh)
    generations_embedding.append(most_likely_emb)
    
    # Generate diverse samples
    for i in range(args.num_generations - 1):
        gen_text, gen_llh, gen_emb = model.predict_prompt_image(
            prompt_with_prefix, args.image_path, 
            temperature=args.temperature, top_p=args.top_p
        )
        generations_text.append(gen_text)
        generations_log_likelihood.append(gen_llh)
        generations_embedding.append(gen_emb)
    
    # Convert embeddings to numpy array
    generations_embedding = np.array([emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb 
                                       for emb in generations_embedding])
    
    # ===== Prepare data for uncertainty metrics =====
    sample = {
        'generations_text': generations_text,
        'generations_log_likelihood': generations_log_likelihood,
        'generations_embedding': generations_embedding,
        'prompt': args.prompt,
        'image': args.image_path
    }
    
    # ===== Compute UMPIRE =====
    print("\nComputing UMPIRE...")
    umpire_score = compute_umpire(
        sample, 
        jitter=args.jitter, 
        alpha=args.alpha, 
        length_normalize=args.length_normalize
    )
    
    # ===== Compute Semantic Entropy =====
    print("Computing Semantic Entropy...")
    example = {
        'question': args.prompt,
        'image_path': args.image_path
    }
    
    # Get semantic clusters
    cluster_ids = get_semantic_ids(
        strings_list=generations_text, 
        model=entailment_model, 
        strict_entailment=True,
        example=example
    )
    
    # Sum log-likelihoods for each token sequence
    llh_sums = [np.sum(llh) for llh in generations_log_likelihood]
    
    # Aggregate by cluster
    log_likelihood_by_cluster = logsumexp_by_id(cluster_ids, llh_sums)
    
    # Compute Semantic Entropy
    semantic_entropy_score = predictive_entropy_rao(log_likelihood_by_cluster)
    
    # ===== Print Results =====
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"\nPrediction (most likely): {most_likely_text}")
    print(f"Multi-sampling: {generations_text}")
    print(f"\nUMPIRE Score: {umpire_score:.4f}")
    print(f"Semantic Entropy Score: {semantic_entropy_score:.4f}")
    print("="*70)
