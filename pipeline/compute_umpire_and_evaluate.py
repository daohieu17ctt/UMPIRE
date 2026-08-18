import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from sklearn.utils.extmath import fast_logdet

# Register tqdm with pandas
tqdm.pandas()

import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.eval_utils import ROC_AUROC, compute_pearsonr, get_calibrate_ece, get_tpr_at_fpr, compute_aurac_from_image_df, \
    df_to_markdown_bold, split_balanced_data

from modules.logdet_utils import normalize_embedding, get_generation_embeddings, get_quad_entropy, \
    get_normalized_entropy, compute_eigenscore

def get_adaptive_alpha_dev_set(image_df, logdet_col, quality_col, dev_ratio=0.1, random_seed=10):
    # Randomly get dev set, and compute the ratio of median logdet and median quality as the adaptive alpha. 
    # Note that we do not use balanced split here as we want the dev set to reflect the real distribution of the data.
    dev_df, test_df = split_balanced_data(image_df, dev_ratio, random_seed, balanced=False)
    quality_values = dev_df[quality_col]
    logdet_values = dev_df[logdet_col]
    return np.abs(logdet_values.median() / quality_values.median())

###### UMPIRE implementation ######
def get_logdet_term(sample, jitter=1e-8):
    embeddings = get_generation_embeddings(sample) # shape: (k, embedding_dim)
    k = embeddings.shape[0]
    kernel = np.dot(embeddings, embeddings.T) # shape: (k, k)
    return 1/(2 * k) * fast_logdet(kernel + np.identity(kernel.shape[0])*jitter)

def compute_umpire(sample, jitter=1e-8, alpha=1, length_normalize=False):
    # get embedding and log-likelihoods
    raw_llh = sample['generations_log_likelihood'] # [[<token1_prob>, <token2_prob>, ...], ...]
    k = len(raw_llh) # number of generations
    # Logdet term
    logdet = get_logdet_term(sample, jitter=jitter)
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

###### Semantic Entropy ######
# Adapted from https://github.com/lorenzkuhn/semantic_uncertainty
def compute_semantic_entropy_from_scratch(sample, entailment_model):
    # Get semantic clusters
    cluster_ids = get_semantic_ids(
        strings_list=sample['generations_text'], 
        model=entailment_model, 
        strict_entailment=True,
        example=None # as Deberta model don't need this.
    )
    # Sum log-likelihoods for each token sequence
    llh_sums = [np.sum(llh) for llh in sample['generations_log_likelihood']]
    # Aggregate by cluster
    log_likelihood_by_cluster = logsumexp_by_id(cluster_ids, llh_sums)
    # Compute Semantic Entropy
    semantic_entropy_score = predictive_entropy_rao(log_likelihood_by_cluster)
    return semantic_entropy_score

def compute_semantic_entropy_from_cluster_ids(sample):
    # Get semantic clusters from pre-computed cluster ids in the sample
    cluster_ids = sample['cluster_ids']
    # Sum log-likelihoods for each token sequence
    llh_sums = [np.sum(llh) for llh in sample['generations_log_likelihood']]
    # Aggregate by cluster
    log_likelihood_by_cluster = logsumexp_by_id(cluster_ids, llh_sums)
    # Compute Semantic Entropy
    semantic_entropy_score = predictive_entropy_rao(log_likelihood_by_cluster)
    return semantic_entropy_score

def update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=15, eval_col='rougeL_to_target', eval_thresold=0.8, conf_col_to_eval_list=[], unc_col_to_eval_list=[]):
    if eval_col == "exact_match":
        image_correct_df = image_df.loc[image_df[eval_col] == 1]
        image_wrong_df = image_df.loc[image_df[eval_col] == 0]
    else:
        image_df['is_correct'] = image_df[eval_col] >= eval_thresold
        image_correct_df = image_df.loc[image_df['is_correct'] == True]
        image_wrong_df = image_df.loc[image_df['is_correct'] == False]
        eval_col = 'is_correct'

    result_dict = {}
    for col in conf_col_to_eval_list + unc_col_to_eval_list:
        if col in conf_col_to_eval_list:
            auc = ROC_AUROC(image_wrong_df[col], image_correct_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece', is_uncertainty=False)
            tpr_at_10_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=False, eval_col=eval_col)
            pearsonr = -compute_pearsonr(image_df[col], image_df[eval_col], num_bins=cpc_num_bins)[0]
        else:
            auc = ROC_AUROC(image_correct_df[col], image_wrong_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece')
            tpr_at_10_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=True, eval_col=eval_col)
            pearsonr = compute_pearsonr(image_df[col], image_df[eval_col], num_bins=cpc_num_bins)[0]
            
        result_dict[col] = {
            'auc': auc,
            'cece': cece,
            'pearsonr': pearsonr, 
            'tpr_at_0.1_fpr': tpr_at_10_fpr,
            'tpr_at_0.01_fpr': tpr_at_1_fpr,
            'aurac': aurac
        }
    return result_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_file', type=str, required=True,
                        help='Path to the generation file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output files')
    parser.add_argument('--jitter', type=float, default=1e-8,
                        help='Jitter value for numerical stability in logdet computation')
    parser.add_argument('--re_cluster_semantic_entropy', action='store_true',
                        help='Whether to re-cluster the generation responses for Semantic Entropy computation. Take note that this will cost few hours to run')
    args = parser.parse_args()

    print("Loading generation file from", args.generation_file, "...")
    # Load generation file
    file_path = args.generation_file
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as r:
            llava_results = pickle.load(r)
    else:
        raise FileNotFoundError(f"Generation file {file_path} not found.")
    image_df = pd.DataFrame().from_dict(llava_results)
    print("Generation file loaded. Number of samples:", len(image_df))

    # Ensure the 'embedding' column exists
    if 'internal_embedding' in image_df.columns:
        image_df = image_df.rename(columns={'internal_embedding': 'embedding'})
    if 'embedding' not in image_df.columns:
        raise ValueError("The 'embedding' column is missing from the DataFrame.")

    ### Compute Uncertainty Metrics ###
    # Normalize embeddings first
    image_df['norm_embedding'] = image_df['embedding'].apply(normalize_embedding)

    # Compute adaptive alpha on calibration set
    image_df['logdet'] = image_df.apply(lambda x: get_logdet_term(x, jitter=args.jitter), axis=1)
    image_df['quad_entropy'] = image_df['generations_log_likelihood'].apply(lambda llh: get_quad_entropy(llh))
    adaptive_alpha = get_adaptive_alpha_dev_set(image_df, logdet_col='logdet', quality_col='quad_entropy', dev_ratio=0.1, random_seed=10)
    print("Adaptive alpha", adaptive_alpha)

    # Compute UMPIRE with adaptive alpha
    tqdm.pandas(desc="Computing UMPIRE")
    image_df['umpire'] = image_df.progress_apply(lambda x: compute_umpire(x, alpha=adaptive_alpha, jitter=args.jitter), axis=1, ) 

    # Compute baselines
    ## length-normalized entropy and eigenscore
    tqdm.pandas(desc="Computing length-normalized entropy")
    image_df['ln_entropy'] = image_df['generations_log_likelihood'].progress_apply(get_normalized_entropy)
    tqdm.pandas(desc="Computing eigenscore")
    image_df['eigen_score'] = image_df.progress_apply(lambda x: compute_eigenscore(x, jitter=args.jitter), axis=1)
    ## Semantic Entropy
    from modules.semantic_entropy import get_semantic_ids, logsumexp_by_id, predictive_entropy_rao    
    if args.re_cluster_semantic_entropy or 'cluster_ids' not in image_df.columns:
        if args.re_cluster_semantic_entropy:
            print("Re-clustering flag on")
        elif 'cluster_ids' not in image_df.columns:
            print("No pre-computed cluster ids found")
        print("Re-clustering generations for semantic entropy computation...")
        from modules.semantic_entropy import EntailmentDeberta
        entailment_model = EntailmentDeberta()
        tqdm.pandas(desc="Computing semantic entropy")
        image_df['semantic_entropy'] = image_df.progress_apply(lambda x: compute_semantic_entropy_from_scratch(x, entailment_model), axis=1)
    else:
        print("Using pre-computed cluster ids for semantic entropy computation...")
        tqdm.pandas(desc="Computing semantic entropy from pre-computed cluster ids")
        image_df['semantic_entropy'] = image_df.progress_apply(lambda x: compute_semantic_entropy_from_cluster_ids(x), axis=1)

    ### Evaluate Uncertainty Metrics ###
    unc_metrics = ['ln_entropy', 'semantic_entropy', 'eigen_score', 'umpire']
    result_dict = update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=50, unc_col_to_eval_list=unc_metrics)
    result_df = pd.DataFrame().from_dict(result_dict, orient='index')
    result_df = result_df.map(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)
    print(df_to_markdown_bold(result_df))

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save result_df as json
    result_json_file = os.path.join(args.output_dir, 'umpire_results.json')
    result_df.to_json(result_json_file, orient='index', indent=4)

    # # Save the updated DataFrame with uncertainty metrics
    # image_df.to_pickle(os.path.join(args.output_dir, 'image_df_with_uncertainty.pkl'))
