import os
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# import dotenv; dotenv.load_dotenv("/home/daohieu/maplecg_nfs/research/VLM/.env")
import sys; sys.path.append(".")

from modules.eval_utils import ROC_AUROC, compute_pearsonr, get_calibrate_ece, get_tpr_at_fpr, compute_aurac_from_image_df, \
    df_to_markdown_bold

from modules.logdet_utils import normalize_embedding, get_normL1_prob, compute_logdet, \
    compute_probL1_logdet, get_normalized_entropy, compute_eigenscore


parser = argparse.ArgumentParser()
parser.add_argument('--generation_file', type=str, required=True,
                    help='Path to the generation file')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to save the output files')
parser.add_argument('--jitter', type=float, default=1e-8,
                    help='Jitter value for numerical stability in logdet computation')

# parser.add_argument('')

args = parser.parse_args()

file_path = args.generation_file
if os.path.isfile(file_path):
    with open(file_path, 'rb') as r:
        llava_results = pickle.load(r)
image_df = pd.DataFrame().from_dict(llava_results)

# Ensure the 'embedding' column exists
if 'internal_embedding' in image_df.columns:
    image_df = image_df.rename(columns={'internal_embedding': 'embedding'})
if 'embedding' not in image_df.columns:
    raise ValueError("The 'embedding' column is missing from the DataFrame.")

### Compute Uncertainty Metrics ###
# Normalize embeddings
image_df['norm_embedding'] = image_df['embedding'].apply(normalize_embedding)

# compute logdet
image_df['logdet'] = image_df['norm_embedding'].apply(lambda x: compute_logdet(np.matmul(x, x.T), alpha=args.jitter))

# Compute adaptive prob_alpha
prob_values = image_df['generations_log_likelihood'].apply(get_normL1_prob)
logdet_values = image_df['logdet']
prob_alpha = np.abs(logdet_values.median() / prob_values.median())
prob_param = prob_alpha
print("apdative prob alpha", prob_param)

# Compute probL1_logdet
image_df['umpire'] = image_df.apply(lambda x: compute_probL1_logdet(x, alpha=prob_param), axis=1)  

### Evaluate Uncertainty Metrics ###
unc_col_to_eval_list = ['umpire']
conf_col_to_eval_list = []
def update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=15, eval_col='exact_match'):
    image_correct_df = image_df.loc[image_df[eval_col] == 1]
    image_wrong_df = image_df.loc[image_df[eval_col] == 0]

    result_dict = {}
    for col in conf_col_to_eval_list + unc_col_to_eval_list:
        if col in conf_col_to_eval_list:
            auc = ROC_AUROC(image_wrong_df[col], image_correct_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece', is_uncertainty=False)
            tpr_at_10_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=False, eval_col=eval_col)
        else:
            auc = ROC_AUROC(image_correct_df[col], image_wrong_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece')
            tpr_at_10_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=True, eval_col=eval_col)
        pearsonr = np.abs(compute_pearsonr(image_df[col], image_df[eval_col], num_bins=cpc_num_bins)[0])
        result_dict[col] = {
            'auc': auc,
            'cece': cece,
            'pearsonr': pearsonr, 
            'tpr_at_0.1_fpr': tpr_at_10_fpr,
            'tpr_at_0.01_fpr': tpr_at_1_fpr,
            'aurac': aurac
        }
    return result_dict

result_dict = update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=15)
result_df = pd.DataFrame().from_dict(result_dict, orient='index')
result_df = result_df.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)
print(df_to_markdown_bold(result_df))

# Save results
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# save result_df as json
result_json_file = os.path.join(args.output_dir, 'umpire_results.json')
result_df.to_json(result_json_file, orient='index', indent=4)

# # Save the updated DataFrame with uncertainty metrics
# image_df.to_pickle(os.path.join(args.output_dir, 'image_df_with_uncertainty.pkl'))
