from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import math
import uncertainty_metrics.numpy as um

from scipy.stats import pointbiserialr

def compute_pcc_biserial(x, y):
    return pointbiserialr(x, y)[0]
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def ROC_AUROC(ori_z, other_z):
    ori_z = ori_z[~np.isnan(ori_z)]
    other_z = other_z[~np.isnan(other_z)]
    labels = np.concatenate([np.zeros(len(ori_z), dtype=bool), np.ones(len(other_z), dtype=bool)])
    pred = np.concatenate([np.array(ori_z), np.array(other_z)])
    fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
    AUROC = metrics.auc(fpr, tpr)
    return fpr, tpr, AUROC #, zs, F1

def AUCPR(ori_z, other_z):
    ori_z = ori_z[~np.isnan(ori_z)]
    other_z = other_z[~np.isnan(other_z)]
    labels = np.concatenate([np.zeros(len(ori_z), dtype=bool), np.ones(len(other_z), dtype=bool)])
    pred = np.concatenate([np.array(ori_z), np.array(other_z)])
    precision, recall, threshold = metrics.precision_recall_curve(labels, pred)
    AUCPR = metrics.auc(recall, precision)
    return recall, precision, AUCPR #, zs, F1

def average_precision(ori_z, other_z):
    ori_z = ori_z[~np.isnan(ori_z)]
    other_z = other_z[~np.isnan(other_z)]
    labels = np.concatenate([np.zeros(len(ori_z), dtype=bool), np.ones(len(other_z), dtype=bool)])
    pred = np.concatenate([np.array(ori_z), np.array(other_z)])
    AP = metrics.average_precision_score(labels, pred)
    return AP

def plot_dist_auc(correct_unc, wrong_unc):
    plt.figure()
    bins = 41
    for data, c, l in zip((correct_unc, wrong_unc), 
                    ('black','g'),
                    ("correct", "wrong")):
        plt.hist(
            data,
            # range = rng, 
            bins = bins, 
            density = 1, 
            alpha = 0.2, 
            color = c,
            lw = 2,
            label = l
            )
    plt.xlabel("uncertainty")
    plt.legend()
    plt.show()

def plot_auc(correct_unc, wrong_unc):
    plt.figure()
    fpr, tpr, auc = ROC_AUROC(correct_unc, wrong_unc)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

def plot_linear_correlation(threshold_list, acc_list, method_name=None, color='blue'):
    plt.figure()
    # plt.scatter(threshold_list, acc_list, label='Data Points')
    X = np.array(threshold_list).reshape(-1, 1)
    y = np.array(acc_list)
    r, p_value = pearsonr(X.flatten(), y)
    print("Pearson: ", r)
    if method_name == None:
        label = f"Linear Fit - Pearson {r:.3}"
    else:
        label = f"{method_name} - Pearson {r:.3}"
    sns.regplot(x=X, y=y, scatter=True, color=color, line_kws={'label':label}, scatter_kws={'s':10})
    plt.xlabel('Uncertainty Score (lower is better)')
    plt.ylabel('Accuracy Score')
    # plt.ylim(0.3, 1)
    plt.title('Accuracy vs Uncertainty correlation')
    plt.legend()
    plt.show()

def plot_pearson(uncertainty_scores, accuracies, num_bins=15):
    # Convert lists to numpy arrays
    uncertainty_scores = np.array(uncertainty_scores)
    accuracies = np.array(accuracies)
    
    # Initialize lists to hold accuracy and thresholds
    acc_list = []
    threshold_list = []

    # Determine uncertainty bins
    min_unc = uncertainty_scores.min()
    max_unc = uncertainty_scores.max()
    bin_boundaries = np.linspace(min_unc, max_unc, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_lowers[0] = min_unc - 1e-5
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of samples that fall within the current bin
        indices = np.where((uncertainty_scores > bin_lower) & (uncertainty_scores <= bin_upper))[0]
        if len(indices) > 0:
            # Calculate accuracy within this threshold
            acc = np.sum(accuracies[indices] == 1) / len(indices)
            acc_list.append(acc)
            threshold_list.append(bin_lower)
    
    # Convert threshold and accuracy lists to numpy arrays for correlation calculation
    X = np.array(threshold_list)
    y = np.array(acc_list)

    plot_linear_correlation(X, y)

def compute_pearsonr_cum(uncertainty_scores, accuracies, num_bins=15):
    """
    Compute the Pearson correlation coefficient between uncertainty thresholds and accuracy.

    Args:
        uncertainty_scores (list): List of uncertainty scores.
        accuracies (list): List of binary accuracy values (1 for correct, 0 for incorrect).
        num_bins (int): Number of bins to divide the uncertainty scores into.

    Returns:
        float: Pearson correlation coefficient.
        float: p-value associated with the Pearson correlation.
    """
    # Convert lists to numpy arrays
    uncertainty_scores = np.array(uncertainty_scores)
    accuracies = np.array(accuracies)
    
    # Initialize lists to hold accuracy and thresholds
    acc_list = []
    threshold_list = []

    # Determine uncertainty bins
    min_unc = uncertainty_scores.min()
    max_unc = uncertainty_scores.max()
    unc_bins = np.linspace(min_unc, max_unc, num_bins)
    
    # Calculate accuracy for each bin
    for threshold in unc_bins:
        # Get indices of entries below the current threshold
        indices = np.where(uncertainty_scores < threshold)[0]
        if len(indices) > 0:
            # Calculate accuracy within this threshold
            acc = np.sum(accuracies[indices] == 1) / len(indices)
            acc_list.append(acc)
            threshold_list.append(threshold)
    
    # Convert threshold and accuracy lists to numpy arrays for correlation calculation
    X = np.array(threshold_list)
    y = np.array(acc_list)
    
    # Calculate Pearson correlation
    r, p_value = pearsonr(X, y)
    
    return r, p_value

def compute_pearsonr(uncertainty_scores, accuracies, num_bins=15):
    """
    Compute the Pearson correlation coefficient between uncertainty thresholds and accuracy.

    Args:
        uncertainty_scores (list): List of uncertainty scores.
        accuracies (list): List of binary accuracy values (1 for correct, 0 for incorrect).
        num_bins (int): Number of bins to divide the uncertainty scores into.

    Returns:
        float: Pearson correlation coefficient.
        float: p-value associated with the Pearson correlation.
    """
    # Convert lists to numpy arrays
    uncertainty_scores = np.array(uncertainty_scores)
    accuracies = np.array(accuracies)
    
    # Initialize lists to hold accuracy and thresholds
    acc_list = []
    threshold_list = []

    # Determine uncertainty bins
    min_unc = uncertainty_scores.min()
    max_unc = uncertainty_scores.max()
    bin_boundaries = np.linspace(min_unc, max_unc, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_lowers[0] = min_unc - 1e-5
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of samples that fall within the current bin
        indices = np.where((uncertainty_scores > bin_lower) & (uncertainty_scores <= bin_upper))[0]
        if len(indices) > 0:
            # Calculate accuracy within this threshold
            acc = np.sum(accuracies[indices] == 1) / len(indices)
            acc = 1 - acc # Invert accuracy to match uncertainty (lower uncertainty should mean higher accuracy)
            acc_list.append(acc)
            threshold_list.append(bin_lower)
    
    # Convert threshold and accuracy lists to numpy arrays for correlation calculation
    X = np.array(threshold_list)
    y = np.array(acc_list)
    
    # Calculate Pearson correlation
    r, p_value = pearsonr(X, y)
    
    return r, p_value



def compute_spearmanr(uncertainty_scores, accuracies, num_bins=15):
    # Convert lists to numpy arrays
    uncertainty_scores = np.array(uncertainty_scores)
    accuracies = np.array(accuracies)
    
    # Initialize lists to hold accuracy and thresholds
    acc_list = []
    threshold_list = []

    # Determine uncertainty bins
    min_unc = uncertainty_scores.min()
    max_unc = uncertainty_scores.max()
    bin_boundaries = np.linspace(min_unc, max_unc, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_lowers[0] = min_unc - 1e-5
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of samples that fall within the current bin
        indices = np.where((uncertainty_scores > bin_lower) & (uncertainty_scores <= bin_upper))[0]
        if len(indices) > 0:
            # Calculate accuracy within this threshold
            acc = np.sum(accuracies[indices] == 1) / len(indices)
            acc_list.append(acc)
            threshold_list.append(bin_lower)
    
    # Convert threshold and accuracy lists to numpy arrays for correlation calculation
    X = np.array(threshold_list)
    y = np.array(acc_list)
    
    
    # Calculate Pearson correlation
    r, p_value = spearmanr(X, y)
    
    return r, p_value


def scale_to_01(arr):
    arr_min = np.min(arr)  # Minimum value of the array
    arr_max = np.max(arr)  # Maximum value of the array
    
    # Avoid division by zero if all values in the array are the same
    if arr_max - arr_min == 0:
        return np.zeros(arr.shape)
    
    # Apply the Min-Max scaling formula
    scaled_arr = (arr - arr_min) / (arr_max - arr_min)

    # # Sigmoid
    # scaled_arr = 1 / (1 + np.exp(-scaled_arr))
    
    return scaled_arr
    
def compute_ece(scores, labels, n_bins=15, is_uncertainty=True):
    labels = np.array(labels, dtype=int)
    scores = scale_to_01(scores)
    # scores = 1 / (1 + np.exp(-scores))
    if is_uncertainty:
        probabilities = 1 - np.array(scores)
        # probabilities = scores
    else:
        probabilities = scores
        # probabilities = 1 - np.array(scores)
    return um.ece(labels=labels, probs=probabilities, num_bins=n_bins)

    
def eval_uncertainty_score(outputs, uncertainty_score_list, num_bins=15):
    exact_list = [x['exact_match'] for x in outputs]
    tmp_df = pd.DataFrame(zip(exact_list, uncertainty_score_list), columns=['acc', 'unc'])
    correct_df = tmp_df.loc[tmp_df['acc'] == 1]
    wrong_df = tmp_df.loc[tmp_df['acc'] == 0]
    correct_unc = correct_df['unc'].to_numpy()
    wrong_unc = wrong_df['unc'].to_numpy()
    acc = len(correct_df)/len(tmp_df)
    auc = ROC_AUROC(correct_unc, wrong_unc)[-1]
    print("ACC: ", acc)
    print("AUROC: ", auc)
    plot_dist_auc(correct_unc, wrong_unc)
    plot_auc(correct_unc, wrong_unc)
    plot_pearson(tmp_df, num_bins)
    return acc, auc

def compute_metric_eval(outputs, uncertainty_score_list):
    exact_list = [x['exact_match'] for x in outputs]
    tmp_df = pd.DataFrame(zip(exact_list, uncertainty_score_list), columns=['acc', 'unc'])
    correct_df = tmp_df.loc[tmp_df['acc'] == 1]
    wrong_df = tmp_df.loc[tmp_df['acc'] == 0]
    correct_unc = correct_df['unc'].to_numpy()
    wrong_unc = wrong_df['unc'].to_numpy()
    acc = len(correct_df)/len(tmp_df)
    auc = ROC_AUROC(correct_unc, wrong_unc)[-1]
    pearsonr, _ = compute_pearsonr(tmp_df['unc'], tmp_df['acc'])
    ece = compute_ece(tmp_df['unc'], tmp_df['acc'])
    return acc, auc, pearsonr, ece

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
import uncertainty_metrics.numpy as um
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def bin_unc_and_acc(image_df, unc_column, num_bins=10):
    # Determine uncertainty bins
    min_unc = image_df[unc_column].min()
    max_unc = image_df[unc_column].max()
    bin_boundaries = np.linspace(min_unc, max_unc, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_lowers[0] = min_unc - 1e-5

    acc_list = []
    threshold_list = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of samples that fall within the current bin
        sliced_df = image_df.loc[(image_df[unc_column] > bin_lower) & (image_df[unc_column] <= bin_upper)]
        if len(sliced_df) > 0:
            # Calculate accuracy within this threshold
            acc = sliced_df['exact_match'].mean()
            acc_list.append(acc)
            threshold_list.append(bin_lower)
    return threshold_list, acc_list

def split_balanced_data(image_df, calibration_ratio, random_seed, eval_col='exact_match', balanced=True):
    if balanced:
        # Step 1: Separate correct and incorrect answers
        correct_df = image_df.loc[image_df[eval_col] == 1]
        incorrect_df = image_df.loc[image_df[eval_col] == 0]
        # Step 2: Shuffle and sample from both subsets
        num_samples = int(len(image_df) * calibration_ratio)
        correct_ratio = len(correct_df) / len(image_df)
        num_correct_samples = int(num_samples * correct_ratio)
        num_incorrect_samples = num_samples - num_correct_samples
        dev_correct_df = correct_df.sample(n=num_correct_samples, random_state=random_seed)
        dev_incorrect_df = incorrect_df.sample(n=num_incorrect_samples, random_state=random_seed)
        # Step 3: Combine sampled correct and incorrect answers for dev set
        dev_df = pd.concat([dev_correct_df, dev_incorrect_df]).reset_index(drop=True)
        # Step 4: Create the test set by excluding the dev set
        test_df = image_df.drop(dev_df.index).reset_index(drop=True)
    else:
        # Step 1: Shuffle the entire dataset
        image_df = image_df.sample(n=len(image_df), random_state=random_seed).reset_index(drop=True)
        # Step 2: Calculate the number of samples for the dev set
        num_samples = int(len(image_df) * calibration_ratio)
        # Step 3: Split the dataset into dev and test sets
        dev_df = image_df.iloc[:num_samples].reset_index(drop=True)
        test_df = image_df.iloc[num_samples:].reset_index(drop=True)
    return dev_df, test_df

def get_calibrate_ece(image_df, unc_column, eval_col='exact_match', num_bins=15, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece', is_uncertainty=True, num_trail=1):
    # split into dev set and test set with balance correct and wrong answers
    random_seed_list = [random_seed + i for i in range(num_trail)]
    metric_list = []
    for i in range(num_trail):
        dev_df, test_df = split_balanced_data(image_df, calibration_ratio, random_seed_list[i], balanced=False)

        X_dev = dev_df[[unc_column]].to_numpy()
        x_dev, y_dev = bin_unc_and_acc(dev_df, unc_column, num_bins=num_bins)
        x_dev = np.array(x_dev).reshape(-1, 1)
        x_test = test_df[[unc_column]].to_numpy().reshape(-1, 1)

        if model_type == 'logistic':
            model = LogisticRegression(random_state=random_seed)
            model.fit(x_dev, y_dev)
            test_df['u_score'] = model.predict_proba(x_test)[:, 1]
        elif model_type == 'linear':
            model = LinearRegression()
            model.fit(x_dev, y_dev)
            test_df['u_score'] = model.predict(x_test)
        elif model_type == 'minmax':
            minmax_scaler = MinMaxScaler() # default 0-1
            X_dev = minmax_scaler.fit(X_dev)
            X_test = minmax_scaler.transform(test_df[[unc_column]].to_numpy())
            if is_uncertainty:
                test_df['u_score'] = 1-X_test
            else:
                test_df['u_score'] = X_test
        elif model_type == 'isotonic':
            # if is_uncertainty:
            #     iso_reg = IsotonicRegression(increasing=False).fit(x_dev, y_dev)
            # else:
            #     iso_reg = IsotonicRegression(increasing=True).fit(x_dev, y_dev)
            iso_reg = IsotonicRegression(increasing='auto', y_max=1, y_min=0, out_of_bounds='clip').fit(x_dev, y_dev)
            test_df['u_score'] = iso_reg.predict(x_test)
        else:
            print("No model")

        if ece_mode == 'ece':
            metric = um.ece(probs=test_df['u_score'], labels=test_df[eval_col].astype(int), num_bins=num_bins)
        elif ece_mode == 'ace':
            metric = um.ace(probs=test_df['u_score'], labels=test_df[eval_col].astype(int), num_bins=num_bins)
        metric_list.append(metric)
    if num_trail == 1:
        return metric_list[0]
    else:
        return metric_list
    
def get_tpr_at_fpr(a, b, fpr_threshold=0.1):
    fpr, tpr, auc = ROC_AUROC(a, b)
    index = np.abs(fpr - fpr_threshold).argmin()
    # idx = np.where(fpr >= fpr_threshold)[0][0]
    return tpr[index]

# AUC-ARC 
def create_selective_plot(df, unc_col, ascending=False, eval_col='exact_match'):
    sort_df = df.sort_values(by=unc_col, ascending=ascending).reset_index(drop=True) # ascending=False
    sort_df['percent_data_rejected'] = ((sort_df.index + 1) / len(sort_df)) * 100
    sort_df['cumulative_correct'] = sort_df[eval_col][::-1].cumsum()[::-1]
    sort_df['remaining_data_count'] = len(sort_df) - sort_df.index
    sort_df['accuracy'] = sort_df['cumulative_correct'] / sort_df['remaining_data_count']
    return sort_df

def smooth_plot(df, time_step=5):
    # Create a reference DataFrame with `percent_list` to match against
    percent_list = pd.DataFrame({'percent_data_rejected': np.arange(0, 100, time_step).astype(float)})
    # Sort both DataFrames by `percent_data_rejected` to enable merge_asof
    df_sorted = df.sort_values(by='percent_data_rejected').reset_index(drop=True)
    percent_list_sorted = percent_list.sort_values(by='percent_data_rejected').reset_index(drop=True)
    # Perform an asof merge to get the closest values from `percent_list`
    df_aligned = pd.merge_asof(percent_list_sorted, df_sorted, on='percent_data_rejected', direction='nearest')
    return df_aligned

from sklearn.metrics import auc
def compute_auc_arc(df):
    """
    Computes the Area Under the Accuracy-Reject Curve (AUC-ARC)
    given a DataFrame with 'percent_data_rejected' and 'accuracy' columns.
    """
    reject_rates = df['percent_data_rejected'].values / 100  # Normalize to [0,1]
    accuracies = df['accuracy'].values  # Accuracy values
    return auc(reject_rates, accuracies)

def compute_aurac_from_image_df(image_df, col, uncertainty=False, eval_col='exact_match'):
    method_df = create_selective_plot(image_df, unc_col=col, ascending=(not uncertainty), eval_col='exact_match')
    auc_arc = compute_auc_arc(method_df)
    return auc_arc

# Print DataFrame in Markdown format with bold values
def df_to_markdown_bold(df, index=True):
    df_markdown = df.copy()
    for col in df.columns:
        if df[col].dtype not in [float, int]:
            continue
        if col == 'ece' or col == 'ace' or col == 'cece':
            min_val = df[col].min()
            df_markdown[col] = df[col].apply(lambda x: f"\033[1m{x:.3f}\033[0m" if x == min_val else f'{x:.3f}')
        else:
            # max_val = df[col].max
            max_val = df[col].max()
            df_markdown[col] = df[col].apply(lambda x: f"\033[1m{x:.3f}\033[0m" if x == max_val else f'{x:.3f}')
    return df_markdown.to_markdown(index=index)

def df_bold_value(df, left_bold="\033[1m", right_bold="\033[0m", take_max=None):
    if take_max is None:
        for col in df.columns:
            if col == 'ece' or col == 'ace' or col == 'cece':
                min_val = df[col].min()
                df[col] = df[col].apply(lambda x: f"{left_bold}{x:.3f}{right_bold}" if x == min_val else f'{x:.3f}')
            else:
                max_val = df[col].max()
                df[col] = df[col].apply(lambda x: f"{left_bold}{x:.3f}{right_bold}" if x == max_val else f'{x:.3f}')
    else:
        if take_max == True:
            for col in df.columns:
                max_val = df[col].max()
                df[col] = df[col].apply(lambda x: f"{left_bold}{x:.3f}{right_bold}" if x == max_val else f'{x:.3f}')
        else:
            for col in df.columns:
                min_val = df[col].min()
                df[col] = df[col].apply(lambda x: f"{left_bold}{x:.3f}{right_bold}" if x == min_val else f'{x:.3f}')
    return df