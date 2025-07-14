import numpy as np
import math
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.utils.extmath import fast_logdet

from scipy.spatial.distance import pdist, squareform
from numpy.linalg import det
from numpy.linalg import norm

def normalize_embedding(x):
    return np.array([e / np.linalg.norm(e, ord=2) for e in x])

def get_predictive_entropy(x):
    x_ = [-np.sum(np.exp(i)*i)for i in x] # sum(p*log(p))
    return np.average(x_)
    
def get_avg_length_response(x):
     x_ = np.array([len(i) for i in x])
     return x_.mean()

def get_normL2_prob(x):
    # x_ = [np.exp(np.sum(i))/len(i) for i in x] # p_sequence / len(sequence)
    x_ = 1-np.array([np.exp(np.sum(i)) for i in x]) # 1-p_sequence
    return np.linalg.norm(x_, ord=2)

def get_normL1_prob(x):
    # x_ = np.array([np.exp(np.sum(i))/len(i) for i in x]) # 1-p_sequence
    x_ = 1-np.array([np.exp(np.sum(i)) for i in x]) # 1-p_sequence
    return np.linalg.norm(x_, ord=1) 

def get_normLk_prob(x, k=1):
    x_ = 1-np.array([np.exp(np.sum(i)) for i in x]) # 1-p_sequence
    return np.linalg.norm(x_, ord=k) 

def get_powerk_prob(x, k=1):
    x_ = 1-np.array([np.exp(np.sum(i)) for i in x]) # 1-p_sequence
    return np.sum(x_**k)

# Length-normalize
def get_normL1_prob_LN(x):
    x_ = 1-np.array([np.exp(np.sum(i)) / len(i) for i in x]) # 1-(p_sequence/len(sequence))
    return np.linalg.norm(x_, ord=1) 

def get_powerk_prob_LN(x, k=1):
    x_ = 1-np.array([np.exp(np.sum(i)) / len(i) for i in x]) # 1-p_sequence / len(sequence)
    return np.sum(x_**k)

###########
def get_normL1_logprob(x):
    # x_ = [np.sum(i) / len(i) for i in x] # p_sequence
    x_ = [np.sum(i) for i in x] # p_sequence
    return np.linalg.norm(x_, ord=1)

def get_normLk_logprob(x, k=1):
    # x_ = [np.sum(i) / len(i) for i in x] # p_sequence
    x_ = [np.sum(i) for i in x] # p_sequence
    return np.linalg.norm(x_, ord=k)

def get_normL2_logprob(x):
    x_ = [np.sum(i) for i in x] # p_sequence
    return np.linalg.norm(x_, ord=2)
###########

def get_normalized_entropy(x):
    x_ = [-np.sum(np.exp(i)*i) *(1/len(i)) for i in x] # sum(p*log(p)) / len(sequence)
    return np.average(x_)

# RBF Kernel function
def rbf_kernel(X, gamma):
    kernel = RBF(length_scale=gamma)
    return kernel(X)

def rbf_kernel_np(X, gamma):
    # Compute the pairwise Euclidean distances in the data
    pairwise_dists = squareform(pdist(X, 'euclidean')) ** 2
    # Compute the RBF kernel (covariance matrix)
    K = np.exp(-gamma * pairwise_dists)
    return K

def marten_kernel(X, gamma, nu):
    kernel = Matern(length_scale=gamma, nu=nu)
    K = kernel(X)
    return K

# Compute volume of ellipsoid (proportional to sqrt(det(K)))
def compute_ellipsoid_volume(K):
    # Compute the determinant of the covariance matrix
    det_K = det(K)
    # Volume is proportional to the square root of the determinant
    volume = np.sqrt(det_K)
    # volume = compute_logdet(K)
    return volume

def compute_logdet(K, alpha=1e-8):
    # seed = np.random.rand()
    logdet_value = fast_logdet(K + np.identity(K.shape[0])*alpha)
    return logdet_value

## For pandas DataFrame rows
def compute_single_prob(row):
    log_likelihoods = row['most_likely_generation_log_likelihood']
    return 1-np.exp(np.sum(log_likelihoods))

def compute_logprobL1_logdet(row, alpha=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL1_logprob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_logprobLk_logdet(row, alpha=1, k=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normLk_logprob(log_likelihoods, k=k)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_logprobL1_logdet_rbf(row, alpha=1):
    ori_logdet = row['logdet_rbf']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL1_logprob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_logprobL2_logdet(row, alpha=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL2_logprob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_logprobL2_logdet_rbf(row, alpha=1):
    ori_logdet = row['logdet_rbf']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL2_logprob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probL1_logdet(row, alpha=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL1_prob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probLk_logdet(row, alpha=1, k=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normLk_prob(log_likelihoods, k=k)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_prob_powerk_logdet(row, alpha=1, k=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_powerk_prob(log_likelihoods, k=k)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probL1_logdet_rbf(row, alpha=1):
    ori_logdet = row['logdet_rbf']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL1_prob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probL2_logdet(row, alpha=1):
    ori_logdet = row['logdet']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL2_prob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probL2_logdet_rbf(row, alpha=1):
    ori_logdet = row['logdet_rbf']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL2_prob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probL1_logdet_marten(row, alpha=1):
    ori_logdet = row['logdet_marten']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL1_prob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_probL2_logdet_marten(row, alpha=1):
    ori_logdet = row['logdet_marten']
    log_likelihoods = row['generations_log_likelihood']
    u = get_normL2_prob(log_likelihoods)
    V_tilde = ori_logdet + alpha * u
    return V_tilde

def compute_eigenscore(row, jitter = 1e-3):
    embedding = row['norm_embedding']
    CovMatrix = np.cov(embedding)
    # CovMatrix = np.matmul(embedding, embedding.T)
    u, s, vT = np.linalg.svd(CovMatrix+jitter*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator

def compute_eigenscore_gram(row, jitter = 1e-3):
    embedding = row['norm_embedding']
    # CovMatrix = np.cov(embedding)
    CovMatrix = np.matmul(embedding, embedding.T)
    u, s, vT = np.linalg.svd(CovMatrix+jitter*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator