import numpy as np
from sklearn.utils.extmath import fast_logdet


def normalize_embedding(x):
    return np.array([e / np.linalg.norm(e, ord=2) for e in x])

def get_generation_embeddings(sample):
    embeddings = np.asarray(sample['norm_embedding']) # take norm_embedding instead of embedding to avoid the need to normalize here
    k = len(sample['generations_log_likelihood'])
    if embeddings.shape[0] == k + 1:
        embeddings = embeddings[1:]
    return embeddings

def get_predictive_entropy(x):
    x_ = [-np.sum(np.exp(i)*i)for i in x] 
    return np.average(x_)

def get_quad_entropy(log_likelihoods, length_normalize=False):
    if length_normalize:
        prob = np.array([np.exp(np.sum(i) / len(i)) for i in log_likelihoods])
    else:
        prob = np.array([np.exp(np.sum(i)) for i in log_likelihoods])
    incoherence_scores = 1 - prob
    return np.sum(incoherence_scores) * (1/(len(log_likelihoods)))

def get_normalized_entropy(x):
    x_ = [-np.sum(np.exp(i)*i) *(1/len(i)) for i in x]
    return np.average(x_)

def compute_logdet(K, jitter=1e-8):
    # seed = np.random.rand()
    logdet_value = fast_logdet(K + np.identity(K.shape[0])*jitter)
    return logdet_value

# Compute the eigenvalue-based score, adapted from https://github.com/D2I-ai/eigenscore/blob/main/func/metric.py
def compute_eigenscore(row, jitter = 1e-3):
    embedding = get_generation_embeddings(row)
    CovMatrix = np.cov(embedding)
    # CovMatrix = np.matmul(embedding, embedding.T)
    u, s, vT = np.linalg.svd(CovMatrix+jitter*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator