#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from scipy.stats import norm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %pip install einops
# from einops import einsum


# In[ ]:


def get_ground_truth_feats(
    distrb_type: str = 'unif',
    model_dims: int = 256,
    feature_dims: int = 512,
    device: torch.device = torch.device('cpu'),
    num_clusters: int = 10,
) -> torch.Tensor:
    dtype = torch.float32

    if distrb_type == 'unif':
        synth_features = torch.randn(model_dims, feature_dims, device=device, dtype=dtype)
    else:
        cluster_data, _ = make_blobs(n_samples=model_dims, centers=num_clusters, n_features=feature_dims)
        synth_features = torch.tensor(cluster_data, device=device, dtype=dtype)
    return synth_features


# In[ ]:


def get_synth_actvs(
        synth_features: torch.Tensor,
        total_data_points: int = 100000,
        avg_active_features: int = 16,
        batch_size: int = 1000
    ) -> torch.Tensor:
    device = synth_features.device
    dtype = torch.float32

    h = synth_features.shape[0]  # model dimensions
    G = synth_features.shape[1]  # number of ground truth features

    # created a random covariance matrix for a multivariate normal distribution with zero mean
    A = np.random.rand(G, G)  # rand correlations for each feature
    cov_matrix = np.dot(A, A.transpose())
    mean = np.zeros(G)

    synth_actvs_batches = []
    for _ in tqdm(range(0, total_data_points, batch_size), desc="Generating Batches"):
        # 1. Correlated: for each feature in sample vector of size G
        # a single sample from a correlated multivariate normal distribution and,
        batch_size_current = min(batch_size, total_data_points - len(synth_actvs_batches))
        # samples are on a scale defined by the normal distribution's probability density fn (PDF)
        samples = np.random.multivariate_normal(mean, cov_matrix, batch_size_current) # (batchNumSamps, G) with correlations

        # for each dimension of that sample, found where that sample lay on the standard normal cumulative distribution function
        uniform_samples = norm.cdf(samples)  # (batchNumSamps, G) is where each samp lies on (0,1) range in cumulative dist fn (CDF)

        # 2. Decayed: for each feature in sample vector of size G
        # probability of the G-dimensional random variable exponentially decayed with the feature’s index
        decay_rate = 0.99  # lambda (put this in loop for ease of code reading)
        indices = np.arange(G)
        decayed_probs = uniform_samples ** (indices * decay_rate)  # prob of each samp's feature expo decays to power of ind*0.99

        # 3. Rescaled: for each feature in sample vector of size G
        # Rescale probabilities to ensure on avg only "avg_active_features" num of ground truth features are active at a time.
        # this changes the avg so (avg_active_features / G) are active
        # scaling_factor: denom is what to cancel out (replace) and numer is what to replace with
        mean_prob = np.mean(decayed_probs) #  calculated the mean probability of all features
        scaling_factor = (avg_active_features / G) / mean_prob #  calculated the ratio of the number of ground truth features that are active at a time to the mean probability
        rescaled_probs = decayed_probs * scaling_factor # multiplied each probability by this ratio to rescale them
        rescaled_probs_tensor = torch.tensor(rescaled_probs, device=device, dtype=dtype)

        # 4. parameterize a vector of Bernoulli random variables (for sparse coefficients):
        # want expectation of this vector to have "avg_active_features" 1s
        # given probs for each index, bernoulli draws a vector of 0s and 1s using those probs
        binary_sparse_coeffs = torch.bernoulli(rescaled_probs_tensor)

        # 5. use the sparse coefficients to linearly combine a sparse selection of the ground truth features
        synth_activations = torch.matmul(binary_sparse_coeffs, synth_features.T.to(dtype))
        # synth_activations = torch.einsum('ij,kj->ik', binary_sparse_coeffs, synth_features.to(dtype))

        synth_actvs_batches.append(synth_activations)

    return torch.cat(synth_actvs_batches, dim=0)  # stack batches along rows (dim=0)


# In[ ]:


grTrue_feats = get_ground_truth_feats('unif', 256, 512, device) # hxG
total_data_points = 10000000
avg_active_features = 32

synth_activations = get_synth_actvs(grTrue_feats, total_data_points, avg_active_features)
print('\n', synth_activations.shape)

