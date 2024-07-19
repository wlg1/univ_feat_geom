#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


# # don't do this; transformerlens or saelens incompat w/ latest numpy?

# %%capture
# !pip install --upgrade numpy pandas
# !pip install umap-learn matplotlib


# In[ ]:


try:
    import google.colab # type: ignore
    from google.colab import output
    COLAB = True
    get_ipython().run_line_magic('pip', 'install sae-lens transformer-lens')
except:
    COLAB = False
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
from tqdm import tqdm

torch.set_grad_enabled(False);


# In[ ]:


# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# # Loading a pretrained Sparse Autoencoder
# 
# Below we load a Transformerlens model, a pretrained SAE and a dataset from huggingface.

# In[4]:


from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2-small", device = device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
    device = device
)


# In[5]:


sae.W_enc.shape


# # get weights

# In[ ]:


weight_matrix_np = sae.W_enc.cpu()


# In[ ]:


import pickle
with open('weight_matrix.pkl', 'wb') as f:
    pickle.dump(weight_matrix_np, f)


# In[ ]:


# from google.colab import files
# files.download('weight_matrix.pkl')


# In[ ]:


# if files doesn't work:

from google.colab import drive
drive.mount('/content/drive')

# Move the file to your Google Drive
get_ipython().system('cp weight_matrix.pkl /content/drive/MyDrive/')


# # get activations

# In[6]:


from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset= dataset,# type: ignore
    tokenizer = model.tokenizer, # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)


# In[7]:


sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    # activation store can give us tokens.
    batch_tokens = token_dataset[:32]["tokens"]
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    feature_acts = sae.encode(cache[sae.cfg.hook_name])
    sae_out = sae.decode(feature_acts)

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    # l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    # print("average l0", l0.mean().item())
    # px.histogram(l0.flatten().cpu().numpy()).show()


# In[10]:


feature_acts.shape


# In[8]:


import pickle
with open('feature_acts.pkl', 'wb') as f:
    pickle.dump(feature_acts, f)


# In[12]:


# this takes a while!

from google.colab import files
files.download('feature_acts.pkl')


# In[13]:


# if files doesn't work: (this is quicker)

from google.colab import drive
drive.mount('/content/drive')

# Move the file to your Google Drive
get_ipython().system('cp feature_acts.pkl /content/drive/MyDrive/')


# # plot weights pca

# In[ ]:


# this also takes a long time

import numpy as np
import pandas as pd

def pca_pandas(data, n_components=2):
    df = pd.DataFrame(data)
    data_mean = df.mean()
    centered_data = df - data_mean

    # Compute covariance matrix
    cov_matrix = centered_data.cov()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    # Select the top n_components eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Project the data onto the top n_components eigenvectors
    reduced_data = np.dot(centered_data, top_eigenvectors)

    return reduced_data

# Example usage
weight_matrix_pca = pca_pandas(weight_matrix_np)


# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(weight_matrix_pca[:, 0], weight_matrix_pca[:, 1], s=1, alpha=0.7)
plt.title('PCA of Weight Matrix')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


# # plot decoder weights

# # plot activations

# In[17]:


reshaped_activations = feature_acts.reshape(32 * 128, 24576)

acts_matrix_pca = pca_pandas(reshaped_activations.cpu())


# In[18]:


import matplotlib.pyplot as plt

plt.scatter(acts_matrix_pca[:, 0], acts_matrix_pca[:, 1], s=1, alpha=0.7)
plt.title('PCA of Matrix')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

