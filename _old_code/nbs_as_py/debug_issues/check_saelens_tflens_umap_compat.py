#!/usr/bin/env python
# coding: utf-8

# # try using %pip on umap

# no error

# In[1]:


get_ipython().run_line_magic('pip', 'install transformer-lens')


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


import umap


# In[4]:


reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')


# In[6]:


from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")


# In[7]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[8]:


from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
)


# # try using !pip on umap

# no error

# In[1]:


get_ipython().run_line_magic('pip', 'install transformer-lens')


# In[2]:


get_ipython().system('pip install umap-learn')


# In[3]:


import umap


# In[4]:


reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')


# In[5]:


from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")


# In[6]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[7]:


from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
)


# # try with device

# error

# In[1]:


get_ipython().system('pip install umap-learn matplotlib')


# In[2]:


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


# In[3]:


# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


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


import umap


# In[ ]:


reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')


# # pinpoint even more by using torch grad false, but no device

# error

# In[1]:


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


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


# # For the most part I'll try to import functions and classes near where they are used
# # to make it clear where they come from.

# if torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Device: {device}")


# In[4]:


import umap


# # try without torch.set_grad_enabled(False); or device

# error

#  so it’s not device var being cuda when loading model or torch grad False either

# In[1]:


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


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


import umap


# # try torch set grad False, no tflens install

# no error

# In[1]:


import torch
from tqdm import tqdm

torch.set_grad_enabled(False);


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


import umap


# # %pip install saelens tflens without try except

# error

# In[1]:


get_ipython().run_line_magic('pip', 'install sae-lens transformer-lens')


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


import umap


# # %pip install transformer-lens

# no error

# In[1]:


get_ipython().run_line_magic('pip', 'install transformer-lens')


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


import umap


# # %pip install sae-lens

# In[1]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[2]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[3]:


import umap


# # %pip install sae-lens after install umap

# In[1]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[2]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[3]:


import umap


# # import umap, install sae-lens, use umap

# no error

# In[1]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[2]:


import umap


# In[3]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Example datasets
data1 = np.random.rand(100, 10)
data2 = np.random.rand(100, 10)
data3 = np.random.rand(100, 10)

# Combine datasets
combined_data = np.vstack((data1, data2, data3))

# Create and fit UMAP reducer on combined dataset
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
combined_embedding = reducer.fit_transform(combined_data)

# Split the combined embedding back to individual datasets
embedding1 = combined_embedding[:100]
embedding2 = combined_embedding[100:200]
embedding3 = combined_embedding[200:300]

# Plot the embeddings
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(embedding1[:, 0], embedding1[:, 1], label='Dataset 1')
plt.title('Dataset 1')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(embedding2[:, 0], embedding2[:, 1], label='Dataset 2')
plt.title('Dataset 2')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(embedding3[:, 0], embedding3[:, 1], label='Dataset 3')
plt.title('Dataset 3')
plt.legend()

plt.tight_layout()
plt.show()


# Restart from here (not disconnect) to run the code

# In[1]:


from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
)

