#!/usr/bin/env python
# coding: utf-8

# ## Setup

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

# In[3]:


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


# In[4]:


sae.W_enc.shape


# # get steering vec- one layer

# In[5]:


# pass in all at once
prompts = ["love",
           "hate"]
tokens = model.to_tokens(prompts, prepend_bos=True)
tokens.shape


# In[6]:


model.reset_hooks(including_permanent=True)
_, cache = model.run_with_cache(tokens)


# In[7]:


hook_point = "blocks.6.hook_resid_pre"  # saelens only has pre, not post
layer = 6
steering_vec = cache[hook_point][0, :, :] - cache[hook_point][1, :, :]
steering_vec = steering_vec.unsqueeze(0)


# # steer by add hook using partial

# In[8]:


from functools import partial

def act_add(
    activation,
    hook,
    steering_vec,
    # initPromptLen
):
    # activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 3
    activation[:, -1, :] += steering_vec[:, -1, :] * 3
    return activation

hook_fn = partial(
        act_add,
        steering_vec=steering_vec,
        # initPromptLen=initPromptLen
    )

# initPromptLen = len(model.tokenizer.encode("I think cats are "))


# # cache L6 to L11 actvs for unst vs st

# In[9]:


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


# In[10]:


batch_tokens = token_dataset[:32]["tokens"]
_, cache = model.run_with_cache(batch_tokens, prepend_bos=True)


# In[11]:


# test_sentence = "I think cats are "
# tokens = model.to_tokens(test_sentence, prepend_bos=True)


# In[12]:


model.reset_hooks(including_permanent=True)
_, unst_cache = model.run_with_cache(batch_tokens)


# In[13]:


model.reset_hooks(including_permanent=True)
model.add_hook(hook_point, hook_fn)
_, steered_cache = model.run_with_cache(batch_tokens)


# In[14]:


# unst_cache_dict = {k: v for k, v in unst_cache.items()}
# steered_cache_dict = {k: v for k, v in steered_cache.items()}


# In[19]:


unst_cache[hook_point].shape


# # obtain SAE feature actvs each L

# In[15]:


get_ipython().run_cell_magic('capture', '', 'unst_feature_acts = {}\nsteered_feature_acts = {}\n\nfor layer_id in range(5, 11):\n    hook_point = f\'blocks.{layer_id}.hook_resid_pre\'\n    # print(hook_point)\n    # sparse_autoencoder = load_sae(hook_point)\n    sae, cfg_dict, sparsity = SAE.from_pretrained(\n        release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml\n        sae_id = hook_point, # won\'t always be a hook point\n        device = device\n    )\n\n    sae.eval()  # prevents error if we\'re expecting a dead neuron mask for who grads\n    with torch.no_grad():\n        # sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(\n        #     unst_cache[hook_point]\n        # )\n        feature_acts = sae.encode(unst_cache[sae.cfg.hook_name])\n        # sae_out = sae.decode(feature_acts)\n    unst_feature_acts[hook_point] = feature_acts\n\n    # print(\'\\n\')\n    sae.eval()\n    with torch.no_grad():\n        # sae_out, feat_acts_steered, loss, mse_loss, l1_loss, _ = sparse_autoencoder(\n        #     steered_cache[hook_point]\n        # )\n        feature_acts = sae.encode(steered_cache[sae.cfg.hook_name])\n    steered_feature_acts[hook_point] = feature_acts\n')


# # save actvs

# In[16]:


import pickle
with open('unst_feature_acts.pkl', 'wb') as f:
    pickle.dump(unst_feature_acts, f)
with open('steered_feature_acts.pkl', 'wb') as f:
    pickle.dump(steered_feature_acts, f)


# In[20]:


from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('cp unst_feature_acts.pkl /content/drive/MyDrive/')
get_ipython().system('cp steered_feature_acts.pkl /content/drive/MyDrive/')

