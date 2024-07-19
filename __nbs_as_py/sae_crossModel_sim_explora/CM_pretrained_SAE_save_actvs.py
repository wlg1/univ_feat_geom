#!/usr/bin/env python
# coding: utf-8

# # Loading and Analysing Pre-Trained Sparse Autoencoders

# ## Imports & Installs

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
import plotly.express as px

# Imports for displaying vis in Colab / notebook
import webbrowser
import http.server
import socketserver
import threading
PORT = 8000

torch.set_grad_enabled(False);


# ## Set Up

# In[2]:


# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# In[3]:


def display_vis_inline(filename: str, height: int = 850):
    '''
    Displays the HTML files in Colab. Uses global `PORT` variable defined in prev cell, so that each
    vis has a unique port without having to define a port within the function.
    '''
    if not(COLAB):
        webbrowser.open(filename);

    else:
        global PORT

        def serve(directory):
            os.chdir(directory)

            # Create a handler for serving files
            handler = http.server.SimpleHTTPRequestHandler

            # Create a socket server with the handler
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"Serving files from {directory} on port {PORT}")
                httpd.serve_forever()

        thread = threading.Thread(target=serve, args=("/content",))
        thread.start()

        output.serve_kernel_port_as_iframe(PORT, path=f"/{filename}", height=height, cache_in_notebook=True)

        PORT += 1


# # hf login

# In[4]:


get_ipython().system('huggingface-cli login')


# # load pretrained SAEs

# In[ ]:


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


# In[ ]:


# sae_2, cfg_dict_2, sparsity_2 = SAE.from_pretrained(
#     release = "gpt2-small-resid-post-v5-32k", # see other options in sae_lens/pretrained_saes.yaml
#     sae_id = "blocks.7.hook_resid_post", # won't always be a hook point
#     device = device
# )


# In[ ]:


model_2_layer = 'blocks.12.hook_resid_post'


# In[ ]:


model_2 = HookedTransformer.from_pretrained("gemma-2b", device = device)
sae_2, cfg_dict_2, sparsity_2 = SAE.from_pretrained(
    release = "gemma-2b-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.12.hook_resid_post", # won't always be a hook point
    device = device
)


# # load dataset

# In[ ]:


from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset= dataset,# type: ignore
    tokenizer = model_2.tokenizer, # type: ignore
    streaming=True,
    max_length=sae_2.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)


# In[ ]:


batch_tokens = token_dataset[:32]["tokens"]
batch_tokens.shape


# # model 1- save sae actvs

# ## get LLM actvs

# In[ ]:


layer_name = 'blocks.8.hook_resid_pre'


# In[ ]:


h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
h_store.shape


# In[ ]:


from torch import nn, Tensor
# import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

def store_h_hook(
    pattern: Float[Tensor, "batch seqlen d_model"],
    hook
):
    h_store[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[ ]:


model.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# ## get SAE actvs

# In[ ]:


sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    feature_acts = sae.encode(h_store)


# Now you have to save actvs, bc saelens not compatible with umap lib

# In[ ]:


import pickle
with open('feature_acts_model_A.pkl', 'wb') as f:
    pickle.dump(feature_acts, f)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('cp feature_acts_model_A.pkl /content/drive/MyDrive/')


# # model 2- save sae actvs

# ## get LLM actvs

# In[ ]:


layer_name = model_2_layer


# In[ ]:


h_store_2 = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model_2.cfg.d_model), device=model_2.cfg.device)
h_store_2.shape


# In[ ]:


def store_h_hook_2(
    pattern: Float[Tensor, "batch seqlen d_model"],
    hook
):
    h_store_2[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[ ]:


model_2.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook_2),
    ]
)


# ## get SAE actvs

# In[ ]:


sae_2.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    feature_acts_2 = sae_2.encode(h_store_2)


# Now you have to save actvs, bc saelens not compatible with umap lib

# In[ ]:


with open('feature_acts_model_B.pkl', 'wb') as f:
    pickle.dump(feature_acts_2, f)


# In[ ]:


get_ipython().system('cp feature_acts_model_B.pkl /content/drive/MyDrive/')


# # gemma 12b, L6- save sae actvs

# ## setup

# In[9]:


from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE


# In[22]:


from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple


# In[29]:


import pickle


# In[33]:


from google.colab import drive
drive.mount('/content/drive')


# ## laod model

# In[10]:


model_2_layer = 'blocks.6.hook_resid_post'


# In[11]:


model_2 = HookedTransformer.from_pretrained("gemma-2b", device = device)
sae_2, cfg_dict_2, sparsity_2 = SAE.from_pretrained(
    release = "gemma-2b-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.12.hook_resid_post", # won't always be a hook point
    device = device
)


# ## get data

# In[17]:


from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset= dataset,# type: ignore
    tokenizer = model_2.tokenizer, # type: ignore
    streaming=True,
    max_length=sae_2.cfg.context_size,
    add_bos_token=sae_2.cfg.prepend_bos,
)


# In[18]:


batch_tokens = token_dataset[:32]["tokens"]
batch_tokens.shape


# ## get LLM actvs

# In[23]:


layer_name = model_2_layer


# In[24]:


h_store_2 = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model_2.cfg.d_model), device=model_2.cfg.device)
h_store_2.shape


# In[25]:


def store_h_hook_2(
    pattern: Float[Tensor, "batch seqlen d_model"],
    hook
):
    h_store_2[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[26]:


model_2.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook_2),
    ]
)


# ## get SAE actvs

# In[27]:


sae_2.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    feature_acts_2 = sae_2.encode(h_store_2)


# Now you have to save actvs, bc saelens not compatible with umap lib

# In[30]:


with open('feature_acts_model_B_L6.pkl', 'wb') as f:
    pickle.dump(feature_acts_2, f)


# In[34]:


get_ipython().system('cp feature_acts_model_B_L6.pkl /content/drive/MyDrive/')


# In[35]:


file_path = '/content/drive/MyDrive/feature_acts_model_B_L6.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)

