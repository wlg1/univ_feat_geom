#!/usr/bin/env python
# coding: utf-8

# # Loading and Analysing Pre-Trained Sparse Autoencoders

# In[11]:


from google.colab import drive
import shutil

drive.mount('/content/drive')


# In[13]:


import pickle


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
# from tqdm import tqdm
# import plotly.express as px

# # Imports for displaying vis in Colab / notebook
# import webbrowser
# import http.server
# import socketserver
# import threading
# PORT = 8000

torch.set_grad_enabled(False);


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

# In[5]:


layer_name = "blocks.0.hook_mlp_out"


# In[6]:


from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-small", device = device)


# In[7]:


from datasets import load_dataset
from sae_lens import SAE

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-mlp-out-v5-32k", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = layer_name, # won't always be a hook point
    device = device
)


# ## save decoder weights

# In[8]:


weight_matrix_np = sae.W_dec.cpu()


# In[10]:


Wdec_filename = 'gpt2sm_mlp0_Wdec.pkl'
with open(Wdec_filename, 'wb') as f:
    pickle.dump(weight_matrix_np, f)

# source_path = f'/path/to/your/file/{file_name}'
source_path = Wdec_filename
# dest_folder = ''
destination_path = f'/content/drive/MyDrive/sae_files/{Wdec_filename}'

shutil.copy(source_path, destination_path) # Copy the file


# # load dataset

# ## get data

# Need load model tokenizer before obtain dataset

# In[ ]:


from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset("roneneldan/TinyStories", streaming=False)
test_dataset = dataset['validation']

token_dataset = tokenize_and_concatenate(
    dataset = test_dataset,
    tokenizer = model.tokenizer, # type: ignore
    streaming=True,
    # max_length=sae.cfg.context_size,
    max_length=128,
    # add_bos_token=sae.cfg.prepend_bos,
    add_bos_token=False,
)


# In[ ]:


batch_tokens = token_dataset[:500]["tokens"]
batch_tokens.shape


# In[9]:


save_data_fn = 'batch_tokens_anySamps_v1.pkl'


# ## save selected data

# In[ ]:


with open(save_data_fn, 'wb') as f:
    pickle.dump(batch_tokens, f)

# source_path = f'/path/to/your/file/{file_name}'
source_path = save_data_fn
# dest_folder = ''
destination_path = f'/content/drive/MyDrive/{save_data_fn}'

shutil.copy(source_path, destination_path) # Copy the file


# ## load selected data

# In[14]:


# check if saved
file_path = '/content/drive/MyDrive/sae_files/' + save_data_fn
with open(file_path, 'rb') as f:
    batch_tokens = pickle.load(f)


# In[15]:


batch_tokens.shape


# # save sae actvs

# ## get LLM actvs

# In[16]:


h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
h_store.shape


# In[17]:


from torch import nn, Tensor
# import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

def store_h_hook(
    pattern: Float[Tensor, "batch seqlen d_model"],
    hook
):
    h_store[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[18]:


model.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# ## get SAE actvs

# In[19]:


sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    feature_acts = sae.encode(h_store)


# Now you have to save actvs, bc saelens not compatible with umap lib

# In[20]:


import pickle
with open('fActs_GPT2sm_MLP0.pkl', 'wb') as f:
    pickle.dump(feature_acts, f)


# In[22]:


test=1
with open('test.pkl', 'wb') as f:
    pickle.dump(test, f)

fActs_filename = 'test.pkl'
# source_path = f'/path/to/your/file/{file_name}'
source_path = fActs_filename
# dest_folder = ''
destination_path = f'/content/drive/MyDrive/sae_files/{fActs_filename}'

shutil.copy(source_path, destination_path) # Copy the file


# In[21]:


fActs_filename = 'fActs_GPT2sm_MLP0.pkl'
# source_path = f'/path/to/your/file/{file_name}'
source_path = fActs_filename
# dest_folder = ''
destination_path = f'/content/drive/MyDrive/sae_files/{fActs_filename}'

shutil.copy(source_path, destination_path) # Copy the file


# In[22]:


# !cp fActs_GPT2sm_MLP0.pkl /content/drive/MyDrive/sae_files/


# In[23]:


# check if saved
file_path = '/content/drive/MyDrive/sae_files/' + 'fActs_GPT2sm_MLP0.pkl'
with open(file_path, 'rb') as f:
    feature_acts = pickle.load(f)


# In[24]:


feature_acts.shape

