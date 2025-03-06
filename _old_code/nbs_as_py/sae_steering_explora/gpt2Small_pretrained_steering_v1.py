#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


get_ipython().run_cell_magic('capture', '', 'try:\n    import google.colab # type: ignore\n    from google.colab import output\n    COLAB = True\n    %pip install sae-lens==1.3.0 transformer-lens==1.17.0\n    # %pip install sae-lens transformer-lens\nexcept:\n    COLAB = False\n    from IPython import get_ipython # type: ignore\n    ipython = get_ipython(); assert ipython is not None\n    ipython.run_line_magic("load_ext", "autoreload")\n    ipython.run_line_magic("autoreload", "2")\n')


# In[2]:


# Standard imports
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


# In[3]:


# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# In[4]:


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


# In[5]:


import torch
import os

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner

# if torch.cuda.is_available():
#     device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cpu"

# print("Using device:", device)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[6]:


from transformer_lens import HookedTransformer
import torch as t


# ## Load Model

# In[7]:


device = t.device("cuda" if t.cuda.is_available() else "cpu")


# In[8]:


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)


# # get steering vec- one layer

# In[9]:


# pass in all at once
prompts = ["love",
           "hate"]
tokens = model.to_tokens(prompts, prepend_bos=True)
tokens.shape


# In[10]:


model.reset_hooks(including_permanent=True)
_, cache = model.run_with_cache(tokens)


# In[11]:


layer = 5
steering_vec = cache[f'blocks.{layer}.hook_resid_post'][0, :, :] - cache[f'blocks.{layer}.hook_resid_post'][1, :, :]
steering_vec.shape


# # failed tries at steering

# ## apply steering vec at layer K

# In[12]:


# def act_add(steering_vec):
#     def hook(activation):
#         # return activation + steering_vec
#         return activation
#     return hook

# test_sentence = "I think cats are "
# model.add_hook(name=cache_name, hook=act_add(steering_vec))
# print(model.generate(test_sentence, max_new_tokens=10))
# print("-"*20)
# model.reset_hooks()
# model.add_hook(name=cache_name, hook=fun_factory(-steering_vec))
# print(model.generate(test_sentence, max_new_tokens=10))
# model.reset_hooks()


# TypeError: act_add.<locals>.hook() got an unexpected keyword argument 'hook'

# ## wrapper steering

# In[13]:


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[14]:


# model_name = "gpt2"
# model_2 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# tokenizer_2 = AutoTokenizer.from_pretrained(model_name)


# In[15]:


# # define wrapper class
# class WrappedModule(torch.nn.Module):
#    def __init__(self, module):
#        super().__init__()
#        self.module = module
#        self.output = None
#        self.steering_vec = None
#    def forward(self, *args, **kwargs):
#        self.output = self.module(*args, **kwargs)
#        if self.steering_vec is not None:
#           return self.output + self.steering_vec
#        else:
#           return self.output

# # wrap a module of your loaded pretrained transformer model
# layer_id = 5
# model_2.layers[layer_id] = WrappedModule(model_2.layers[layer_id])

# # define a steering vector
# _ = model_2("Love")
# act_love = model_2.layers[layer_id].output
# _ = model_2("Hate")
# act_hate = model_2.layers[layer_id].output
# steering_vec = act_love-act_hate

# # set the steering vector in the WrappedModule and generate some steered text
# test_sentence = "I think dogs are "
# model_2.layers[layer_id].steering_vec = steering_vec
# print(model_2.generate(test_sentence, max_new_tokens=10))
# print("-"*20)
# model_2.layers[layer_id].steering_vec = -steering_vec
# print(model_2.generate(test_sentence, max_new_tokens=10))


# AttributeError: 'GPT2LMHeadModel' object has no attribute 'layers'

# # steer by add hook using partial

# In[12]:


# Get list of arguments to pass to `generate` (specifically these are the ones relating to sampling)
generate_kwargs = dict(
    do_sample = False, # deterministic output so we can compare it to the HF model
    top_p = 1.0, # suppresses annoying output errors
    temperature = 1.0, # suppresses annoying output errors
)


# In[13]:


test_sentence = "I think cats are "
model.reset_hooks(including_permanent=True)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# In[14]:


# unsqueeze to get batch pos so can add to actvs
print(steering_vec.shape)
steering_vec = steering_vec.unsqueeze(0)
steering_vec.shape


# In[15]:


from functools import partial

def act_add(
    # z: Float[Tensor, "batch seq head d_head"],
    # hook: HookPoint,
    activation,
    hook,
    steering_vec
):
    # print(activation[:, -1, :].shape)
    # print(steering_vec[:, -1, :].shape)
    activation[:, -1, :] += steering_vec[:, -1, :]
    return activation

hook_fn = partial(
        act_add,
        steering_vec=steering_vec
    )

test_sentence = "I think cats are "
cache_name = 'blocks.5.hook_resid_post'
model.reset_hooks(including_permanent=True)
model.add_hook(cache_name, hook_fn)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# In[16]:


list(range(4))


# In[17]:


list(range(4))[1:]


# In[18]:


initPromptLen = len(model.tokenizer.encode("I think cats are "))
initPromptLen


# In[19]:


t.empty(1, 8, 3)[:, initPromptLen:, :]


# In[20]:


from functools import partial

def act_add(
    activation,
    hook,
    steering_vec,
    initPromptLen
):
    activation[:, initPromptLen:, :] += steering_vec[:, -1, :]
    return activation

hook_fn = partial(
        act_add,
        steering_vec=steering_vec,
        initPromptLen=initPromptLen
    )

test_sentence = "I think cats are "
initPromptLen = len(model.tokenizer.encode("I think cats are "))
cache_name = 'blocks.5.hook_resid_post'
model.reset_hooks(including_permanent=True)
model.add_hook(cache_name, hook_fn)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# In[21]:


from functools import partial

def act_add(
    activation,
    hook,
    steering_vec,
    initPromptLen
):
    activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 3
    return activation

hook_fn = partial(
        act_add,
        steering_vec=steering_vec,
        initPromptLen=initPromptLen
    )

test_sentence = "I think cats are "
initPromptLen = len(model.tokenizer.encode("I think cats are "))
cache_name = 'blocks.5.hook_resid_post'
model.reset_hooks(including_permanent=True)
model.add_hook(cache_name, hook_fn)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# # cache L6 to L11 actvs for unst vs st

# In[22]:


cache_name


# In[23]:


test_sentence = "I think cats are "
tokens = model.to_tokens(test_sentence, prepend_bos=True)

model.reset_hooks(including_permanent=True)
_, unst_cache = model.run_with_cache(tokens)


# In[24]:


model.reset_hooks(including_permanent=True)
model.add_hook(cache_name, hook_fn)
_, steered_cache = model.run_with_cache(tokens)


# In[25]:


# these are not steering vecs, they're diffs between unst vs steer, NOT b/w love & hate

actv_diffs_dict = {}

layer = 4
for i in range(layer, model.cfg.n_layers):
    cache_name = 'blocks.'+str(i)+'.hook_resid_pre'
    actv_diffs_dict[i] = unst_cache[cache_name] - steered_cache[cache_name]
actv_diffs_dict[6].shape


# # get steering vec L 11

# In[30]:


# pass in all at once
prompts = ["love",
           "hate"]
tokens = model.to_tokens(prompts, prepend_bos=True)
model.reset_hooks(including_permanent=True)
_, cache = model.run_with_cache(tokens)


# In[31]:


hook_point = "blocks.11.hook_resid_pre"  # saelens only has pre, not post
layer = 11
steering_vec = cache[hook_point][0, :, :] - cache[hook_point][1, :, :]
steering_vec = steering_vec.unsqueeze(0)


# In[32]:


test_sentence = "I think cats are "
initPromptLen = len(model.tokenizer.encode("I think cats are "))


# In[33]:


def act_add(
    activation,
    hook,
    steering_vec,
    initPromptLen
):
    activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 3
    return activation

hook_fn = partial(
        act_add,
        steering_vec = steering_vec,
        initPromptLen=initPromptLen
    )


# In[34]:


model.reset_hooks(including_permanent=True)
model.add_hook(hook_point, hook_fn)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# # get feature actvs- blocks.6.hook_resid_pre

# ## get steering vec

# In[26]:


# pass in all at once
prompts = ["love",
           "hate"]
tokens = model.to_tokens(prompts, prepend_bos=True)
model.reset_hooks(including_permanent=True)
_, cache = model.run_with_cache(tokens)


# In[27]:


hook_point = "blocks.6.hook_resid_pre"  # saelens only has pre, not post
layer = 6
steering_vec = cache[hook_point][0, :, :] - cache[hook_point][1, :, :]
steering_vec = steering_vec.unsqueeze(0)


# In[28]:


test_sentence = "I think cats are "
initPromptLen = len(model.tokenizer.encode("I think cats are "))


# In[29]:


def act_add(
    activation,
    hook,
    steering_vec,
    initPromptLen
):
    activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 3
    return activation

hook_fn = partial(
        act_add,
        steering_vec = steering_vec,
        initPromptLen=initPromptLen
    )


# In[30]:


model.reset_hooks(including_permanent=True)
model.add_hook(hook_point, hook_fn)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# ## Load sae

# In[31]:


from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

# if the SAEs were stored with precomputed feature sparsities,
#  those will be return in a dictionary as well.
saes, sparsities = get_gpt2_res_jb_saes(hook_point)

print(saes.keys())


# In[32]:


from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

sparse_autoencoder = saes[hook_point]
sparse_autoencoder.to(device)
sparse_autoencoder.cfg.device = device

loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

# don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
model_0, _, activation_store = loader.load_sae_training_group_session()

# TODO: We should have the session loader go straight to huggingface.


# In[33]:


# _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
# cache[sparse_autoencoder.cfg.hook_point].shape


# ## compare num nonzero feats before/after

# In[34]:


def count_nonzero_features(feature_acts):
    # Count the number of 0s in the tensor
    num_zeros = (feature_acts == 0).sum().item()

    # Count the number of nonzeroes in the tensor
    num_ones = (feature_acts > 0).sum().item()

    # Calculate the percentage of 1s over 0s
    if num_zeros > 0:
        perc_ones_over_total = (num_ones / (num_ones + num_zeros)) * 100
    else:
        perc_ones_over_total = float('inf')  # Handle division by zero

    print(f"Number of 0s: {num_zeros}")
    print(f"Number of nonzeroes: {num_ones}")
    print(f"Percentage of 1s over 0s: {perc_ones_over_total:.2f}%")


# In[35]:


sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        unst_cache[hook_point]
    )
count_nonzero_features(feature_acts)


# In[36]:


sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    sae_out, feat_acts_steered, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        steered_cache[hook_point]
    )
count_nonzero_features(feat_acts_steered)


# In[37]:


sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    # activation store can give us tokens.
    # batch_tokens = activation_store.get_batch_tokens()
    # _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    sae_out, feature_acts_diffs, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        # cache[sparse_autoencoder.cfg.hook_point]
        actv_diffs_dict[6]
    )
count_nonzero_features(feature_acts_diffs)


# # generalize nonzero feats functions

# In[38]:


def count_nonzero_features(feature_acts):
    # Count the number of 0s in the tensor
    num_zeros = (feature_acts == 0).sum().item()

    # Count the number of nonzeroes in the tensor
    num_ones = (feature_acts > 0).sum().item()

    # Calculate the percentage of 1s over 0s
    if num_zeros > 0:
        perc_ones_over_total = (num_ones / (num_ones + num_zeros)) * 100
    else:
        perc_ones_over_total = float('inf')  # Handle division by zero
    return perc_ones_over_total


# In[39]:


from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

def load_sae(hook_point):
    # if the SAEs were stored with precomputed feature sparsities,
    #  those will be return in a dictionary as well.
    saes, sparsities = get_gpt2_res_jb_saes(hook_point)

    sparse_autoencoder = saes[hook_point]
    sparse_autoencoder.to(device)
    sparse_autoencoder.cfg.device = device

    # loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

    # # don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
    # model_0, _, activation_store = loader.load_sae_training_group_session()
    return sparse_autoencoder


# # nonzero feats L6 to L11

# In[44]:


get_ipython().run_cell_magic('capture', '', "unst_nonzero_feats = {}\nsteered_nonzero_feats = {}\n\nfor layer_id in range(4, 12):\n    hook_point = f'blocks.{layer_id}.hook_resid_pre'\n    # print(hook_point)\n    sparse_autoencoder = load_sae(hook_point)\n\n    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads\n    with torch.no_grad():\n        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(\n            unst_cache[hook_point]\n        )\n    unst_nonzero_feats[hook_point] = count_nonzero_features(feature_acts)\n\n    # print('\\n')\n    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads\n    with torch.no_grad():\n        sae_out, feat_acts_steered, loss, mse_loss, l1_loss, _ = sparse_autoencoder(\n            steered_cache[hook_point]\n        )\n    steered_nonzero_feats[hook_point] = count_nonzero_features(feat_acts_steered)\n")


# In[45]:


for layer_id in range(4, 12):
    hook_point = f'blocks.{layer_id}.hook_resid_pre'
    print(hook_point)
    print(f"unst: % of 1s: {unst_nonzero_feats[hook_point]:.2f}%")
    print(f"steered: % of 1s: {steered_nonzero_feats[hook_point]:.2f}%")
    print('\n')


# # get highest actvs, L6

# In[46]:


hook_point = 'blocks.6.hook_resid_pre'


# In[47]:


sparse_autoencoder = saes[hook_point]
sparse_autoencoder.to(device)
sparse_autoencoder.cfg.device = device

loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)
# don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
model_0, _, activation_store = loader.load_sae_training_group_session()


# In[48]:


activation_store.get_batch_tokens().shape


# In[49]:


sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        steered_cache[hook_point]
    )
steered_nonzero_feats[hook_point] = count_nonzero_features(feature_acts)
feature_acts.shape


# In[50]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
# if want to get highest batch, use dim=0
feat_k = 5
top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

print(top_acts_indices.shape)
top_acts_values.shape


# In[51]:


# we only want the top feature indices of the LAST pos for the prompt

top_acts_indices[0, -1, :]


# [ 4583,  6581, 12645, 16292,  9182], device='cuda:0')

# # interpret top L6 feats after steer

# feature_acts should be dims (batch * seq, num_autoen = 1, feature_dims) in ARENA, then we take one slice on feature_dims so it's only (batch*seq). But SAELens uses (batch, seq, feature_dims) already!
# 
# So transform it in btach*seq, making it 1D

# In[52]:


# IMPT: steered prompt, not activation store, bc want features for actvs before/after steering

import pdb
from jaxtyping import Float, Int
from typing import Optional, Union, Callable, List, Tuple
from torch import nn, Tensor
from rich import print as rprint

# IMPT: activation store, not steered prompt, bc want examples on large dataset

def act_add(
    activation,
    hook,
    steering_vec,
    # initPromptLen
):
    # activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 3
    activation[:, -1, :] += steering_vec[:, -1, :] * 3
    return activation

@t.inference_mode()
def highest_activating_tokens(
    # tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder,
    feature_idx: int,
    activation_store,
    # autoencoder_B: bool = False,
    k: int = 10,  # num batch_seq samples
    # steer_bool = False,
    cache_name = None,
    steering_vec = None,
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    # batch_size, seq_len = tokens.shape
    # instance_idx = 1 if autoencoder_B else 0

    # # Get the post activations from the clean run
    # cache = model.run_with_cache(tokens, names_filter=["blocks.0.mlp.hook_post"])[1]
    # post = cache["blocks.0.mlp.hook_post"]
    # post_reshaped = einops.rearrange(post, "batch seq d_mlp -> (batch seq) d_mlp")

    # # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    # h_cent = post_reshaped - autoencoder.b_dec[instance_idx]
    # acts = einops.einsum(
    #     h_cent, autoencoder.W_enc[instance_idx, :, feature_idx],
    #     "batch_size n_input_ae, n_input_ae -> batch_size"
    # )

    batch_tokens = activation_store.get_batch_tokens()
    batch_size, seq_len = batch_tokens.shape

    if cache_name:
        hook_fn = partial(
                act_add,
                steering_vec = steering_vec,
                # initPromptLen=initPromptLen
            )

    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    with torch.no_grad():
        # unsteered run
        model.reset_hooks(including_permanent=True)
        if cache_name:
            model.add_hook(cache_name, hook_fn)
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
            cache[sparse_autoencoder.cfg.hook_point]

        )

    # Get the top k largest activations for only targeted feature
    # need to flatten (batch,seq) into batch*seq first because it's ANY batch_seq, even if in same batch or same pos
    flattened_feature_acts = feature_acts[:, :, feature_idx].reshape(-1)

    top_acts_values, top_acts_indices = flattened_feature_acts.topk(k)
    # top_acts_values should be 1D
    # top_acts_indices should be also be 1D. Now, turn it back to 2D
    # Convert the indices into (batch, seq) indices
    top_acts_batch = top_acts_indices // seq_len
    top_acts_seq = top_acts_indices % seq_len

    return t.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values

def display_top_sequences(top_acts_indices, top_acts_values, activation_store):
    s = ""
    batch_tokens = activation_store.get_batch_tokens()
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # Get the sequence as a string (with some padding on either side of our sequence)
        seq_start = max(seq_idx - 5, 0)
        # seq_end = min(seq_idx + 5, all_tokens.shape[1])
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        # Loop over the sequence, adding each token to the string (highlighting the token with the large activations)
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # Print the sequence, and the activation value
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'

    rprint(s)


# In[53]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
layer_st = 5
cache_name = f'blocks.{layer_st}.hook_resid_post'
steering_vec = cache[cache_name][0, :, :] - cache[cache_name][1, :, :]
steering_vec = steering_vec.unsqueeze(0)

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)

    # unsteered run
    print('Unsteered Top Dataset Examples for feature')
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=None, steering_vec=None)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store)

    # steered run
    print('Steered Top Dataset Examples for feature')
    layer_st = 5
    cache_name = f'blocks.{layer_st}.hook_resid_post'
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store)


# Compare steering at resid_post 5 vs at resid_pre 6

# In[54]:


comparison_result = (unst_cache['blocks.6.hook_resid_pre'] == unst_cache['blocks.5.hook_resid_post'])
num_true = torch.sum(comparison_result).item()
total_elements = comparison_result.numel()
num_true / total_elements


# ## non-random get_batch_tokens

# SOLN: they’re the same, but each time you run dataset examples, you get different results

# The code exhibits randomness due to the use of activation_store.get_batch_tokens(), which is likely fetching a random batch of tokens from a dataset each time it is called. This introduces variability in the examples displayed each run.

# In[55]:


activation_store.get_batch_tokens().shape


# In[99]:


from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer

# Load the dataset in streaming mode
dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the maximum sequence length for the model
# max_length = 1024
max_length = 200

# Define a function to get tokens in batches with truncation and padding
def get_batch_tokens(dataset, tokenizer, batch_size=32, max_length=1024):
    sequences = []
    iterator = iter(dataset)  # Create an iterator from the streamed dataset
    for _ in range(batch_size):
        try:
            # Get the next example from the dataset
            example = next(iterator)
            # Tokenize the text with truncation and padding
            tokens = tokenizer.encode(example['text'], max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
            sequences.append(tokens)
        except StopIteration:
            # If the dataset ends before reaching the batch size
            break
    if sequences:
        batch_tokens = torch.stack(sequences, dim=0).squeeze(1)
        return batch_tokens
    else:
        return None


# In[57]:


# activation_store.store_batch_size_prompts
# sparse_autoencoder.cfg.store_batch_size_prompts


# In[58]:


# activation_store.get_batch_tokens(1024).shape


# Fix it so that batch_tokens corresponds to the same tokens in each fn

# In[98]:


# IMPT: steered prompt, not activation store, bc want features for actvs before/after steering

import pdb
from jaxtyping import Float, Int
from typing import Optional, Union, Callable, List, Tuple
from torch import nn, Tensor
from rich import print as rprint

# IMPT: activation store, not steered prompt, bc want examples on large dataset

def act_add(
    activation,
    hook,
    steering_vec,
    # initPromptLen
):
    # activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 3
    activation[:, -1, :] += steering_vec[:, -1, :] * 3
    return activation

@t.inference_mode()
def highest_activating_tokens(
    # tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder,
    feature_idx: int,
    activation_store,
    # autoencoder_B: bool = False,
    k: int = 10,  # num batch_seq samples
    # steer_bool = False,
    cache_name = None,
    steering_vec = None,
    batch_tokens=None
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    # batch_tokens = activation_store.get_batch_tokens()
    batch_size, seq_len = batch_tokens.shape

    if cache_name:
        hook_fn = partial(
                act_add,
                steering_vec = steering_vec,
                # initPromptLen=initPromptLen
            )

    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    with torch.no_grad():
        # unsteered run
        model.reset_hooks(including_permanent=True)
        if cache_name:
            model.add_hook(cache_name, hook_fn)
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
            cache[sparse_autoencoder.cfg.hook_point]

        )

    # Get the top k largest activations for only targeted feature
    # need to flatten (batch,seq) into batch*seq first because it's ANY batch_seq, even if in same batch or same pos
    flattened_feature_acts = feature_acts[:, :, feature_idx].reshape(-1)

    top_acts_values, top_acts_indices = flattened_feature_acts.topk(k)
    # top_acts_values should be 1D
    # top_acts_indices should be also be 1D. Now, turn it back to 2D
    # Convert the indices into (batch, seq) indices
    top_acts_batch = top_acts_indices // seq_len
    top_acts_seq = top_acts_indices % seq_len

    return t.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values

def display_top_sequences(top_acts_indices, top_acts_values, activation_store,
                          batch_tokens):
    s = ""
    # batch_tokens = activation_store.get_batch_tokens()
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # s += f'{batch_idx}\n'
        s += f'batchID: {batch_idx}, '
        # Get the sequence as a string (with some padding on either side of our sequence)
        seq_start = max(seq_idx - 5, 0)
        # seq_end = min(seq_idx + 5, all_tokens.shape[1])
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        # Loop over the sequence, adding each token to the string (highlighting the token with the large activations)
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # Print the sequence, and the activation value
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'

    rprint(s)


# In[100]:


max_length = 128
batch_tokens = get_batch_tokens(dataset, model.tokenizer, batch_size=32, max_length=max_length)
if batch_tokens is not None:
    print(batch_tokens.shape)
else:
    print("No data to load.")


# In[101]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
layer_st = 5
cache_name = f'blocks.{layer_st}.hook_resid_post'
steering_vec = cache[cache_name][0, :, :] - cache[cache_name][1, :, :]
steering_vec = steering_vec.unsqueeze(0)

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)

    # unsteered run
    print('Unsteered Top Dataset Examples for feature')
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=None, steering_vec=None,
                                            batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)

    # steered run
    print('Steered Top Dataset Examples for feature')
    layer_st = 5
    cache_name = f'blocks.{layer_st}.hook_resid_post'
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec,
                                            batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)


# In[102]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
layer_st = 6
cache_name = f'blocks.{layer_st}.hook_resid_pre'
steering_vec = cache[cache_name][0, :, :] - cache[cache_name][1, :, :]
steering_vec = steering_vec.unsqueeze(0)

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)

    # unsteered run
    print('Unsteered Top Dataset Examples for feature')
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=None, steering_vec=None,
                                            batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)

    # steered run
    print('Steered Top Dataset Examples for feature')
    layer_st = 5
    cache_name = f'blocks.{layer_st}.hook_resid_post'
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec,
                                            batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)


# ## samples that contain ‘love’ or ‘hate’ in them

# In[103]:


# Load the dataset in streaming mode
dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# Define the maximum sequence length for the model
max_length = 128

# Function to check if text contains the words "love" or "hate"
def contains_love_or_hate(text):
    # return "love" in text.lower() or "hate" in text.lower()
    return "love" in text.lower()

# Define a function to get tokens in batches with truncation and padding
def get_batch_tokens(dataset, tokenizer, batch_size=32, max_length=128):
    sequences = []
    love_hate_sequences = []
    other_sequences = []
    iterator = iter(dataset)  # Create an iterator from the streamed dataset

    # Separate sequences into those containing "love" or "hate" and those that do not
    for _ in range(batch_size * 2):  # Load more to ensure we get enough samples
        try:
            # Get the next example from the dataset
            example = next(iterator)
            text = example['text']
            if contains_love_or_hate(text):
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
                love_hate_sequences.append(tokens)
            else:
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
                other_sequences.append(tokens)
        except StopIteration:
            # If the dataset ends before reaching the required amount
            break

    # Ensure we have enough samples of each type
    min_length = min(len(love_hate_sequences), len(other_sequences))
    love_hate_sequences = love_hate_sequences[:min_length]
    other_sequences = other_sequences[:min_length]

    # Combine sequences to form the batch
    sequences = love_hate_sequences[:batch_size//2] + other_sequences[:batch_size//2]

    # pdb.set_trace()

    if sequences:
        batch_tokens = torch.cat(sequences, dim=0).squeeze(1)
        return batch_tokens
    else:
        return None

# Get a batch of tokens
batch_tokens = get_batch_tokens(dataset, model.tokenizer, batch_size=32, max_length=max_length)
if batch_tokens is not None:
    print(batch_tokens.shape)
else:
    print("No data to load.")


# len(love_hate_sequences):
# 8

# In[ ]:


# # Load the dataset in streaming mode
# dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# # Define the maximum sequence length for the model
# max_length = 128

# # Function to check if text contains the words "love" or "hate"
# def contains_love_or_hate(text):
#     return "love" in text.lower() or "hate" in text.lower()

# # Define a function to get tokens in batches with truncation and padding
# def get_batch_tokens(dataset, tokenizer, batch_size=32):
#     sequences = []
#     love_hate_sequences = []
#     other_sequences = []
#     iterator = iter(dataset)  # Create an iterator from the streamed dataset

#     # Separate sequences into those containing "love" or "hate" and those that do not
#     for _ in range(batch_size * 2):  # Load more to ensure we get enough samples
#         try:
#             # Get the next example from the dataset
#             example = next(iterator)
#             text = example['text']
#             if contains_love_or_hate(text):
#                 tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
#                 love_hate_sequences.append((tokens, text))
#             else:
#                 tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
#                 other_sequences.append((tokens, text))
#         except StopIteration:
#             # If the dataset ends before reaching the required amount
#             break

#     # Ensure we have enough samples of each type
#     min_length = min(len(love_hate_sequences), len(other_sequences))
#     love_hate_sequences = love_hate_sequences[:min_length]
#     other_sequences = other_sequences[:min_length]

#     # Combine sequences to form the batch
#     sequences = love_hate_sequences[:batch_size//2] + other_sequences[:batch_size//2]

#     if sequences:
#         batch_tokens = torch.cat([seq[0] for seq in sequences], dim=0).squeeze(1)
#         return batch_tokens, [seq[1] for seq in sequences]
#     else:
#         return None, None

# # Get a batch of tokens
# batch_tokens, texts = get_batch_tokens(dataset, model.tokenizer, batch_size=32)
# if batch_tokens is not None:
#     print(batch_tokens.shape)

#     # Decode and check for "love" or "hate"
#     decoded_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in batch_tokens]
#     for i, (tokens, original_text) in enumerate(zip(decoded_texts, texts)):
#         contains_keywords = contains_love_or_hate(original_text)
#         print(f"Sample {i+1} (contains 'love' or 'hate': {contains_keywords}):\n{original_text}\n")
# else:
#     print("No data to load.")


# In[ ]:


# decoded_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in batch_tokens]
# # for i, text in enumerate(decoded_texts):
# #     print(f"Sample {i+1}:\n{text}\n")
# for i, text in enumerate(decoded_texts):
#     contains_keywords = contains_love_or_hate(text)
#     print(f"Sample {i+1} (contains 'love' or 'hate': {contains_keywords}):\n{text}\n")


# In[104]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
layer_st = 6
cache_name = f'blocks.{layer_st}.hook_resid_pre'
steering_vec = cache[cache_name][0, :, :] - cache[cache_name][1, :, :]
steering_vec = steering_vec.unsqueeze(0)

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)

    # unsteered run
    print('Unsteered Top Dataset Examples for feature')
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=None, steering_vec=None,
                                            batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)

    # steered run
    print('Steered Top Dataset Examples for feature')
    layer_st = 5
    cache_name = f'blocks.{layer_st}.hook_resid_post'
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec,
                                            batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)


# # interpret top L7 to L11

# In[105]:


# for layer_id in range(4, 12):
def top_samps_layer(layer_id):
    feat_k = 5
    samp_m = 5
    layer_st = 5

    hook_point = f'blocks.{layer_id}.hook_resid_pre'
    # print(hook_point)
    sparse_autoencoder = load_sae(hook_point)

    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    with torch.no_grad():
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
            steered_cache[hook_point]
        )
    steered_nonzero_feats[hook_point] = count_nonzero_features(feature_acts)

    # Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
    top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

    # get top samp_m tokens for all top feat_k feature neurons
    cache_name = f'blocks.{layer_st}.hook_resid_post'
    steering_vec = cache[cache_name][0, :, :] - cache[cache_name][1, :, :]
    steering_vec = steering_vec.unsqueeze(0)

    for feature_idx in top_acts_indices[0, -1, :]:
        feature_idx = feature_idx.item()
        print('Feature: ', feature_idx)

        # unsteered run
        print('Unsteered Top Dataset Examples for feature')
        ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                                activation_store, k=samp_m, cache_name=None, steering_vec=None,
                                                                            batch_tokens=batch_tokens)
        display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)

        # steered run
        print('Steered Top Dataset Examples for feature')
        ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                                activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec,
                                                                            batch_tokens=batch_tokens)
        display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)


# In[106]:


top_samps_layer(7)


# In[107]:


top_samps_layer(8)


# In[108]:


top_samps_layer(9)


# In[109]:


top_samps_layer(10)


# In[110]:


top_samps_layer(11)


# # decompose love-hate vector into features

# ## get steering vec- one layer

# "resid post 5" is the same as "resid pre 6"

# In[111]:


hook_point = 'blocks.6.hook_resid_pre'


# In[112]:


# pass in all at once
prompts = ["love",
           "hate"]
tokens = model.to_tokens(prompts, prepend_bos=True)
tokens.shape


# In[113]:


model.reset_hooks(including_permanent=True)
_, cache = model.run_with_cache(tokens)


# In[114]:


steering_vec = cache[hook_point][0, :, :] - cache[hook_point][1, :, :]
steering_vec = steering_vec.unsqueeze(0)
steering_vec.shape


# ## test steering vec

# In[115]:


test_sentence = "I think cats are "
initPromptLen = len(model.tokenizer.encode("I think cats are "))


# In[116]:


model.reset_hooks(including_permanent=True)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# In[117]:


from functools import partial

# def act_add(
#     activation,
#     hook,
#     steering_vec,
#     initPromptLen
# ):
#     activation[:, initPromptLen:, :] += steering_vec[:, -1, :] * 10
#     return activation

# hook_fn = partial(
#         act_add,
#         steering_vec = steering_vec,
#         initPromptLen=initPromptLen
#     )

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
        steering_vec = steering_vec,
        # initPromptLen=initPromptLen
    )


# In[118]:


model.reset_hooks(including_permanent=True)
model.add_hook(hook_point, hook_fn)
print(model.generate(test_sentence, max_new_tokens=10, **generate_kwargs))


# ## Load sae

# In[119]:


from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

# if the SAEs were stored with precomputed feature sparsities,
#  those will be return in a dictionary as well.
saes, sparsities = get_gpt2_res_jb_saes(hook_point)

print(saes.keys())


# In[120]:


from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

sparse_autoencoder = saes[hook_point]
sparse_autoencoder.to(device)
sparse_autoencoder.cfg.device = device

loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)

# don't overwrite the sparse autoencoder with the loader's sae (newly initialized)
model_0, _, activation_store = loader.load_sae_training_group_session()

# TODO: We should have the session loader go straight to huggingface.


# ## get top features for SV

# In[121]:


sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        steering_vec
    )
count_nonzero_features(feature_acts)


# In[126]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
# if want to get highest batch, use dim=0
feat_k = 15
top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

print(top_acts_indices.shape)
top_acts_values.shape


# ## use larger dataset with 8 samples 'love'

# In[128]:


# Load the dataset in streaming mode
dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# Define the maximum sequence length for the model
max_length = 128

# Function to check if text contains the words "love" or "hate"
def contains_love_or_hate(text):
    # return "love" in text.lower() or "hate" in text.lower()
    return "love" in text.lower()

# Define a function to get tokens in batches with truncation and padding
def get_batch_tokens(dataset, tokenizer, batch_size=32, max_length=128):
    sequences = []
    love_hate_sequences = []
    other_sequences = []
    iterator = iter(dataset)  # Create an iterator from the streamed dataset

    # Separate sequences into those containing "love" or "hate" and those that do not
    for _ in range(batch_size * 2):  # Load more to ensure we get enough samples
        try:
            # Get the next example from the dataset
            example = next(iterator)
            text = example['text']
            if contains_love_or_hate(text):
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
                love_hate_sequences.append(tokens)
            else:
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
                other_sequences.append(tokens)
        except StopIteration:
            # If the dataset ends before reaching the required amount
            break

    # Ensure we have enough samples of each type
    # min_length = min(len(love_hate_sequences), len(other_sequences))
    # love_hate_sequences = love_hate_sequences[:min_length]
    # other_sequences = other_sequences[:min_length]

    others_len = batch_size - len(love_hate_sequences)
    other_sequences = other_sequences[:others_len]

    # Combine sequences to form the batch
    # sequences = love_hate_sequences[:batch_size//2] + other_sequences[:batch_size//2]
    sequences = love_hate_sequences + other_sequences

    # pdb.set_trace()

    if sequences:
        batch_tokens = torch.cat(sequences, dim=0).squeeze(1)
        return batch_tokens
    else:
        return None

# Get a batch of tokens
batch_tokens = get_batch_tokens(dataset, model.tokenizer, batch_size=100, max_length=max_length)
if batch_tokens is not None:
    print(batch_tokens.shape)
else:
    print("No data to load.")


# ## interpret top features by dataset examples

# In[129]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
# layer_st = 5
# cache_name = f'blocks.{layer_st}.hook_resid_pre'

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)

    # unsteered run
    print('Unsteered Top Dataset Examples for feature')
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=None, steering_vec=None,
                                                                        batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)

    # steered run
    print('Steered Top Dataset Examples for feature')
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                            activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec,
                                                                        batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)


# # interpret top L7 to L11 using larger dataset

# In[136]:


# for layer_id in range(4, 12):
def top_samps_layer(layer_id):
    feat_k = 5
    samp_m = 5
    layer_st = 6

    hook_point = f'blocks.{layer_id}.hook_resid_pre'
    # print(hook_point)
    sparse_autoencoder = load_sae(hook_point)

    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    with torch.no_grad():
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
            steered_cache[hook_point]
        )
    steered_nonzero_feats[hook_point] = count_nonzero_features(feature_acts)

    # Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
    top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

    # get top samp_m tokens for all top feat_k feature neurons
    cache_name = hook_point
    steering_vec = cache[cache_name][0, :, :] - cache[cache_name][1, :, :]
    steering_vec = steering_vec.unsqueeze(0)

    for feature_idx in top_acts_indices[0, -1, :]:
        feature_idx = feature_idx.item()
        print('Feature: ', feature_idx)

        # unsteered run
        print('Unsteered Top Dataset Examples for feature')
        ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                                activation_store, k=samp_m, cache_name=None, steering_vec=None,
                                                                            batch_tokens=batch_tokens)
        display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)

        # steered run
        print('Steered Top Dataset Examples for feature')
        ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(model, sparse_autoencoder, feature_idx,
                                                activation_store, k=samp_m, cache_name=cache_name, steering_vec=steering_vec,
                                                                            batch_tokens=batch_tokens)
        display_top_sequences(ds_top_acts_indices, ds_top_acts_values, activation_store, batch_tokens=batch_tokens)


# In[137]:


top_samps_layer(7)


# In[138]:


top_samps_layer(8)


# In[139]:


top_samps_layer(9)


# In[140]:


top_samps_layer(10)


# In[141]:


top_samps_layer(11)

