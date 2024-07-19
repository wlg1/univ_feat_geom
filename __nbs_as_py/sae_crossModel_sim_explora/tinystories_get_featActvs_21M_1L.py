#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


from google.colab import drive
import shutil

drive.mount('/content/drive')


# In[2]:


try:
    #import google.colab # type: ignore
    #from google.colab import output
    get_ipython().run_line_magic('pip', 'install sae-lens transformer-lens')
except:
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# In[3]:


import torch
import os

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # load model

# In[4]:


from transformer_lens import HookedTransformer


# In[5]:


model = HookedTransformer.from_pretrained("tiny-stories-1L-21M")


# # load sae using saelens

# To get feature actvs, use sae_lens class method. To do this, you must load the sae as the sae class (wrapper over torch model).

# In[6]:


sae_layer = "blocks.0.hook_mlp_out"


# In[7]:


total_training_steps = 30_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="tiny-stories-1L-21M",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name="blocks.0.hook_mlp_out",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    hook_layer=0,  # Only one layer in the model.
    d_in=1024,  # the width of the mlp output.
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    is_dataset_tokenized=True,
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
    apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",
    # Training Parameters
    lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="sae_lens_tutorial",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)


# In[8]:


# https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L13
from sae_lens import SAEConfig

cfg_dict = cfg.to_dict()
cfg_dict['finetuning_scaling_factor'] = 0

sae_cfg = SAEConfig.from_dict(cfg_dict)
# sae = cls(sae_cfg)


# In[9]:


from sae_lens import SAE
sae = SAE(sae_cfg)
# sae.cfg = sae_cfg


# In[17]:


state_dict = torch.load('/content/drive/MyDrive/tiny-stories-1L-21M_sae_v1.pth')
sae.load_state_dict(state_dict)


# # load dataset

# In[18]:


from datasets import load_dataset


# In[19]:


# dataset = load_dataset("apollo-research/roneneldan-TinyStories-tokenizer-gpt2", streaming=True) # loads token ids

# # Get an iterator for the train split
# train_iter = iter(dataset['train'])

# # Fetch a batch of 32 tokens
# batch_size = 1
# batch = []

# for _ in range(batch_size):
#     try:
#         sample = next(train_iter)['input_ids']
#         batch.append(sample)
#     except StopIteration:
#         break

# torch.tensor(batch[0]).shape


# In[20]:


from transformer_lens.utils import tokenize_and_concatenate

# dataset = load_dataset(
#     path = "NeelNanda/pile-10k",
#     split="train",
#     streaming=False,
# )

dataset = load_dataset("roneneldan/TinyStories", streaming=False)


# In[21]:


test_dataset = dataset['validation']

token_dataset = tokenize_and_concatenate(
    dataset = test_dataset,
    tokenizer = model.tokenizer, # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)


# In[22]:


batch_tokens = token_dataset[:32]["tokens"]
batch_tokens.shape


# In[23]:


# as text

# dataset = load_dataset("roneneldan/TinyStories", streaming=True)

# # Get an iterator for the train split
# train_iter = iter(dataset['train'])

# # Fetch a batch of 32 tokens
# batch_size = 1
# batch = []

# for _ in range(batch_size):
#     try:
#         sample = next(train_iter)['text']
#         batch.append(sample)
#     except StopIteration:
#         break

# batch[0]


# # interpret features

# In[24]:


from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple


# ## get LLM actvs

# In[25]:


layer_name = sae_layer


# In[26]:


h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
h_store.shape


# In[27]:


def store_h_hook(
    pattern: Float[Tensor, "batch seqlen d_model"],
    hook
):
    h_store[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[28]:


model.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# ## get SAE actvs

# In[29]:


sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    feature_acts = sae.encode(h_store)


# Now you have to save actvs, bc saelens not compatible with umap lib

# In[30]:


# with open('feature_acts_model_B_L6.pkl', 'wb') as f:
#     pickle.dump(feature_acts, f)


# In[31]:


# !cp feature_acts_model_B_L6.pkl /content/drive/MyDrive/


# In[32]:


# file_path = '/content/drive/MyDrive/feature_acts_model_B_L6.pkl'
# with open(file_path, 'rb') as f:
#     feature_acts_model_B = pickle.load(f)


# In[33]:


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
count_nonzero_features(feature_acts)


# ## get top features

# In[34]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
# if want to get highest batch, use dim=0
feat_k = 15
top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

print(top_acts_indices.shape)
top_acts_values.shape


# ## interpret top features by dataset examples

# In[35]:


def highest_activating_tokens(
    feature_acts,
    feature_idx: int,
    k: int = 10,  # num batch_seq samples
    batch_tokens=None
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    batch_size, seq_len = batch_tokens.shape

    # Get the top k largest activations for only targeted feature
    # need to flatten (batch,seq) into batch*seq first because it's ANY batch_seq, even if in same batch or same pos
    flattened_feature_acts = feature_acts[:, :, feature_idx].reshape(-1)

    top_acts_values, top_acts_indices = flattened_feature_acts.topk(k)
    # top_acts_values should be 1D
    # top_acts_indices should be also be 1D. Now, turn it back to 2D
    # Convert the indices into (batch, seq) indices
    top_acts_batch = top_acts_indices // seq_len
    top_acts_seq = top_acts_indices % seq_len

    return torch.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values


# In[36]:


from rich import print as rprint
def display_top_sequences(top_acts_indices, top_acts_values, batch_tokens):
    s = ""
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


# In[38]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# # Generating Feature Interfaces
# 
# Feature dashboards are an important part of SAE Evaluation. They work by:
# - 1. Collecting feature activations over a larger number of examples.
# - 2. Aggregating feature specific statistics (such as max activating examples).
# - 3. Representing that information in a standardized way

# In[ ]:


get_ipython().run_cell_magic('capture', '', '%pip install sae-dashboard sae-vis\n')


# In[ ]:


hook_name = sae.cfg.hook_name
hook_name


# In[ ]:


from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

# test_feature_idx_gpt = list(range(10)) + [14057]
test_feature_idx_gpt = 0

feature_vis_config_gpt = SaeVisConfig(
    hook_point=hook_name,
    features=test_feature_idx_gpt,
    batch_size=2048,
    minibatch_size_tokens=128,
    verbose=True,
)

sae_vis_data_gpt = SaeVisData.create(
    encoder=sae,
    model=model, # type: ignore
    tokens=batch_tokens,  # type: ignore
    cfg=feature_vis_config_gpt,
)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = sae.to(device)
model = model.to(device)  # If 'model' is a PyTorch model

sae_vis_data_gpt = SaeVisData.create(
    encoder=sae,
    model=model,
    tokens=batch_tokens,
    cfg=feature_vis_config_gpt,
)


# In[ ]:


import os
import torch

# Set CUDA_LAUNCH_BLOCKING for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Ensure you are using the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Your existing code
try:
    sae_vis_data_gpt = SaeVisData.create(
        encoder=sae,
        model=model, # type: ignore
        tokens=batch_tokens,  # type: ignore
        cfg=feature_vis_config_gpt,
    )
except RuntimeError as e:
    print("RuntimeError:", e)

# Additional debugging steps
if device.type == 'cuda':
    print("CUDA is available")
else:
    print("Running on CPU")


# In[ ]:


for feature in test_feature_idx_gpt:
    filename = f"{feature}_feature_vis_demo_gpt.html"
    sae_vis_data_gpt.save_feature_centric_vis(filename, feature)
    display_vis_inline(filename)

