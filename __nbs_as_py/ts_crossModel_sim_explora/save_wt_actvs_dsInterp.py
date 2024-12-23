#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


from google.colab import drive
import shutil

drive.mount('/content/drive')


# ## for weights only

# In[2]:


get_ipython().run_cell_magic('capture', '', '%pip install sae-lens\n')


# In[3]:


from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner


# In[8]:


import pickle
from google.colab import files

import torch
import os

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ## for actvs and labels

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'try:\n    #import google.colab # type: ignore\n    #from google.colab import output\n    %pip install sae-lens transformer-lens\nexcept:\n    from IPython import get_ipython # type: ignore\n    ipython = get_ipython(); assert ipython is not None\n    ipython.run_line_magic("load_ext", "autoreload")\n    ipython.run_line_magic("autoreload", "2")\n')


# In[ ]:


from transformer_lens import HookedTransformer
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

from datasets import load_dataset
from rich import print as rprint  # printing highlights in data samples
import pickle
from google.colab import files

import torch
import os

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # change input params here

# In[18]:


# model_name = "tiny-stories-1L-21M"
model_name = "tiny-stories-2L-33M"
# short_model_name = 'ts_1L_21M'
short_model_name = 'ts_2L_33M'
layer_name = "blocks.0.hook_mlp_out"
hook_layer = 0
d_in = 1024
expa_fac = 16  # 8, 16, 32
total_training_steps = 100000 # 30_000
activation_fn_str = "topk"
activation_fn_kwargs = {"k":32}

# wandb_project = "sae_" + model_name+"_MLP" + str(hook_layer) + "_df-" + str(d_in * expa_fac)
# wandb_project = "sae_" + model_name + "_MLP" + str(hook_layer) + "_df" + str(d_in * expa_fac) + "_steps100k" + "_topK"
wandb_project = "sae_" + model_name + "_MLP" + str(hook_layer) + "_df" + str(d_in * expa_fac) + "_steps100k"
    # sae_tiny-stories-1L-21M_MLP0_df16384_steps100k
drive_save_path = '/content/drive/MyDrive/'+wandb_project+'.pth'

save_data_fn = 'batch_tokens_anySamps_v1.pkl'
# Wdec_filename = 'Wdec_' + short_model_name + "_df" + str(d_in * expa_fac) + '.pkl'
# acts_save_path = 'fActs_' + short_model_name + "_df_" + str(d_in * expa_fac)  + '_anySamps_v1.pkl'
# Wdec_filename = 'Wdec_' + short_model_name + "_df" + str(d_in * expa_fac)  + "_steps100k" + "_topK" + '.pkl'
# acts_save_path = 'fActs_' + short_model_name + "_df_" + str(d_in * expa_fac)  + "_steps100k" + "_topK" + '_anySamps_v1.pkl'
Wdec_filename = 'Wdec_' + short_model_name + "_df" + str(d_in * expa_fac)  + "_steps100k" + '.pkl'
# acts_save_path = 'fActs_' + short_model_name + "_df_" + str(d_in * expa_fac)  + "_steps100k" + "_topK" + '_anySamps_v1.pkl'


# # load model

# In[ ]:


model = HookedTransformer.from_pretrained(model_name)


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

# In[ ]:


# check if saved
file_path = '/content/drive/MyDrive/' + save_data_fn
with open(file_path, 'rb') as f:
    batch_tokens = pickle.load(f)


# # load sae using saelens

# ## load

# To get feature actvs, use sae_lens class method. To do this, you must load the sae as the sae class (wrapper over torch model).

# In[19]:


batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name=model_name,  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name=layer_name,  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    hook_layer=hook_layer,  # Only one layer in the model.
    d_in=d_in,  # the width of the mlp output.
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    is_dataset_tokenized=True,
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    # activation_fn = activation_fn_str,
    # activation_fn_kwargs = activation_fn_kwargs,
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor= expa_fac,  # the width of the SAE. Larger will result in better stats but slower training.
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
    wandb_project=wandb_project,
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)


# In[20]:


# https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L13
from sae_lens import SAEConfig

cfg_dict = cfg.to_dict()
cfg_dict['finetuning_scaling_factor'] = 0

sae_cfg = SAEConfig.from_dict(cfg_dict)

from sae_lens import SAE
sae = SAE(sae_cfg)

state_dict = torch.load(drive_save_path)
sae.load_state_dict(state_dict)


# ## save decoder weights

# In[21]:


weight_matrix_np = sae.W_dec.cpu()


# In[22]:


with open(Wdec_filename, 'wb') as f:
    pickle.dump(weight_matrix_np, f)

# source_path = f'/path/to/your/file/{file_name}'
source_path = Wdec_filename
# dest_folder = ''
destination_path = f'/content/drive/MyDrive/{Wdec_filename}'

shutil.copy(source_path, destination_path) # Copy the file


# # save top ds exm for features

# ## get LLM actvs

# In[ ]:


h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
h_store.shape


# In[ ]:


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


# In[ ]:


def count_nonzero_features(feature_acts):
    # Count the number of 0s in the tensor
    # num_zeros = (feature_acts == 0).sum().item() # OOM
    # chunk
    num_zeros = 0
    for i in range(feature_acts.size(0)):
        for j in range(feature_acts.size(1)):
            num_zeros += (feature_acts[i, j, :] == 0).sum().item()

    # Count the number of nonzeroes in the tensor
    # num_ones = (feature_acts > 0).sum().item()
    num_ones = 0
    for i in range(feature_acts.size(0)):
        for j in range(feature_acts.size(1)):
            num_ones += (feature_acts[i, j, :] > 0).sum().item()

    # Calculate the percentage of 1s over 0s
    if num_zeros > 0:
        perc_ones_over_total = (num_ones / (num_ones + num_zeros)) * 100
    else:
        perc_ones_over_total = float('inf')  # Handle division by zero
    return perc_ones_over_total
count_nonzero_features(feature_acts)


# ## save actvs

# Now you have to save actvs, bc saelens not compatible with umap OR cosine sim lib

# In[ ]:


with open(acts_save_path, 'wb') as f:
    pickle.dump(feature_acts, f)

# source_path = f'/path/to/your/file/{file_name}'
source_path = acts_save_path
# dest_folder = ''
destination_path = f'/content/drive/MyDrive/{acts_save_path}'

shutil.copy(source_path, destination_path) # Copy the file


# In[ ]:


# check if saved
file_path = '/content/drive/MyDrive/' + acts_save_path
with open(file_path, 'rb') as f:
    feature_acts = pickle.load(f)


# ## interpret by dataset examples, keyword search

# In[ ]:


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


# In[ ]:


def store_top_sequences(top_acts_indices, top_acts_values, batch_tokens):
    s = ""
    feat_samps = []
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        s += f'batchID: {batch_idx}, '
        seq_start = max(seq_idx - 5, 0)
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        for i in range(seq_start, seq_end):
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'
        feat_samps.append(seq)
    return s, feat_samps


# In[ ]:


# store feature : str of top 5 snippets (contains batchID, val, seq in each substr)
feat_snip_dict_strs = {}
feat_snip_dict_lst = {}
samp_m = 5

for feature_idx in range(feature_acts.shape[-1]):
    if feature_idx % 1000 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, samp_m, batch_tokens=batch_tokens)
    disp_str, feat_samps = store_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens)
    feat_snip_dict_strs[feature_idx] = disp_str
    feat_snip_dict_lst[feature_idx] = feat_samps


# In[ ]:


relevant_features = {}

# keyword = 'princess'
keyword = '[bold u dark_orange] princess[/]'
relevant_features[keyword] = []
for f_id, str_output in feat_snip_dict_strs.items():
    if keyword in str_output:
        relevant_features[keyword].append(f_id)
        print(f'Feature: {f_id}')
        rprint(str_output)


# ## save datasamps for features

# In[ ]:


with open('feat_snip_dict_strs.pkl', 'wb') as f:
    pickle.dump(feat_snip_dict_strs, f)
files.download('feat_snip_dict_strs.pkl')


# In[ ]:


import json

# Transform the dictionary into a list of dictionaries
feature_top_samps_lst = [{"feature": feature, "strings": strings} for feature, strings in feat_snip_dict_lst.items()]

# Save to a JSON file
output_file = 'feature_top_samps_lst.json'
with open(output_file, 'w') as f:
    json.dump(feature_top_samps_lst, f, indent=4)

files.download(output_file)

