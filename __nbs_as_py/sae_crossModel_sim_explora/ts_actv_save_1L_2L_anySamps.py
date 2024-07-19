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


# In[4]:


from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple


# # load model

# In[5]:


from transformer_lens import HookedTransformer


# In[6]:


model = HookedTransformer.from_pretrained("tiny-stories-1L-21M")


# # load sae using saelens

# ## load

# To get feature actvs, use sae_lens class method. To do this, you must load the sae as the sae class (wrapper over torch model).

# In[ ]:


sae_layer = "blocks.0.hook_mlp_out"


# In[ ]:


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


# In[ ]:


# https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L13
from sae_lens import SAEConfig

cfg_dict = cfg.to_dict()
cfg_dict['finetuning_scaling_factor'] = 0

sae_cfg = SAEConfig.from_dict(cfg_dict)
# sae = cls(sae_cfg)


# In[ ]:


from sae_lens import SAE
sae = SAE(sae_cfg)
# sae.cfg = sae_cfg


# In[ ]:


state_dict = torch.load('/content/drive/MyDrive/tiny-stories-1L-21M_sae_v1.pth')
sae.load_state_dict(state_dict)


# ## save decoder weights

# In[ ]:


weight_matrix_np = sae.W_dec.cpu()


# In[ ]:


import pickle
with open('ts-1L-21M_Wdec.pkl', 'wb') as f:
    pickle.dump(weight_matrix_np, f)


# In[ ]:


# from google.colab import files
# files.download('weight_matrix.pkl')


# In[ ]:


get_ipython().system('cp ts-1L-21M_Wdec.pkl /content/drive/MyDrive/')


# # load dataset

# Need load model tokenizer and sae params before obtain dataset

# In[7]:


from datasets import load_dataset


# ## any samples

# In[8]:


# sae.cfg.context_size


# In[9]:


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


# In[10]:


# sae.cfg.prepend_bos


# In[11]:


batch_tokens = token_dataset[:500]["tokens"]
batch_tokens.shape


# ### save data

# In[12]:


import pickle


# In[13]:


with open('batch_tokens_anySamps_v1.pkl', 'wb') as f:
    pickle.dump(batch_tokens, f)


# In[14]:


get_ipython().system('cp batch_tokens_anySamps_v1.pkl /content/drive/MyDrive/')


# In[15]:


# check if saved
file_path = '/content/drive/MyDrive/batch_tokens_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    batch_tokens = pickle.load(f)


# ## only get samples with specific concepts/words

# In[ ]:


# # Load the dataset in streaming mode
# dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# # Define the maximum sequence length for the model
# max_length = 128

# def contains_key_word(text):
#     words = ["king", "queen", "princess", "mother", "father", "daughter"]
#     # words = ["she"]
#     text_lower = text.lower()
#     for word in words:
#         if word in text_lower:
#             return True, word
#     return False, _

# # Define a function to get tokens in batches with truncation and padding
# def get_batch_tokens(dataset, tokenizer, batch_size=32, max_length=128):
#     sequences = []
#     love_hate_sequences = []
#     other_sequences = []
#     iterator = iter(dataset)  # Create an iterator from the streamed dataset
#     str_output = []
#     all_keywords = []

#     # Separate sequences into those containing "love" or "hate" and those that do not
#     # for _ in range(batch_size * 2):  # Load more to ensure we get enough samples
#     for _ in range(batch_size * 2):  # Load more to ensure we get enough samples
#         try:
#             # Get the next example from the dataset
#             example = next(iterator)
#             text = example['text']
#             containsBool, keyword = contains_key_word(text)
#             if containsBool:
#                 tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
#                 love_hate_sequences.append(tokens)
#                 str_output.append(text)
#                 all_keywords.append(keyword)
#             # else:
#             #     tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
#             #     other_sequences.append(tokens)
#         except StopIteration:
#             # If the dataset ends before reaching the required amount
#             break

#     # Ensure we have enough samples of each type
#     # min_length = min(len(love_hate_sequences), len(other_sequences))
#     # love_hate_sequences = love_hate_sequences[:min_length]
#     # other_sequences = other_sequences[:min_length]

#     # others_len = batch_size - len(love_hate_sequences)
#     # other_sequences = other_sequences[:others_len]

#     # Combine sequences to form the batch
#     # sequences = love_hate_sequences[:batch_size//2] + other_sequences[:batch_size//2]
#     sequences = love_hate_sequences

#     if sequences:
#         batch_tokens = torch.cat(sequences, dim=0).squeeze(1)
#         return batch_tokens, str_output, all_keywords
#     else:
#         return None

# # Get a batch of tokens
# batch_tokens, input_strs, all_keywords = get_batch_tokens(dataset, model.tokenizer, batch_size=100, max_length=max_length)
# if batch_tokens is not None:
#     print(batch_tokens.shape)
# else:
#     print("No data to load.")


# In[ ]:


# Load the dataset in streaming mode
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Define the maximum sequence length for the model
max_length = 128

def contains_key_word(text):
    words = ["king", "queen", "princess", "mother", "father", "daughter"]
    text_lower = text.lower()
    for word in words:
        if word in text_lower:
            return True, word
    return False, None

# Define a function to get tokens in batches with truncation and padding
def get_batch_tokens(dataset, tokenizer, batch_size=32, max_length=128):
    sequences = []
    love_hate_sequences = []
    other_sequences = []
    iterator = iter(dataset)  # Create an iterator from the streamed dataset
    str_output = []
    all_keywords = []

    # Separate sequences into those containing key words and those that do not
    while True:
        try:
            # Get the next example from the dataset
            example = next(iterator)
            text = example['text']
            containsBool, keyword = contains_key_word(text)
            if containsBool:
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
                love_hate_sequences.append(tokens)
                str_output.append(text)
                all_keywords.append(keyword)
            if len(love_hate_sequences) >= batch_size:
                break
        except StopIteration:
            # If the dataset ends before reaching the required amount
            break

    sequences = love_hate_sequences

    if sequences:
        batch_tokens = torch.cat(sequences, dim=0).squeeze(1)
        return batch_tokens, str_output, all_keywords
    else:
        return None, [], []

# Get a batch of tokens
batch_tokens, input_strs, all_keywords = get_batch_tokens(dataset, model.tokenizer, batch_size=10000, max_length=max_length)
if batch_tokens is not None:
    print(batch_tokens.shape)
else:
    print("No data to load.")


# In[ ]:


from collections import Counter
Counter(all_keywords)


# In[ ]:


# Load the dataset in streaming mode
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

samps_per_word = 20
max_length = 100

# Define a function to get tokens in batches with truncation and padding
def get_batch_tokens(dataset, tokenizer, samps_per_word=20, max_length=100):
    iterator = iter(dataset)  # Create an iterator from the streamed dataset
    sequences = []
    str_output = []

    keywords = ["king", "queen", "princess", "mother", "father", "daughter"]
    keyword_count = {word:0 for word in keywords}

    for word in keywords:
        while True:
            try:
                # don't reset it to prevent duplicate samples
                example = next(iterator) # Get the next example from the dataset
                text = example['text'].lower()
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
                # pdb.set_trace()
                text_trunc = model.tokenizer.decode(tokens[0])
                if word in text_trunc:
                    sequences.append(tokens)
                    str_output.append(text_trunc)
                    keyword_count[word] += 1
                if keyword_count[word] >= samps_per_word:
                    print(keyword_count[word])
                    break
            except StopIteration:
                # If the dataset ends before reaching the required amount
                break

    if sequences:
        batch_tokens = torch.cat(sequences, dim=0).squeeze(1)
        return batch_tokens, str_output, keyword_count
    else:
        return None, [], []

batch_tokens, input_strs, keyword_count = get_batch_tokens(dataset, model.tokenizer, samps_per_word=samps_per_word, max_length=max_length)
if batch_tokens is not None:
    print(batch_tokens.shape)
else:
    print("No data to load.")


# In[ ]:


from collections import Counter
Counter(keyword_count)


# # model 1- interpret features

# ## get LLM actvs

# In[ ]:


layer_name = sae_layer


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


# ## save actvs

# Now you have to save actvs, bc saelens not compatible with umap OR cosine sim lib

# In[ ]:


import pickle


# In[ ]:


with open('fActs_ts_1L_21M_anySamps_v1.pkl', 'wb') as f:
    pickle.dump(feature_acts, f)


# In[ ]:


get_ipython().system('cp fActs_ts_1L_21M_anySamps_v1.pkl /content/drive/MyDrive/')


# In[ ]:


# check if saved
file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts = pickle.load(f)


# ## get top features

# In[ ]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
# if want to get highest batch, use dim=0
feat_k = 10
top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

print(top_acts_indices.shape)
top_acts_values.shape


# ## interpret top features by dataset examples

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


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model B
# top_feats = [3383, 8341]

for feature_idx in top_acts_indices[0, -1, :]:
# for feature_idx in top_feats:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# ## only get features with keyword

# In[ ]:


from rich import print as rprint


# In[ ]:


def store_top_sequences(top_acts_indices, top_acts_values, batch_tokens):
    s = ""
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        s += f'batchID: {batch_idx}, '
        seq_start = max(seq_idx - 5, 0)
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'
    return s


# In[ ]:


# store feature : str of top 5 snippets (contains batchID, val, seq in each substr)
feat_snip_dict = {}
samp_m = 5

for feature_idx in range(feature_acts.shape[-1]):
    if feature_idx % 500 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, samp_m, batch_tokens=batch_tokens)
    feat_snip_dict[feature_idx] = store_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens)


# In[ ]:


relevant_features = {}


# In[ ]:


# keyword = 'princess'
keyword = '[bold u dark_orange] princess[/]'
relevant_features[keyword] = []
for f_id, str_output in feat_snip_dict.items():
    if keyword in str_output:
        relevant_features[keyword].append(f_id)
        print(f'Feature: {f_id}')
        rprint(str_output)


# In[ ]:


len(relevant_features[keyword])


# ## save datasamps for features

# In[ ]:


def store_top_sequences_asLst(top_acts_indices, top_acts_values, batch_tokens):
    # s = ""
    feat_samps = []
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # s += f'batchID: {batch_idx}, '
        seq_start = max(seq_idx - 5, 0)
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # s += f'Act = {value:.2f}, Seq = "{seq}"\n'
        feat_samps.append(seq)
    return feat_samps


# In[ ]:


# store feature : lst of top strs
feat_snip_dict = {}
samp_m = 5

for feature_idx in range(feature_acts.shape[-1]):
    if feature_idx % 500 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, samp_m, batch_tokens=batch_tokens)
    feat_snip_dict[feature_idx] = store_top_sequences_asLst(ds_top_acts_indices, ds_top_acts_values, batch_tokens)


# In[ ]:


# store feature : str of top 5 snippets (contains batchID, val, seq in each substr)
feat_snip_dict_strs = {}
samp_m = 5

for feature_idx in range(feature_acts.shape[-1]):
    if feature_idx % 500 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, samp_m, batch_tokens=batch_tokens)
    feat_snip_dict_strs[feature_idx] = store_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens)


# In[ ]:


import json

# Transform the dictionary into a list of dictionaries
feature_top_samps_lst = [{"feature": feature, "strings": strings} for feature, strings in feat_snip_dict.items()]

# Save to a JSON file
output_file = 'feature_top_samps_lst.json'
with open(output_file, 'w') as f:
    json.dump(feature_top_samps_lst, f, indent=4)

print(f"Data saved to {output_file}")


# In[ ]:


from google.colab import files
files.download(output_file)


# In[ ]:


import pickle
with open('feat_snip_dict_strs.pkl', 'wb') as f:
    pickle.dump(feat_snip_dict_strs, f)
files.download('feat_snip_dict_strs.pkl')


# # model 2- interpret features

# ## load model and sae

# In[ ]:


model_2 = HookedTransformer.from_pretrained("tiny-stories-2L-33M")


# In[ ]:


sae_layer = "blocks.0.hook_mlp_out"

model_name = "tiny-stories-2L-33M"
layer_name = "blocks.1.hook_mlp_out"
hook_layer = 1
d_in = 1024
wandb_project = model_name+"_MLP1_sae"


# In[ ]:


total_training_steps = 30_000  # probably we should do more
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


# In[ ]:


# https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L13
from sae_lens import SAEConfig

cfg_dict = cfg.to_dict()
cfg_dict['finetuning_scaling_factor'] = 0

sae_cfg = SAEConfig.from_dict(cfg_dict)


# In[ ]:


from sae_lens import SAE
sae_2 = SAE(sae_cfg)


# In[ ]:


state_dict = torch.load('/content/drive/MyDrive/tiny-stories-2L-33M_MLP0_sae.pth')
sae_2.load_state_dict(state_dict)


# ## save decoder weights

# In[ ]:


weight_matrix_np = sae_2.W_dec.cpu()


# In[ ]:


import pickle
with open('ts-2L-33M_Wdec.pkl', 'wb') as f:
    pickle.dump(weight_matrix_np, f)


# In[ ]:


# from google.colab import files
# files.download('weight_matrix.pkl')


# In[ ]:


get_ipython().system('cp ts-2L-33M_Wdec.pkl /content/drive/MyDrive/')


# ## get LLM actvs

# In[ ]:


layer_name = sae_layer


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


model_2.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# ## get SAE actvs

# In[ ]:


sae_2.eval()  # prevents error if we're expecting a dead neuron mask for who grads
with torch.no_grad():
    feature_acts_2 = sae_2.encode(h_store)


# In[ ]:


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
count_nonzero_features(feature_acts_2)


# ## save actvs

# Now you have to save actvs, bc saelens not compatible with umap OR cosine sim lib

# In[ ]:


with open('fActs_ts_2L_33M_anySamps_v1.pkl', 'wb') as f:
    pickle.dump(feature_acts_2, f)


# In[ ]:


get_ipython().system('cp fActs_ts_2L_33M_anySamps_v1.pkl /content/drive/MyDrive/')


# In[ ]:


# check if saved
file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_2 = pickle.load(f)


# ## get top features

# In[ ]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
# if want to get highest batch, use dim=0
feat_k = 100
top_acts_values, top_acts_indices = feature_acts_2.topk(feat_k, dim=-1)

print(top_acts_indices.shape)
top_acts_values.shape


# ## interpret top features by dataset examples

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


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_2, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# ## only get features with keyword

# In[ ]:


from rich import print as rprint


# In[ ]:


def store_top_sequences(top_acts_indices, top_acts_values, batch_tokens):
    s = ""
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        s += f'batchID: {batch_idx}, '
        seq_start = max(seq_idx - 5, 0)
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'
    return s


# In[ ]:


# store feature : str of top 5 snippets (contains batchID, val, seq in each substr)
feat_snip_dict_2 = {}
samp_m = 5

for feature_idx in range(feature_acts_2.shape[-1]):
    if feature_idx % 500 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_2, feature_idx, samp_m, batch_tokens=batch_tokens)
    feat_snip_dict_2[feature_idx] = store_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens)


# In[ ]:


relevant_features = {}
# keyword = 'princess'
keyword = '[bold u dark_orange] princess[/]'
relevant_features[keyword] = []
for f_id, str_output in feat_snip_dict_2.items():
    if keyword in str_output:
        relevant_features[keyword].append(f_id)
        print(f'Feature: {f_id}')
        rprint(str_output)


# In[ ]:


len(relevant_features[keyword])


# ## save ds for features

# In[ ]:


def store_top_sequences_asLst(top_acts_indices, top_acts_values, batch_tokens):
    # s = ""
    feat_samps = []
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # s += f'batchID: {batch_idx}, '
        seq_start = max(seq_idx - 5, 0)
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # s += f'Act = {value:.2f}, Seq = "{seq}"\n'
        feat_samps.append(seq)
    return feat_samps


# In[ ]:


# store feature : lst of top strs
feat_snip_dict = {}
samp_m = 5

for feature_idx in range(feature_acts_2.shape[-1]):
    if feature_idx % 500 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_2, feature_idx, samp_m, batch_tokens=batch_tokens)
    feat_snip_dict[feature_idx] = store_top_sequences_asLst(ds_top_acts_indices, ds_top_acts_values, batch_tokens)


# In[ ]:


import json

# Transform the dictionary into a list of dictionaries
feature_top_samps_lst = [{"feature": feature, "strings": strings} for feature, strings in feat_snip_dict.items()]

# Save to a JSON file
output_file = 'feature_top_samps_lst_2L_MLP0.json'
with open(output_file, 'w') as f:
    json.dump(feature_top_samps_lst, f, indent=4)

print(f"Data saved to {output_file}")


# In[ ]:


from google.colab import files
files.download(output_file)


# In[ ]:


# store feature : str of top 5 snippets (contains batchID, val, seq in each substr)
feat_snip_dict_strs = {}
samp_m = 5

for feature_idx in range(feature_acts_2.shape[-1]):
    if feature_idx % 500 == 0:
        print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_2, feature_idx, samp_m, batch_tokens=batch_tokens)
    feat_snip_dict_strs[feature_idx] = store_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens)


# In[ ]:


import pickle
with open('feat_snip_dict_strs_2L_MLP0.pkl', 'wb') as f:
    pickle.dump(feat_snip_dict_strs, f)
files.download('feat_snip_dict_strs_2L_MLP0.pkl')

