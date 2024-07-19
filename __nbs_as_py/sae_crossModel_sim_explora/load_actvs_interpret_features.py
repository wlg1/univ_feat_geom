#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_line_magic('pip', 'install transformer-lens')


# In[ ]:


import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple


# # load model

# In[ ]:


from transformer_lens import HookedTransformer


# In[ ]:


model = HookedTransformer.from_pretrained("tiny-stories-1L-21M")


# # load dataset

# only get samples with specific concepts/words

# In[ ]:


from datasets import load_dataset


# In[ ]:


# Load the dataset in streaming mode
# dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Define the maximum sequence length for the model
max_length = 128

# Function to check if text contains the words "love" or "hate"
def contains_love_or_hate(text):
    # return "love" in text.lower() or "hate" in text.lower()
    return "she" in text.lower() or "her" in text.lower()

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


# In[ ]:


import pickle


# In[ ]:


with open('batch_tokens_sheHer.pkl', 'wb') as f:
    pickle.dump(batch_tokens, f)


# In[ ]:


get_ipython().system('cp batch_tokens_sheHer.pkl /content/drive/MyDrive/')


# In[ ]:


# check if saved
file_path = '/content/drive/MyDrive/batch_tokens_sheHer.pkl'
with open(file_path, 'rb') as f:
    batch_tokens = pickle.load(f)


# # load sae features

# In[ ]:


file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_sheHer.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_A = pickle.load(f)


# In[ ]:


file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_sheHer.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)


# # interpret top features by dataset examples

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


# 16244 in model A

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model B
top_feats = [15028,  3580,  7634,  2208,  9768,  7110,  8427,   239,  3023,
        5859]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


# feature 0 in model A

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model B
top_feats = [ 7219,  2090, 13160,  5793,  9053,  3751,  1351,  1786, 13994,
       10304]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# 1786 comes from above

# In[ ]:


# feature 1 in model A

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model B
top_feats = [  657,  4223,  9950, 12712,  4451,  7606,  5630,  2945, 12266,
        5642]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# # B to A

# In[ ]:


# feature 1786 in model B

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model A
top_feats = [16251, 13152, 14923, 10373, 16144,  7364,  9425, 15399,  5912,
        3655]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_A, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


top_feats = [1786]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


# feature 3103 in model B

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model A
top_feats = [14923, 16251, 13152, 13166, 10373, 16144,  7364, 15399,  9425,
        5912]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_A, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


top_feats = [3103]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

