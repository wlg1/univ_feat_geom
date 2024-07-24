#!/usr/bin/env python
# coding: utf-8

# # setup

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install umap-learn')


# In[ ]:


import pickle
import numpy as np
from google.colab import files

import umap
import matplotlib.pyplot as plt


# # load weight mats

# In[ ]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts_1L_21M_Wdec_df32768.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_np = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_np.shape)


# In[ ]:


weight_matrix_np = weight_matrix_np.detach().numpy()


# In[ ]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts_2L_33M_Wdec_df32768.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_2.shape)


# In[ ]:


weight_matrix_2 = weight_matrix_2.detach().numpy()


# # load sae f actvs

# In[69]:


file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps__df-32768_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_A = pickle.load(f)


# In[70]:


file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_df_32768_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)


# In[71]:


feature_acts_model_B.shape


# In[72]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[73]:


reshaped_activations_B.shape


# # load feature labels

# In[ ]:


import json
with open('feature_top_samps_lst_1L.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[ ]:


with open('feature_top_samps_lst_2L_MLP0.json', 'rb') as f:
    feat_snip_dict_2 = json.load(f)


# In[ ]:


import re

def extract_tagged_word(s):
    # Define the regex pattern to match the tagged word
    pattern = r'\[bold u dark_orange\](.*?)\[/\]'

    # Search for the pattern in the string
    match = re.search(pattern, s)

    # If a match is found, return the captured group (the word inside the tags)
    if match:
        return match.group(1)
    else:
        return None


# In[ ]:


fList_model_A = []
for feat_dict in feat_snip_dict:
    # text = feat_dict['strings'][0]
    # result = extract_tagged_word(text)
    # fList_model_A.append(result)
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_A.append(out_str)


# In[ ]:


fList_model_B = []
for feat_dict in feat_snip_dict_2:
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_B.append(out_str)


# # umap

# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# Fit and transform the data by rows
embedding1 = reducer.fit_transform(weight_matrix_np)
embedding2 = reducer.fit_transform(weight_matrix_2)


# In[ ]:


with open('embedding1.pkl', 'wb') as f:
    pickle.dump(embedding1, f)
files.download('embedding1.pkl')

with open('embedding2.pkl', 'wb') as f:
    pickle.dump(embedding2, f)
files.download('embedding2.pkl')


# ## load

# In[ ]:


import pickle
with open('embedding1.pkl', 'rb') as f:
    embedding1 = pickle.load(f)
with open('embedding2.pkl', 'rb') as f:
    embedding2 = pickle.load(f)


# # load corr

# only load if obtained corr in cells below already

# In[ ]:


# import pickle
# with open('highest_correlations_indices_v1.pkl', 'rb') as f:
#     highest_correlations_indices = pickle.load(f)
# with open('highest_correlations_values_v1.pkl', 'rb') as f:
#     highest_correlations_values = pickle.load(f)


# # corr mat

# ## free memory

# In[ ]:


import torch
torch.cuda.empty_cache()


# In[ ]:


del feature_acts_model_A
del feature_acts_model_B


# In[ ]:


import gc
gc.collect()


# ## get all actv corrs

# In[ ]:


# import torch
# import numpy as np

# def top_ind_from_B(ind, reshaped_activations_A, reshaped_activations_B):
#     # Select a column from matrix B
#     column_A = reshaped_activations_B[:, ind]

#     # Ensure tensors are on GPU
#     if torch.cuda.is_available():
#         reshaped_activations_A = reshaped_activations_A.to('cuda')
#         reshaped_activations_B = reshaped_activations_B.to('cuda')
#         column_A = column_A.to('cuda')

#     # Calculate means and standard deviations
#     mean_A = column_A.mean()
#     std_A = column_A.std()

#     # Mask columns with zero standard deviation
#     std_B = reshaped_activations_A.std(dim=0)
#     valid_columns_mask = std_B != 0

#     # Compute correlations for valid columns
#     valid_reshaped_activations_A = reshaped_activations_A[:, valid_columns_mask]
#     mean_B = valid_reshaped_activations_A.mean(dim=0)
#     std_B = valid_reshaped_activations_A.std(dim=0)

#     covariance = ((valid_reshaped_activations_A - mean_B) * (column_A - mean_A).unsqueeze(1)).mean(dim=0)
#     correlations = covariance / (std_A * std_B)

#     # Fill correlations with -inf where columns were invalid
#     all_correlations = torch.full((reshaped_activations_A.shape[1],), float('-inf')).to(correlations.device)
#     all_correlations[valid_columns_mask] = correlations

#     # Get the indices of the top 10 columns in B with the highest correlations
#     top_10_indices = torch.topk(all_correlations, 1).indices.cpu().numpy()
#     top_10_correlations = all_correlations[top_10_indices].cpu().numpy()

#     return top_10_indices, top_10_correlations


# In[ ]:


import torch

def find_all_highest_correlations(reshaped_activations_A, reshaped_activations_B, batch_size=1024):
    # Ensure tensors are on GPU
    if torch.cuda.is_available():
        reshaped_activations_A = reshaped_activations_A.to('cuda', dtype=torch.float16)
        reshaped_activations_B = reshaped_activations_B.to('cuda', dtype=torch.float16)

    # Normalize columns of A
    mean_A = reshaped_activations_A.mean(dim=0, keepdim=True)
    std_A = reshaped_activations_A.std(dim=0, keepdim=True)
    normalized_A = (reshaped_activations_A - mean_A) / (std_A + 1e-8)  # Avoid division by zero

    # Normalize columns of B
    mean_B = reshaped_activations_B.mean(dim=0, keepdim=True)
    std_B = reshaped_activations_B.std(dim=0, keepdim=True)
    normalized_B = (reshaped_activations_B - mean_B) / (std_B + 1e-8)  # Avoid division by zero

    # Determine the number of batches
    num_batches = (normalized_A.shape[1] + batch_size - 1) // batch_size  # Round up division

    highest_correlations_values = torch.full((normalized_B.shape[1],), float('-inf'), device=normalized_B.device, dtype=torch.float16)
    highest_correlations_indices = torch.full((normalized_B.shape[1],), -1, device=normalized_B.device, dtype=torch.long)

    # Process in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, normalized_A.shape[1])

        # Compute the correlation for the current batch
        correlation_matrix_batch = torch.matmul(normalized_A[:, start_idx:end_idx].t(), normalized_B) / normalized_A.shape[0]

        # Handle NaNs by setting them to -inf
        correlation_matrix_batch = torch.where(torch.isnan(correlation_matrix_batch), torch.tensor(float('-inf')).to(correlation_matrix_batch.device), correlation_matrix_batch)

        # Compare and update the highest correlation values and indices
        batch_values, batch_indices = correlation_matrix_batch.max(dim=0)
        mask = batch_values > highest_correlations_values
        highest_correlations_values[mask] = batch_values[mask]
        highest_correlations_indices[mask] = batch_indices[mask] + start_idx

    # Move results back to CPU
    highest_correlations_indices = highest_correlations_indices.cpu().numpy()
    highest_correlations_values = highest_correlations_values.cpu().numpy()

    return highest_correlations_indices, highest_correlations_values


# In[ ]:


# import torch

# def remove_outliers(tensor, threshold=3.0):
#     mean = tensor.mean(dim=0, keepdim=True)
#     std = tensor.std(dim=0, keepdim=True)
#     z_scores = (tensor - mean) / (std + 1e-8)  # Avoid division by zero
#     mask = torch.abs(z_scores) < threshold
#     return tensor * mask + mean * (~mask)

# def find_all_highest_correlations(reshaped_activations_A, reshaped_activations_B, batch_size=1024):
#     # Ensure tensors are on GPU
#     if torch.cuda.is_available():
#         reshaped_activations_A = reshaped_activations_A.to('cuda', dtype=torch.float16)
#         reshaped_activations_B = reshaped_activations_B.to('cuda', dtype=torch.float16)

#     # Remove outliers in A and B
#     reshaped_activations_A = remove_outliers(reshaped_activations_A)
#     reshaped_activations_B = remove_outliers(reshaped_activations_B)

#     # Normalize columns of A
#     mean_A = reshaped_activations_A.mean(dim=0, keepdim=True)
#     std_A = reshaped_activations_A.std(dim=0, keepdim=True)
#     normalized_A = (reshaped_activations_A - mean_A) / (std_A + 1e-8)  # Avoid division by zero

#     # Normalize columns of B
#     mean_B = reshaped_activations_B.mean(dim=0, keepdim=True)
#     std_B = reshaped_activations_B.std(dim=0, keepdim=True)
#     normalized_B = (reshaped_activations_B - mean_B) / (std_B + 1e-8)  # Avoid division by zero

#     # Determine the number of batches
#     num_batches = (normalized_A.shape[1] + batch_size - 1) // batch_size  # Round up division

#     highest_correlations_values = torch.full((normalized_B.shape[1],), float('-inf'), device=normalized_B.device, dtype=torch.float16)
#     highest_correlations_indices = torch.full((normalized_B.shape[1],), -1, device=normalized_B.device, dtype=torch.long)

#     # Process in batches
#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, normalized_A.shape[1])

#         # Compute the correlation for the current batch
#         correlation_matrix_batch = torch.matmul(normalized_A[:, start_idx:end_idx].t(), normalized_B) / normalized_A.shape[0]

#         # Handle NaNs by setting them to -inf
#         correlation_matrix_batch = torch.where(torch.isnan(correlation_matrix_batch), torch.tensor(float('-inf'), device=correlation_matrix_batch.device), correlation_matrix_batch)

#         # Compare and update the highest correlation values and indices
#         batch_values, batch_indices = correlation_matrix_batch.max(dim=0)
#         mask = batch_values > highest_correlations_values
#         highest_correlations_values[mask] = batch_values[mask]
#         highest_correlations_indices[mask] = batch_indices[mask] + start_idx

#     # Move results back to CPU
#     highest_correlations_indices = highest_correlations_indices.cpu().numpy()
#     highest_correlations_values = highest_correlations_values.cpu().numpy()

#     return highest_correlations_indices, highest_correlations_values


# In[ ]:


highest_correlations_indices, highest_correlations_values = find_all_highest_correlations(reshaped_activations_A, reshaped_activations_B)
print(f'Highest correlations indices: {len(highest_correlations_indices)}')
print(f'Highest correlations values: {len(highest_correlations_values)}')


# In[ ]:


highest_correlations_indices[:100]


# In[ ]:


highest_correlations_values[:100]


# ### save corrs

# In[ ]:


with open('highest_correlations_indices_v1.pkl', 'wb') as f:
    pickle.dump(highest_correlations_indices, f)
with open('highest_correlations_values_v1.pkl', 'wb') as f:
    pickle.dump(highest_correlations_values, f)


# In[ ]:


files.download('highest_correlations_indices_v1.pkl')
files.download('highest_correlations_values_v1.pkl')


# In[ ]:


# !cp batch_tokens_anySamps_v1.pkl /content/drive/MyDrive/


# # load tokenizer

# In[ ]:


from transformers import AutoTokenizer

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1Layer-21M")


# # interpret paired features

# ## load dataset tokens

# In[ ]:


import pickle
file_path = '/content/drive/MyDrive/batch_tokens_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    batch_tokens = pickle.load(f)


# ## interpret

# In[ ]:


get_ipython().run_line_magic('pip', 'install jaxtyping')


# In[ ]:


import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple


# In[66]:


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
    # flattened_feature_acts = feature_acts[:, feature_idx]

    top_acts_values, top_acts_indices = flattened_feature_acts.topk(k)
    # top_acts_values should be 1D
    # top_acts_indices should be also be 1D. Now, turn it back to 2D
    # Convert the indices into (batch, seq) indices
    top_acts_batch = top_acts_indices // seq_len
    top_acts_seq = top_acts_indices % seq_len

    return torch.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values


# In[67]:


from rich import print as rprint
def display_top_sequences(top_acts_indices, top_acts_values, batch_tokens):
    s = ""
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # s += f'{batch_idx}\n'
        s += f'batchID: {batch_idx}, '
        # Get the sequence as a string (with some padding on either side of our sequence)
        seq_start = max(seq_idx - 5, 0)
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        # Loop over the sequence, adding each token to the string (highlighting the token with the large activations)
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = tokenizer.decode([batch_tokens[batch_idx, i].item()]).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # Print the sequence, and the activation value
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'

    rprint(s)


# In[77]:


samp_m = 5 # get top samp_m tokens for all top feat_k feature neurons

for feature_idx_B, feature_idx_A in enumerate(highest_correlations_indices[:5]):
    print(f'Correlation: {highest_correlations_values[feature_idx_B]}')
    print('Model A Feature: ', feature_idx_A)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_A, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    # ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(reshaped_activations_A, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

    print('Model B Feature: ', feature_idx_B)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx_B, samp_m, batch_tokens=batch_tokens)
    # ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(reshaped_activations_B, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

    print('-'*50)


# ## interpret feats with highest corr

# In[75]:


# def get_top_n_correlated_indices(highest_correlations_indices, highest_correlations_values, n=10):
#     # Get the indices of the top n correlation values
#     top_n_indices = np.argsort(highest_correlations_values)[-n:][::-1]

#     # Retrieve the corresponding indices from A and B
#     top_n_indices_A = highest_correlations_indices[top_n_indices]
#     top_n_indices_B = top_n_indices

#     # Retrieve the corresponding top n correlation values
#     top_n_values = highest_correlations_values[top_n_indices]

#     return top_n_indices_A, top_n_indices_B, top_n_values

def get_top_n_correlated_indices(highest_correlations_indices, highest_correlations_values, n=10):
    # Filter out correlation values that are >= 1
    valid_indices = np.where(highest_correlations_values < 0.9)[0]
    filtered_values = highest_correlations_values[valid_indices]
    filtered_indices = highest_correlations_indices[valid_indices]

    # Get the indices of the top n correlation values
    top_n_indices = np.argsort(filtered_values)[-n:][::-1]

    # Retrieve the corresponding indices from A and B
    top_n_indices_A = filtered_indices[top_n_indices]
    top_n_indices_B = valid_indices[top_n_indices]

    # Retrieve the corresponding top n correlation values
    top_n_values = filtered_values[top_n_indices]

    return top_n_indices_A, top_n_indices_B, top_n_values

top_indices_A, top_indices_B, top_values = get_top_n_correlated_indices(highest_correlations_indices, highest_correlations_values, n=10)

for i in range(len(top_values)):
    print(f"Pair {i+1}: Index A = {top_indices_A[i]}, Index B = {top_indices_B[i]}, Correlation Value = {top_values[i]}")


# In[61]:


highest_correlations_indices[6088]


# In[62]:


highest_correlations_values[6088]


# In[63]:


fList_model_B[6088]


# In[65]:


fList_model_A[32578]


# In[76]:


samp_m = 5 # get top samp_m tokens for all top feat_k feature neurons

for feature_idx_A, feature_idx_B in zip(top_indices_A, top_indices_B):
# for feature_idx_B, feature_idx_A in enumerate(highest_correlations_indices[:3]):
    # print(f'Correlation: {highest_correlations_values[feature_idx_B]}')
    print(f'Correlation: {highest_correlations_values[feature_idx_B]}')
    print('Model A Feature: ', feature_idx_A)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_A, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    # ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(reshaped_activations_A, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

    print('Model B Feature: ', feature_idx_B)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx_B, samp_m, batch_tokens=batch_tokens)
    # ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(reshaped_activations_B, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

    print('-'*50)


# ## plot feature actv corrs

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


model_A_f_ind = 32055
model_B_f_ind = 25723

feature_0_actvs_A = reshaped_activations_A[:, model_A_f_ind].numpy()
feature_0_actvs_B = reshaped_activations_B[:, model_B_f_ind].numpy()

corr = np.corrcoef(feature_0_actvs_A, feature_0_actvs_B)[0, 1]
print(corr)

plt.scatter(feature_0_actvs_A, feature_0_actvs_B, alpha=0.5)

plt.xlabel('Feature Activations (Model A)')
plt.ylabel('Feature Activations (Model B)')
plt.title('Feature Activations (A/16251 vs B/11654)\n Corr = ' + str(corr))

plt.tight_layout()
plt.show()


# In[51]:


model_A_f_ind = 27935
model_B_f_ind = 22040

feature_0_actvs_A = reshaped_activations_A[:, model_A_f_ind].numpy()
feature_0_actvs_B = reshaped_activations_B[:, model_B_f_ind].numpy()

corr = np.corrcoef(feature_0_actvs_A, feature_0_actvs_B)[0, 1]
print(corr)

plt.scatter(feature_0_actvs_A, feature_0_actvs_B, alpha=0.5)

plt.xlabel('Feature Activations (Model A)')
plt.ylabel('Feature Activations (Model B)')
plt.title('Feature Activations (A/16251 vs B/11654)\n Corr = ' + str(corr))

plt.tight_layout()
plt.show()


# In[52]:


model_A_f_ind = 15556
model_B_f_ind = 18992

feature_0_actvs_A = reshaped_activations_A[:, model_A_f_ind].numpy()
feature_0_actvs_B = reshaped_activations_B[:, model_B_f_ind].numpy()

corr = np.corrcoef(feature_0_actvs_A, feature_0_actvs_B)[0, 1]
print(corr)

plt.scatter(feature_0_actvs_A, feature_0_actvs_B, alpha=0.5)

plt.xlabel('Feature Activations (Model A)')
plt.ylabel('Feature Activations (Model B)')
plt.title('Feature Activations (A/16251 vs B/11654)\n Corr = ' + str(corr))

plt.tight_layout()
plt.show()


# # search modB features with keyword, get modA f pair

# ISSUE WITH SEARCH: ‘king’ appears to be part of ‘talking’, etc

# In[ ]:


def find_indices_with_keyword(fList, keyword):
    """
    Find all indices of fList which contain the keyword in the string at those indices.

    Args:
    fList (list of str): List of strings to search within.
    keyword (str): Keyword to search for within the strings of fList.

    Returns:
    list of int: List of indices where the keyword is found within the strings of fList.
    """
    index_list = []
    for index, string in enumerate(fList):
        split_list = string.split(',')
        no_space_list = [i.replace(' ', '').lower() for i in split_list]
        if keyword in no_space_list:
            index_list.append(index)
    return index_list

def get_values_from_indices(indices, values_list):
    """
    Get the values from values_list at the specified indices.

    Args:
    indices (list of int): List of indices to retrieve values from.
    values_list (list): List of values from which to retrieve the specified indices.

    Returns:
    list: List of values from values_list at the specified indices.
    """
    return [values_list[index] for index in indices]

keyword = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices)
print(modA_feats)
print(modB_feats)


# In[ ]:


keyword = "king"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices)
print(modA_feats)
print(modB_feats)


# In[ ]:


print(fList_model_A[30967])
print(fList_model_B[5920])


# In[ ]:


keyword = "spot"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices)
print(modA_feats)
print(modB_feats)


# In[ ]:


print(fList_model_A[15713])
print(fList_model_B[3])


# # statically color points on 2 plots

# ## search and plot fn

# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1):
    modB_feats = find_indices_with_keyword(fList_model_B, keyword)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)

    df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
    df1['Feature ID'] = range(len(embedding1))
    df1['Feature Description'] = fList_model_A[:len(embedding1)]
    df1['Color'] = ['red' if i in modA_feats else 'blue' for i in df1['Feature ID']]

    df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
    df2['Feature ID'] = range(len(embedding2))
    df2['Feature Description'] = fList_model_B[:len(embedding2)]
    df2['Color'] = ['red' if i in modB_feats else 'blue' for i in df2['Feature ID']]

    # Create side by side plots using Plotly subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('UMAP Projection of SAE 1', 'UMAP Projection of SAE 2'))

    # Add first scatter plot
    fig.add_trace(
        go.Scatter(
            x=df1['UMAP Component 1'], y=df1['UMAP Component 2'],
            mode='markers', marker=dict(color=df1['Color']),
            text=df1['Feature ID'], customdata=np.array(df1[['Feature Description']]),
            hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}'
        ),
        row=1, col=1
    )

    # Add second scatter plot
    fig.add_trace(
        go.Scatter(
            x=df2['UMAP Component 1'], y=df2['UMAP Component 2'],
            mode='markers', marker=dict(color=df2['Color']),
            text=df2['Feature ID'], customdata=np.array(df2[['Feature Description']]),
            hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text='UMAP Projections of Feature Decoder Weights',
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        showlegend=False
    )

    fig.show()


# In[48]:


keyword = "king"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices)


# ## try other keywords

# In[49]:


keyword = "upon"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices)


# In[ ]:


keyword = "let"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices)


# In[ ]:


keyword = "spot"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices)

