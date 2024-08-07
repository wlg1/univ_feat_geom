#!/usr/bin/env python
# coding: utf-8

# # setup

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system('pip install umap-learn')


# In[3]:


import pickle
import numpy as np
from google.colab import files

import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# # load weight mats

# In[4]:


# file_path = '/content/drive/MyDrive/ts-1L-21M_Wdec.pkl'
file_path = '/content/drive/MyDrive/Wdec_ts_1L_21M_df8192_steps100k_topK.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_1L_16384 = pickle.load(f)
print(weight_matrix_1L_16384.shape)
weight_matrix_1L_16384 = weight_matrix_1L_16384.detach().numpy()


# In[ ]:


file_path = '/content/drive/MyDrive/ts-2L-33M_Wdec.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_2L_16384 = pickle.load(f)
print(weight_matrix_2L_16384.shape)
weight_matrix_2L_16384 = weight_matrix_2L_16384.detach().numpy()


# In[5]:


# file_path = '/content/drive/MyDrive/ts_1L_21M_Wdec_df32768.pkl'
file_path = '/content/drive/MyDrive/Wdec_ts_1L_21M_df16384_steps100k_topK.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_1L_32768 = pickle.load(f)
print(weight_matrix_1L_32768.shape)
weight_matrix_1L_32768 = weight_matrix_1L_32768.detach().numpy()


# In[ ]:


file_path = '/content/drive/MyDrive/ts_2L_33M_Wdec_df32768.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_2L_32768 = pickle.load(f)
print(weight_matrix_2L_32768.shape)
weight_matrix_2L_32768 = weight_matrix_2L_32768.detach().numpy()


# # load feature labels

# In[9]:


import json
# with open('feature_top_samps_lst_1L_16k.json', 'rb') as f:
with open('feature_top_samps_lst_1L_8k_topk32_100ksteps.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[10]:


# with open('feature_top_samps_lst_1L_32k.json', 'rb') as f:
with open('feature_top_samps_lst_1L_16k_topk32_100ksteps.json', 'rb') as f:
    feat_snip_dict_2 = json.load(f)


# In[11]:


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


# In[12]:


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


# In[13]:


fList_model_B = []
for feat_dict in feat_snip_dict_2:
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_B.append(out_str)


# # umap combined data

# In[6]:


# combined_data = np.vstack((weight_matrix_1L_16384, data2, data3))
combined_data = np.vstack((weight_matrix_1L_16384, weight_matrix_1L_32768))

# Create and fit UMAP reducer on combined dataset with a fixed random seed
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean', random_state=42)
reducer.fit(combined_data)

# Transform each dataset using the same reducer
embedding_1_16384_32768 = reducer.transform(weight_matrix_1L_16384)
# embedding_2_16384_32768 = reducer.transform(weight_matrix_2L_16384)
embedding_3_16384_32768 = reducer.transform(weight_matrix_1L_32768)
# embedding_4_16384_32768 = reducer.transform(weight_matrix_2L_32768)


# In[7]:


weight_matrix_1L_16384.shape


# In[8]:


combined_data.shape


# In[14]:


# Create DataFrames for each embedding
df_A0 = pd.DataFrame(embedding_1_16384_32768, columns=['UMAP Component 1', 'UMAP Component 2'])
df_A0['Feature ID'] = range(len(fList_model_A))
df_A0['Feature Description'] = fList_model_A
df_A0['Run'] = 'Run A/0'

df_A1 = pd.DataFrame(embedding_3_16384_32768, columns=['UMAP Component 1', 'UMAP Component 2'])
df_A1['Feature ID'] = range(len(fList_model_B))
df_A1['Feature Description'] = fList_model_B
df_A1['Run'] = 'Run A/1'

# Combine the DataFrames
df_combined = pd.concat([df_A0, df_A1])

# Plot using Plotly
fig = px.scatter(df_combined, x='UMAP Component 1', y='UMAP Component 2', color='Run', text='Feature ID')

# Customize hover information
fig.update_traces(
    hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}<br><b>Run:</b> %{customdata[1]}',
    customdata=np.array(df_combined[['Feature Description', 'Run']])
)

fig.update_layout(
    title='UMAP of Decoder Weights',
    xaxis_title='UMAP Component 1',
    yaxis_title='UMAP Component 2'
)

fig.show()


# # load sae f actvs

# In[17]:


# file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps_v1.pkl'
file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_df_8192_steps100k_topK_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_A = pickle.load(f)


# In[19]:


# file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps__df-32768_v1.pkl'
file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_df_16384_steps100k_topK_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)


# In[20]:


feature_acts_model_B.shape


# In[21]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[22]:


reshaped_activations_A.shape


# In[23]:


reshaped_activations_B.shape


# # corr between fs saes

# In[24]:


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


# In[25]:


highest_correlations_indices, highest_correlations_values = find_all_highest_correlations(reshaped_activations_A, reshaped_activations_B)
print(f'Highest correlations indices: {len(highest_correlations_indices)}')
print(f'Highest correlations values: {len(highest_correlations_values)}')


# # load tokenizer

# In[26]:


from transformers import AutoTokenizer

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1Layer-21M")


# # interpret paired features

# ## load dataset tokens

# In[27]:


import pickle
file_path = '/content/drive/MyDrive/batch_tokens_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    batch_tokens = pickle.load(f)


# ## interpret

# In[28]:


get_ipython().run_line_magic('pip', 'install jaxtyping')


# In[29]:


import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple


# In[30]:


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


# In[31]:


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


# In[32]:


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

# In[33]:


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


# In[34]:


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


# # color corr fs features

# ## search and interpret

# In[35]:


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


# In[36]:


samp_m = 5 # get top samp_m tokens for all top feat_k feature neurons

for feature_idx_A, feature_idx_B in list(zip(modA_feats, modB_feats))[:5]:
# for feature_idx_A, feature_idx_B in zip(modA_feats, modB_feats):
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


# ## search and plot fn

# In[37]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1):
    # Find the indices with the keyword in model B
    modB_feats = find_indices_with_keyword(fList_model_B, keyword)
    # Get the corresponding indices in model A
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)

    # Create DataFrame for embedding 1
    df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
    df1['Feature ID'] = range(len(embedding1))
    df1['Feature Description'] = fList_model_A[:len(embedding1)]
    df1['Run'] = 'Run A/0'
    df1['Color'] = ['red' if i in modA_feats else 'blue' for i in df1['Feature ID']]

    # Create DataFrame for embedding 2
    df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
    df2['Feature ID'] = range(len(embedding2))
    df2['Feature Description'] = fList_model_B[:len(embedding2)]
    df2['Run'] = 'Run A/1'
    df2['Color'] = ['red' if i in modB_feats else 'green' for i in df2['Feature ID']]

    # Combine the DataFrames
    df_combined = pd.concat([df1, df2])

    # Create a scatter plot using Plotly
    fig = px.scatter(df_combined, x='UMAP Component 1', y='UMAP Component 2', color='Run', text='Feature ID',
                     color_discrete_map={'Run A/0': 'blue', 'Run A/1': 'green'})

    # Customize hover information
    fig.update_traces(
        hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}<br><b>Run:</b> %{customdata[1]}',
        customdata=np.array(df_combined[['Feature Description', 'Run']])
    )

    # Highlight the keyword features
    fig.add_trace(
        go.Scatter(
            x=df_combined[df_combined['Color'] == 'red']['UMAP Component 1'],
            y=df_combined[df_combined['Color'] == 'red']['UMAP Component 2'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='star'),
            text=df_combined[df_combined['Color'] == 'red']['Feature ID'],
            customdata=np.array(df_combined[df_combined['Color'] == 'red'][['Feature Description', 'Run']]),
            hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}<br><b>Run:</b> %{customdata[1]}',
            name='Keyword Features'
        )
    )

    # Update layout
    fig.update_layout(
        title='UMAP Projections of Feature Decoder Weights',
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        showlegend=True
    )

    fig.show()


# In[38]:


keyword = "upon"
keyword_umaps(keyword, embedding_1_16384_32768, embedding_3_16384_32768, fList_model_A, fList_model_B, highest_correlations_indices)


# In[39]:


print(modA_feats)
print(modB_feats)


# In[40]:


keyword = "time"
keyword_umaps(keyword, embedding_1_16384_32768, embedding_3_16384_32768, fList_model_A, fList_model_B, highest_correlations_indices)


# # plot from diff LLMs on same umap

# In[41]:


# combined_data = np.vstack((weight_matrix_1L_16384, weight_matrix_2L_16384, weight_matrix_1L_32768))
combined_data = np.vstack((weight_matrix_1L_16384, weight_matrix_2L_16384))

# Create and fit UMAP reducer on combined dataset with a fixed random seed # , random_state=42
reducer_2 = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')
reducer_2.fit(combined_data)

embedding_1_16384_32768 = reducer_2.transform(weight_matrix_1L_16384)
embedding_2_16384_32768 = reducer_2.transform(weight_matrix_2L_16384)
# embedding_3_16384_32768 = reducer.transform(weight_matrix_1L_32768)


# In[ ]:


with open('feature_top_samps_lst_2L_MLP0_16k.json', 'rb') as f:
    feat_snip_dict_3 = json.load(f)

fList_model_C = []
for feat_dict in feat_snip_dict_3:
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_C.append(out_str)


# In[ ]:


file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_C = pickle.load(f)


# In[ ]:


first_dim_reshaped = feature_acts_model_C.shape[0] * feature_acts_model_C.shape[1]
reshaped_activations_C = feature_acts_model_C.reshape(first_dim_reshaped, feature_acts_model_C.shape[-1]).cpu()


# In[ ]:


highest_correlations_indices_AC, highest_correlations_values_AC = find_all_highest_correlations(reshaped_activations_A, reshaped_activations_C)
print(f'Highest correlations indices: {len(highest_correlations_indices_AC)}')
print(f'Highest correlations values: {len(highest_correlations_values_AC)}')


# In[ ]:


keyword = "upon"
keyword_umaps(keyword, embedding_1_16384_32768, embedding_2_16384_32768, fList_model_A, fList_model_C, highest_correlations_indices_AC)


# # plot fs from diff LLMs on same umap

# In[ ]:


combined_data = np.vstack((weight_matrix_1L_16384, weight_matrix_2L_16384, weight_matrix_1L_32768))

# Create and fit UMAP reducer on combined dataset with a fixed random seed # , random_state=42
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')
reducer.fit(combined_data)

embedding_1_16384_32768 = reducer.transform(weight_matrix_1L_16384)
embedding_2_16384_32768 = reducer.transform(weight_matrix_2L_16384)
embedding_3_16384_32768 = reducer.transform(weight_matrix_1L_32768)


# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def keyword_umaps_3(keyword, embedding1, embedding2, embedding3, fList_model_A, fList_model_B, fList_model_C, highest_correlations_indices_v1):
    # Find the indices with the keyword in model B
    modB_feats = find_indices_with_keyword(fList_model_B, keyword)
    # Get the corresponding indices in model A
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)

    modC_feats = find_indices_with_keyword(fList_model_C, keyword)

    # Create DataFrame for embedding 1
    df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
    df1['Feature ID'] = range(len(embedding1))
    df1['Feature Description'] = fList_model_A[:len(embedding1)]
    df1['Run'] = '1L 16k'
    df1['Color'] = ['red' if i in modA_feats else 'blue' for i in df1['Feature ID']]

    # Create DataFrame for embedding 2
    df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
    df2['Feature ID'] = range(len(embedding2))
    df2['Feature Description'] = fList_model_B[:len(embedding2)]
    df2['Run'] = '1L 32k'
    df2['Color'] = ['red' if i in modB_feats else 'green' for i in df2['Feature ID']]

    df3 = pd.DataFrame(embedding3, columns=['UMAP Component 1', 'UMAP Component 2'])
    df3['Feature ID'] = range(len(embedding3))
    df3['Feature Description'] = fList_model_C[:len(embedding3)]
    df3['Run'] = '2L 16k'
    df3['Color'] = ['red' if i in modC_feats else 'orange' for i in df3['Feature ID']]
    # df3['Color'] = 'orange'

    # Combine the DataFrames
    df_combined = pd.concat([df1, df2, df3])

    # Create a scatter plot using Plotly
    fig = px.scatter(df_combined, x='UMAP Component 1', y='UMAP Component 2', color='Run', text='Feature ID',
                     color_discrete_map={'1L 16k': 'blue', '1L 32k': 'green', '2L 16k': 'orange'})

    # Customize hover information
    fig.update_traces(
        hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}<br><b>Run:</b> %{customdata[1]}',
        customdata=np.array(df_combined[['Feature Description', 'Run']])
    )

    # Highlight the keyword features
    fig.add_trace(
        go.Scatter(
            x=df_combined[df_combined['Color'] == 'red']['UMAP Component 1'],
            y=df_combined[df_combined['Color'] == 'red']['UMAP Component 2'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='star'),
            text=df_combined[df_combined['Color'] == 'red']['Feature ID'],
            customdata=np.array(df_combined[df_combined['Color'] == 'red'][['Feature Description', 'Run']]),
            hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}<br><b>Run:</b> %{customdata[1]}',
            name='Keyword Features'
        )
    )

    # Update layout
    fig.update_layout(
        title='UMAP Projections of Feature Decoder Weights',
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        showlegend=True
    )

    fig.show()


# In[ ]:


embedding_3_16384_32768.shape


# In[ ]:


len(fList_model_C)


# In[ ]:


keyword = "upon"
keyword_umaps_3(keyword, embedding_1_16384_32768, embedding_3_16384_32768, embedding_2_16384_32768, fList_model_A, fList_model_B, fList_model_C, highest_correlations_indices)


# In[ ]:




