#!/usr/bin/env python
# coding: utf-8

# # setup

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# !pip install umap-learn


# In[7]:


import pickle
import numpy as np

# import umap
import matplotlib.pyplot as plt


# # load weight mats

# In[8]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-1L-21M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_np = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_np.shape)


# In[9]:


weight_matrix_np = weight_matrix_np.detach().numpy()


# In[10]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-2L-33M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_2.shape)


# # mnn

# In[ ]:


import torch
from sklearn.neighbors import NearestNeighbors

# Number of nearest neighbors to find
k = 5

# Find k-nearest neighbors for matrix1 in matrix2
nn1 = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(weight_matrix_2)
distances1, indices1 = nn1.kneighbors(weight_matrix_np)

# Find k-nearest neighbors for matrix2 in matrix1
nn2 = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(weight_matrix_np)
distances2, indices2 = nn2.kneighbors(weight_matrix_2)

# Identify mutual nearest neighbors
mutual_neighbors = []
for i in range(len(indices1)):
    for neighbor in indices1[i]:
        if i in indices2[neighbor]:
            mutual_neighbors.append((i, neighbor))

# mutual_neighbors now contains pairs of indices that are mutual nearest neighbors
print(f"Found {len(mutual_neighbors)} mutual nearest neighbors.")


# In[ ]:


# Assume mutual_neighbors is a list of pairs (i, j)
mutual_neighbors = np.array(mutual_neighbors)

# Print some of the mutual nearest neighbor pairs
print("Some mutual nearest neighbor pairs (indices from matrix1, indices from matrix2):")
print(mutual_neighbors[:10])  # Printing the first 10 pairs

# If you want to visualize or analyze these pairs further, you can use the indices to extract corresponding points
# Example: Extracting the first mutual nearest neighbor pair
i, j = mutual_neighbors[0]
point_from_matrix1 = weight_matrix_np[i]
point_from_matrix2 = weight_matrix_2[j]

print(f"Point from matrix1 (index {i}): {point_from_matrix1}")
print(f"Point from matrix2 (index {j}): {point_from_matrix2}")


# In[11]:


weight_matrix_2 = weight_matrix_2.detach().numpy()


# # load sae f actvs

# In[ ]:


file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_A = pickle.load(f)


# In[ ]:


file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)


# In[ ]:


feature_acts_model_B.shape


# In[ ]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[ ]:


reshaped_activations_B.shape


# # load feature labels

# In[ ]:


import json
with open('feature_top_samps_lst_1L_16k.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[ ]:


with open('feature_top_samps_lst_2L_MLP0_16k.json', 'rb') as f:
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


# ## save spliced labels

# In[ ]:


with open('fList_model_A.pkl', 'wb') as f:
    pickle.dump(fList_model_A, f)
with open('fList_model_B.pkl', 'wb') as f:
    pickle.dump(fList_model_B, f)


# In[ ]:


from google.colab import files
files.download('fList_model_A.pkl')
files.download('fList_model_B.pkl')


# In[ ]:


import pickle
with open('fList_model_A.pkl', 'rb') as f:
    fList_model_A = pickle.load(f)
with open('fList_model_B.pkl', 'rb') as f:
    fList_model_B = pickle.load(f)


# # load corr

# In[ ]:


import pickle
with open('highest_corr_inds_1L_2L_MLP0_16k_30k_relu.pkl', 'rb') as f:
    highest_correlations_indices_v1 = pickle.load(f)
with open('highest_corr_vals_1L_2L_MLP0_16k_30k_relu.pkl', 'rb') as f:
    highest_correlations_values_v1 = pickle.load(f)


# # cca on two models

# In[ ]:


from sklearn.cross_decomposition import CCA


# ## directly

# In[ ]:


n_comp=2 #choose number of canonical variates pairs
cca = CCA(scale=False, n_components=n_comp) #define CCA
cca.fit(weight_matrix_np, weight_matrix_2)

# Transform the data
A_c, B_c = cca.transform(weight_matrix_np, weight_matrix_2)

comp_corr = [np.corrcoef(A_c[:, i], B_c[:, i])[1][0] for i in range(n_comp)]


# In[ ]:


comp_corr


# In[ ]:


plt.bar(['CC1', 'CC2'], comp_corr, color='lightgrey', width = 0.8, edgecolor='k')


# ## on umap embed

# In[ ]:


cca = CCA(n_components=2)
cca.fit(embedding1, embedding2) # Fit the model after UMAP

# Transform the data
A_c, B_c = cca.transform(embedding1, embedding2)

# A_c and B_c are the transformed data in the canonical space
print("Canonical Correlations:")
print(cca.score(embedding1, embedding2))

# Optional: To see the correlation coefficients
corrs = [np.corrcoef(A_c[:, i], B_c[:, i])[0, 1] for i in range(A_c.shape[1])]
print("Correlation Coefficients:", corrs)


# # corr mat

# ## plot feature actv corrs

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


model_A_f_ind = 11654
model_B_f_ind = 3103

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


# ## get all actv corrs

# In[ ]:


import torch
import numpy as np

def top_ind_from_B(ind, reshaped_activations_A, reshaped_activations_B):
    # Select a column from matrix B
    column_A = reshaped_activations_B[:, ind]

    # Ensure tensors are on GPU
    if torch.cuda.is_available():
        reshaped_activations_A = reshaped_activations_A.to('cuda')
        reshaped_activations_B = reshaped_activations_B.to('cuda')
        column_A = column_A.to('cuda')

    # Calculate means and standard deviations
    mean_A = column_A.mean()
    std_A = column_A.std()

    # Mask columns with zero standard deviation
    std_B = reshaped_activations_A.std(dim=0)
    valid_columns_mask = std_B != 0

    # Compute correlations for valid columns
    valid_reshaped_activations_A = reshaped_activations_A[:, valid_columns_mask]
    mean_B = valid_reshaped_activations_A.mean(dim=0)
    std_B = valid_reshaped_activations_A.std(dim=0)

    covariance = ((valid_reshaped_activations_A - mean_B) * (column_A - mean_A).unsqueeze(1)).mean(dim=0)
    correlations = covariance / (std_A * std_B)

    # Fill correlations with -inf where columns were invalid
    all_correlations = torch.full((reshaped_activations_A.shape[1],), float('-inf')).to(correlations.device)
    all_correlations[valid_columns_mask] = correlations

    # Get the indices of the top 10 columns in B with the highest correlations
    top_10_indices = torch.topk(all_correlations, 1).indices.cpu().numpy()
    top_10_correlations = all_correlations[top_10_indices].cpu().numpy()

    return top_10_indices, top_10_correlations


# In[ ]:


def find_all_highest_correlations(reshaped_activations_A, reshaped_activations_B):
    # Ensure tensors are on GPU
    if torch.cuda.is_available():
        reshaped_activations_A = reshaped_activations_A.to('cuda')
        reshaped_activations_B = reshaped_activations_B.to('cuda')

    # Normalize columns of A
    mean_A = reshaped_activations_A.mean(dim=0, keepdim=True)
    std_A = reshaped_activations_A.std(dim=0, keepdim=True)
    normalized_A = (reshaped_activations_A - mean_A) / (std_A + 1e-8)  # Avoid division by zero

    # Normalize columns of B
    mean_B = reshaped_activations_B.mean(dim=0, keepdim=True)
    std_B = reshaped_activations_B.std(dim=0, keepdim=True)
    normalized_B = (reshaped_activations_B - mean_B) / (std_B + 1e-8)  # Avoid division by zero

    # Compute correlation matrix
    correlation_matrix = torch.matmul(normalized_A.t(), normalized_B) / normalized_A.shape[0]

    # Handle NaNs by setting them to -inf
    correlation_matrix = torch.where(torch.isnan(correlation_matrix), torch.tensor(float('-inf')).to(correlation_matrix.device), correlation_matrix)

    # Get the highest correlation indices and values
    highest_correlations_values, highest_correlations_indices = correlation_matrix.max(dim=0)

    # Move results back to CPU
    highest_correlations_indices = highest_correlations_indices.cpu().numpy()
    highest_correlations_values = highest_correlations_values.cpu().numpy()

    return highest_correlations_indices, highest_correlations_values

highest_correlations_indices, highest_correlations_values = find_all_highest_correlations(reshaped_activations_A, reshaped_activations_B)
print(f'Highest correlations indices: {highest_correlations_indices}')
print(f'Highest correlations values: {highest_correlations_values}')


# In[ ]:


highest_correlations_indices[:100]


# ### save corrs

# In[ ]:


import pickle


# In[ ]:


with open('highest_correlations_indices_v1.pkl', 'wb') as f:
    pickle.dump(highest_correlations_indices, f)
with open('highest_correlations_values_v1.pkl', 'wb') as f:
    pickle.dump(highest_correlations_values, f)


# In[ ]:


from google.colab import files
files.download('highest_correlations_indices_v1.pkl')
files.download('highest_correlations_values_v1.pkl')


# In[ ]:


# !cp batch_tokens_anySamps_v1.pkl /content/drive/MyDrive/


# # load model

# In[ ]:


from transformers import AutoTokenizer

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1Layer-21M")


# In[ ]:


# Convert a single string to a token ID
single_string = "example"
token_id = tokenizer.encode(single_string, add_special_tokens=False)[0]
decoded_string = tokenizer.decode([token_id])
print(decoded_string)


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


# In[ ]:


samp_m = 5 # get top samp_m tokens for all top feat_k feature neurons

for feature_idx_B, feature_idx_A in enumerate(highest_correlations_indices[:10]):
    print(f'Correlation: {highest_correlations_values[feature_idx_B]}')
    print('Model A Feature: ', feature_idx_A)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_A, feature_idx_A, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

    print('Model B Feature: ', feature_idx_B)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx_B, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)

    print('-'*50)


# # search modB features with keyword, get modA f pair

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
    return [index for index, string in enumerate(fList) if keyword in string]

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
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
print(modA_feats)
print(modB_feats)


# ISSUE WITH SEARCH: ‘king’ appears to be part of ‘talking’, etc

# In[ ]:


split_list = fList_model_B[0].split(',')
[i.replace(' ', '').lower() for i in split_list]


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


# In[ ]:


keyword = "king"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
print(modA_feats)
print(modB_feats)


# In[ ]:


print(fList_model_A[11920])
print(fList_model_B[5430])


# In[ ]:


keyword = "spot"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
print(modA_feats)
print(modB_feats)


# In[ ]:


print(fList_model_A[0])
print(fList_model_B[12])


# # statically color points on 2 plots

# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Create DataFrames for Plotly
df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
df1['Feature ID'] = range(len(embedding1))
df1['Feature Description'] = fList_model_A[:len(embedding1)]  # Adjust this if needed
# df1['Color'] = ['yellow' if i in highest_correlations_indices_v1[0:16000] else 'blue' for i in df1['Feature ID']]
df1['Color'] = ['red' if i in modA_feats else 'blue' for i in df1['Feature ID']]

df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
df2['Feature ID'] = range(len(embedding2))
df2['Feature Description'] = fList_model_B[:len(embedding2)]  # Adjust this if needed
# df2['Color'] = ['yellow' if i in list(range(16000)) else 'red' for i in df2['Feature ID']]
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


# In[ ]:


keyword = "princess"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# ## try other keywords

# In[ ]:


keyword = "let"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# In[ ]:


keyword = "saw"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# In[ ]:


keyword = "spot"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# In[ ]:


print(fList_model_A[0])
print(fList_model_B[12])


# In[ ]:


keyword = "king"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# # save umap as html

# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def keyword_umaps_html(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1, output_filename='umap_plot.html'):
    """
    keyword is a string for which points should be in red instead of blue
    embedding1 is an array of (x,y) pts for scattterplot 1; WLOG for scatterplot 2
        Each embedding has the same number of pts
        Each pt is an SAE feature decoder weight vector embedded in 2D umap space
        Each embedding is for an SAE feature decoder weight matrix
    fList_model_A is a list of strings (labels) for every pt in embedding1
        Each pt has a string of the top 5 tokens that the feature activates highest on
        WLOG, fList_model_B is a list of strings for every pt in embedding2
        when a point is hovered over, its ID and string of top 5 tokens is displayed
    highest_correlations_indices_v1 is a list in which indices are point ids in scatter plot 2
        and the values are the mapped indices in scatter plot 1
        Eg) [5 3 0] : feature 0 in embedding2 is mapped to feature 5 in embedding1
        This mapping means feature 0's highest correlated feature in embedding1 is feature 5
        So given a set of tokens, these two features have the highest corr score
    output_filename is the name of the html file to save the plot to

    (vars within fn)
    modB_feats: a list of features that contain the keyword in its string from fList_model_B
        Eg) keyword is "Upon", and for feature 3, fList_model_A[3] = 'upon Upon king upon up'
         Thus, we include 3 in modB_feats. If it didn't have 'upon', we don't include it.
    modA_feats: the value in highest_correlations_indices_v1 for every ind in modB_feats
        Eg) 3 is in modB_feats. highest_correlations_indices_v1[3] = 6, so 6 is in modA_feats

    AIM: modify so hovering over a point in one scatterplot will create a hover box over another a
        point in a second hover box, based on a list "highest_correlations_indices" in which indices
        are point ids in scatter plot 2 and the values are the mapped indices in scatter plot 1.

    """
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

    # Save the figure as an HTML file
    fig.write_html(output_filename)

    print(f"Plot saved as {output_filename}")


# In[ ]:


keyword = "once"
outputFN = 'ts_1L_2L_MLP0_16k_30k_relu.html'
keyword_umaps_html(keyword, embedding1, embedding2, fList_model_A, fList_model_B,
                   highest_correlations_indices_v1, output_filename = outputFN)
files.download(outputFN)


# In[ ]:


keyword = "upon"
outputFN = 'ts_1L_2L_MLP0_16k_30k_relu.html'
keyword_umaps_html(keyword, embedding1, embedding2, fList_model_A, fList_model_B,
                   highest_correlations_indices_v1, output_filename = outputFN)
files.download(outputFN)


# ## color two keywords

# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from google.colab import files

def keyword_umaps_html_2(keyword, kw2, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1, output_filename='umap_plot.html'):
    """
    keyword is a string for which points should be in red instead of blue
    embedding1 is an array of (x,y) pts for scattterplot 1; WLOG for scatterplot 2
        Each embedding has the same number of pts
        Each pt is an SAE feature decoder weight vector embedded in 2D umap space
        Each embedding is for an SAE feature decoder weight matrix
    fList_model_A is a list of strings (labels) for every pt in embedding1
        Each pt has a string of the top 5 tokens that the feature activates highest on
        WLOG, fList_model_B is a list of strings for every pt in embedding2
        when a point is hovered over, its ID and string of top 5 tokens is displayed
    highest_correlations_indices_v1 is a list in which indices are point ids in scatter plot 2
        and the values are the mapped indices in scatter plot 1
        Eg) [5 3 0] : feature 0 in embedding2 is mapped to feature 5 in embedding1
        This mapping means feature 0's highest correlated feature in embedding1 is feature 5
        So given a set of tokens, these two features have the highest corr score
    output_filename is the name of the html file to save the plot to

    (vars within fn)
    modB_feats: a list of features that contain the keyword in its string from fList_model_B
        Eg) keyword is "Upon", and for feature 3, fList_model_A[3] = 'upon Upon king upon up'
         Thus, we include 3 in modB_feats. If it didn't have 'upon', we don't include it.
    modA_feats: the value in highest_correlations_indices_v1 for every ind in modB_feats
        Eg) 3 is in modB_feats. highest_correlations_indices_v1[3] = 6, so 6 is in modA_feats

    AIM: modify so hovering over a point in one scatterplot will create a hover box over another a
        point in a second hover box, based on a list "highest_correlations_indices" in which indices
        are point ids in scatter plot 2 and the values are the mapped indices in scatter plot 1.

    """
    modB_feats = find_indices_with_keyword(fList_model_B, keyword)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    modB_feats_2 = find_indices_with_keyword(fList_model_B, kw2)
    modA_feats_2 = get_values_from_indices(modB_feats_2, highest_correlations_indices_v1)

    df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
    df1['Feature ID'] = range(len(embedding1))
    df1['Feature Description'] = fList_model_A[:len(embedding1)]
    df1['Color'] = [
        'red' if i in modA_feats else 'green' if i in modA_feats_2 else 'blue'
        for i in df1['Feature ID']
    ]

    df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
    df2['Feature ID'] = range(len(embedding2))
    df2['Feature Description'] = fList_model_B[:len(embedding2)]
    df2['Color'] = [
        'red' if i in modB_feats else 'green' if i in modB_feats_2 else 'blue'
        for i in df2['Feature ID']
    ]

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

    # Save the figure as an HTML file
    fig.write_html(output_filename)

    print(f"Plot saved as {output_filename}")


# In[ ]:


keyword = "once"
kw2 = "upon"
outputFN = 'ts_1L_2L_MLP0_16k_30k_relu_v2.html'
keyword_umaps_html_2(keyword, kw2, embedding1, embedding2, fList_model_A, fList_model_B,
                   highest_correlations_indices_v1, output_filename = outputFN)
files.download(outputFN)


# ## try hover sync both subplots

# In[ ]:


modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)

df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
df1['Feature ID'] = range(len(embedding1))
df1['Feature Description'] = fList_model_A[:len(embedding1)]
df1['Color'] = ['red' if i in modA_feats else 'blue' for i in df1['Feature ID']]
df1['Correlated ID'] = highest_correlations_indices_v1[:len(embedding1)]

df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
df2['Feature ID'] = range(len(embedding2))
df2['Feature Description'] = fList_model_B[:len(embedding2)]
df2['Color'] = ['red' if i in modB_feats else 'blue' for i in df2['Feature ID']]
df2['Correlated ID'] = [i for i in range(len(embedding2))]

# Create side by side plots using Plotly subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('UMAP Projection of SAE 1', 'UMAP Projection of SAE 2'))

# Add first scatter plot
fig.add_trace(
    go.Scatter(
        x=df1['UMAP Component 1'], y=df1['UMAP Component 2'],
        mode='markers', marker=dict(color=df1['Color']),
        text=df1['Feature ID'], customdata=np.array(df1[['Feature Description', 'Correlated ID']]),
        hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}'
    ),
    row=1, col=1
)

# Add second scatter plot
fig.add_trace(
    go.Scatter(
        x=df2['UMAP Component 1'], y=df2['UMAP Component 2'],
        mode='markers', marker=dict(color=df2['Color']),
        text=df2['Feature ID'], customdata=np.array(df2[['Feature Description', 'Correlated ID']]),
        hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text='UMAP Projections of Feature Decoder Weights',
    showlegend=False
)

# Save the figure as an HTML file
fig.write_html(outputFN)


# In[ ]:


fig.data[0].to_plotly_json()


# In[ ]:





# In[ ]:


keyword = "once"
outputFN = 'ts_1L_2L_MLP0_16k_30k_relu.html'
keyword_umaps_html(keyword, embedding1, embedding2, fList_model_A, fList_model_B,
                   highest_correlations_indices_v1, output_filename = outputFN)
files.download(outputFN)


# # cca on feature subset

# ## directly

# In[ ]:


keyword = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]


# In[ ]:


n_comp=2 #choose number of canonical variates pairs
cca = CCA(scale=False, n_components=n_comp) #define CCA
cca.fit(X_subset, Y_subset)

# Transform the data
A_c, B_c = cca.transform(X_subset, Y_subset)

comp_corr = [np.corrcoef(A_c[:, i], B_c[:, i])[1][0] for i in range(n_comp)]
comp_corr


# ### compare to random

# In[ ]:


rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=len(modA_feats)).tolist()
rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=len(modB_feats)).tolist()
X_subset = weight_matrix_np[rand_modA_feats, :]
Y_subset = weight_matrix_2[rand_modB_feats, :]

n_comp=2 #choose number of canonical variates pairs
cca = CCA(scale=False, n_components=n_comp) #define CCA
cca.fit(X_subset, Y_subset)

# Transform the data
A_c, B_c = cca.transform(X_subset, Y_subset)

comp_corr = [np.corrcoef(A_c[:, i], B_c[:, i])[1][0] for i in range(n_comp)]
comp_corr


# In[ ]:


rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=len(modA_feats)).tolist()
rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=len(modB_feats)).tolist()
X_subset = weight_matrix_np[rand_modA_feats, :]
Y_subset = weight_matrix_2[rand_modB_feats, :]

n_comp=2 #choose number of canonical variates pairs
cca = CCA(scale=False, n_components=n_comp) #define CCA
cca.fit(X_subset, Y_subset)

# Transform the data
A_c, B_c = cca.transform(X_subset, Y_subset)

comp_corr = [np.corrcoef(A_c[:, i], B_c[:, i])[1][0] for i in range(n_comp)]
comp_corr


# In[ ]:


len(rand_modA_feats)


# In[ ]:


rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=10000).tolist()
rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=10000).tolist()
X_subset = weight_matrix_np[rand_modA_feats, :]
Y_subset = weight_matrix_2[rand_modB_feats, :]

n_comp=2 #choose number of canonical variates pairs
cca = CCA(scale=False, n_components=n_comp) #define CCA
cca.fit(X_subset, Y_subset)

# Transform the data
A_c, B_c = cca.transform(X_subset, Y_subset)

comp_corr = [np.corrcoef(A_c[:, i], B_c[:, i])[1][0] for i in range(n_comp)]
comp_corr


# ## after umap

# In[ ]:


keyword = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
print(modA_feats)
print(modB_feats)


# In[ ]:


import numpy as np
import umap
from sklearn.cross_decomposition import CCA

# Assume X and Y are your input matrices of size (16384, 1024)
# Assume indices_X and indices_Y are lists of indices for subsetting X and Y respectively

# Step 1: Subset the matrices
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]

# Step 2: Apply UMAP to each subset
# umap_model = umap.UMAP(n_components=2)  # you can adjust n_components as needed
# X_umap = umap_model.fit_transform(X_subset)
# Y_umap = umap_model.fit_transform(Y_subset)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')
X_umap = reducer.fit_transform(X_subset)
Y_umap = reducer.fit_transform(Y_subset)

# Step 3: Apply CCA to the UMAP-transformed data
cca = CCA(n_components=2)  # you can adjust n_components as needed
X_c, Y_c = cca.fit_transform(X_umap, Y_umap)

# Now X_c and Y_c are the CCA-transformed data
# You can analyze these to find the similarity between the subspaces

# To find the correlation between the CCA components:
correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
print("Canonical correlations:", correlations)


# In[ ]:


# Compute the average canonical correlation score
average_score = np.mean(correlations)
print("Average canonical correlation score:", average_score)


# In[ ]:


print("Canonical Correlations:")
print(cca.score(X_umap, Y_umap))


# In[ ]:


print("Canonical Correlations:")
print(cca.score(X_c, Y_c))


# ## normalize?

# In[ ]:


import numpy as np
import umap
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Assuming X and Y are your input matrices of size (16384, 1024)
# Assuming indices_X and indices_Y are lists of indices for subsetting X and Y respectively

# Step 1: Subset the matrices
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]

# Step 2: Standardize the subsets
scaler = StandardScaler()
X_subset = scaler.fit_transform(X_subset)
Y_subset = scaler.fit_transform(Y_subset)

# Step 2: Apply UMAP to each subset
# umap_model = umap.UMAP(n_components=2)  # you can adjust n_components as needed
# X_umap = umap_model.fit_transform(X_subset)
# Y_umap = umap_model.fit_transform(Y_subset)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')
X_umap = reducer.fit_transform(X_subset)
Y_umap = reducer.fit_transform(Y_subset)

# Step 4: Standardize the UMAP-transformed data
X_umap = scaler.fit_transform(X_umap)
Y_umap = scaler.fit_transform(Y_umap)

# Step 5: Fit and transform using CCA
cca = CCA(n_components=2)  # you can adjust n_components as needed
X_c, Y_c = cca.fit_transform(X_umap, Y_umap)

# Step 6: Calculate canonical correlations manually
correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
print("Canonical correlations:", correlations)

# Calculate the average canonical correlation score manually
average_correlation_score = np.mean(correlations)
print("Average canonical correlation score:", average_correlation_score)

# Step 7: Compute the average canonical correlation score using CCA's score method
cca_score = cca.score(X_umap, Y_umap)
print("CCA score:", cca_score)


# ## turn into fn

# In[ ]:


# def cca_subspace(keyword):
def cca_subspace(keyword):
    modB_feats = find_indices_with_keyword(fList_model_B, keyword)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)

    # Step 1: Subset the matrices
    X_subset = weight_matrix_np[modA_feats, :]
    Y_subset = weight_matrix_2[modB_feats, :]

    # Step 2: Apply UMAP to each subset
    # umap_model = umap.UMAP(n_components=2)  # you can adjust n_components as needed
    # X_umap = umap_model.fit_transform(X_subset)
    # Y_umap = umap_model.fit_transform(Y_subset)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')
    X_umap = reducer.fit_transform(X_subset)
    Y_umap = reducer.fit_transform(Y_subset)

    # Step 3: Apply CCA to the UMAP-transformed data
    cca = CCA(n_components=2)  # you can adjust n_components as needed
    X_c, Y_c = cca.fit_transform(X_umap, Y_umap) # Now X_c and Y_c are the CCA-transformed data

    print("Canonical Correlations:")
    print(cca.score(X_umap, Y_umap))

    # To find the correlation between the CCA components:
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
    print("Canonical correlations:", correlations)


# In[ ]:


keyword = "let"
cca_subspace(keyword)


# In[ ]:


keyword = "saw"
cca_subspace(keyword)


# In[ ]:


keyword = "king"
cca_subspace(keyword)


# In[ ]:


keyword = "dragon"
cca_subspace(keyword)


# In[ ]:


keyword = "he"
cca_subspace(keyword)


# In[ ]:


keyword = "she"
cca_subspace(keyword)

