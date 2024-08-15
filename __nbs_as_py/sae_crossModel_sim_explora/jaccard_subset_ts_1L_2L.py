#!/usr/bin/env python
# coding: utf-8

# # setup

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


# !pip install umap-learn


# In[3]:


import pickle
import numpy as np

# import umap
import matplotlib.pyplot as plt


# # load weight mats

# In[ ]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-1L-21M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_np = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_np.shape)


# In[ ]:


weight_matrix_np = weight_matrix_np.detach().numpy()


# In[ ]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-2L-33M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_2.shape)


# In[ ]:


weight_matrix_2 = weight_matrix_2.detach().numpy()


# # load sae f actvs

# In[4]:


file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_A = pickle.load(f)


# In[5]:


file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)


# In[6]:


feature_acts_model_B.shape


# In[7]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[8]:


reshaped_activations_B.shape


# # load feature labels

# ## load

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


# In[ ]:


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

# ## fns

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


# ## test

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

# ## umap

# In[ ]:


# import umap
# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np

# reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# # Fit and transform the data by rows
# embedding1 = reducer.fit_transform(weight_matrix_np)
# embedding2 = reducer.fit_transform(weight_matrix_2)


# In[ ]:


# with open('embedding1.pkl', 'wb') as f:
#     pickle.dump(embedding1, f)
# files.download('embedding1.pkl')

# with open('embedding2.pkl', 'wb') as f:
#     pickle.dump(embedding2, f)
# files.download('embedding2.pkl')


# ## load

# In[ ]:


import pickle
with open('embedding_1L_16384.pkl', 'rb') as f:
    embedding1 = pickle.load(f)
with open('embedding_2L_16384.pkl', 'rb') as f:
    embedding2 = pickle.load(f)


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


# # jaccard on feature subset

# ## fns

# In[ ]:


import functools
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch


def to_numpy_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[npt.NDArray]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> npt.NDArray:
        return x if isinstance(x, np.ndarray) else x.numpy()

    return list(map(convert, args))


def to_torch_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[torch.Tensor]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    return list(map(convert, args))


def adjust_dimensionality(
    R: npt.NDArray, Rp: npt.NDArray, strategy="zero_pad"
) -> Tuple[npt.NDArray, npt.NDArray]:
    D = R.shape[1]
    Dp = Rp.shape[1]
    if strategy == "zero_pad":
        if D - Dp == 0:
            return R, Rp
        elif D - Dp > 0:
            return R, np.concatenate((Rp, np.zeros((Rp.shape[0], D - Dp))), axis=1)
        else:
            return np.concatenate((R, np.zeros((R.shape[0], Dp - D))), axis=1), Rp
    else:
        raise NotImplementedError()


def center_columns(R: npt.NDArray) -> npt.NDArray:
    return R - R.mean(axis=0)[None, :]


def normalize_matrix_norm(R: npt.NDArray) -> npt.NDArray:
    return R / np.linalg.norm(R, ord="fro")


def sim_random_baseline(
    rep1: torch.Tensor, rep2: torch.Tensor, sim_func: Callable, n_permutations: int = 10
) -> Dict[str, Any]:
    torch.manual_seed(1234)
    scores = []
    for _ in range(n_permutations):
        perm = torch.randperm(rep1.size(0))

        score = sim_func(rep1[perm, :], rep2)
        score = score if isinstance(score, float) else score["score"]

        scores.append(score)

    return {"baseline_scores": np.array(scores)}


class Pipeline:
    def __init__(
        self,
        preprocess_funcs: List[Callable[[npt.NDArray], npt.NDArray]],
        similarity_func: Callable[[npt.NDArray, npt.NDArray], Dict[str, Any]],
    ) -> None:
        self.preprocess_funcs = preprocess_funcs
        self.similarity_func = similarity_func

    def __call__(self, R: npt.NDArray, Rp: npt.NDArray) -> Dict[str, Any]:
        for preprocess_func in self.preprocess_funcs:
            R = preprocess_func(R)
            Rp = preprocess_func(Rp)
        return self.similarity_func(R, Rp)

    def __str__(self) -> str:
        def func_name(func: Callable) -> str:
            return (
                func.__name__
                if not isinstance(func, functools.partial)
                else func.func.__name__
            )

        def partial_keywords(func: Callable) -> str:
            if not isinstance(func, functools.partial):
                return ""
            else:
                return str(func.keywords)

        return (
            "Pipeline("
            + (
                "+".join(map(func_name, self.preprocess_funcs))
                + "+"
                + func_name(self.similarity_func)
                + partial_keywords(self.similarity_func)
            )
            + ")"
        )


# In[ ]:


from typing import List, Set, Union

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

# from llmcomp.measures.utils import to_numpy_if_needed


def _jac_sim_i(idx_R: Set[int], idx_Rp: Set[int]) -> float:
    return len(idx_R.intersection(idx_Rp)) / len(idx_R.union(idx_Rp))


def jaccard_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    k: int = 10,
    inner: str = "cosine",
    n_jobs: int = 8,
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    indices_R = nn_array_to_setlist(top_k_neighbors(R, k, inner, n_jobs))
    indices_Rp = nn_array_to_setlist(top_k_neighbors(Rp, k, inner, n_jobs))

    return float(
        np.mean(
            [_jac_sim_i(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]
        )
    )


def top_k_neighbors(
    R: npt.NDArray,
    k: int,
    inner: str,
    n_jobs: int,
) -> npt.NDArray:
    # k+1 nearest neighbors, because we pass in all the data, which means that a point
    # will be the nearest neighbor to itself. We remove this point from the results and
    # report only the k nearest neighbors distinct from the point itself.
    nns = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, metric=inner, n_jobs=n_jobs
    )
    nns.fit(R)
    _, nns = nns.kneighbors(R)
    return nns[:, 1:]


def nn_array_to_setlist(nn: npt.NDArray) -> List[Set[int]]:
    return [set(idx) for idx in nn]


# ## entire space

# slow

# In[ ]:


jaccard_similarity(weight_matrix_np, weight_matrix_2)


# In[ ]:


jaccard_similarity(weight_matrix_np[highest_correlations_indices_v1], weight_matrix_2)


# In[ ]:


weight_matrix_2.shape[0]


# In[ ]:


len(list(set(highest_correlations_indices_v1)))


# ## single token subspaces

# In[ ]:


def get_rand_scores(modA_feats, modB_feats, k: int = 10):
    total_scores = 0
    for i in range(100):
        # if i % 20 == 0:
        #     print(i)
        rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=len(modA_feats)).tolist()
        rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=len(modB_feats)).tolist()
        X_subset = weight_matrix_np[rand_modA_feats, :]
        Y_subset = weight_matrix_2[rand_modB_feats, :]

        total_scores += jaccard_similarity(X_subset, Y_subset, k)

    return total_scores / 100


# In[ ]:


keyword = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "once"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "let"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset, k=4))
print(get_rand_scores(modA_feats, modB_feats, k=4))


# ## multiple token subspaces

# In[ ]:


keyword_1 = "she"
keyword_2 = "princess"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "she"
keyword_2 = "he"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "princess"
keyword_2 = "dragon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "princess"
keyword_2 = "dragon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset, k = 3))
print(get_rand_scores(modA_feats, modB_feats, k =3))


# In[ ]:


keyword_1 = "once"
keyword_2 = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "once"
keyword_2 = "upon"
keyword_3 = "time"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2) + \
                find_indices_with_keyword(fList_model_B, keyword_3)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "once"
keyword_2 = "she"
keyword_3 = "."
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2) + \
                find_indices_with_keyword(fList_model_B, keyword_3)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "time"
keyword_2 = "she"
keyword_3 = "."
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2) + \
                find_indices_with_keyword(fList_model_B, keyword_3)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "time"
keyword_2 = "she"
keyword_3 = "let"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2) + \
                find_indices_with_keyword(fList_model_B, keyword_3)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "time"
keyword_2 = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword_1 = "once"
keyword_2 = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# ## compare feats from diff clusters to check

# In[ ]:


keyword_1 = "time"
modB_feats_1 = find_indices_with_keyword(fList_model_B, keyword_1)
len(modB_feats_1)


# In[ ]:


keyword_2 = "she"
modB_feats_2 = find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats_2 = get_values_from_indices(modB_feats_2, highest_correlations_indices_v1)
len(modA_feats_2)


# In[ ]:


minInd = min(len(modB_feats_1), len(modA_feats_2))
X_subset = weight_matrix_np[modA_feats_2[:minInd], :]
Y_subset = weight_matrix_2[modB_feats_1[:minInd], :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))


# ### more exmaples

# In[ ]:


keyword_1 = "once"
modB_feats_1 = find_indices_with_keyword(fList_model_B, keyword_1)
len(modB_feats_1)

keyword_2 = "he"
modB_feats_2 = find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats_2 = get_values_from_indices(modB_feats_2, highest_correlations_indices_v1)
len(modA_feats_2)

minInd = min(len(modB_feats_1), len(modA_feats_2))
X_subset = weight_matrix_np[modA_feats_2[:minInd], :]
Y_subset = weight_matrix_2[modB_feats_1[:minInd], :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset, k=4))
# print(get_rand_scores(modA_feats, modB_feats))


# ## corr explora

# In[ ]:


modA_feats[:3]


# In[ ]:


weight_matrix_np[modA_feats[:3], :]


# In[ ]:


weight_matrix_np[13316, :]


# In[ ]:


weight_matrix_np[[1,1,2], :]


# In[ ]:


keyword_1 = "once"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(list(set(modA_feats))))
len(list(set(modB_feats)))


# In[ ]:


keyword_1 = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(list(set(modA_feats))))
len(list(set(modB_feats)))


# In[ ]:


# Generate 50 unique indices
unique_feats_A = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=50).tolist()

# Create a list with repeats by duplicating the unique elements
rand_modA_feats = unique_feats_A + unique_feats_A

# Shuffle the list to mix unique and repeated elements
np.random.shuffle(rand_modA_feats)

# Ensure rand_modB_feats remains fully random
rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=100).tolist()

# Subset the weight matrices using the selected features
X_subset = weight_matrix_np[rand_modA_feats, :]
Y_subset = weight_matrix_2[rand_modB_feats, :]

# Update total_scores with the Jaccard similarity between the subsets
jaccard_similarity(X_subset, Y_subset)


# In[ ]:


# Generate 50 unique indices
unique_feats_A = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=10).tolist()

# Create a list with repeats by duplicating the unique elements
rand_modA_feats = unique_feats_A + unique_feats_A

# Shuffle the list to mix unique and repeated elements
np.random.shuffle(rand_modA_feats)

# Ensure rand_modB_feats remains fully random
rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=20).tolist()

# Subset the weight matrices using the selected features
X_subset = weight_matrix_np[rand_modA_feats, :]
Y_subset = weight_matrix_2[rand_modB_feats, :]

# Update total_scores with the Jaccard similarity between the subsets
jaccard_similarity(X_subset, Y_subset, k=3)


# ## jaccard explora

# In[ ]:





# ## sel one from each category

# This prob won't succeed at first so you need to refine this.

# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["once", "upon", "a", "time", "let", "she", "he", "princess", "dragon", "king", "."]

for kw in keywords:
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
jaccard_similarity(X_subset, Y_subset, k=3)


# In[ ]:


len(mixed_modA_feats)


# In[ ]:


print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=3))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["once", "upon", "a", "time", "let", "she", "he", "princess", "dragon", "king", "."]

for kw in keywords:
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.extend(modA_feats[0:2])
    mixed_modB_feats.extend(modB_feats[0:2])

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=3))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["once", "upon", "a", "time", "let", "she", "he", "princess", "dragon", "king", "."]

for kw in keywords:
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.extend(modA_feats[0:5])
    mixed_modB_feats.extend(modB_feats[0:5])

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=3))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["once", "upon", "a", "time", "let", "she", "he", "princess", "dragon", "king", "."]

for kw in keywords:
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.extend(modA_feats[0:5])
    mixed_modB_feats.extend(modB_feats[0:5])

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=10))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=10))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["once", "upon", "a", "time", "let", "she", "he", "princess", "dragon", "king", ".", "family"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=3))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["once", "upon", "a", "time"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "princess", "dragon"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "princess", "dragon", "she", "he"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time", "let", "."]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time", "let", "."]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=3))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time", "let", "."]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=4))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=4))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time", "let", "."]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=5))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=5))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time", "let"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=4))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=4))


# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=4))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=4))


# ## see what feat neighs they have in common

# In[ ]:


kw_dict = {}
for i, kw in enumerate(["girl", "boy", "she", "he", "her", "his", "it", "once", "upon", "a", "time"]):
    kw_dict[i] = kw


# In[ ]:


indices_R = nn_array_to_setlist(top_k_neighbors(X_subset, 4, "cosine", 8))
replaced_sets_list = [{kw_dict.get(ind, ind) for ind in s} for s in indices_R]
for i, kw_neighs in enumerate(replaced_sets_list):
    print(kw_dict[i], ': ', kw_neighs)


# In[ ]:


indices_Rp = nn_array_to_setlist(top_k_neighbors(Y_subset, 4, "cosine", 8))
replaced_sets_list_2 = [{kw_dict.get(ind, ind) for ind in s} for s in indices_Rp]
for i, kw_neighs in enumerate(replaced_sets_list_2):
    print(kw_dict[i], ': ', kw_neighs)


# In[ ]:


R_Rp = [set.intersection(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]
replaced_sets_list_both = [{kw_dict.get(ind, ind) for ind in s} for s in R_Rp]
for i, kw_neighs in enumerate(replaced_sets_list_both):
    print(kw_dict[i], ': ', kw_neighs)


# In[ ]:


[_jac_sim_i(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]


# In[ ]:


rand_modA_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=len(mixed_modA_feats)).tolist()
rand_X_subset = weight_matrix_np[rand_modA_feats, :]
indices_R = nn_array_to_setlist(top_k_neighbors(rand_X_subset, 4, "cosine", 8))
indices_R


# ### another

# In[ ]:


mixed_modA_feats = []
mixed_modB_feats = []

keywords = ["girl", "boy", "she", "he", "her", "his", "it"]

for kw in keywords:
    print(kw + ": ")
    modB_feats = find_indices_with_keyword(fList_model_B, kw)
    modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
    mixed_modA_feats.append(modA_feats[0])
    mixed_modB_feats.append(modB_feats[0])
    print(len(modB_feats), len(list(set(modA_feats))))

X_subset = weight_matrix_np[mixed_modA_feats, :]
Y_subset = weight_matrix_2[mixed_modB_feats, :]
print(len(mixed_modA_feats))
print(jaccard_similarity(X_subset, Y_subset, k=2))

# print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:


kw_dict = {}
for i, kw in enumerate(keywords):
    kw_dict[i] = kw


# In[ ]:


indices_R = nn_array_to_setlist(top_k_neighbors(X_subset, 2, "cosine", 8))
replaced_sets_list = [{kw_dict.get(ind, ind) for ind in s} for s in indices_R]
for i, kw_neighs in enumerate(replaced_sets_list):
    print(kw_dict[i], ': ', kw_neighs)


# In[ ]:


indices_Rp = nn_array_to_setlist(top_k_neighbors(Y_subset, 2, "cosine", 8))
replaced_sets_list_2 = [{kw_dict.get(ind, ind) for ind in s} for s in indices_Rp]
for i, kw_neighs in enumerate(replaced_sets_list_2):
    print(kw_dict[i], ': ', kw_neighs)


# In[ ]:


R_Rp = [set.intersection(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]
replaced_sets_list_both = [{kw_dict.get(ind, ind) for ind in s} for s in R_Rp]
for i, kw_neighs in enumerate(replaced_sets_list_both):
    print(kw_dict[i], ': ', kw_neighs)


# ## only take NN of 1-1 features

# get rid of features in B that map to features in A that were already mapped to

# In[ ]:


keyword = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)

new_modB_feats = []
new_modA_feats = []
for ind_B, ind_A in zip(modB_feats, modA_feats):
    if ind_A not in new_modA_feats:
        new_modA_feats.append(ind_A)
        new_modB_feats.append(ind_B)

print(len(new_modA_feats), len(list(set(new_modA_feats))))
print(len(new_modB_feats), len(list(set(new_modB_feats))))


# In[ ]:


X_subset = weight_matrix_np[new_modA_feats, :]
Y_subset = weight_matrix_2[new_modB_feats, :]

print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(new_modA_feats, new_modB_feats))


# # jaccard on token actvs

# ## fns

# In[ ]:


from typing import List, Set, Union

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

# from llmcomp.measures.utils import to_numpy_if_needed


def _jac_sim_i(idx_R: Set[int], idx_Rp: Set[int]) -> float:
    return len(idx_R.intersection(idx_Rp)) / len(idx_R.union(idx_Rp))


def jaccard_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    k: int = 10,
    inner: str = "cosine",
    n_jobs: int = 8,
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    indices_R = nn_array_to_setlist(top_k_neighbors(R, k, inner, n_jobs))
    indices_Rp = nn_array_to_setlist(top_k_neighbors(Rp, k, inner, n_jobs))

    # all_scores = []
    # for i, idx_R, idx_Rp in enumerate(zip(indices_R, indices_Rp)):
    #     if i % 1000 == 0:
    #         print(idx_R, idx_Rp)
    #     score = _jac_sim_i(idx_R, idx_Rp)
    #     all_scores.append(score)
    # return float(np.mean(all_scores))

    return float(
        np.mean(
            [_jac_sim_i(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]
        )
    )


def top_k_neighbors(
    R: npt.NDArray,
    k: int,
    inner: str,
    n_jobs: int,
) -> npt.NDArray:
    # k+1 nearest neighbors, because we pass in all the data, which means that a point
    # will be the nearest neighbor to itself. We remove this point from the results and
    # report only the k nearest neighbors distinct from the point itself.
    nns = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, metric=inner, n_jobs=n_jobs
    )
    nns.fit(R)
    _, nns = nns.kneighbors(R)
    return nns[:, 1:]


def nn_array_to_setlist(nn: npt.NDArray) -> List[Set[int]]:
    return [set(idx) for idx in nn]


# ## test

# In[ ]:


reshaped_activations_A.shape


# In[ ]:


# jaccard_similarity(reshaped_activations_A, reshaped_activations_B)

jaccard_similarity(reshaped_activations_A[:10000, :], reshaped_activations_B[:10000, :])


# In[ ]:


jaccard_similarity(reshaped_activations_A[:10000, :], reshaped_activations_B[10000:20000, :])


# In[ ]:


jaccard_similarity(reshaped_activations_A[:10000, :], torch.rand(10000, reshaped_activations_A.shape[1]))


# In[ ]:


jaccard_similarity(reshaped_activations_A[:1000, :], reshaped_activations_B[:1000, :])


# In[ ]:


jaccard_similarity(reshaped_activations_A[:, :], reshaped_activations_B[:, :])


# In[ ]:


import random
row_idxs = list(range(reshaped_activations_A.shape[0]))
random.shuffle(row_idxs)
jaccard_similarity(reshaped_activations_A[row_idxs, :], reshaped_activations_B)


# ## by corr features

# In[ ]:


jaccard_similarity(reshaped_activations_A[:1000, :].t(), reshaped_activations_B[:1000, :].t())


# In[ ]:


jaccard_similarity(reshaped_activations_A[:1000, :].t()[highest_correlations_indices_v1], reshaped_activations_B[:1000, :].t())


# In[ ]:


jaccard_similarity(reshaped_activations_A[:10000, :].t(), reshaped_activations_B[:10000, :].t())


# In[ ]:


jaccard_similarity(reshaped_activations_A[:10000, :].t()[highest_correlations_indices_v1], reshaped_activations_B[:10000, :].t())


# In[ ]:


jaccard_similarity(reshaped_activations_A[:, :].t(), reshaped_activations_B[:, :].t())


# In[ ]:


jaccard_similarity(reshaped_activations_A[:, :].t()[highest_correlations_indices_v1], reshaped_activations_B[:, :].t())


# # perm procr

# ## fn

# In[10]:


import functools
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import get_args
from typing import List
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
# from loguru import logger

log = logging.getLogger(__name__)


SHAPE_TYPE = Literal["nd", "ntd", "nchw"]

ND_SHAPE, NTD_SHAPE, NCHW_SHAPE = get_args(SHAPE_TYPE)[0], get_args(SHAPE_TYPE)[1], get_args(SHAPE_TYPE)[2]


class SimilarityFunction(Protocol):
    def __call__(  # noqa: E704
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
    ) -> float: ...


class RSMSimilarityFunction(Protocol):
    def __call__(  # noqa: E704
        self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE, n_jobs: int
    ) -> float: ...


@dataclass
class BaseSimilarityMeasure(ABC):
    larger_is_more_similar: bool
    is_symmetric: bool

    is_metric: bool | None = None
    invariant_to_affine: bool | None = None
    invariant_to_invertible_linear: bool | None = None
    invariant_to_ortho: bool | None = None
    invariant_to_permutation: bool | None = None
    invariant_to_isotropic_scaling: bool | None = None
    invariant_to_translation: bool | None = None
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class FunctionalSimilarityMeasure(BaseSimilarityMeasure):
    @abstractmethod
    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> float:
        raise NotImplementedError


@dataclass(kw_only=True)
class RepresentationalSimilarityMeasure(BaseSimilarityMeasure):
    sim_func: SimilarityFunction

    def __call__(
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
    ) -> float:
        return self.sim_func(R, Rp, shape)


class RSMSimilarityMeasure(RepresentationalSimilarityMeasure):
    sim_func: RSMSimilarityFunction

    @staticmethod
    def estimate_good_number_of_jobs(R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray) -> int:
        # RSMs in are NxN (or DxD) so the number of jobs should roughly scale quadratically with increase in N (or D).
        # False! As long as sklearn-native metrics are used, they will use parallel implementations regardless of job
        # count. Each job would spawn their own threads, which leads to oversubscription of cores and thus slowdown.
        # This seems to be not fully correct (n_jobs=2 seems to actually use two cores), but using n_jobs=1 seems the
        # fastest.
        return 1

    def __call__(
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
        n_jobs: Optional[int] = None,
    ) -> float:
        if n_jobs is None:
            n_jobs = self.estimate_good_number_of_jobs(R, Rp)
        return self.sim_func(R, Rp, shape, n_jobs=n_jobs)


def to_numpy_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[npt.NDArray]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> npt.NDArray:
        return x if isinstance(x, np.ndarray) else x.numpy()

    return list(map(convert, args))


def to_torch_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[torch.Tensor]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    return list(map(convert, args))


def adjust_dimensionality(R: npt.NDArray, Rp: npt.NDArray, strategy="zero_pad") -> Tuple[npt.NDArray, npt.NDArray]:
    D = R.shape[1]
    Dp = Rp.shape[1]
    if strategy == "zero_pad":
        if D - Dp == 0:
            return R, Rp
        elif D - Dp > 0:
            return R, np.concatenate((Rp, np.zeros((Rp.shape[0], D - Dp))), axis=1)
        else:
            return np.concatenate((R, np.zeros((R.shape[0], Dp - D))), axis=1), Rp
    else:
        raise NotImplementedError()


def center_columns(R: npt.NDArray) -> npt.NDArray:
    return R - R.mean(axis=0)[None, :]


def normalize_matrix_norm(R: npt.NDArray) -> npt.NDArray:
    return R / np.linalg.norm(R, ord="fro")


def normalize_row_norm(R: npt.NDArray) -> npt.NDArray:
    return R / np.linalg.norm(R, ord=2, axis=1, keepdims=True)


def standardize(R: npt.NDArray) -> npt.NDArray:
    return (R - R.mean(axis=0, keepdims=True)) / R.std(axis=0)


def double_center(x: npt.NDArray) -> npt.NDArray:
    return x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()


def align_spatial_dimensions(R: npt.NDArray, Rp: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Aligns spatial representations by resizing them to the smallest spatial dimension.
    Subsequent aligned spatial representations are flattened, with the spatial aligned representations
    moving into the *sample* dimension.
    """
    R_re, Rp_re = resize_wh_reps(R, Rp)
    R_re = rearrange(R_re, "n c h w -> (n h w) c")
    Rp_re = rearrange(Rp_re, "n c h w -> (n h w) c")
    if R_re.shape[0] > 5000:
        logger.info(f"Got {R_re.shape[0]} samples in N after flattening. Subsampling to reduce compute.")
        subsample = R_re.shape[0] // 5000
        R_re = R_re[::subsample]
        Rp_re = Rp_re[::subsample]

    return R_re, Rp_re


def average_pool_downsample(R, resize: bool, new_size: tuple[int, int]):
    if not resize:
        return R  # do nothing
    else:
        is_numpy = isinstance(R, np.ndarray)
        R_torch = torch.from_numpy(R) if is_numpy else R
        R_torch = torch.nn.functional.adaptive_avg_pool2d(R_torch, new_size)
        return R_torch.numpy() if is_numpy else R_torch


def resize_wh_reps(R: npt.NDArray, Rp: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Function for resizing spatial representations that are not the same size.
    Does through fourier transform and resizing.

    Args:
        R: numpy array of shape  [batch_size, height, width, num_channels]
        RP: numpy array of shape [batch_size, height, width, num_channels]

    Returns:
        fft_acts1: numpy array of shape [batch_size, (new) height, (new) width, num_channels]
        fft_acts2: numpy array of shape [batch_size, (new) height, (new) width, num_channels]

    """
    height1, width1 = R.shape[2], R.shape[3]
    height2, width2 = Rp.shape[2], Rp.shape[3]
    if height1 != height2 or width1 != width2:
        height = min(height1, height2)
        width = min(width1, width2)
        new_size = [height, width]
        resize = True
    else:
        height = height1
        width = width1
        new_size = None
        resize = False

    # resize and preprocess with fft
    avg_ds1 = average_pool_downsample(R, resize=resize, new_size=new_size)
    avg_ds2 = average_pool_downsample(Rp, resize=resize, new_size=new_size)
    return avg_ds1, avg_ds2


def fft_resize(images, resize=False, new_size=None):
    """Function for applying DFT and resizing.

    This function takes in an array of images, applies the 2-d fourier transform
    and resizes them according to new_size, keeping the frequencies that overlap
    between the two sizes.

    Args:
              images: a numpy array with shape
                      [batch_size, height, width, num_channels]
              resize: boolean, whether or not to resize
              new_size: a tuple (size, size), with height and width the same

    Returns:
              im_fft_downsampled: a numpy array with shape
                           [batch_size, (new) height, (new) width, num_channels]
    """
    assert len(images.shape) == 4, "expecting images to be" "[batch_size, height, width, num_channels]"
    if resize:
        # FFT --> remove high frequencies --> inverse FFT
        im_complex = images.astype("complex64")
        im_fft = np.fft.fft2(im_complex, axes=(1, 2))
        im_shifted = np.fft.fftshift(im_fft, axes=(1, 2))

        center_width = im_shifted.shape[2] // 2
        center_height = im_shifted.shape[1] // 2
        half_w = new_size[0] // 2
        half_h = new_size[1] // 2
        cropped_fft = im_shifted[
            :, center_height - half_h : center_height + half_h, center_width - half_w : center_width + half_w, :
        ]
        cropped_fft_shifted_back = np.fft.ifft2(cropped_fft, axes=(1, 2))
        return cropped_fft_shifted_back.real
    else:
        return images


class Pipeline:
    def __init__(
        self,
        preprocess_funcs: List[Callable[[npt.NDArray], npt.NDArray]],
        similarity_func: Callable[[npt.NDArray, npt.NDArray, SHAPE_TYPE], float],
    ) -> None:
        self.preprocess_funcs = preprocess_funcs
        self.similarity_func = similarity_func

    def __call__(self, R: npt.NDArray, Rp: npt.NDArray, shape: SHAPE_TYPE) -> float:
        try:
            for preprocess_func in self.preprocess_funcs:
                R = preprocess_func(R)
                Rp = preprocess_func(Rp)
            return self.similarity_func(R, Rp, shape)
        except ValueError as e:
            log.info(f"Pipeline failed: {e}")
            return np.nan

    def __str__(self) -> str:
        def func_name(func: Callable) -> str:
            return func.__name__ if not isinstance(func, functools.partial) else func.func.__name__

        def partial_keywords(func: Callable) -> str:
            if not isinstance(func, functools.partial):
                return ""
            else:
                return str(func.keywords)

        return (
            "Pipeline("
            + (
                "+".join(map(func_name, self.preprocess_funcs))
                + "+"
                + func_name(self.similarity_func)
                + partial_keywords(self.similarity_func)
            )
            + ")"
        )


def flatten(*args: Union[torch.Tensor, npt.NDArray], shape: SHAPE_TYPE) -> List[Union[torch.Tensor, npt.NDArray]]:
    if shape == "ntd":
        return list(map(flatten_nxtxd_to_ntxd, args))
    elif shape == "nd":
        return list(args)
    elif shape == "nchw":
        return list(map(flatten_nxcxhxw_to_nxchw, args))  # Flattening non-trivial for nchw
    else:
        raise ValueError("Unknown shape of representations. Must be one of 'ntd', 'nchw', 'nd'.")


def flatten_nxtxd_to_ntxd(R: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
    R = to_torch_if_needed(R)[0]
    log.debug("Shape before flattening: %s", str(R.shape))
    R = torch.flatten(R, start_dim=0, end_dim=1)
    log.debug("Shape after flattening: %s", str(R.shape))
    return R


def flatten_nxcxhxw_to_nxchw(R: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
    R = to_torch_if_needed(R)[0]
    log.debug("Shape before flattening: %s", str(R.shape))
    R = torch.reshape(R, (R.shape[0], -1))
    log.debug("Shape after flattening: %s", str(R.shape))
    return R


# In[11]:


import scipy.optimize

def permutation_procrustes(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    optimal_permutation_alignment: Optional[Tuple[npt.NDArray, npt.NDArray]] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)

    if not optimal_permutation_alignment:
        PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments
        optimal_permutation_alignment = (PR, PRp)
    PR, PRp = optimal_permutation_alignment
    return float(np.linalg.norm(R[:, PR] - Rp[:, PRp], ord="fro"))


# ## all decoder weights

# In[ ]:


permutation_procrustes(weight_matrix_np, weight_matrix_2, "nd")


# In[ ]:


permutation_procrustes(weight_matrix_np[highest_correlations_indices_v1], weight_matrix_2, "nd")


# # RSA

# ## fns

# In[ ]:


from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import scipy.stats
import sklearn.metrics
import torch
# from repsim.measures.utils import flatten
# from repsim.measures.utils import RSMSimilarityMeasure
# from repsim.measures.utils import SHAPE_TYPE
# from repsim.measures.utils import to_numpy_if_needed


def representational_similarity_analysis(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    inner="correlation",
    outer="spearman",
    n_jobs: Optional[int] = None,
) -> float:
    """Representational similarity analysis

    Args:
        R (Union[torch.Tensor, npt.NDArray]): N x D representation
        Rp (Union[torch.Tensor, npt.NDArray]): N x D' representation
        inner (str, optional): inner similarity function for RSM. Must be one of
            scipy.spatial.distance.pdist identifiers . Defaults to "correlation".
        outer (str, optional): outer similarity function that compares RSMs. Defaults to
             "spearman". Must be one of "spearman", "euclidean"

    Returns:
        float: _description_
    """
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    if inner == "correlation":
        # n_jobs only works if metric is in PAIRWISE_DISTANCES as defined in sklearn, i.e., not for correlation.
        # But correlation = 1 - cosine dist of row-centered data, so we use the faster cosine metric and center the data.
        R = R - R.mean(axis=1, keepdims=True)
        S = scipy.spatial.distance.squareform(  # take the lower triangle of RSM
            1 - sklearn.metrics.pairwise_distances(R, metric="cosine", n_jobs=n_jobs),  # type:ignore
            checks=False,
        )
        Rp = Rp - Rp.mean(axis=1, keepdims=True)
        Sp = scipy.spatial.distance.squareform(
            1 - sklearn.metrics.pairwise_distances(Rp, metric="cosine", n_jobs=n_jobs),  # type:ignore
            checks=False,
        )
    elif inner == "euclidean":
        # take the lower triangle of RSM
        S = scipy.spatial.distance.squareform(
            sklearn.metrics.pairwise_distances(R, metric=inner, n_jobs=n_jobs), checks=False
        )
        Sp = scipy.spatial.distance.squareform(
            sklearn.metrics.pairwise_distances(Rp, metric=inner, n_jobs=n_jobs), checks=False
        )
    else:
        raise NotImplementedError(f"{inner=}")

    if outer == "spearman":
        return scipy.stats.spearmanr(S, Sp).statistic  # type:ignore
    elif outer == "euclidean":
        return float(np.linalg.norm(S - Sp, ord=2))
    else:
        raise ValueError(f"Unknown outer similarity function: {outer}")


class RSA(RSMSimilarityMeasure):
    def __init__(self):
        # choice of inner/outer in __call__ if fixed to default values, so these values are always the same
        super().__init__(
            sim_func=representational_similarity_analysis,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )


# ## all decoder weights

# slow

# In[ ]:


representational_similarity_analysis(weight_matrix_np, weight_matrix_2, "nd")


# In[ ]:


representational_similarity_analysis(weight_matrix_np[highest_correlations_indices_v1], weight_matrix_2, "nd")


# # cca

# ## fns

# In[12]:


##################################################################################
# Copied from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/cca_core.py
# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The core code for applying Canonical Correlation Analysis to deep networks.

This module contains the core functions to apply canonical correlation analysis
to deep neural networks. The main function is get_cca_similarity, which takes in
two sets of activations, typically the neurons in two layers and their outputs
on all of the datapoints D = [d_1,...,d_m] that have been passed through.

Inputs have shape (num_neurons1, m), (num_neurons2, m). This can be directly
applied used on fully connected networks. For convolutional layers, the 3d block
of neurons can either be flattened entirely, along channels, or alternatively,
the dft_ccas (Discrete Fourier Transform) module can be used.

See:
https://arxiv.org/abs/1706.05806
https://arxiv.org/abs/1806.05759
for full details.

"""
import numpy as np
# from repsim.measures.utils import align_spatial_dimensions

num_cca_trials = 5


def positivedef_matrix_sqrt(array):
    """Stable method for computing matrix square roots, supports complex matrices.

    Args:
              array: A numpy 2d array, can be complex valued that is a positive
                     definite symmetric (or hermitian) matrix

    Returns:
              sqrtarray: The matrix square root of array
    """
    w, v = np.linalg.eigh(array)
    #  A - np.dot(v, np.dot(np.diag(w), v.T))
    wsqrt = np.sqrt(w)
    sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
    return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
    """Takes covariance between X, Y, and removes values of small magnitude.

    Args:
              sigma_xx: 2d numpy array, variance matrix for x
              sigma_xy: 2d numpy array, crossvariance matrix for x,y
              sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                        (conjugate) transpose of sigma_xy
              sigma_yy: 2d numpy array, variance matrix for y
              epsilon : cutoff value for norm below which directions are thrown
                         away

    Returns:
              sigma_xx_crop: 2d array with low x norm directions removed
              sigma_xy_crop: 2d array with low x and y norm directions removed
              sigma_yx_crop: 2d array with low x and y norm directiosn removed
              sigma_yy_crop: 2d array with low y norm directions removed
              x_idxs: indexes of sigma_xx that were removed
              y_idxs: indexes of sigma_yy that were removed
    """

    x_diag = np.abs(np.diagonal(sigma_xx))
    y_diag = np.abs(np.diagonal(sigma_yy))
    x_idxs = x_diag >= epsilon
    y_idxs = y_diag >= epsilon

    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop, x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon, verbose=True):
    """Main cca computation function, takes in variances and crossvariances.

    This function takes in the covariances and cross covariances of X, Y,
    preprocesses them (removing small magnitudes) and outputs the raw results of
    the cca computation, including cca directions in a rotated space, and the
    cca correlation coefficient values.

    Args:
              sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                        variance matrix for x
              sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                        crossvariance matrix for x,y
              sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                        crossvariance matrix for x,y (conj) transpose of sigma_xy
              sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                        variance matrix for y
              epsilon:  small float to help with stabilizing computations
              verbose:  boolean on whether to print intermediate outputs

    Returns:
              [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                            ux and vx are (conj) transposes of each other, being
                            the canonical directions in the X subspace.
                            sx is the set of canonical correlation coefficients-
                            how well corresponding directions in vx, Vy correlate
                            with each other.
              [uy, sy, vy]: Same as above, but for Y space
              invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                            directions back to original space
              invsqrt_yy:   Same as above but for sigma_yy
              x_idxs:       The indexes of the input sigma_xx that were pruned
                            by remove_small
              y_idxs:       Same as above but for sigma_yy
    """

    (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = remove_small(
        sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon
    )

    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]

    if numx == 0 or numy == 0:
        return (
            [0, 0, 0],
            [0, 0, 0],
            np.zeros_like(sigma_xx),
            np.zeros_like(sigma_yy),
            x_idxs,
            y_idxs,
        )

    if verbose:
        print("adding eps to diagonal and taking inverse")
    sigma_xx += epsilon * np.eye(numx)
    sigma_yy += epsilon * np.eye(numy)
    inv_xx = np.linalg.pinv(sigma_xx)
    inv_yy = np.linalg.pinv(sigma_yy)

    if verbose:
        print("taking square root")
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

    if verbose:
        print("dot products...")
    arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

    if verbose:
        print("trying to take final svd")
    u, s, v = np.linalg.svd(arr)

    if verbose:
        print("computed everything!")

    return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
    """Computes threshold index of decreasing nonnegative array by summing.

    This function takes in a decreasing array nonnegative floats, and a
    threshold between 0 and 1. It returns the index i at which the sum of the
    array up to i is threshold*total mass of the array.

    Args:
              array: a 1d numpy array of decreasing, nonnegative floats
              threshold: a number between 0 and 1

    Returns:
              i: index at which np.sum(array[:i]) >= threshold
    """
    assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

    for i in range(len(array)):
        if np.sum(array[:i]) / np.sum(array) >= threshold:
            return i


def create_zero_dict(compute_dirns, dimension):
    """Outputs a zero dict when neuron activation norms too small.

    This function creates a return_dict with appropriately shaped zero entries
    when all neuron activations are very small.

    Args:
              compute_dirns: boolean, whether to have zero vectors for directions
              dimension: int, defines shape of directions

    Returns:
              return_dict: a dict of appropriately shaped zero entries
    """
    return_dict = {}
    return_dict["mean"] = (np.asarray(0), np.asarray(0))
    return_dict["sum"] = (np.asarray(0), np.asarray(0))
    return_dict["cca_coef1"] = np.asarray(0)
    return_dict["cca_coef2"] = np.asarray(0)
    return_dict["idx1"] = 0
    return_dict["idx2"] = 0

    if compute_dirns:
        return_dict["cca_dirns1"] = np.zeros((1, dimension))
        return_dict["cca_dirns2"] = np.zeros((1, dimension))

    return return_dict


def get_cca_similarity(
    acts1,
    acts2,
    epsilon=0.0,
    threshold=0.98,
    compute_coefs=True,
    compute_dirns=False,
    verbose=True,
):
    """The main function for computing cca similarities.

    This function computes the cca similarity between two sets of activations,
    returning a dict with the cca coefficients, a few statistics of the cca
    coefficients, and (optionally) the actual directions.

    Args:
              acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                     datapoints where entry (i,j) is the output of neuron i on
                     datapoint j.
              acts2: (num_neurons2, data_points) same as above, but (potentially)
                     for a different set of neurons. Note that acts1 and acts2
                     can have different numbers of neurons, but must agree on the
                     number of datapoints

              epsilon: small float to help stabilize computations

              threshold: float between 0, 1 used to get rid of trailing zeros in
                         the cca correlation coefficients to output more accurate
                         summary statistics of correlations.


              compute_coefs: boolean value determining whether coefficients
                             over neurons are computed. Needed for computing
                             directions

              compute_dirns: boolean value determining whether actual cca
                             directions are computed. (For very large neurons and
                             datasets, may be better to compute these on the fly
                             instead of store in memory.)

              verbose: Boolean, whether intermediate outputs are printed

    Returns:
              return_dict: A dictionary with outputs from the cca computations.
                           Contains neuron coefficients (combinations of neurons
                           that correspond to cca directions), the cca correlation
                           coefficients (how well aligned directions correlate),
                           x and y idxs (for computing cca directions on the fly
                           if compute_dirns=False), and summary statistics. If
                           compute_dirns=True, the cca directions are also
                           computed.
    """

    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
    # check that acts1, acts2 are transposition
    assert acts1.shape[0] < acts1.shape[1], "input must be number of neurons" "by datapoints"
    return_dict = {}

    # compute covariance with numpy function for extra stability
    numx = acts1.shape[0]
    numy = acts2.shape[0]

    covariance = np.cov(acts1, acts2)
    sigmaxx = covariance[:numx, :numx]
    sigmaxy = covariance[:numx, numx:]
    sigmayx = covariance[numx:, :numx]
    sigmayy = covariance[numx:, numx:]

    # rescale covariance to make cca computation more stable
    xmax = np.max(np.abs(sigmaxx))
    ymax = np.max(np.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= np.sqrt(xmax * ymax)
    sigmayx /= np.sqrt(xmax * ymax)

    ([u, s, v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs) = compute_ccas(
        sigmaxx, sigmaxy, sigmayx, sigmayy, epsilon=epsilon, verbose=verbose
    )

    # if x_idxs or y_idxs is all false, return_dict has zero entries
    if (not np.any(x_idxs)) or (not np.any(y_idxs)):
        return create_zero_dict(compute_dirns, acts1.shape[1])

    if compute_coefs:
        # also compute full coefficients over all neurons
        x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
        y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

        return_dict["coef_x"] = u.T
        return_dict["invsqrt_xx"] = invsqrt_xx
        return_dict["full_coef_x"] = np.zeros((numx, numx))
        np.place(return_dict["full_coef_x"], x_mask, return_dict["coef_x"])
        return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
        np.place(return_dict["full_invsqrt_xx"], x_mask, return_dict["invsqrt_xx"])

        return_dict["coef_y"] = v
        return_dict["invsqrt_yy"] = invsqrt_yy
        return_dict["full_coef_y"] = np.zeros((numy, numy))
        np.place(return_dict["full_coef_y"], y_mask, return_dict["coef_y"])
        return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
        np.place(return_dict["full_invsqrt_yy"], y_mask, return_dict["invsqrt_yy"])

        # compute means
        neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
        neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
        return_dict["neuron_means1"] = neuron_means1
        return_dict["neuron_means2"] = neuron_means2

    if compute_dirns:
        # orthonormal directions that are CCA directions
        cca_dirns1 = (
            np.dot(
                np.dot(return_dict["full_coef_x"], return_dict["full_invsqrt_xx"]),
                (acts1 - neuron_means1),
            )
            + neuron_means1
        )
        cca_dirns2 = (
            np.dot(
                np.dot(return_dict["full_coef_y"], return_dict["full_invsqrt_yy"]),
                (acts2 - neuron_means2),
            )
            + neuron_means2
        )

    # get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold(s, threshold)
    idx2 = sum_threshold(s, threshold)

    return_dict["cca_coef1"] = s
    return_dict["cca_coef2"] = s
    return_dict["x_idxs"] = x_idxs
    return_dict["y_idxs"] = y_idxs
    # summary statistics
    return_dict["mean"] = (np.mean(s[:idx1]), np.mean(s[:idx2]))
    return_dict["sum"] = (np.sum(s), np.sum(s))

    if compute_dirns:
        return_dict["cca_dirns1"] = cca_dirns1
        return_dict["cca_dirns2"] = cca_dirns2

    return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-6, compute_dirns=True):
    """Calls get_cca_similarity multiple times while adding noise.

    This function is very similar to get_cca_similarity, and can be used if
    get_cca_similarity doesn't converge for some pair of inputs. This function
    adds some noise to the activations to help convergence.

    Args:
              acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                     datapoints where entry (i,j) is the output of neuron i on
                     datapoint j.
              acts2: (num_neurons2, data_points) same as above, but (potentially)
                     for a different set of neurons. Note that acts1 and acts2
                     can have different numbers of neurons, but must agree on the
                     number of datapoints

              threshold: float between 0, 1 used to get rid of trailing zeros in
                         the cca correlation coefficients to output more accurate
                         summary statistics of correlations.

              epsilon: small float to help stabilize computations

              compute_dirns: boolean value determining whether actual cca
                             directions are computed. (For very large neurons and
                             datasets, may be better to compute these on the fly
                             instead of store in memory.)

    Returns:
              return_dict: A dictionary with outputs from the cca computations.
                           Contains neuron coefficients (combinations of neurons
                           that correspond to cca directions), the cca correlation
                           coefficients (how well aligned directions correlate),
                           x and y idxs (for computing cca directions on the fly
                           if compute_dirns=False), and summary statistics. If
                           compute_dirns=True, the cca directions are also
                           computed.
    """

    for trial in range(num_cca_trials):
        try:
            return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
        except np.linalg.LinAlgError:
            acts1 = acts1 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
            acts2 = acts2 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
            if trial + 1 == num_cca_trials:
                raise

    return return_dict
    # End of copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/cca_core.py


def top_k_pca_comps(singular_values, threshold=0.99):
    total_variance = np.sum(singular_values**2)
    explained_variance = (singular_values**2) / total_variance
    cumulative_variance = np.cumsum(explained_variance)
    return np.argmax(cumulative_variance >= threshold * total_variance) + 1


def _svcca_original(acts1, acts2):
    # Copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/tutorials/001_Introduction.ipynb
    # Modification: get_cca_similarity is in the same file.
    # Modification: top-k PCA component selection s.t. explained variance > 0.99 total variance
    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    # top-k PCA components only
    k1 = top_k_pca_comps(s1)
    k2 = top_k_pca_comps(s2)

    svacts1 = np.dot(s1[:k1] * np.eye(k1), V1[:k1])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:k2] * np.eye(k2), V2[:k2])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    # End of copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/tutorials/001_Introduction.ipynb
    return np.mean(svcca_results["cca_coef1"])


# Copied from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/pwcca.py
# Modification: get_cca_similarity is in the same file.
def compute_pwcca(acts1, acts2, epsilon=0.0):
    """Computes projection weighting for weighting CCA coefficients

    Args:
         acts1: 2d numpy array, shaped (neurons, num_datapoints)
         acts2: 2d numpy array, shaped (neurons, num_datapoints)

    Returns:
         Original cca coefficient mean and weighted mean

    """
    sresults = get_cca_similarity(
        acts1,
        acts2,
        epsilon=epsilon,
        compute_dirns=False,
        compute_coefs=True,
        verbose=False,
    )
    if np.sum(sresults["x_idxs"]) <= np.sum(sresults["y_idxs"]):
        dirns = (
            np.dot(
                sresults["coef_x"],
                (acts1[sresults["x_idxs"]] - sresults["neuron_means1"][sresults["x_idxs"]]),
            )
            + sresults["neuron_means1"][sresults["x_idxs"]]
        )
        coefs = sresults["cca_coef1"]
        acts = acts1
        idxs = sresults["x_idxs"]
    else:
        dirns = (
            np.dot(
                sresults["coef_y"],
                (acts1[sresults["y_idxs"]] - sresults["neuron_means2"][sresults["y_idxs"]]),
            )
            + sresults["neuron_means2"][sresults["y_idxs"]]
        )
        coefs = sresults["cca_coef2"]
        acts = acts2
        idxs = sresults["y_idxs"]
    P, _ = np.linalg.qr(dirns.T)
    weights = np.sum(np.abs(np.dot(P.T, acts[idxs].T)), axis=1)
    weights = weights / np.sum(weights)

    return np.sum(weights * coefs), weights, coefs
    # End of copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/pwcca.py


##################################################################################

from typing import Union  # noqa:e402

import numpy.typing as npt  # noqa:e402
import torch  # noqa:e402

# from repsim.measures.utils import (
#     SHAPE_TYPE,
#     flatten,
#     resize_wh_reps,
#     to_numpy_if_needed,
#     RepresentationalSimilarityMeasure,
# )  # noqa:e402


def svcca(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    return _svcca_original(R.T, Rp.T)


def pwcca(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    return compute_pwcca(R.T, Rp.T)[0]


class SVCCA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=svcca,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class PWCCA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=pwcca,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=False,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=False,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


# ## all decoder weights

# fast

# In[ ]:


svcca(weight_matrix_np, weight_matrix_2, "nd")


# In[ ]:


svcca(weight_matrix_np[highest_correlations_indices_v1], weight_matrix_2, "nd")


# ## single token subspaces

# In[ ]:


def get_rand_scores(modA_feats, modB_feats):
    total_scores = 0
    for i in range(100):
        # if i % 20 == 0:
        #     print(i)
        rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=len(modA_feats)).tolist()
        rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=len(modB_feats)).tolist()
        X_subset = weight_matrix_np[rand_modA_feats, :]
        Y_subset = weight_matrix_2[rand_modB_feats, :]

        total_scores += svcca(X_subset, Y_subset, "nd")

    return total_scores / 100


# In[ ]:


keyword = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "once"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "let"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# ## on token actvs

# In[13]:


svcca(reshaped_activations_A[:10000, :], reshaped_activations_B[:10000, :], "nd")


# In[ ]:


svcca(reshaped_activations_A, reshaped_activations_B, "nd")


# # cka

# ## fns

# In[ ]:


from typing import Union

import numpy.typing as npt
import torch
# from repsim.measures.utils import flatten
# from repsim.measures.utils import RepresentationalSimilarityMeasure
# from repsim.measures.utils import SHAPE_TYPE
# from repsim.measures.utils import to_torch_if_needed


def centered_kernel_alignment(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Kornblith et al. (2019)"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_torch_if_needed(R, Rp)
    N, D = R.size()

    R = R - R.mean(dim=0)[None, :]
    Rp = Rp - Rp.mean(dim=0)[None, :]

    if N < D:
        S = R @ R.T
        Sp = Rp @ Rp.T  # noqa: E741
        return (hsic(S, Sp) / torch.sqrt(hsic(S, S) * hsic(Sp, Sp))).item()
    else:
        return (
            torch.linalg.norm(Rp.T @ R, ord="fro") ** 2
            / (torch.linalg.norm(R.T @ R, ord="fro") * torch.linalg.norm(Rp.T @ Rp, ord="fro"))
        ).item()


def hsic(S: torch.Tensor, Sp: torch.Tensor) -> torch.Tensor:  # noqa: E741
    S = S - S.mean(dim=0)[:, None]
    Sp = Sp - Sp.mean(dim=0)[:, None]  # noqa: E741
    return torch.trace(S @ Sp) / (S.size(0) - 1) ** 2


class CKA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=centered_kernel_alignment,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )


# ## all decoder weights

# fast

# In[ ]:


centered_kernel_alignment(weight_matrix_np, weight_matrix_2, "nd")


# In[ ]:


centered_kernel_alignment(weight_matrix_np[highest_correlations_indices_v1], weight_matrix_2, "nd")


# ## single token subspaces

# In[ ]:


def get_rand_scores(modA_feats, modB_feats):
    total_scores = 0
    for i in range(100):
        # if i % 20 == 0:
        #     print(i)
        rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=len(modA_feats)).tolist()
        rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=len(modB_feats)).tolist()
        X_subset = weight_matrix_np[rand_modA_feats, :]
        Y_subset = weight_matrix_2[rand_modB_feats, :]

        total_scores += centered_kernel_alignment(X_subset, Y_subset, "nd")

    return total_scores / 100


# In[ ]:


keyword = "upon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "once"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# In[ ]:


keyword = "let"
modB_feats = find_indices_with_keyword(fList_model_B, keyword)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(svcca(X_subset, Y_subset, "nd"))
print(get_rand_scores(modA_feats, modB_feats))


# # weight cosine sim vs actv corr

# ## fn

# In[ ]:


import numpy as np

def find_all_highest_cosine_similarities(reshaped_activations_A, reshaped_activations_B):
    # Normalize rows of A (each vector) for cosine similarity
    norms_A = np.linalg.norm(reshaped_activations_A, axis=1, keepdims=True)
    normalized_A = reshaped_activations_A / (norms_A + 1e-8)  # Avoid division by zero

    # Normalize rows of B (each vector) for cosine similarity
    norms_B = np.linalg.norm(reshaped_activations_B, axis=1, keepdims=True)
    normalized_B = reshaped_activations_B / (norms_B + 1e-8)  # Avoid division by zero

    # Compute cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_A, normalized_B.T)

    # Get the highest cosine similarity indices and values
    highest_cosine_values = np.max(cosine_similarity_matrix, axis=1)
    highest_cosine_indices = np.argmax(cosine_similarity_matrix, axis=1)

    return highest_cosine_indices, highest_cosine_values


# In[ ]:


highest_cosine_indices, highest_cosine_values = find_all_highest_cosine_similarities(weight_matrix_np, weight_matrix_2)


# In[ ]:


len(highest_cosine_indices)


# In[ ]:


import torch
learned_norm = torch.nn.functional.normalize(torch.from_numpy(weight_matrix_np), p=2, dim=0)
ground_truth_norm = torch.nn.functional.normalize(torch.from_numpy(weight_matrix_2), p=2, dim=0)
cos_sims = torch.matmul(learned_norm, ground_truth_norm.t())
highest_cosine_values_v2, highest_cosine_indices_v2 = cos_sims.max(dim=0)


# In[ ]:


highest_cosine_indices_v2.shape


# In[ ]:


highest_cosine_indices_v2 = highest_cosine_indices_v2.cpu().numpy()


# ## metrics on cos sim paired

# In[ ]:


jaccard_similarity(weight_matrix_np[highest_cosine_indices], weight_matrix_2)


# In[ ]:


jaccard_similarity(weight_matrix_np[highest_cosine_indices_v2], weight_matrix_2)


# In[ ]:


centered_kernel_alignment(weight_matrix_np[highest_cosine_indices], weight_matrix_2, "nd")


# In[ ]:


centered_kernel_alignment(weight_matrix_np[highest_cosine_indices_v2], weight_matrix_2, "nd")


# In[ ]:


svcca(weight_matrix_np[highest_cosine_indices], weight_matrix_2, "nd")


# In[ ]:


svcca(weight_matrix_np[highest_cosine_indices_v2], weight_matrix_2, "nd")

