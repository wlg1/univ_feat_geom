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

# In[4]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-1L-21M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_np = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_np.shape)


# In[5]:


weight_matrix_np = weight_matrix_np.detach().numpy()


# In[6]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-2L-33M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_2.shape)


# In[7]:


weight_matrix_2 = weight_matrix_2.detach().numpy()


# # load sae f actvs

# In[8]:


file_path = '/content/drive/MyDrive/fActs_ts_1L_21M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_A = pickle.load(f)


# In[9]:


file_path = '/content/drive/MyDrive/fActs_ts_2L_33M_anySamps_v1.pkl'
with open(file_path, 'rb') as f:
    feature_acts_model_B = pickle.load(f)


# In[10]:


feature_acts_model_B.shape


# In[11]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[12]:


reshaped_activations_B.shape


# # load feature labels

# ## load

# In[18]:


import json
with open('feature_top_samps_lst_1L_16k.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[19]:


with open('feature_top_samps_lst_2L_MLP0_16k.json', 'rb') as f:
    feat_snip_dict_2 = json.load(f)


# In[20]:


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


# In[21]:


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


# In[22]:


fList_model_B = []
for feat_dict in feat_snip_dict_2:
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_B.append(out_str)


# ## save spliced labels

# In[23]:


with open('fList_model_A.pkl', 'wb') as f:
    pickle.dump(fList_model_A, f)
with open('fList_model_B.pkl', 'wb') as f:
    pickle.dump(fList_model_B, f)


# In[24]:


from google.colab import files
files.download('fList_model_A.pkl')
files.download('fList_model_B.pkl')


# In[25]:


import pickle
with open('fList_model_A.pkl', 'rb') as f:
    fList_model_A = pickle.load(f)
with open('fList_model_B.pkl', 'rb') as f:
    fList_model_B = pickle.load(f)


# # load corr

# In[13]:


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

# In[26]:


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


# In[27]:


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

# In[28]:


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


# In[29]:


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

# In[ ]:


jaccard_similarity(weight_matrix_np, weight_matrix_2)


# In[74]:


jaccard_similarity(weight_matrix_np[highest_correlations_indices_v1], weight_matrix_2)


# In[77]:


weight_matrix_2.shape[0]


# In[76]:


len(list(set(highest_correlations_indices_v1)))


# ## single token subspaces

# In[65]:


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


# In[70]:


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


# In[62]:


keyword_1 = "princess"
keyword_2 = "dragon"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[67]:


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


# In[71]:


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


# In[32]:


keyword_1 = "time"
keyword_2 = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1) + find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))
print(get_rand_scores(modA_feats, modB_feats))


# In[42]:


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

# In[46]:


keyword_1 = "time"
modB_feats_1 = find_indices_with_keyword(fList_model_B, keyword_1)
len(modB_feats_1)


# In[47]:


keyword_2 = "she"
modB_feats_2 = find_indices_with_keyword(fList_model_B, keyword_2)
modA_feats_2 = get_values_from_indices(modB_feats_2, highest_correlations_indices_v1)
len(modA_feats_2)


# In[48]:


minInd = min(len(modB_feats_1), len(modA_feats_2))
X_subset = weight_matrix_np[modA_feats_2[:minInd], :]
Y_subset = weight_matrix_2[modB_feats_1[:minInd], :]
print(len(X_subset))
print(jaccard_similarity(X_subset, Y_subset))


# ### more exmaples

# In[73]:


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


# In[ ]:





# ## corr explora

# In[39]:


modA_feats[:3]


# In[40]:


weight_matrix_np[modA_feats[:3], :]


# In[41]:


weight_matrix_np[13316, :]


# In[49]:


weight_matrix_np[[1,1,2], :]


# In[57]:


keyword_1 = "once"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(list(set(modA_feats))))
len(list(set(modB_feats)))


# In[58]:


keyword_1 = "she"
modB_feats = find_indices_with_keyword(fList_model_B, keyword_1)
modA_feats = get_values_from_indices(modB_feats, highest_correlations_indices_v1)
X_subset = weight_matrix_np[modA_feats, :]
Y_subset = weight_matrix_2[modB_feats, :]
print(len(list(set(modA_feats))))
len(list(set(modB_feats)))


# In[60]:


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


# In[78]:


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

# In[80]:


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


# In[81]:


len(mixed_modA_feats)


# In[82]:


print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=3))


# In[84]:


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


# In[85]:


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


# In[101]:


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


# In[93]:


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


# In[95]:


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
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[97]:


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
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[98]:


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
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[99]:


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
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[103]:


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
print(jaccard_similarity(X_subset, Y_subset, k=3))

print(get_rand_scores(mixed_modA_feats, mixed_modB_feats, k=2))


# In[ ]:



