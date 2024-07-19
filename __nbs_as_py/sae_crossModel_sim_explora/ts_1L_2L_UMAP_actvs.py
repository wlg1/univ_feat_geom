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

import umap
import matplotlib.pyplot as plt


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

# In[11]:


import json
with open('feature_top_samps_lst.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[12]:


with open('feature_top_samps_lst_2L_MLP0.json', 'rb') as f:
    feat_snip_dict_2 = json.load(f)


# In[13]:


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


# In[14]:


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


# In[15]:


fList_model_B = []
for feat_dict in feat_snip_dict_2:
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_B.append(out_str)


# # umap

# In[16]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# In[ ]:


# Transpose the data to treat columns as points
reshaped_activations_A_T = reshaped_activations_A.T
reshaped_activations_B_T = reshaped_activations_B.T

# Apply UMAP
reducer = umap.UMAP()
embedding1 = reducer.fit_transform(reshaped_activations_A_T)
embedding2 = reducer.fit_transform(reshaped_activations_B_T)


# In[ ]:


# Create DataFrames for Plotly
df1 = pd.DataFrame(embedding1, columns=['UMAP Component 1', 'UMAP Component 2'])
df1['Feature ID'] = range(len(embedding1))
df1['Feature Description'] = fList_model_A[:len(embedding1)]  # Adjust this if needed

df2 = pd.DataFrame(embedding2, columns=['UMAP Component 1', 'UMAP Component 2'])
df2['Feature ID'] = range(len(embedding2))
df2['Feature Description'] = fList_model_B[:len(embedding2)]  # Adjust this if needed

# Create side by side plots using Plotly subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('UMAP Projection of Dataset 1', 'UMAP Projection of Dataset 2'))

# Add first scatter plot
fig.add_trace(
    go.Scatter(
        x=df1['UMAP Component 1'], y=df1['UMAP Component 2'],
        mode='markers', marker=dict(color='blue'),
        text=df1['Feature ID'], customdata=np.array(df1[['Feature Description']]),
        hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}'
    ),
    row=1, col=1
)

# Add second scatter plot
fig.add_trace(
    go.Scatter(
        x=df2['UMAP Component 1'], y=df2['UMAP Component 2'],
        mode='markers', marker=dict(color='green'),
        text=df2['Feature ID'], customdata=np.array(df2[['Feature Description']]),
        hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text='UMAP Projections of Datasets',
    xaxis_title='UMAP Component 1',
    yaxis_title='UMAP Component 2',
    showlegend=False
)

fig.show()

