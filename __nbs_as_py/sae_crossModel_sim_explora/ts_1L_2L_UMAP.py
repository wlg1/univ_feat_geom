#!/usr/bin/env python
# coding: utf-8

# # setup

# In[1]:


get_ipython().system('pip install umap-learn')


# In[2]:


import pickle
import numpy as np

import umap
import matplotlib.pyplot as plt


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


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


# # load feature labels

# In[ ]:


# features_list = ['amongus' for i in range(weight_matrix_np.shape[0])]


# In[ ]:


import pickle
with open('feat_snip_dict_strs.pkl', 'rb') as f:
    feat_snip_dict_strs = pickle.load(f)


# In[ ]:


features_list = [i for i in feat_snip_dict_strs.values()]


# In[ ]:


features_list[0]


# In[ ]:


import json
with open('feature_top_samps_lst.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[ ]:


len(feat_snip_dict)


# In[ ]:


features_list_v2 = []
for feat_dict in feat_snip_dict:
    features_list_v2.append(feat_dict['strings'][0])


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

features_list_v2 = []
for feat_dict in feat_snip_dict:
    # text = feat_dict['strings'][0]
    # result = extract_tagged_word(text)
    # features_list_v2.append(result)
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    features_list_v2.append(out_str)


# ## Use the code below for label plots

# In[43]:


import json
with open('feature_top_samps_lst.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[44]:


with open('feature_top_samps_lst_2L_MLP0.json', 'rb') as f:
    feat_snip_dict_2 = json.load(f)


# In[45]:


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


# In[46]:


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


# In[47]:


fList_model_B = []
for feat_dict in feat_snip_dict_2:
    out_str = ''
    for text in feat_dict['strings']:
        result = extract_tagged_word(text)
        out_str += result + ', '
    fList_model_B.append(out_str)


# # umap decoder weights

# In[ ]:


umap_model = umap.UMAP(n_components=2)
weight_matrix_umap = umap_model.fit_transform(weight_matrix_np)


# In[ ]:


plt.subplot(1, 2, 2)
plt.scatter(weight_matrix_umap[:, 0], weight_matrix_umap[:, 1], s=1, alpha=0.7)
plt.title('UMAP of Weight Matrix')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

plt.tight_layout()
plt.show()


# In[ ]:


# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# Fit and transform the data by rows
embedding = reducer.fit_transform(weight_matrix_np)

# Plot the UMAP result
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral')
plt.title('UMAP of Weight Matrix')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()


# # plot two models umap

# In[ ]:


# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# Fit and transform the data by rows
embedding1 = reducer.fit_transform(weight_matrix_np)
embedding2 = reducer.fit_transform(weight_matrix_2)

# Create Side-by-Side Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for the first dataset
axes[0].scatter(embedding1[:, 0], embedding1[:, 1], c='blue', label='Dataset 1')
axes[0].set_title('UMAP Projection of Dataset 1')
axes[0].legend()

# Plot for the second dataset
axes[1].scatter(embedding2[:, 0], embedding2[:, 1], c='green', label='Dataset 2')
axes[1].set_title('UMAP Projection of Dataset 2')
axes[1].legend()

plt.show()


# # cca on two models

# In[ ]:


from sklearn.cross_decomposition import CCA


# In[ ]:


# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# Fit and transform the data by rows
embedding1 = reducer.fit_transform(weight_matrix_np)
embedding2 = reducer.fit_transform(weight_matrix_2)

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

# Create Side-by-Side Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for the first dataset
axes[0].scatter(embedding1[:, 0], embedding1[:, 1], c='blue', label='Dataset 1')
axes[0].set_title('UMAP Projection of Dataset 1')
axes[0].legend()

# Plot for the second dataset
axes[1].scatter(embedding2[:, 0], embedding2[:, 1], c='green', label='Dataset 2')
axes[1].set_title('UMAP Projection of Dataset 2')
axes[1].legend()

plt.show()


# # interactve umap with feature labels

# ## umap

# In[ ]:


import umap
umap_model = umap.UMAP(n_components=2)
embedding = umap_model.fit_transform(weight_matrix_np)


# ## plot

# In[ ]:


import plotly.express as px
import numpy as np

# Create a DataFrame for Plotly
import pandas as pd
df = pd.DataFrame(embedding, columns=['UMAP Component 1', 'UMAP Component 2'])
df['Feature ID'] = range(len(features_list_v2))
df['Feature Description'] = features_list_v2

# Plot using Plotly
fig = px.scatter(df, x='UMAP Component 1', y='UMAP Component 2', text='Feature ID')

# Customize hover information
fig.update_traces(
    hovertemplate='<b>Feature ID:</b> %{text}<br><b>Description:</b> %{customdata[0]}',
    customdata=np.array(df[['Feature Description']])
)

fig.update_layout(
    title='UMAP of Decoder Weights',
    xaxis_title='UMAP Component 1',
    yaxis_title='UMAP Component 2'
)

fig.show()


# In[ ]:


import umap
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, Div, CustomJS
from bokeh.layouts import layout
output_notebook()

# Create DataFrame for Bokeh
df = pd.DataFrame(embedding, columns=['UMAP Component 1', 'UMAP Component 2'])
df['Feature ID'] = range(len(features_list_v2))
df['Feature Description'] = features_list_v2

source = ColumnDataSource(df)

# Create the plot
plot = figure(title='UMAP of Decoder Weights',
              x_axis_label='UMAP Component 1',
              y_axis_label='UMAP Component 2',
              tools="pan,wheel_zoom,reset")

# Add points to the plot
plot.circle('UMAP Component 1', 'UMAP Component 2', size=5, source=source)

# Create a HoverTool
hover = HoverTool()
hover.tooltips = [("Feature ID", "@{Feature ID}"), ("Description", "@{Feature Description}")]

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create a Div to show the detailed feature information
div = Div(width=800)

# CustomJS callback to update the Div text
callback = CustomJS(args=dict(source=source, div=div), code="""
    const indices = cb_obj.indices;
    if (indices.length > 0) {
        const index = indices[0];
        const data = source.data;
        const feature_id = data['Feature ID'][index];
        const description = data['Feature Description'][index];
        div.text = "<b>Feature:</b> " + feature_id + "<br><pre>" + description + "</pre>";
    } else {
        div.text = "Hover over a point to see details here.";
    }
""")
source.selected.js_on_change('indices', callback)

# Layout to arrange the plot and the div
layout = layout([
    [plot],
    [div]
])

# Show the plot
show(layout)


# # two models interactive umap

# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Assume weight_matrix_np, weight_matrix_2, and features_list_v2 are already defined

# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# Fit and transform the data by rows
embedding1 = reducer.fit_transform(weight_matrix_np)
embedding2 = reducer.fit_transform(weight_matrix_2)

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


# In[ ]:


import umap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Assume weight_matrix_np, weight_matrix_2, and features_list_v2 are already defined

# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, metric='euclidean')

# Fit and transform the data by rows
embedding1 = reducer.fit_transform(weight_matrix_np)
embedding2 = reducer.fit_transform(weight_matrix_2)

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


# In[17]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[18]:


reshaped_activations_B.shape


# # corr mat

# ## weight corr

# In[ ]:


import torch
import numpy as np

reshaped_activations_A = weight_matrix_np
reshaped_activations_B = weight_matrix_2

def top_ind_from_B(ind):
    # Select a row from matrix A
    column_A = reshaped_activations_B[ind, :] #.numpy()

    # Compute the correlation of row_A with all rows in B
    correlations = []

    for i in range(reshaped_activations_A.shape[1]):
        column_B = reshaped_activations_A[i, :] #.numpy()

        if np.std(column_A) == 0 or np.std(column_B) == 0:
            # Skip columns with zero standard deviation
            correlations.append(np.nan)
        else:
            correlation = np.corrcoef(column_A, column_B)[0, 1]
            correlations.append(correlation)

    # Convert the list of correlations to a NumPy array for easy manipulation
    correlations = np.array(correlations)

    # Remove nan values for the purpose of finding the highest correlation
    valid_correlations = np.where(np.isnan(correlations), -np.inf, correlations)

    # Get the index of the column in B with the highest correlation
    # highest_correlation_index = np.argmax(valid_correlations)
    # highest_correlation_value = correlations[highest_correlation_index]

    # # Extract the column with the highest correlation
    # highest_correlation_column = reshaped_activations_B[:, highest_correlation_index]

    # print(f'Highest correlation value: {highest_correlation_value}')
    # print(f'Index of the column with highest correlation: {highest_correlation_index}')
    # print(f'Column with highest correlation:\n {highest_correlation_column}')

    # Get the indices of the top 10 columns in B with the highest correlations
    top_10_indices = np.argsort(valid_correlations)[-1:][::-1] # [-10:][::-1]
    top_10_correlations = correlations[top_10_indices]
    return top_10_indices, top_10_correlations


# In[ ]:


corrs = []
for i in range(1024):
    # print(top_ind_from_B(i))
    top_10_indices, top_10_correlations = top_ind_from_B(i)
    corrs.append(top_10_correlations)


# In[ ]:


from matplotlib import pyplot as plt
plt.hist(corrs, 10)
plt.show()


# We see weights have VERY low correlations, so try activations now

# ## actv corrs one feature

# In[ ]:


# import torch
# import numpy as np

# def top_ind_from_B(ind):
#     # Select a column from matrix A (e.g., the first column)
#     column_A = reshaped_activations_B[:, ind].numpy()

#     # Compute the correlation of column_A with all columns in B
#     correlations = []

#     for i in range(reshaped_activations_A.shape[1]):
#         column_B = reshaped_activations_A[:, i].numpy()

#         if np.std(column_A) == 0 or np.std(column_B) == 0:
#             # Skip columns with zero standard deviation
#             correlations.append(np.nan)
#         else:
#             correlation = np.corrcoef(column_A, column_B)[0, 1]
#             correlations.append(correlation)

#     # Convert the list of correlations to a NumPy array for easy manipulation
#     correlations = np.array(correlations)

#     # Remove nan values for the purpose of finding the highest correlation
#     valid_correlations = np.where(np.isnan(correlations), -np.inf, correlations)

#     # Get the index of the column in B with the highest correlation
#     # highest_correlation_index = np.argmax(valid_correlations)
#     # highest_correlation_value = correlations[highest_correlation_index]

#     # # Extract the column with the highest correlation
#     # highest_correlation_column = reshaped_activations_B[:, highest_correlation_index]

#     # print(f'Highest correlation value: {highest_correlation_value}')
#     # print(f'Index of the column with highest correlation: {highest_correlation_index}')
#     # print(f'Column with highest correlation:\n {highest_correlation_column}')

#     # Get the indices of the top 10 columns in B with the highest correlations
#     top_10_indices = np.argsort(valid_correlations)[-10:][::-1]
#     top_10_correlations = correlations[top_10_indices]
#     return top_10_indices, top_10_correlations


# In[ ]:


import torch
import numpy as np

def top_ind_from_B(ind, reshaped_activations_A, reshaped_activations_B):
    # Select a column from matrix B (e.g., the first column)
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
    top_10_indices = torch.topk(all_correlations, 10).indices.cpu().numpy()
    top_10_correlations = all_correlations[top_10_indices].cpu().numpy()

    return top_10_indices, top_10_correlations


# In[ ]:


top_10_indices, top_10_correlations = top_ind_from_B(0, reshaped_activations_A, reshaped_activations_B)
print(f'Top 10 indices: {top_10_indices}')
print(f'Top 10 correlations: {top_10_correlations}')


# In[ ]:


top_10_indices, top_10_correlations = top_ind_from_B(3103, reshaped_activations_A, reshaped_activations_B)
print(f'Top 10 indices: {top_10_indices}')
print(f'Top 10 correlations: {top_10_correlations}')


# In[ ]:


top_10_indices, top_10_correlations = top_ind_from_B(3103)
print(top_10_indices)
print(top_10_correlations)


# ## plot feature actv corrs

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


model_A_f_ind = 16251
model_B_f_ind = 3103

feature_0_actvs_A = reshaped_activations_A[:, model_A_f_ind].numpy()
feature_0_actvs_B = reshaped_activations_B[:, model_B_f_ind].numpy()

corr = np.corrcoef(feature_0_actvs_A, feature_0_actvs_B)[0, 1]
print(corr)

plt.scatter(feature_0_actvs_A, feature_0_actvs_B, alpha=0.5)

plt.xlabel('Feature Activations (Model A)')
plt.ylabel('Feature Activations (Model B)')
plt.title('Feature Activations (A/16251 vs B/3103)\n Corr = ' + str(corr))

plt.tight_layout()
plt.show()


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


# In[ ]:


model_A_f_ind = 5561
model_B_f_ind = 0

feature_0_actvs_A = reshaped_activations_A[:, model_A_f_ind].numpy()
feature_0_actvs_B = reshaped_activations_B[:, model_B_f_ind].numpy()

corr = np.corrcoef(feature_0_actvs_A, feature_0_actvs_B)[0, 1]
print(corr)

plt.scatter(feature_0_actvs_A, feature_0_actvs_B, alpha=0.5)

plt.xlabel('Feature Activations (Model A)')
plt.ylabel('Feature Activations (Model B)')
plt.title('Feature Activations (A vs B)\n Corr = ' + str(corr))

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


# ### load

# In[32]:


import pickle
with open('highest_correlations_indices_v1.pkl', 'rb') as f:
    highest_correlations_indices_v1 = pickle.load(f)
with open('highest_correlations_values_v1.pkl', 'rb') as f:
    highest_correlations_values_v1 = pickle.load(f)


# # load model

# In[ ]:


get_ipython().run_line_magic('pip', 'install transformer-lens')


# In[ ]:


from transformer_lens import HookedTransformer


# In[ ]:


model = HookedTransformer.from_pretrained("tiny-stories-1L-21M")


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


# 0 in model B

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model A
top_feats = [5561]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_A, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


# 5561 in model A

# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

# top features in matching pair with model B
top_feats = [0]

# for feature_idx in top_acts_indices[0, -1, :]:
for feature_idx in top_feats:
    # feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts_model_B, feature_idx, samp_m, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


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


# # pair hover umaps

# ## umap

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


# ### load

# In[20]:


import pickle
with open('embedding1.pkl', 'rb') as f:
    embedding1 = pickle.load(f)
with open('embedding2.pkl', 'rb') as f:
    embedding2 = pickle.load(f)


# ## try just highlight a pt

# In[22]:


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

# Display the figure
fig.show()

# Add the following JavaScript code to change the color of points on hover
display(HTML("""
<script>
require(["plotly"], function(Plotly) {
    var myPlot = document.getElementsByClassName('js-plotly-plot');

    myPlot[0].on('plotly_hover', function(data) {
        var pn = data.points[0].pointNumber;
        var trace = data.points[0].curveNumber;
        var update = {'marker': {color: 'yellow'}};
        update['marker.color'] = new Array(myPlot[0].data[trace].x.length).fill(myPlot[0].data[trace].marker.color);
        update['marker.color'][pn] = 'yellow';
        Plotly.restyle(myPlot[0], update, [trace]);
    });

    myPlot[0].on('plotly_unhover', function(data) {
        var pn = data.points[0].pointNumber;
        var trace = data.points[0].curveNumber;
        var update = {'marker': {color: 'blue'}};
        if(trace == 1) update = {'marker': {color: 'green'}};
        Plotly.restyle(myPlot[0], update, [trace]);
    });
});
</script>
"""))


# ## failed attempt using dash

# In[ ]:


# feat_map = {i:i for i in range(len(fList_model_A))}
# highest_correlations_indices


# In[ ]:


feat_map = {i:i for i in range(len(fList_model_A))}


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install dash jupyter-dash\n')


# In[ ]:


from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn.datasets import make_blobs

app = JupyterDash(__name__)

# Create Plotly scatter plots
def create_figure(embedding, highlight_index=None):
    colors = ['blue']*100
    if highlight_index is not None:
        colors[highlight_index] = 'yellow'
    return {
        'data': [go.Scatter(x=embedding[:,0], y=embedding[:,1], mode='markers',
                            marker=dict(color=colors, size=10),
                            hoverinfo='text', text=[f'Point {i}' for i in range(100)])],
        'layout': go.Layout(hovermode='closest')
    }

app.layout = html.Div([
    dcc.Graph(id='graph1', figure=create_figure(embedding1)),
    dcc.Graph(id='graph2', figure=create_figure(embedding2)),
    html.Pre(id='hover-data')
])

@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Input('graph1', 'hoverData'),
    Input('graph2', 'hoverData')
)
def update_graph(hover_data1, hover_data2):
    ctx = dash.callback_context
    if not ctx.triggered:
        point_index = None
    else:
        point_index = ctx.triggered[0]['value']['points'][0]['pointIndex'] if ctx.triggered[0]['value'] else None

    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'graph1.hoverData':
        mapped_index = feat_map.get(point_index, None)
        return create_figure(embedding1, point_index), create_figure(embedding2, mapped_index)
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'graph2.hoverData':
        inv_map = {v: k for k, v in feat_map.items()}
        mapped_index = inv_map.get(point_index, None)
        return create_figure(embedding1, mapped_index), create_figure(embedding2, point_index)
    return create_figure(embedding1), create_figure(embedding2)

# Run the Dash app within a Jupyter environment
app.run_server(mode='inline')


# # search modB features with keyword, get modA f pair

# In[54]:


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


# # statically color points on 2 plots

# In[53]:


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

# In[56]:


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


# In[57]:


keyword = "princess"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# ## try other keywords

# In[58]:


keyword = "let"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# In[59]:


keyword = "saw"
keyword_umaps(keyword, embedding1, embedding2, fList_model_A, fList_model_B, highest_correlations_indices_v1)


# In[ ]:




