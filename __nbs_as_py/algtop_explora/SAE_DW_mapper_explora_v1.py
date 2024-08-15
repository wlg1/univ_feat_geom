#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


# %%capture
get_ipython().system('pip install kmapper matplotlib numpy scikit_learn umap umap-learn')


# In[3]:


import pickle

import kmapper as km
from kmapper.jupyter import display
import umap
import sklearn
import sklearn.manifold as manifold
import numpy as np
import matplotlib.pyplot as plt


# # load sae features

# In[4]:


fn = 'ts-1L-21M_Wdec'
# fn = 'ts-2L-33M_Wdec'
file_path = f'/content/drive/MyDrive/{fn}.pkl'
with open(file_path, 'rb') as f:
    feature_weights = pickle.load(f)


# In[5]:


data = feature_weights.detach().cpu().numpy()
data.shape


# # load labels

# In[9]:


import json
with open('feature_top_samps_lst_1L_16k.json', 'rb') as f:
# with open('feature_top_samps_lst_2L_MLP0.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[10]:


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


# In[11]:


fList_model_A = []
for feat_dict in feat_snip_dict:
    text = feat_dict['strings'][0]
    result = extract_tagged_word(text)
    fList_model_A.append(result)
    # out_str = ''
    # for text in feat_dict['strings']:
    #     result = extract_tagged_word(text)
    #     out_str += result + ', '
    # fList_model_A.append(out_str)


# In[12]:


fList_model_A[:5]


# In[13]:


fList_model_A = np.array(fList_model_A)
len(fList_model_A)


# # Mapper

# In[14]:


from sklearn.decomposition import PCA


# In[20]:


mapper = km.KeplerMapper(verbose=1) # initialize mapper

# project data into 2D subspace via 2 step transformation, 1)isomap 2)UMAP
# projected_data = mapper.fit_transform(data, projection=[manifold.Isomap(n_components=100, n_jobs=-1), umap.UMAP(n_components=2,random_state=1)])
# projected_data = mapper.fit_transform(data, projection=PCA(n_components=2))
projected_data = mapper.fit_transform(data, projection="l2norm")

# cluster data using DBSCAN
graph = mapper.map(projected_data, data, clusterer=sklearn.cluster.DBSCAN(metric="cosine"))


# In[21]:


# define an excessively long filename (helpful if saving multiple Mapper variants for single dataset)
fileID = fn + '_projection=' + graph['meta_data']['projection'].split('(')[0] + '_' + \
'n_cubes=' + str(graph['meta_data']['n_cubes']) + '_' + \
'perc_overlap=' + str(graph['meta_data']['perc_overlap']) + '_' + \
'clusterer=' + graph['meta_data']['clusterer'].split('(')[0] + '_' + \
'scaler=' + graph['meta_data']['scaler'].split('(')[0]

fileID


# In[22]:


labels = list(range(data.shape[0]))
labels = np.array(labels)


# In[23]:


mapper.visualize(graph,
                path_html=fileID + ".html",
                title=fileID,
                custom_tooltips =  fList_model_A,
                # custom_tooltips = labels,
                # color_values = np.log(per_return+1),
                color_function_name = 'test',
                node_color_function = np.array(['average', 'std', 'sum', 'max', 'min']))


# In[24]:


from google.colab import files
files.download(fileID + ".html")

