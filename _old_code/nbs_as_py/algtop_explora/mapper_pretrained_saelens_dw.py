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

# In[7]:


# fn = 'ts-1L-21M_Wdec'
# fn = 'ts-2L-33M_Wdec'
fn = 'gpt2-small-8-res-jb_Wdec'
file_path = f'/content/drive/MyDrive/{fn}.pkl'
with open(file_path, 'rb') as f:
    feature_weights = pickle.load(f)


# In[8]:


data = feature_weights.detach().cpu().numpy()
data.shape


# # load labels

# In[9]:


import json
with open('gpt2-small-8-res-jb-explanations.json', 'rb') as f:
    feat_snip_dict = json.load(f)


# In[10]:


# can't just loop over dict as it's not in order
# labels = []
# for feat_dict in feat_snip_dict['explanations']:
#     labels.append(feat_dict['description'])

labels = [0] * len(feat_snip_dict['explanations'])
for feat_dict in feat_snip_dict['explanations']:
    labels[int(feat_dict['index'])] = feat_dict['description']


# In[11]:


labels[41]


# In[12]:


fList_model_A = np.array(labels)


# # Mapper

# In[13]:


mapper = km.KeplerMapper(verbose=1) # initialize mapper

# project data into 2D subspace via 2 step transformation, 1)isomap 2)UMAP
projected_data = mapper.fit_transform(data, projection=[manifold.Isomap(n_components=100, n_jobs=-1), umap.UMAP(n_components=2,random_state=1)])

# cluster data using DBSCAN
graph = mapper.map(projected_data, data, clusterer=sklearn.cluster.DBSCAN(metric="cosine"))


# In[14]:


# define an excessively long filename (helpful if saving multiple Mapper variants for single dataset)
fileID = fn + '_projection=' + graph['meta_data']['projection'].split('(')[0] + '_' + \
'n_cubes=' + str(graph['meta_data']['n_cubes']) + '_' + \
'perc_overlap=' + str(graph['meta_data']['perc_overlap']) + '_' + \
'clusterer=' + graph['meta_data']['clusterer'].split('(')[0] + '_' + \
'scaler=' + graph['meta_data']['scaler'].split('(')[0]

fileID


# In[15]:


labels = list(range(data.shape[0]))
labels = np.array(labels)


# In[16]:


mapper.visualize(graph,
                path_html=fileID + ".html",
                title=fileID,
                custom_tooltips =  fList_model_A,
                # custom_tooltips = labels,
                # color_values = np.log(per_return+1),
                color_function_name = 'test',
                node_color_function = np.array(['average', 'std', 'sum', 'max', 'min']))


# In[17]:


from google.colab import files
files.download(fileID + ".html")

