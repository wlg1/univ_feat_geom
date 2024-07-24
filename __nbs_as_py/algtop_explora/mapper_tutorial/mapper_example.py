#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# %%capture
get_ipython().system('pip install kmapper==2.0.1 matplotlib==3.4.1 numpy scikit_learn==1.0.2 umap umap-learn')


# In[ ]:


# numpy==1.19.4
# yfinance==0.1.63
# umap==0.1.1 umap-learn==0.4.6


# In[ ]:


get_ipython().system('pip show kmapper')


# In[ ]:


import yfinance as yf
import kmapper as km
from kmapper.jupyter import display
import umap
import sklearn
import sklearn.manifold as manifold
import numpy as np
import matplotlib.pyplot as plt


# # Get data

# In[2]:


# read text file with ticker names
filename = open("SP500_tickernames.txt", "r")
raw_tickernames = filename.read()
ticker_names = raw_tickernames.split("\n")
ticker_names = ticker_names[:len(ticker_names)-1]


# In[8]:


import yfinance as yf
import numpy as np


# In[5]:


# define date range
start_date_string = "2020-01-01"
end_date_string = "2022-04-02"

# pull historical data
raw_data = yf.download(ticker_names, start=start_date_string, end=end_date_string)


# In[6]:


# get daily close prices and drop missing columns
df_close = raw_data['Adj Close'].dropna(axis='columns')


# In[9]:


# convert pandas dataframe to numpy array, standardize ticker data, and transpose array
data = df_close.to_numpy()
data = data-np.mean(data, axis=0)/np.std(data, axis=0)
data = data.transpose()


# In[ ]:


# calculate percent return of each ticker over date range
per_return = (df_close.to_numpy().transpose()[:,504] - df_close.to_numpy().transpose()[:,0])/df_close.to_numpy().transpose()[:,0]


# In[11]:


data.shape


# In[13]:


df_close.columns.to_numpy()


# # Mapper

# In[ ]:


# initialize mapper
mapper = km.KeplerMapper(verbose=1)

# project data into 2D subsapce via 2 step transformation, 1)isomap 2)UMAP
projected_data = mapper.fit_transform(data, projection=[manifold.Isomap(n_components=100, n_jobs=-1), umap.UMAP(n_components=2,random_state=1)])

# cluster data using DBSCAN
# G = mapper.map(projected_data, data, clusterer=sklearn.cluster.DBSCAN(metric="cosine"))
graph = mapper.map(projected_data, data, clusterer=sklearn.cluster.DBSCAN(metric="cosine"))


# In[ ]:


# define an excessively long filename (helpful if saving multiple Mapper variants for single dataset)
fileID = 'projection=' + G['meta_data']['projection'].split('(')[0] + '_' + \
'n_cubes=' + str(G['meta_data']['n_cubes']) + '_' + \
'perc_overlap=' + str(G['meta_data']['perc_overlap']) + '_' + \
'clusterer=' + G['meta_data']['clusterer'].split('(')[0] + '_' + \
'scaler=' + G['meta_data']['scaler'].split('(')[0]


# In[ ]:


mapper.visualize(G,
                path_html="mapper_example_" + fileID + ".html",
                title=fileID,
                custom_tooltips = df_close.columns.to_numpy(),
                color_values = np.log(per_return+1),
                color_function_name = 'Log Percent Returns',
                node_color_function = np.array(['average', 'std', 'sum', 'max', 'min']))


# In[ ]:


# https://colab.research.google.com/github/shizuo-kaji/TutorialTopologicalDataAnalysis/blob/master/TopologicalDataAnalysisWithPython.ipynb#scrollTo=w41pQN-zHPcB

import kmapper, kmapper.jupyter
from sklearn.preprocessing import StandardScaler
import networkx as nx

# km = kmapper.KeplerMapper()
# X_std = StandardScaler().fit_transform(X)   # standardise
# f = km.project(X_std,projection="sum") # we have to choose a 1-dim projection
# graph = km.map(lens=f, X=X_std, clusterer=sklearn.cluster.DBSCAN(metric="cosine",algorithm='auto', eps=0.3, leaf_size=10))

nx.draw(kmapper.adapter.to_nx(graph))
#km.visualize(graph, color_function=Y, custom_tooltips=Y, path_html="kepler-mapper-output.html") # To download the output, look at the beginning of this note
# km.visualize(graph, custom_tooltips=Y, path_html="kepler-mapper-output.html") # To download the output, look at the beginning of this note
mapper.visualize(graph, path_html="kepler-mapper-output.html") # To download the output, look at the beginning of this note
print("Download kepler-mapper-output.html and open it with a browser to see the results.")
# the following rich visualisation does not work with Google Colab.
# download "kepler-mapper-output.html" and open it locally with a browser
#kmapper.jupyter.display(path_html="kepler-mapper-output.html")


# In[ ]:


from google.colab import files
files.download("mapper_example_" + fileID + ".html")


# In[ ]:


nodeid = 'cube0_cluster0'
node = G['nodes'][nodeid]


plt.figure(figsize=(18, 8), dpi=80)
plt.rcParams.update({'font.size': 22})

for i in node:
    plt.plot(df_close.iloc[:,i], linewidth=2)

plt.legend(list(df_close.columns[node]), fontsize=18)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title(nodeid)

plt.savefig("mapper_example_" + fileID + ".png", dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='white', edgecolor='auto')


# In[ ]:


# convert notebook to python script
get_ipython().system('jupyter nbconvert --to script mapper_example.ipynb')


# In[ ]:




