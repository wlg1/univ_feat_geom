#!/usr/bin/env python
# coding: utf-8

# # install sae-lens, install umap, import umap

# error

# In[ ]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[ ]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[ ]:


import umap


# # install umap, install sae-lens, import umap

# error

# In[ ]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[ ]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[ ]:


import umap


# # import umap, install sae-lens, use umap and saelens

# no error

# In[ ]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[ ]:


import umap


# In[ ]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

n_samples = 100
n_features = 10
data = np.random.rand(n_samples, n_features)

reducer = umap.UMAP(random_state=42)
data_umap = reducer.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(data_umap[:, 0], data_umap[:, 1], cmap='viridis')
plt.title('UMAP projection of random data')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()


# Restart from here (not disconnect) to run the code

# In[ ]:


from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
)


# # numpy vers update after saelens install

# no error

# In[1]:


get_ipython().run_line_magic('pip', 'install sae-lens')


# In[2]:


get_ipython().run_line_magic('pip', 'install numpy==1.25.2')


# In[1]:


get_ipython().run_line_magic('pip', 'install umap-learn')


# In[2]:


import umap


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

n_samples = 100
n_features = 10
data = np.random.rand(n_samples, n_features)

reducer = umap.UMAP(random_state=42)
data_umap = reducer.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(data_umap[:, 0], data_umap[:, 1], cmap='viridis')
plt.title('UMAP projection of random data')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()


# In[5]:


from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
)

