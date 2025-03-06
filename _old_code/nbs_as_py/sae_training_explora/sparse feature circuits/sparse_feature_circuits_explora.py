#!/usr/bin/env python
# coding: utf-8

# # setup

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!git clone https://github.com/saprmarks/feature-circuits.git\n%cd feature-circuits\n!pip install -r requirements.txt\n')


# In[ ]:


# import sys
# sys.path.append('/content/feature-circuits')


# In[ ]:


# !git submodule update --init  # this will download an older vers that doesn't use from_pretrained


# In[ ]:


get_ipython().system('git clone https://github.com/saprmarks/dictionary_learning.git')


# In[ ]:


get_ipython().system('bash dictionary_learning/pretrained_dictionary_downloader.sh')


# In[ ]:


import os
# Create the directory if it doesn't exist
os.makedirs('circuits', exist_ok=True)


# In[ ]:


get_ipython().system('bash scripts/get_circuit.sh rc_train 0.1 0.01 none 6 10')


# In[ ]:


get_ipython().system('bash scripts/evaluate_circuit.sh circuits/rc_train_dict10_node0.1_edge0.01_n100_aggnone.pt rc_test 0.1 6 10')

