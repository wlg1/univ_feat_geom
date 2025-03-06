#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


get_ipython().system('git clone https://github.com/wlg1/univ_feat_geom.git')


# In[2]:


import pickle
import numpy as np
import matplotlib.pyplot as plt


# ## load weight mats

# Temporarily loading data from repo for convenience; larger files in the future will used a better storage system and not a repo

# In[3]:


file_path = '/content/univ_feat_geom/data/Wdec_ts_1L_21M_df16384_steps100k.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_1 = pickle.load(f)
weight_matrix_1 = weight_matrix_1.detach().numpy()
print(weight_matrix_1.shape)


# In[4]:


file_path = '/content/univ_feat_geom/data/Wdec_ts_2L_33M_df16384_steps100k.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)
weight_matrix_2  = weight_matrix_2.detach().numpy()
print(weight_matrix_2.shape)


# ## Comparison Functions

# In[5]:


def compare_singular_values(S1, S2):
    l2_distance = np.linalg.norm(S1 - S2)

    # print("Singular values of the first matrix:", S1)
    # print("Singular values of the second matrix:", S2)
    print("L2 distance between singular values:", l2_distance)

    # print('\n')

    # Normalize singular values
    S1_normalized = S1 / np.linalg.norm(S1)
    S2_normalized = S2 / np.linalg.norm(S2)

    l2_distance_normalized = np.linalg.norm(S1_normalized - S2_normalized)

    # print("Normalized Singular values of the second matrix:", S1_normalized)
    # print("Normalized Singular values of the second matrix:", S2_normalized)
    print("L2 distance between normalized singular values:", l2_distance_normalized)

    return l2_distance, l2_distance_normalized


# In[6]:


import numpy as np
import random
random.seed(3)

def generate_rand_svd(n, m):
    # Generate two random matrices of size n x m
    matrix1 = np.random.rand(n, m)
    matrix2 = np.random.rand(n, m)

    # Perform SVD on both matrices
    U1, S1_rand, Vt1 = np.linalg.svd(matrix1)
    U2, S2_rand, Vt2 = np.linalg.svd(matrix2)

    return(S1_rand, S2_rand)


# In[7]:


matPair_to_l2Dist = {}
matPair_to_l2Dist_norma = {}


# # Compare SAEs on 2 LLMs

# In[8]:


U1, S1_SAE, Vt1 = np.linalg.svd(weight_matrix_1)
U2, S2_SAE, Vt2 = np.linalg.svd(weight_matrix_2)


# In[9]:


matPair_to_l2Dist['SAE1_SAE2'], matPair_to_l2Dist_norma['SAE1_SAE2'] = compare_singular_values(S1_SAE, S2_SAE)


# # Compare to random weights

# In[10]:


n, m = weight_matrix_1.shape[0], weight_matrix_1.shape[1]
S1_rand, S2_rand = generate_rand_svd(n, m)


# In[11]:


# matPair_to_l2Dist['rand_rand'], matPair_to_l2Dist_norma['rand_rand'] = compare_singular_values(S1_rand, S2_rand)
compare_singular_values(S1_rand, S2_rand)


# In[12]:


matPair_to_l2Dist['SAE_rand'], matPair_to_l2Dist_norma['SAE_rand'] = compare_singular_values(S1_SAE, S1_rand)


# In[49]:


n, m = 3, 3
S1_rand_small, S2_rand_small = generate_rand_svd(n, m)
compare_singular_values(S1_rand_small, S2_rand_small)


# The large size of the matrices (16384x1024) may caused the singular values to stabilize and appear similar? No; even small values are similar. Why?

# In[50]:


n, m = 50, 50
S1_rand_med, S2_rand_med = generate_rand_svd(n, m)
compare_singular_values(S1_rand_med, S2_rand_med)


# # Compare weight matrices of orig LLMs

# ## load and get svd

# In[13]:


from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1Layer-21M")
mlp_weights = model.transformer.h[0].mlp.c_proj.weight
mlp_weights.shape


# In[14]:


model_2 = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-2Layers-33M")
mlp_weights_2 = model_2.transformer.h[0].mlp.c_proj.weight
mlp_weights_2.shape


# In[15]:


U1, S1_LLM, Vt1 = np.linalg.svd(mlp_weights.detach().numpy())
U2, S2_LLM, Vt2 = np.linalg.svd(mlp_weights_2.detach().numpy())


# ## compare LLMs (MLP0)

# In[16]:


matPair_to_l2Dist['ts1_ts2'], matPair_to_l2Dist_norma['ts1_ts2'] = compare_singular_values(S1_LLM, S2_LLM)


# In[17]:


# ts1_rand
matPair_to_l2Dist['ts_rand'], matPair_to_l2Dist_norma['ts_rand'] = compare_singular_values(S1_LLM, S1_rand)


# # Compare to GPT2 med

# ## load and get svd

# In[18]:


gpt2_med = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")


# In[19]:


mlp0_weights_gpt2_med = gpt2_med.transformer.h[0].mlp.c_proj.weight
mlp0_weights_gpt2_med.shape


# In[20]:


U2, S_GPT2_0, Vt2 = np.linalg.svd(mlp0_weights_gpt2_med.detach().numpy())


# In[21]:


S_GPT2_0.shape


# ## Compare MLP0

# In[22]:


matPair_to_l2Dist['ts_GPT2'], matPair_to_l2Dist_norma['ts_GPT2'] = compare_singular_values(S1_LLM, S_GPT2_0)


# In[23]:


matPair_to_l2Dist['GPT2_rand'], matPair_to_l2Dist_norma['GPT2_rand'] = compare_singular_values(S_GPT2_0, S1_rand)


# # Compare to pythia

# ## load and get svd

# In[24]:


from transformers import GPTNeoXForCausalLM, AutoTokenizer

model_pythia410 = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m",
)


# In[25]:


model_pythia410_weights_mlp0 = model_pythia410.gpt_neox.layers[0].mlp.dense_4h_to_h.weight
model_pythia410_weights_mlp0.shape


# In[26]:


U1, S_pythia410_mlp0, Vt1 = np.linalg.svd(model_pythia410_weights_mlp0.detach().numpy())


# ## Compare MLP0

# In[27]:


matPair_to_l2Dist['ts_pythia'], matPair_to_l2Dist_norma['ts_pythia'] = compare_singular_values(S_pythia410_mlp0, S1_LLM)


# In[28]:


matPair_to_l2Dist['pythia_GPT'], matPair_to_l2Dist['pythia_GPT'] = compare_singular_values(S_pythia410_mlp0, S_GPT2_0)


# In[29]:


matPair_to_l2Dist['pythia_rand'], matPair_to_l2Dist_norma['pythia_rand'] = compare_singular_values(S_pythia410_mlp0, S1_rand)


# # Summarize results

# ## labels next to point

# (better for normalized)

# In[45]:


def plot1D_L2dist_singvals(data, norma_bool=False):
    sorted_data = sorted(data.items(), key=lambda x: x[1])
    keys = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    # Create a custom color list
    colors = []
    for label in keys:
        if 'SAE' in label and 'rand' not in label:
            colors.append('green')
        elif 'rand' in label:
            colors.append('red')
        else:
            colors.append('blue')

    fig, ax = plt.subplots(figsize=(10, 2))
    scatter_plots = []  # Store scatter plot handles for legend
    for i, (key, value) in enumerate(zip(keys, values)):
        # Append each scatter plot handle to the list for the legend
        scatter = ax.scatter(value, 0, color=colors[i], s=100)
        scatter_plots.append(scatter)

        # Put labels next to points
        if i % 2 == 0:
            ax.text(value, 0.02, f'{key}', ha='center', color=colors[i])
        else:
            if norma_bool:  # xlabel shift is proportional to x values scale
                ax.text(value + 0.05, -0.02, f'{key}', ha='center', color=colors[i])
            else:
                ax.text(value + max(values) * (0.05), -0.02, f'{key}', ha='center', color=colors[i])

        # Dotted vertical line from point to the horizontal axis
        ax.axvline(x=value, ymin=0, ymax=0.35, color=colors[i], linewidth=1, linestyle='--')

    if norma_bool:
        xticks = np.arange(0, 1.1, 0.1)
        ax.set_xticks(xticks)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.1)
    ax.yaxis.set_visible(False)
    ax.set_title('L2 dist for singular vals of matrix pairs')
    for spine in ax.spines.values():
        if spine.spine_type != 'bottom':
            spine.set_visible(False)
    ax.grid(False)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    ]
    legend_labels = ['LLM', 'SAE', 'Random']
    ax.legend(legend_handles, legend_labels, title='', title_fontsize='13', fontsize='11', loc='upper right', bbox_to_anchor=(1, 1.3))

    plt.show()


# In[46]:


plot1D_L2dist_singvals(matPair_to_l2Dist_norma, norma_bool=True)


# In[47]:


matPair_to_l2Dist_norma_LST = [ (v,k) for k,v in matPair_to_l2Dist_norma.items() ]
matPair_to_l2Dist_norma_LST.sort()
for v, k in matPair_to_l2Dist_norma_LST:
    print(v, k)


# In[48]:


matPair_to_l2Dist_norma_2 = matPair_to_l2Dist_norma.copy()
del matPair_to_l2Dist_norma_2['ts_pythia']
del matPair_to_l2Dist_norma_2['pythia_rand']
del matPair_to_l2Dist_norma_2['GPT2_rand']

plot1D_L2dist_singvals(matPair_to_l2Dist_norma_2)


# ## labels in legend

# (better for unnormalized)

# In[34]:


def plot_L2dist_singvals(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1])
    keys, values = zip(*sorted_data)

    colors = plt.cm.viridis(np.linspace(0, 1, len(keys)))

    fig, ax = plt.subplots(figsize=(10, 4.5))  # Reduced height
    for i, (key, value) in enumerate(zip(keys, values)):
        ax.scatter(value, 0, color=colors[i], label=key, s=100)  # 's' adjusts the size of the point

    # for i, (value, key) in enumerate(zip(values, keys)):
    #     ax.annotate(key, (value, 0), xytext=(0, 10),
    #                 textcoords="offset points", ha='center', va='bottom',
    #                 rotation=45, fontsize=8)

    ax.set_ylim(-0.5, 0.5)  # Tighten y-axis
    ax.yaxis.set_visible(False)
    ax.set_title('L2 dist for singular vals of matrix pairs')
    ax.grid(True, axis='x')
    ax.legend(title="Key", loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    plt.show()


# In[35]:


plot_L2dist_singvals(matPair_to_l2Dist)


# In[36]:


plot_L2dist_singvals(matPair_to_l2Dist_norma)

