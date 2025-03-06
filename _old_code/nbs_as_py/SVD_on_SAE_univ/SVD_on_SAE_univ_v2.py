#!/usr/bin/env python
# coding: utf-8

# List of singular value vectors generated:
# 
# 
# *   List item
# *   List item
# 

# # setup

# In[1]:


get_ipython().system('git clone https://github.com/wlg1/univ_feat_geom.git')


# In[2]:


import pickle
import numpy as np
import matplotlib.pyplot as plt


# # load weight mats

# Temporarily loading data from repo for convenience; larger files in the future will used a better storage system and not a repo

# In[3]:


file_path = '/content/univ_feat_geom/data/ts-1L-21M_Wdec.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_1 = pickle.load(f)
weight_matrix_1 = weight_matrix_1.detach().numpy()
print(weight_matrix_1.shape)


# In[4]:


file_path = '/content/univ_feat_geom/data/ts-2L-33M_Wdec.pkl'
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)
weight_matrix_2 = weight_matrix_2.detach().numpy()
print(weight_matrix_2.shape)


# # Comparison Functions

# In[163]:


def compare_singular_values(S1, S2):
    l2_distance = np.linalg.norm(S1 - S2)

    # Normalization by norm
    frobenius_norm1 = np.linalg.norm(S1)
    frobenius_norm2 = np.linalg.norm(S2)
    normalized_l2_distance = l2_distance / (frobenius_norm1 + frobenius_norm2)

    # Normalization by avg sum
    sum_singular_values1 = np.sum(S1)
    sum_singular_values2 = np.sum(S2)
    average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2
    percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

    print("(Singular values of the first matrix):", S1)
    print("(Singular values of the second matrix):", S2)
    print("L2 distance between singular values:", l2_distance)
    print("Normalized L2 distance between singular values:", normalized_l2_distance)
    print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)

    print('\n')

    # Normalize singular values
    S1_normalized = S1 / np.linalg.norm(S1)
    S2_normalized = S2 / np.linalg.norm(S2)

    l2_distance_normalized = np.linalg.norm(S1_normalized - S2_normalized)

    # Normalize the L2 distance
    frobenius_norm1 = np.linalg.norm(S1_normalized)
    frobenius_norm2 = np.linalg.norm(S2_normalized)
    normalized_l2_distance = l2_distance_normalized / (frobenius_norm1 + frobenius_norm2)

    # Normalization by avg sum
    sum_singular_values1 = np.sum(S1_normalized)
    sum_singular_values2 = np.sum(S2_normalized)
    average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2
    percentage_l2_distance = (l2_distance_normalized / average_sum_singular_values) * 100

    print("Normalized Singular Values (Matrix 1):", S1_normalized)
    print("Normalized Singular Values (Matrix 2):", S2_normalized)
    print("L2 distance between normalized singular values:", l2_distance_normalized)
    print("Normalized L2 distance between normalized singular values:", normalized_l2_distance)
    print("Percentage L2 distance relative to the sum of normalized singular values:", percentage_l2_distance)

    return l2_distance, l2_distance_normalized


# In[6]:


import numpy as np

def generate_rand_svd(n, m):
    # Generate two random matrices of size n x m
    matrix1 = np.random.rand(n, m)
    matrix2 = np.random.rand(n, m)

    # Perform SVD on both matrices
    U1, S1_rand, Vt1 = np.linalg.svd(matrix1)
    U2, S2_rand, Vt2 = np.linalg.svd(matrix2)

    return(S1_rand, S2_rand)


# In[165]:


matPair_to_l2Dist = {}
matPair_to_l2Dist_norma = {}


# # compare SAEs on 2 LLMs

# In[7]:


U1, S1_SAE, Vt1 = np.linalg.svd(weight_matrix_1)
U2, S2_SAE, Vt2 = np.linalg.svd(weight_matrix_2)


# In[166]:


matPair_to_l2Dist['SAE_SAE'], matPair_to_l2Dist_norma['SAE_SAE'] = compare_singular_values(S1_SAE, S2_SAE)


# # compare to random weights

# ## compare two rand

# In[16]:


n, m = weight_matrix_1.shape[0], weight_matrix_1.shape[1]
S1_rand, S2_rand = generate_rand_svd(n, m)


# In[167]:


matPair_to_l2Dist['rand_rand'], matPair_to_l2Dist_norma['rand_rand'] = compare_singular_values(S1_rand, S2_rand)


# While we generally do not expect two random matrices to have highly similar singular values, the large size of the matrices (16384x1024) causes the singular values to stabilize and appear similar due to the reasons outlined above.

# ## compare rand to SAE

# In[168]:


matPair_to_l2Dist['SAE_rand'], matPair_to_l2Dist_norma['SAE_rand'] = compare_singular_values(S1_SAE, S2_rand)


# # compare weight matrices of orig LLMs

# ## load and get svd

# In[20]:


from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1Layer-21M")
mlp_weights = model.transformer.h[0].mlp.c_fc.weight
mlp_weights.shape


# In[21]:


model_2 = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-2Layers-33M")
mlp_weights_2 = model_2.transformer.h[0].mlp.c_fc.weight
mlp_weights_2.shape


# In[22]:


U1, S1_LLM, Vt1 = np.linalg.svd(mlp_weights.detach().numpy())
U2, S2_LLM, Vt2 = np.linalg.svd(mlp_weights_2.detach().numpy())


# ## compare LLMs (MLP0)

# In[169]:


matPair_to_l2Dist['LLM_LLM_samelayer'], matPair_to_l2Dist_norma['LLM_LLM_samelayer'] = compare_singular_values(S1_LLM, S2_LLM)


# ## LLM to rand

# In[170]:


matPair_to_l2Dist['LLM_rand'], matPair_to_l2Dist_norma['LLM_rand'] = compare_singular_values(S1_LLM, S1_rand)


# ## LLM_1 (MLP0) to LLM_2 (MLP1)

# In[25]:


mlp_weights_2b = model_2.transformer.h[1].mlp.c_fc.weight  # Example for GPT-like models
mlp_weights_2b.shape


# In[26]:


U2, S2_LLM_MLP1, Vt2 = np.linalg.svd(mlp_weights_2b.detach().numpy())


# In[171]:


matPair_to_l2Dist['LLM_LLM_difflayer'], matPair_to_l2Dist_norma['LLM_LLM_difflayer'] = compare_singular_values(S1_LLM, S2_LLM_MLP1)


# ## compare saes to LLMs

# In[172]:


matPair_to_l2Dist['LLM_SAE_sameMod'], matPair_to_l2Dist_norma['LLM_SAE_sameMod'] = compare_singular_values(S1_LLM, S1_SAE)


# In[173]:


matPair_to_l2Dist['LLM_SAE_diffMod'], matPair_to_l2Dist_norma['LLM_SAE_diffMod'] = compare_singular_values(S1_LLM, S2_SAE)


# # compare to gpt2 med

# ## load and get svd

# In[30]:


gpt2_med = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")


# In[31]:


mlp0_weights_gpt2_med = gpt2_med.transformer.h[0].mlp.c_fc.weight
mlp0_weights_gpt2_med.shape


# In[32]:


U2, S_GPT2_0, Vt2 = np.linalg.svd(mlp0_weights_gpt2_med.detach().numpy())


# In[33]:


S_GPT2_0.shape


# ## GPT2 L0

# In[174]:


matPair_to_l2Dist['LLM_GPT2_sameLayer'], matPair_to_l2Dist_norma['LLM_GPT2_sameLayer'] = compare_singular_values(S2_LLM, S_GPT2_0)


# In[175]:


matPair_to_l2Dist['LLM_GPT2_diffLayer'], matPair_to_l2Dist_norma['LLM_GPT2_diffLayer'] = compare_singular_values(S2_LLM_MLP1, S_GPT2_0)


# ## to mid layer of GPT2_med

# In[36]:


mlp7_weights_gpt2_med = gpt2_med.transformer.h[7].mlp.c_fc.weight
mlp7_weights_gpt2_med.shape


# In[37]:


U2, S_GPT2_7, Vt2 = np.linalg.svd(mlp7_weights_gpt2_med.detach().numpy())


# In[38]:


compare_singular_values(S1_LLM, S_GPT2_7)


# In[39]:


compare_singular_values(S_GPT2_0, S_GPT2_7)


# # compare to pythia

# ## load and get svd

# In[40]:


from transformers import GPTNeoXForCausalLM, AutoTokenizer

model_pythia410 = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m",
)


# In[41]:


model_pythia410.gpt_neox.layers[0].mlp


# In[42]:


model_pythia410_weights_mlp0 = model_pythia410.gpt_neox.layers[0].mlp.dense_4h_to_h.weight
model_pythia410_weights_mlp0.shape


# In[43]:


U1, S_pythia410_mlp0, Vt1 = np.linalg.svd(model_pythia410_weights_mlp0.detach().numpy())


# In[176]:


matPair_to_l2Dist_pythia = matPair_to_l2Dist.copy()
matPair_to_l2Dist_norma_pythia = matPair_to_l2Dist_norma.copy()


# ## L0

# In[177]:


matPair_to_l2Dist_pythia['pythia_rand'], matPair_to_l2Dist_norma_pythia['pythia_rand'] = compare_singular_values(S_pythia410_mlp0, S2_rand)


# In[178]:


matPair_to_l2Dist_pythia['pythia_LLM'], matPair_to_l2Dist_norma_pythia['pythia_LLM'] = compare_singular_values(S_pythia410_mlp0, S1_LLM)


# In[179]:


matPair_to_l2Dist_pythia['pythia_GPT'], matPair_to_l2Dist_norma_pythia['pythia_GPT'] = compare_singular_values(S_pythia410_mlp0, S_GPT2_0)


# In[180]:


matPair_to_l2Dist_pythia['pythia_GPT.L7'], matPair_to_l2Dist_norma_pythia['pythia_GPT.L7'] = compare_singular_values(S_pythia410_mlp0, S_GPT2_7)


# ## L17

# In[49]:


model_pythia410_weights_mlp17 = model_pythia410.gpt_neox.layers[17].mlp.dense_4h_to_h.weight  # Example for GPT-like models
model_pythia410_weights_mlp17.shape


# In[50]:


U1, S_pythia410_mlp17, Vt1 = np.linalg.svd(model_pythia410_weights_mlp17.detach().numpy())


# In[51]:


compare_singular_values(S_pythia410_mlp0, S_pythia410_mlp17)


# In[52]:


compare_singular_values(S_pythia410_mlp17, S1_LLM)


# In[53]:


compare_singular_values(S_pythia410_mlp17, S_GPT2_0)


# In[54]:


compare_singular_values(S_pythia410_mlp17, S1_SAE)


# # summarize results

# In[183]:


def plot_dict_on_number_line(data):
    # Sort the dictionary by its values
    sorted_data = sorted(data.items(), key=lambda x: x[1])

    # Extract keys and values for plotting
    keys = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    # Assign a unique color to each key
    colors = plt.cm.viridis(np.linspace(0, 1, len(keys)))

    # Create a figure and a plot
    fig, ax = plt.subplots()

    # Plot each point on the number line with a unique color
    for i, (key, value) in enumerate(zip(keys, values)):
        ax.scatter(value, 0, color=colors[i], label=key, s=100)  # 's' adjusts the size of the point

    # Hide y-axis as it's not needed
    ax.yaxis.set_visible(False)

    # Set title and grid
    ax.set_title('Values on a Number Line')
    ax.grid(True)

    # Create a legend
    ax.legend(title="Key", loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Show the plot
    plt.show()


# In[77]:


plot_dict_on_number_line(matPair_to_l2Dist)


# In[184]:


plot_dict_on_number_line(matPair_to_l2Dist_norma)


# In[79]:


matPair_to_l2Dist_2 = matPair_to_l2Dist.copy()
del matPair_to_l2Dist_2['rand_rand']
del matPair_to_l2Dist_2['SAE_rand']
del matPair_to_l2Dist_2['LLM_rand']

plot_dict_on_number_line(matPair_to_l2Dist_2)


# In[80]:


matPair_to_l2Dist_2 = matPair_to_l2Dist.copy()
del matPair_to_l2Dist_2['rand_rand']
del matPair_to_l2Dist_2['SAE_rand']
del matPair_to_l2Dist_2['LLM_rand']
del matPair_to_l2Dist_2['LLM_SAE_sameMod']
del matPair_to_l2Dist_2['LLM_SAE_diffMod']

plot_dict_on_number_line(matPair_to_l2Dist_2)


# In[192]:


matPair_to_l2Dist_2 = matPair_to_l2Dist.copy()
del matPair_to_l2Dist_2['rand_rand']
# del matPair_to_l2Dist_2['SAE_rand']
# del matPair_to_l2Dist_2['LLM_rand']
del matPair_to_l2Dist_2['LLM_LLM_difflayer']
del matPair_to_l2Dist_2['LLM_GPT2_diffLayer']
del matPair_to_l2Dist_2['LLM_SAE_sameMod']
del matPair_to_l2Dist_2['LLM_SAE_diffMod']

plot_dict_on_number_line(matPair_to_l2Dist_2)


# In[191]:


matPair_to_l2Dist_norma_2 = matPair_to_l2Dist_norma.copy()
del matPair_to_l2Dist_norma_2['rand_rand']
# del matPair_to_l2Dist_norma_2['SAE_rand']
# del matPair_to_l2Dist_norma_2['LLM_rand']
del matPair_to_l2Dist_norma_2['LLM_LLM_difflayer']
del matPair_to_l2Dist_norma_2['LLM_GPT2_diffLayer']
del matPair_to_l2Dist_norma_2['LLM_SAE_sameMod']
del matPair_to_l2Dist_norma_2['LLM_SAE_diffMod']

plot_dict_on_number_line(matPair_to_l2Dist_norma_2)


# In[193]:


matPair_to_l2Dist_norma_pythia_2 = matPair_to_l2Dist_norma_pythia.copy()
del matPair_to_l2Dist_norma_pythia_2['rand_rand']
# del matPair_to_l2Dist_norma_pythia_2['SAE_rand']
# del matPair_to_l2Dist_norma_pythia_2['LLM_rand']
del matPair_to_l2Dist_norma_pythia_2['LLM_LLM_difflayer']
del matPair_to_l2Dist_norma_pythia_2['LLM_GPT2_diffLayer']
del matPair_to_l2Dist_norma_pythia_2['LLM_SAE_sameMod']
del matPair_to_l2Dist_norma_pythia_2['LLM_SAE_diffMod']

plot_dict_on_number_line(matPair_to_l2Dist_norma_pythia_2)


# ## single num line plots

# In[196]:


def plot1D_dict_on_number_line(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1])
    keys = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]
    colors = plt.cm.viridis(np.linspace(0, 1, len(keys))) # Assign a unique color to each key

    fig, ax = plt.subplots(figsize=(10, 2))
    for i, (key, value) in enumerate(zip(keys, values)):
        ax.scatter(value, 0, color=colors[i], s=100)  # 's' adjusts the size of the point
        # Labels next to pts
        if i % 2 == 0:
            ax.text(value, 0.02, f'{key}', ha='center', color=colors[i])
        else:
            ax.text(value + 1, 0.05, f'{key}', ha='center', color=colors[i])
        # Draw vertical line from point to the horizontal axis
        ax.axvline(x=value, ymin=0, ymax=0.4, color=colors[i], linewidth=1, linestyle='--')

    ax.set_ylim(-0.05, 0.1)
    ax.yaxis.set_visible(False)
    ax.set_title('L2 dist for singular vals of matrix pairs')
    for spine in ax.spines.values():
        if spine.spine_type != 'bottom':  # Keep only the bottom spine
            spine.set_visible(False)
    ax.grid(False)

    fig.tight_layout(pad=2) # make less big

    plt.show()


# In[197]:


plot1D_dict_on_number_line(matPair_to_l2Dist_2)


# In[198]:


matPair_to_l2Dist_norma_2 = matPair_to_l2Dist_norma.copy()
del matPair_to_l2Dist_norma_2['rand_rand']
del matPair_to_l2Dist_norma_2['SAE_rand']
del matPair_to_l2Dist_norma_2['LLM_rand']
del matPair_to_l2Dist_norma_2['LLM_SAE_sameMod']
del matPair_to_l2Dist_norma_2['LLM_SAE_diffMod']

plot1D_dict_on_number_line(matPair_to_l2Dist_norma_2)


# In[162]:


plot1D_dict_on_number_line(matPair_to_l2Dist)


# In[199]:


matPair_to_l2Dist_norma_2 = matPair_to_l2Dist_norma.copy()
del matPair_to_l2Dist_norma_2['rand_rand']
# del matPair_to_l2Dist_norma_2['SAE_rand']
# del matPair_to_l2Dist_norma_2['LLM_rand']
del matPair_to_l2Dist_norma_2['LLM_LLM_difflayer']
del matPair_to_l2Dist_norma_2['LLM_GPT2_diffLayer']
del matPair_to_l2Dist_norma_2['LLM_SAE_sameMod']
del matPair_to_l2Dist_norma_2['LLM_SAE_diffMod']

plot1D_dict_on_number_line(matPair_to_l2Dist_norma_2)

