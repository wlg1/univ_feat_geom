#!/usr/bin/env python
# coding: utf-8

# # setup

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# !pip install umap-learn


# In[ ]:


import pickle
import numpy as np

# import umap
import matplotlib.pyplot as plt


# # load weight mats

# In[ ]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-1L-21M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_np = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_np.shape)


# In[ ]:


weight_matrix_np = weight_matrix_np.detach().numpy()


# In[ ]:


# Define the path to your pickle file in Google Drive
file_path = '/content/drive/MyDrive/ts-2L-33M_Wdec.pkl'  # Change the path if necessary

# Load the weight matrix from the pickle file
with open(file_path, 'rb') as f:
    weight_matrix_2 = pickle.load(f)

# Optionally, check the shape of the loaded weight matrix
print(weight_matrix_2.shape)


# In[ ]:


weight_matrix_2 = weight_matrix_2.detach().numpy()


# # SVD

# In[ ]:


U1, S1, Vt1 = np.linalg.svd(weight_matrix_np)
U2, S2, Vt2 = np.linalg.svd(weight_matrix_2)


# In[ ]:


l2_distance = np.linalg.norm(S1 - S2)

# Step 4: Normalize the L2 distance
frobenius_norm1 = np.linalg.norm(S1)
frobenius_norm2 = np.linalg.norm(S2)
normalized_l2_distance = l2_distance / (frobenius_norm1 + frobenius_norm2)

print("L2 distance between singular values:", l2_distance)
print("Normalized L2 distance:", normalized_l2_distance)


# # SVD on transpose

# In[ ]:


weight_matrix_np.T.shape


# In[ ]:


U1, S1_T, Vt1 = np.linalg.svd(weight_matrix_np.T)
U2, S2_T, Vt2 = np.linalg.svd(weight_matrix_2.T)


# In[ ]:


S1_T.shape


# In[ ]:


S1_T


# In[ ]:


l2_distance = np.linalg.norm(S1_T - S2_T)

# Step 4: Normalize the L2 distance
frobenius_norm1 = np.linalg.norm(S1_T)
frobenius_norm2 = np.linalg.norm(S2_T)
normalized_l2_distance = l2_distance / (frobenius_norm1 + frobenius_norm2)

print("L2 distance between singular values:", l2_distance)
print("Normalized L2 distance:", normalized_l2_distance)


# In[ ]:


# Step 1: Compute the sum of singular values for both matrices
sum_singular_values1 = np.sum(S1_T)
sum_singular_values2 = np.sum(S2_T)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Step 2: Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


l2_distance / average_sum_singular_values


# That's 1%, which means highly similar.

# # fns

# In[ ]:


def compare_singular_values(S1, S2):
    # Compute the L2 distance between the singular values
    l2_distance = np.linalg.norm(S1 - S2)

    # Compute the sum of singular values for normalization
    sum_singular_values1 = np.sum(S1)
    sum_singular_values2 = np.sum(S2)
    average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

    # Calculate the percentage L2 distance
    percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

    # Output results
    print("(Singular values of the first matrix):", S1)
    print("(Singular values of the second matrix):", S2)
    print("L2 distance between singular values:", l2_distance)
    print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


def normalize_singular_values(singular_values):
    """Normalize singular values by their sum."""
    # return singular_values / np.sum(singular_values)
    return singular_values / np.linalg.norm(singular_values)

def compare_normalized_singular_values(S1, S2):
    """Compare normalized singular values using L2 distance."""
    # Normalize singular values
    S1_normalized = normalize_singular_values(S1)
    S2_normalized = normalize_singular_values(S2)

    # Calculate L2 distance between normalized singular values
    l2_distance_normalized = np.linalg.norm(S1_normalized - S2_normalized)

    # Normalize the L2 distance
    frobenius_norm1 = np.linalg.norm(S1_normalized)
    frobenius_norm2 = np.linalg.norm(S2_normalized)
    normalized_l2_distance = l2_distance_normalized / (frobenius_norm1 + frobenius_norm2)

    print("Normalized Singular Values (Matrix 1):", S1_normalized)
    print("Normalized Singular Values (Matrix 2):", S2_normalized)
    print("L2 distance between normalized singular values:", l2_distance_normalized)
    print("Normalized L2 distance:", normalized_l2_distance)


# In[ ]:


compare_normalized_singular_values(S1_T, S2_T)


# # compare two rand

# In[ ]:


import numpy as np

def compare_svd(n, m):
    # Generate two random matrices of size n x m
    matrix1 = np.random.rand(n, m)
    matrix2 = np.random.rand(n, m)

    # Perform SVD on both matrices
    U1, S1_rand, Vt1 = np.linalg.svd(matrix1)
    U2, S2_rand, Vt2 = np.linalg.svd(matrix2)

    # Compute the L2 distance between the singular values
    l2_distance = np.linalg.norm(S1_rand - S2_rand)

    # Compute the sum of singular values for normalization
    sum_singular_values1 = np.sum(S1_rand)
    sum_singular_values2 = np.sum(S2_rand)
    average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

    # Calculate the percentage L2 distance
    percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

    # Output results
    print("S1_rand (Singular values of the first random matrix):", S1_rand)
    print("S2_rand (Singular values of the second random matrix):", S2_rand)
    print("L2 distance between singular values:", l2_distance)
    print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)

# Example usage:
n = 5  # specify the number of rows
m = 4  # specify the number of columns
compare_svd(weight_matrix_2.shape[0], weight_matrix_2.shape[1])


# In[ ]:


n, m = weight_matrix_2.shape[0], weight_matrix_2.shape[1]

# Generate two random matrices of size n x m
matrix1 = np.random.rand(n, m)
matrix2 = np.random.rand(n, m)

# # Perform SVD on both matrices
U1, S1_rand, Vt1 = np.linalg.svd(matrix1)
U2, S2_rand, Vt2 = np.linalg.svd(matrix2)


# In[ ]:


compare_normalized_singular_values(S1_rand, S2_rand)


# While we generally do not expect two random matrices to have highly similar singular values, the large size of the matrices (16384x1024) causes the singular values to stabilize and appear similar due to the reasons outlined above.

# # compare rand to SAE

# In[ ]:


compare_singular_values(S1_T, S2_rand)


# In[ ]:


compare_singular_values(S2_T, S2_rand)


# In[ ]:


# from scipy import spatial
# 1 - spatial.distance.cosine(S2_T, S2_rand)


# In[ ]:


compare_normalized_singular_values(S2_T, S2_rand)


# In[ ]:


compare_normalized_singular_values(S1_T, S1_rand)


# In[ ]:


compare_normalized_singular_values(S2_T, S1_rand)


# # compare weight matrices of orig LLMs

# In[ ]:


get_ipython().run_cell_magic('capture', '', '%pip install transformer-lens\n')


# In[ ]:


from transformer_lens import HookedTransformer


# In[ ]:


from transformers import AutoModelForCausalLM

# Load the TinyStories-1Layer-21M model
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1Layer-21M")


# In[ ]:


# Accessing the first (and only) transformer's MLP layer weights
# Depending on the architecture, the MLP might be accessible via different paths
mlp_weights = model.transformer.h[0].mlp.c_fc.weight  # Example for GPT-like models
mlp_weights.shape


# In[ ]:


model_2 = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-2Layers-33M")


# In[ ]:


# Accessing the first (and only) transformer's MLP layer weights
# Depending on the architecture, the MLP might be accessible via different paths
mlp_weights_2 = model_2.transformer.h[0].mlp.c_fc.weight  # Example for GPT-like models
mlp_weights_2.shape


# In[ ]:


U1, S1_LLM, Vt1 = np.linalg.svd(mlp_weights.detach().numpy())
U2, S2_LLM, Vt2 = np.linalg.svd(mlp_weights_2.detach().numpy())


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S1_LLM - S2_LLM)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S1_LLM)
sum_singular_values2 = np.sum(S2_LLM)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S1_LLM)
print("(Singular values of the second matrix):", S2_LLM)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


# from scipy import spatial
# 1 - spatial.distance.cosine(S1_LLM, S2_LLM)


# In[ ]:


compare_normalized_singular_values(S1_LLM, S2_LLM)


# # LLM to rand

# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S1_LLM - S1_rand)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S1_LLM)
sum_singular_values2 = np.sum(S1_rand)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S1_LLM)
print("(Singular values of the second matrix):", S1_rand)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


compare_normalized_singular_values(S1_LLM, S1_rand)


# In[ ]:


compare_normalized_singular_values(S2_LLM, S2_rand)


# ## to MLP1

# In[ ]:


mlp_weights_2b = model_2.transformer.h[1].mlp.c_fc.weight  # Example for GPT-like models
mlp_weights_2b.shape


# In[ ]:


U2, S2_LLM_MLP1, Vt2 = np.linalg.svd(mlp_weights_2b.detach().numpy())


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S1_LLM - S2_LLM_MLP1)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S1_LLM)
sum_singular_values2 = np.sum(S2_LLM_MLP1)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S1_LLM)
print("(Singular values of the second matrix):", S2_LLM_MLP1)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


S1_LLM


# In[ ]:


from scipy import spatial
1 - spatial.distance.cosine(S1_LLM, S2_LLM_MLP1)


# In[ ]:


compare_normalized_singular_values(S1_LLM, S2_LLM_MLP1)


# # compare saes to LLMs

# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S1_LLM - S1_T)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S1_LLM)
sum_singular_values2 = np.sum(S1_T)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S1_LLM)
print("(Singular values of the second matrix):", S1_T)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


compare_normalized_singular_values(S1_LLM, S1_T)


# In[ ]:


compare_normalized_singular_values(S2_LLM, S2_T)


# # compare to ts LLMs to gpt2 med

# In[ ]:


gpt2_med = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")


# In[ ]:


mlp0_weights_gpt2_med = gpt2_med.transformer.h[0].mlp.c_fc.weight
mlp0_weights_gpt2_med.shape


# In[ ]:


U2, S2_GPT2_0, Vt2 = np.linalg.svd(mlp0_weights_gpt2_med.detach().numpy())


# In[ ]:


S2_GPT2_0.shape


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S1_LLM - S2_GPT2_0)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S1_LLM)
sum_singular_values2 = np.sum(S2_GPT2_0)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S1_LLM)
print("(Singular values of the second matrix):", S2_GPT2_0)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S2_LLM - S2_GPT2_0)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S2_LLM)
sum_singular_values2 = np.sum(S2_GPT2_0)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S2_LLM)
print("(Singular values of the second matrix):", S2_GPT2_0)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S2_LLM_MLP1 - S2_GPT2_0)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S2_LLM_MLP1)
sum_singular_values2 = np.sum(S2_GPT2_0)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S2_LLM_MLP1)
print("(Singular values of the second matrix):", S2_GPT2_0)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


compare_normalized_singular_values(S2_LLM_MLP1, S2_GPT2_0)


# In[ ]:


compare_normalized_singular_values(S2_LLM, S2_GPT2_0)


# In[ ]:


compare_normalized_singular_values(S1_LLM, S2_GPT2_0)


# # to mid layer of GPT2_med

# In[ ]:


mlp7_weights_gpt2_med = gpt2_med.transformer.h[7].mlp.c_fc.weight
mlp7_weights_gpt2_med.shape


# In[ ]:


U2, S2_GPT2_7, Vt2 = np.linalg.svd(mlp7_weights_gpt2_med.detach().numpy())


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S2_LLM_MLP1 - S2_GPT2_7)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S2_LLM_MLP1)
sum_singular_values2 = np.sum(S2_GPT2_7)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S2_LLM_MLP1)
print("(Singular values of the second matrix):", S2_GPT2_7)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


compare_normalized_singular_values(S2_LLM_MLP1, S2_GPT2_7)


# In[ ]:


# Compute the L2 distance between the singular values
l2_distance = np.linalg.norm(S2_GPT2_0 - S2_GPT2_7)

# Compute the sum of singular values for normalization
sum_singular_values1 = np.sum(S2_GPT2_0)
sum_singular_values2 = np.sum(S2_GPT2_7)
average_sum_singular_values = (sum_singular_values1 + sum_singular_values2) / 2

# Calculate the percentage L2 distance
percentage_l2_distance = (l2_distance / average_sum_singular_values) * 100

# Output results
print("(Singular values of the first matrix):", S2_GPT2_0)
print("(Singular values of the second matrix):", S2_GPT2_7)
print("L2 distance between singular values:", l2_distance)
print("Percentage L2 distance relative to the sum of singular values:", percentage_l2_distance)


# In[ ]:


compare_normalized_singular_values(S2_GPT2_0, S2_GPT2_7)


# # pythia

# In[ ]:


from transformers import GPTNeoXForCausalLM, AutoTokenizer

model_pythia410 = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m",
)


# In[ ]:


model_pythia410.gpt_neox.layers[0].mlp


# In[ ]:


# Accessing the first (and only) transformer's MLP layer weights
# Depending on the architecture, the MLP might be accessible via different paths
model_pythia410_weights_mlp0 = model_pythia410.gpt_neox.layers[0].mlp.dense_4h_to_h.weight  # Example for GPT-like models
model_pythia410_weights_mlp0.shape


# In[ ]:


U1, S_pythia410_mlp0, Vt1 = np.linalg.svd(model_pythia410_weights_mlp0.detach().numpy())


# In[ ]:


compare_singular_values(S_pythia410_mlp0, S2_rand)


# In[ ]:


compare_singular_values(S_pythia410_mlp0, S1_LLM)


# In[ ]:


compare_singular_values(S_pythia410_mlp0, S1_T)


# In[ ]:


compare_singular_values(S_pythia410_mlp0, S2_GPT2_0)


# In[ ]:


compare_singular_values(S_pythia410_mlp0, S2_GPT2_7)


# In[ ]:


model_pythia410_weights_mlp17 = model_pythia410.gpt_neox.layers[17].mlp.dense_4h_to_h.weight  # Example for GPT-like models
model_pythia410_weights_mlp17.shape


# In[ ]:


U1, S_pythia410_mlp17, Vt1 = np.linalg.svd(model_pythia410_weights_mlp17.detach().numpy())


# In[ ]:


compare_singular_values(S_pythia410_mlp0, S_pythia410_mlp17)


# In[ ]:


compare_singular_values(S_pythia410_mlp17, S1_LLM)


# In[ ]:


compare_singular_values(S_pythia410_mlp17, S1_T)


# In[ ]:


compare_singular_values(S_pythia410_mlp17, S2_GPT2_0)


# In[ ]:


compare_normalized_singular_values(S_pythia410_mlp17, S2_GPT2_0)


# In[ ]:


compare_normalized_singular_values(S_pythia410_mlp17, S2_rand)


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


# In[ ]:


first_dim_reshaped = feature_acts_model_A.shape[0] * feature_acts_model_A.shape[1]
reshaped_activations_A = feature_acts_model_A.reshape(first_dim_reshaped, feature_acts_model_A.shape[-1]).cpu()
reshaped_activations_B = feature_acts_model_B.reshape(first_dim_reshaped, feature_acts_model_B.shape[-1]).cpu()


# In[ ]:


reshaped_activations_B.shape

