import torch
from torch.utils.data import DataLoader, TensorDataset

import gc
import numpy as np

from sparsify.config import SaeConfig
from sparsify.utils import decoder_impl
from sparsify import Sae

def get_sae_actvs(model, sae_name, inputs, layer_id, batch_size=32):
    """
    Process the sae activations in batches to avoid OOM errors.
    
    Args:
        model (torch.nn.Module): The model to process.
        sae_name (str): The SAE model name to load from the hub.
        layer_id (int): The layer index to process.
        batch_size (int): The number of samples per batch.
    
    Returns:
        weight_matrix_np (numpy.ndarray): The decoder weights.
        reshaped_activations_A (torch.Tensor): The pre-activation outputs reshaped.
        orig (torch.Tensor): The original batched pre-activations.
    """    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LLM_outputs = None
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in loader:
        input_ids, attention_mask = batch

        batch_inputs = {'input_ids': input_ids.to(model.device), 'attention_mask': attention_mask.to(model.device)}
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            if LLM_outputs is None:
                LLM_outputs = outputs.hidden_states[layer_id]
            else:
                LLM_outputs = torch.cat((LLM_outputs, outputs.hidden_states[layer_id]), dim= 0)

        del batch_inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    ### Get SAE actvs ###
    hookpoint = "layers." + str(layer_id)
    
    # Load the sae model on the appropriate device.
    sae = Sae.load_from_hub(sae_name, hookpoint=hookpoint, device=device)
    weight_matrix_np = sae.W_dec.cpu().detach().numpy()
    
    # Init vars to store the hidden states for the given layer.
    num_samples = LLM_outputs.size(0)
    pre_act_batches = []
    
    # Process the activations in batches.
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = LLM_outputs[start:end].to(device)
        with torch.inference_mode():
            batch_pre_acts = sae.pre_acts(batch)
        pre_act_batches.append(batch_pre_acts.cpu())
        
        # Free up GPU memory.
        del batch, batch_pre_acts
        torch.cuda.empty_cache()
        gc.collect()
    
    del LLM_outputs
    torch.cuda.empty_cache()
    gc.collect()

    # Concatenate the processed batches.
    orig_actvs = torch.cat(pre_act_batches, dim=0)
    
    first_dim_reshaped = orig_actvs.shape[0] * orig_actvs.shape[1]
    reshaped_activations_A = orig_actvs.reshape(first_dim_reshaped, orig_actvs.shape[-1]).cpu()
    
    return weight_matrix_np, reshaped_activations_A, orig_actvs


####
def get_llm_actvs_batch(model, inputs, layerID):
    accumulated_outputs = None
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in loader:
        input_ids, attention_mask = batch

        batch_inputs = {'input_ids': input_ids.to(model.device), 'attention_mask': attention_mask.to(model.device)}
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            if accumulated_outputs is None:
                accumulated_outputs = outputs.hidden_states[layerID]
            else:
                accumulated_outputs = torch.cat((accumulated_outputs, outputs.hidden_states[layerID]), dim= 0)

        del batch_inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    return accumulated_outputs

##### saes #####

def get_weights_and_acts_byLayer_batched(name, layer_id, layer_outputs, batch_size=32):
    """
    Process the sae activations in batches to avoid OOM errors.
    
    Args:
        name (str): The model name to load from the hub.
        layer_id (int): The layer index to process.
        outputs (Namespace): The model outputs containing hidden_states.
        batch_size (int): The number of samples per batch.
    
    Returns:
        weight_matrix_np (numpy.ndarray): The decoder weights.
        reshaped_activations_A (torch.Tensor): The pre-activation outputs reshaped.
        orig (torch.Tensor): The original batched pre-activations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hookpoint = "layers." + str(layer_id)
    
    # Load the sae model on the appropriate device.
    sae = Sae.load_from_hub(name, hookpoint=hookpoint, device=device)
    weight_matrix_np = sae.W_dec.cpu().detach().numpy()
    
    # Init vars to store the hidden states for the given layer.
    num_samples = layer_outputs.size(0)
    pre_act_batches = []
    
    # Process the activations in batches.
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = layer_outputs[start:end].to(device)
        with torch.inference_mode():
            batch_pre_acts = sae.pre_acts(batch)
        pre_act_batches.append(batch_pre_acts.cpu())
        
        # Free up GPU memory.
        del batch, batch_pre_acts
        torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate the processed batches.
    orig = torch.cat(pre_act_batches, dim=0)
    
    first_dim_reshaped = orig.shape[0] * orig.shape[1]
    reshaped_activations_A = orig.reshape(first_dim_reshaped, orig.shape[-1]).cpu()
    
    return weight_matrix_np, reshaped_activations_A, orig


## old sae fns:
def get_weights_and_acts(name, layer_id, outputs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hookpoint = "layers." + str(layer_id)

    sae = Sae.load_from_hub(name, hookpoint=hookpoint, device=device)

    weight_matrix_np = sae.W_dec.cpu().detach().numpy()

    with torch.inference_mode():
        # reshaped_activations_A = sae.pre_acts(outputs.hidden_states[layer_id].to("cuda"))
        orig = sae.pre_acts(outputs.to("cuda"))

    first_dim_reshaped = orig.shape[0] * orig.shape[1]
    reshaped_activations_A = orig.reshape(first_dim_reshaped, orig.shape[-1]).cpu()

    return weight_matrix_np, reshaped_activations_A, orig
    # return weight_matrix_np, reshaped_activations_A

def get_weights_and_acts_byLayer(name, layer_id, outputs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hookpoint = "layers." + str(layer_id)

    sae = Sae.load_from_hub(name, hookpoint=hookpoint, device=device)

    weight_matrix_np = sae.W_dec.cpu().detach().numpy()

    with torch.inference_mode():
        orig = sae.pre_acts(outputs.hidden_states[layer_id].to("cuda"))

    first_dim_reshaped = orig.shape[0] * orig.shape[1]
    reshaped_activations_A = orig.reshape(first_dim_reshaped, orig.shape[-1]).cpu()

    return weight_matrix_np, reshaped_activations_A, orig
    # return weight_matrix_np, reshaped_activations_A

def count_zero_columns(tensor):
    # Check if all elements in each column are zero
    zero_columns = np.all(tensor == 0, axis=0)
    # Count True values in the zero_columns array
    zero_cols_indices = np.where(zero_columns)[0]
    return np.sum(zero_columns), zero_cols_indices