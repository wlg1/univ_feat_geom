import torch
from torch.utils.data import DataLoader, TensorDataset
import gc
import numpy as np

from sparsify.config import SaeConfig
from sparsify.utils import decoder_impl
from sparsify import Sae

# Also support alternate SAE loading via sae_lens.
from sae_lens import SAE

def get_sae_actvs(model, model_name, sae_name, inputs, layer_id, batch_size=32, 
                  sae_lib='eleuther', compare_MLPs_bool=False):
    """
    Process the SAE activations in batches to avoid OOM errors.
    
    Args:
        model (torch.nn.Module): The model to process.
        sae_name (str): The SAE model name to load from the hub.
        inputs (dict): Tokenized inputs.
        layer_id (int): The layer index to process.
        batch_size (int): The number of samples per batch.
        sae_lib (str): Which library to use ('eleuther' or 'sae_lens').
        custom_hookpoint (str, optional): If provided, overrides the default hookpoint.
    
    Returns:
        weight_matrix_np (numpy.ndarray): The decoder weights.
        reshaped_activations (torch.Tensor): The pre-activation outputs reshaped.
        orig_actvs (torch.Tensor): The original batched pre-activations.
    """    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ### Get LLM Residual Stream activations ###
    if compare_MLPs_bool:
        # _, _, LLM_actvs = get_LLM_MLP_actvs(model, model_name, layer_id, inputs)
        _, _, LLM_actvs = get_LLM_MLP_actvs(model, model_name, layer_id, inputs, batch_size)
    else:
        LLM_actvs = None
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch_data in loader:
            input_ids, attention_mask = batch_data
            batch_inputs = {
                'input_ids': input_ids.to(model.device), 
                'attention_mask': attention_mask.to(model.device)
            }
            with torch.no_grad():
                outputs = model(**batch_inputs, output_hidden_states=True)
                if LLM_actvs is None:
                    LLM_actvs = outputs.hidden_states[layer_id]
                else:
                    LLM_actvs = torch.cat((LLM_actvs, outputs.hidden_states[layer_id]), dim=0)

            del batch_inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

    # Load the SAE from the hub.
    if sae_lib == 'eleuther':
        if compare_MLPs_bool:
            filename_suffix = '.mlp'
        else:
            filename_suffix = ''
        if 'wlog' in sae_name:
            hookpoint = f"gpt_neox.layers.{layer_id}" + filename_suffix
        elif 'EleutherAI' in sae_name:
            hookpoint = f"layers.{layer_id}" + filename_suffix
        sae = Sae.load_from_hub(sae_name, hookpoint=hookpoint, device=device)
    elif sae_lib == 'sae_lens':
        """
        release = "gemma-2b-res-jb",
        sae_id = f"blocks.{layer_id}.hook_resid_post",

        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = f"layer_{layer_id}/width_16k/canonical",
        """
        if 'scope' in sae_name: # gemma-2
            sae_id = f"layer_{layer_id}/width_16k/canonical" # gemma-2
        else:
            sae_id = f"blocks.{layer_id}.hook_resid_post" # gemma-1

        sae, _, _ = SAE.from_pretrained(
            release = sae_name, # "gemma-2b-res-jb"
            sae_id = sae_id,
        )
        sae = sae.to('cuda')
        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    weight_matrix_np = sae.W_dec.cpu().detach().numpy()
    
    pre_act_batches = []
    num_samples = LLM_actvs.size(0)
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        LLM_actvs_batch = LLM_actvs[start:end].to(device)
        with torch.inference_mode():
            if sae_lib == 'eleuther':
                batch_pre_acts = sae.pre_acts(LLM_actvs_batch)
            elif sae_lib == 'sae_lens':                
                batch_pre_acts = sae.encode(LLM_actvs_batch)
        pre_act_batches.append(batch_pre_acts.cpu())
        
        del LLM_actvs_batch, batch_pre_acts
        torch.cuda.empty_cache()
        gc.collect()
    
    del LLM_actvs
    torch.cuda.empty_cache()
    gc.collect()

    orig_actvs = torch.cat(pre_act_batches, dim=0)
    first_dim_reshaped  = orig_actvs.shape[0] * orig_actvs.shape[1]
    reshaped_activations = orig_actvs.reshape(first_dim_reshaped , orig_actvs.shape[-1]).cpu()
    
    return weight_matrix_np, reshaped_activations, orig_actvs

def count_zero_columns(tensor):
    zero_columns = np.all(tensor == 0, axis=0)
    zero_cols_indices = np.where(zero_columns)[0]
    return np.sum(zero_columns), zero_cols_indices


def get_LLM_res_stream_actvs(model, layer_id, inputs, batch_size):
    LLM_actvs = None
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_data in loader:
        input_ids, attention_mask = batch_data
        batch_inputs = {
            'input_ids': input_ids.to(model.device), 
            'attention_mask': attention_mask.to(model.device)
        }
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            if LLM_actvs is None:
                LLM_actvs = outputs.hidden_states[layer_id]
            else:
                LLM_actvs = torch.cat((LLM_actvs, outputs.hidden_states[layer_id]), dim=0)

        del batch_inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    return LLM_actvs

# def get_LLM_MLP_actvs(model, model_name, layer_id, inputs):
#     if 'pythia' in model_name:
#         weight_matrix = model.gpt_neox.layers[layer_id].mlp.dense_4h_to_h.weight
#     elif 'gemma' in model_name:
#         weight_matrix = model.model.layers[layer_id].mlp.down_proj.weight
#     weight_matrix = weight_matrix.cpu().detach().numpy()

#     for module in model.modules():
#         if hasattr(module, "_forward_hooks"):
#             module._forward_hooks.clear()

#     orig_actvs = []
#     def hook_fn(module, input, output):
#         orig_actvs.append(output)

#     if 'pythia' in model_name:
#         handle = model.gpt_neox.layers[layer_id].mlp.dense_4h_to_h.register_forward_hook(hook_fn)
#     elif 'gemma' in model_name:
#         handle = model.model.layers[layer_id].mlp.down_proj.register_forward_hook(hook_fn)
#     with torch.inference_mode():
#         model(**inputs)
#     handle.remove()
    
#     orig_actvs = orig_actvs[0]

#     first_dim_reshaped = orig_actvs.shape[0] * orig_actvs.shape[1]
#     reshaped_activations = orig_actvs.reshape(first_dim_reshaped, orig_actvs.shape[-1]).cpu()

#     return weight_matrix, reshaped_activations, orig_actvs
    
def get_LLM_MLP_actvs(model, model_name, layer_id, inputs, batch_size):
    if 'pythia' in model_name:
        weight_matrix = model.gpt_neox.layers[layer_id].mlp.dense_4h_to_h.weight
    elif 'gemma' in model_name:
        weight_matrix = model.model.layers[layer_id].mlp.down_proj.weight
    weight_matrix = weight_matrix.cpu().detach().numpy()

    # List to collect activations from the MLP module.
    mlp_activation_list = []

    # Define a hook function to capture the MLP output.
    def mlp_hook(module, input, output):
        # Append the output (detached and moved to CPU if desired)
        mlp_activation_list.append(output.detach().cpu())

    # Register the hook on the MLP module of the specified layer.
    if 'pythia' in model_name:
        handle = model.gpt_neox.layers[layer_id].mlp.dense_4h_to_h.register_forward_hook(mlp_hook)
    elif 'gemma' in model_name:
        handle = model.model.layers[layer_id].mlp.down_proj.register_forward_hook(mlp_hook)

    # Create a DataLoader over your dataset.
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Run inference; the hook will collect MLP activations.
    for batch_data in loader:
        input_ids, attention_mask = batch_data
        batch_inputs = {
            'input_ids': input_ids.to(model.device), 
            'attention_mask': attention_mask.to(model.device)
        }
        with torch.no_grad():
            _ = model(**batch_inputs)
        
        # Cleanup
        del batch_inputs
        torch.cuda.empty_cache()
        gc.collect()

    # Remove the hook to avoid side effects.
    handle.remove()

    # Concatenate all collected activations.
    orig_actvs = torch.cat(mlp_activation_list, dim=0)

    first_dim_reshaped = orig_actvs.shape[0] * orig_actvs.shape[1]
    reshaped_activations = orig_actvs.reshape(first_dim_reshaped, orig_actvs.shape[-1]).cpu()

    return weight_matrix, reshaped_activations, orig_actvs