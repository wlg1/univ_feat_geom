import torch
from torch.utils.data import DataLoader, TensorDataset

import gc
import numpy as np

from sparsify.config import SaeConfig
from sparsify.utils import decoder_impl
from sparsify import Sae

# Note: also importing the SAE from sae_lens for alternate usage.
from sae_lens import SAE

def get_sae_actvs(model, sae_name, inputs, layer_id, batch_size=32, 
                  sae_lib='eleuther', custom_hookpoint=None):
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
        reshaped_activations_A (torch.Tensor): The pre-activation outputs reshaped.
        orig (torch.Tensor): The original batched pre-activations.
    """    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LLM_actvs = None
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_data in loader:
        input_ids, attention_mask = batch_data
        batch_inputs = {'input_ids': input_ids.to(model.device), 
                        'attention_mask': attention_mask.to(model.device)}
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            if LLM_actvs is None:
                LLM_actvs = outputs.hidden_states[layer_id]
            else:
                LLM_actvs = torch.cat((LLM_actvs, outputs.hidden_states[layer_id]), dim=0)

        del batch_inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    ### Get SAE activations ###
    if sae_lib == 'eleuther':
        # Use the custom hookpoint if provided; otherwise, default to "layers.<layer_id>"
        hookpoint = custom_hookpoint if custom_hookpoint is not None else "layers." + str(layer_id)
        sae = Sae.load_from_hub(sae_name, hookpoint=hookpoint, device=device)
    elif sae_lib == 'sae_lens':
        # For SAE-Lens, determine sae_id based on the release name.
        if 'scope' in sae_name:
            sae_id = f"layer_{layer_id}/width_16k/canonical"
        else:
            sae_id = f"blocks.{layer_id}.hook_resid_post"
        sae, _, _ = SAE.from_pretrained(
            release=sae_name,
            sae_id=sae_id,
        )
        sae = sae.to('cuda')
        sae.eval()  # Ensure evaluation mode.
    
    weight_matrix_np = sae.W_dec.cpu().detach().numpy()
    
    # Process the activations in batches.
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

    # Concatenate the processed batches.
    orig_actvs = torch.cat(pre_act_batches, dim=0)
    reshaped_dim = orig_actvs.shape[0] * orig_actvs.shape[1]
    reshaped_activations_A = orig_actvs.reshape(reshaped_dim, orig_actvs.shape[-1]).cpu()
    
    return weight_matrix_np, reshaped_activations_A, orig_actvs

def count_zero_columns(tensor):
    # Check if all elements in each column are zero.
    zero_columns = np.all(tensor == 0, axis=0)
    zero_cols_indices = np.where(zero_columns)[0]
    return np.sum(zero_columns), zero_cols_indices
