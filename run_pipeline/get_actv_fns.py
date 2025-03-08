import torch
from torch.utils.data import DataLoader, TensorDataset

import gc
import numpy as np

from sparsify.config import SaeConfig
from sparsify.utils import decoder_impl
from sparsify import Sae

def get_llm_actvs_batch(model, inputs, layerID, batch_size=100, maxseqlen=300):
    accumulated_outputs = None
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_hidden_states = []
    for batch in loader:
        input_ids, attention_mask = batch

        batch_inputs = {'input_ids': input_ids.to(model.device), 'attention_mask': attention_mask.to(model.device)}
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            if accumulated_outputs is None:
                accumulated_outputs = outputs.hidden_states[layerID]
                # for layer_id in len(model.gpt_neox.layers):
                    # outputs_by_layer[layer_id] = outputs.hidden_states[layer_id]
            else:
                accumulated_outputs = torch.cat((accumulated_outputs, outputs.hidden_states[layerID]), dim= 0)

        del batch_inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    return accumulated_outputs

##### saes #####

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