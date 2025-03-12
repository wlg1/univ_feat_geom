import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from typing import NamedTuple, Optional, Callable, Union, List, Tuple
from collections import Counter
import seaborn as sns
import pandas as pd

import einops
import torch
from torch import Tensor, nn
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.config import SaeConfig
from sparsify.utils import decoder_impl
from sparsify import Sae

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load local helper functions
from correlation_fns import *
from sim_fns import *
from get_rand_fns import *
from interpret_fns import *
from get_actv_fns import get_sae_actvs
from run_expm_fns import *
from plot_fns import *

import argparse

# Import our rerandomization wrapper and experiment configuration.
from rerandomized_model import RerandomizedModel
from experiment_config import config

def main():
    # --- Set model and experiment parameters --- 
    # For this experiment we compare the SAE used for pythia-70m at layer 3.
    model_name = "EleutherAI/pythia-70m"
    
    # Toggle which SAE to use:
    use_custom_sae = True   # If True, use custom SAE from wlog/random_sae_pythia_70m.
    # For model A (e.g. custom SAE) and model B (original SAE) you can set separately:
    use_custom_sae_A = True
    use_custom_sae_B = False

    # Define SAE parameters based on the flag.
    if use_custom_sae_A:
        sae_name_A = "wlog/random_sae_pythia_70m"
        hookpoint_A = "gpt_neox.layers.3"
        sae_lib_A = "eleuther"
    else:
        sae_name_A = "EleutherAI/sae-pythia-70m-32k"
        hookpoint_A = "layers.3"
        sae_lib_A = "eleuther"
        
    if use_custom_sae_B:
        sae_name_B = "wlog/random_sae_pythia_70m"
        hookpoint_B = "gpt_neox.layers.3"
        sae_lib_B = "eleuther"
    else:
        sae_name_B = "EleutherAI/sae-pythia-70m-32k"
        hookpoint_B = "layers.3"
        sae_lib_B = "eleuther"

    # Since we are comparing layer 3 SAEs, set the start and end layers accordingly.
    model_A_startLayer = 3
    model_A_endLayer = 4  # non-inclusive, so only layer 3
    model_B_startLayer = 3
    model_B_endLayer = 4
    layer_step_size = 1

    batch_size = 150
    max_length = 150
    num_rand_runs = 1
    oneToOne_bool = True

    ### Load base language models and tokenizers
    model_A = AutoModelForCausalLM.from_pretrained(model_name)
    model_B = AutoModelForCausalLM.from_pretrained(model_name)

    # Re-randomize models using the original seed from config.
    model_A = RerandomizedModel(
        model_A,
        # rerandomize_embeddings=config.rerandomize_embeddings,
        rerandomize_embeddings=True,
        rerandomize_layer_norm=config.rerandomize_layer_norm,
        seed=config.random_seed
    ).model
    # model_B = RerandomizedModel(
    #     model_B,
    #     rerandomize_embeddings=config.rerandomize_embeddings,
    #     rerandomize_layer_norm=config.rerandomize_layer_norm,
    #     seed=config.random_seed
    # ).model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ### Load data using a streaming dataset.
    from datasets import load_dataset
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

    def get_next_batch(dataset, batch_size=100, max_length=100):
        batch = []
        dataset_iter = iter(dataset)
        for _ in range(batch_size):
            try:
                sample = next(dataset_iter)
                batch.append(sample['text'])
            except StopIteration:
                break
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return batch, inputs

    batch, inputs = get_next_batch(dataset, batch_size=batch_size, max_length=max_length)

    ### Process SAE activations for each model.
    print("Storing SAE activations for Model A")
    saeActvs_by_layer_A = {}
    for layer_id in range(model_A_startLayer, model_A_endLayer, layer_step_size): 
        print("Model A Layer: " + str(layer_id))
        with torch.inference_mode():
            weight_matrix, reshaped_activations, feature_acts_model = get_sae_actvs(
                model_A, 
                sae_name_A, 
                inputs, 
                layer_id, 
                batch_size=32,
                sae_lib=sae_lib_A,
                custom_hookpoint=hookpoint_A  # custom hookpoint used if custom SAE is desired.
            )
            saeActvs_by_layer_A[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

    with open('saeActvs_by_layer_A.pkl', 'wb') as f:
        pickle.dump(saeActvs_by_layer_A, f)

    print("Storing SAE activations for Model B")
    saeActvs_by_layer_B = {}
    for layer_id in range(model_B_startLayer, model_B_endLayer, layer_step_size): 
        print("Model B Layer: " + str(layer_id))
        with torch.inference_mode():
            weight_matrix, reshaped_activations, feature_acts_model = get_sae_actvs(
                model_B, 
                sae_name_B, 
                inputs, 
                layer_id, 
                batch_size=32,
                sae_lib=sae_lib_B,
                custom_hookpoint=hookpoint_B
            )
            saeActvs_by_layer_B[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

    with open('saeActvs_by_layer_B.pkl', 'wb') as f:
        pickle.dump(saeActvs_by_layer_B, f)

    ### Run experiment comparing the two models’ SAE activations.
    print("Running experiment")

    sae_name_A_clean = sae_name_A.replace('/', '_')
    sae_name_B_clean = sae_name_B.replace('/', '_')

    model_layer_to_dictscores = {}

    model_A_layers = list(range(model_A_startLayer, model_A_endLayer, layer_step_size))
    model_B_layers = list(range(model_B_startLayer, model_B_endLayer, layer_step_size))
    for layer_id in model_A_layers:
        print("Model A Layer: " + str(layer_id))
        model_layer_to_dictscores[layer_id] = {}
        for layer_id_2 in model_B_layers:
            print("Model B Layer: " + str(layer_id_2))
            model_layer_to_dictscores[layer_id][layer_id_2] = run_expm(
                inputs, tokenizer, layer_id, 
                saeActvs_by_layer_A[layer_id],
                saeActvs_by_layer_B[layer_id_2], 
                num_rand_runs=num_rand_runs, 
                oneToOne_bool=oneToOne_bool
            )
            for key, value in model_layer_to_dictscores[layer_id][layer_id_2].items():
                print(key + ": " + str(value))
            print("\n")
            with open(f'{sae_name_A_clean}_{sae_name_B_clean}_multL_scores.pkl', 'wb') as f:
                pickle.dump(model_layer_to_dictscores, f)

if __name__ == "__main__":
    main()
