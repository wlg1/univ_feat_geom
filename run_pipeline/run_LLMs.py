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

# from sparsify.config import SaeConfig
# from sparsify.utils import decoder_impl
# from sparsify import Sae

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
    # model_name_A = "EleutherAI/pythia-160m"
    # model_name_A = "EleutherAI/pythia-70m"
    model_name_A = "google/gemma-2-2b"

    # model_name_B = "EleutherAI/pythia-70m"
    # model_name_B = "EleutherAI/pythia-160m"
    # model_name_B = "google/gemma-2-9b-it"
    model_name_B = "google/gemma-2-9b"

    ## Use SAE with fewer features as model B to be "one" (when using manyA_1B_bool=True in run_expms.py)
    
    # sae_name_A = "wlog/sae_pythia_70m_32k_seed32"
    # sae_name_A = "wlog/random_sae_pythia_70m_32k"
    # sae_name_A = "EleutherAI/sae-pythia-70m-32k"
    # sae_name_A = "EleutherAI/sae-pythia-160m-32k"
    # sae_name_A = "wlog/sae_pythia_70m_64k"
    # sae_name_A = "wlog/sae_pythia_70m_32k"
    # sae_name_A = "wlog/sae_pythia_160m_24k"
    # sae_lib_A = "eleuther"
    # sae_name_A = "gemma-scope-9b-pt-res-canonical"
    sae_name_A = "gemma-scope-2b-pt-mlp-canonical"
    sae_lib_A = "sae_lens"

    # sae_name_B = "wlog/sae_pythia_70m_32k_seed32"
    # sae_name_B = "EleutherAI/sae-pythia-70m-32k"
    # sae_name_B = "wlog/sae_pythia_70m_32k"
    # sae_name_B = "wlog/sae_pythia_70m_16k"
    # sae_name_B = "wlog/sae_pythia_70m_32k_OWT"
    # sae_lib_B = "eleuther"
    # sae_name_B = "gemma-scope-9b-it-res-canonical"
    sae_name_B = "gemma-scope-9b-pt-mlp-canonical"
    sae_lib_B = "sae_lens"

    dataset = "Skylion007/openwebtext"
    # dataset = "togethercomputer/RedPajama-Data-1T-Sample"

    # Since we are comparing layer 3 SAEs, set the start and end layers accordingly.
    ## pythia
    # model_A_startLayer = 2
    # model_A_endLayer = 5
    # layer_step_size_A = 1
    # model_A_startLayer = 2
    # model_A_endLayer = 11
    # layer_step_size_A = 2
    # model_A_startLayer = 1
    # model_A_endLayer = 6
    # layer_step_size_A = 1
    ## gemma
    model_A_startLayer = 1
    model_A_endLayer = 22
    layer_step_size_A = 5
    # model_A_startLayer = 11
    # model_A_endLayer = 12
    # layer_step_size_A = 1
    model_A_layers = list(range(model_A_startLayer, model_A_endLayer, layer_step_size_A))
    # model_A_layers = [9, 20, 31]

    ## pythia
    # model_B_startLayer = 2
    # model_B_endLayer = 7
    # layer_step_size_B = 2
    # model_B_startLayer = 1
    # model_B_endLayer = 6
    # # layer_step_size_B = 1
    # model_B_startLayer = 2
    # model_B_endLayer = 11
    # layer_step_size_B = 2
    ## gemma
    model_B_startLayer = 1
    model_B_endLayer = 42
    layer_step_size_B = 10
    # model_B_startLayer = 21
    # model_B_endLayer = 22
    # layer_step_size_B = 1
    model_B_layers = list(range(model_B_startLayer, model_B_endLayer, layer_step_size_B))
    # model_B_layers = [9, 20, 31]

    batch_size = 100
    max_length = 100
    num_rand_runs = 1
    oneToOne_bool = True
    compare_SAEs_bool = True
    compare_MLPs_bool = True

    ### Load base language models and tokenizers
    model_A = AutoModelForCausalLM.from_pretrained(model_name_A)
    model_B = AutoModelForCausalLM.from_pretrained(model_name_B)

    if compare_SAEs_bool:
        # Re-randomize models using the original seed from config.
        if 'random' in sae_name_A:
            model_A = RerandomizedModel(
                model_A,
                rerandomize_embeddings=config.rerandomize_embeddings,
                rerandomize_layer_norm=config.rerandomize_layer_norm,
                seed=config.random_seed # 42
            ).model
        if 'random' in sae_name_B:
            model_B = RerandomizedModel(
                model_B,
                rerandomize_embeddings=config.rerandomize_embeddings,
                rerandomize_layer_norm=config.rerandomize_layer_norm,
                seed=32
            ).model

    tokenizer = AutoTokenizer.from_pretrained(model_name_A)
    tokenizer.pad_token = tokenizer.eos_token

    ### Load data using a streaming dataset.
    from datasets import load_dataset
    dataset = load_dataset(dataset, split="train", streaming=True, trust_remote_code=True)

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

    _, inputs = get_next_batch(dataset, batch_size=batch_size, max_length=max_length)

    ### Process SAE activations for each model.
    if compare_SAEs_bool:
        print("Storing SAE activations for Model A")
        actvs_by_layer_A = {}
        for layer_id in model_A_layers:
            print("Model A Layer: " + str(layer_id))
            with torch.inference_mode():
                weight_matrix, reshaped_activations, feature_acts_model = get_sae_actvs(
                    model_A, 
                    model_name_A,
                    sae_name_A, 
                    inputs, 
                    layer_id, 
                    batch_size=32,
                    sae_lib=sae_lib_A,
                    compare_MLPs_bool=compare_MLPs_bool
                )
                actvs_by_layer_A[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

        # with open('actvs_by_layer_A.pkl', 'wb') as f:
        #     pickle.dump(actvs_by_layer_A, f)

        print("Storing SAE activations for Model B")
        actvs_by_layer_B = {}
        for layer_id in model_B_layers:
            print("Model B Layer: " + str(layer_id))
            with torch.inference_mode():
                weight_matrix, reshaped_activations, feature_acts_model = get_sae_actvs(
                    model_B, 
                    model_name_B,
                    sae_name_B, 
                    inputs, 
                    layer_id, 
                    batch_size=32,
                    sae_lib=sae_lib_B,
                    compare_MLPs_bool=compare_MLPs_bool
                )
                actvs_by_layer_B[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

        # with open('actvs_by_layer_B.pkl', 'wb') as f:
        #     pickle.dump(actvs_by_layer_B, f)
        
        sae_name_A = sae_name_A.replace('/', '_')
        sae_name_B = sae_name_B.replace('/', '_')
    else:
        actvs_by_layer_A = {}
        for layer_id in model_A_layers:
            print("Model A Layer: " + str(layer_id))
            with torch.inference_mode():
                weight_matrix, reshaped_activations, feature_acts_model = get_LLM_MLP_actvs(model_A, model_name_A, 
                                                                                            layer_id, inputs)
                actvs_by_layer_A[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

        actvs_by_layer_B = {}
        for layer_id in model_B_layers:
            print("Model B Layer: " + str(layer_id))
            with torch.inference_mode():
                weight_matrix, reshaped_activations, feature_acts_model = get_LLM_MLP_actvs(model_B, model_name_B,
                                                                                            layer_id, inputs)
                actvs_by_layer_B[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

        model_name_A = model_name_A.replace('/', '_')
        model_name_B = model_name_B.replace('/', '_')

    ### Run experiment comparing the two models’ SAE activations.
    print("Running experiment")

    model_layer_to_dictscores = {}

    for layer_id in model_A_layers:
        print("Model A Layer: " + str(layer_id))
        model_layer_to_dictscores[layer_id] = {}
        for layer_id_2 in model_B_layers:
            print("Model B Layer: " + str(layer_id_2))
            model_layer_to_dictscores[layer_id][layer_id_2] = run_expm(
                inputs, tokenizer, 
                actvs_by_layer_A[layer_id],
                actvs_by_layer_B[layer_id_2], 
                num_rand_runs=num_rand_runs, 
                oneToOne_bool=oneToOne_bool
            )
            for key, value in model_layer_to_dictscores[layer_id][layer_id_2].items():
                print(key + ": " + str(value))
            print("\n")
            if compare_SAEs_bool:
                with open(f'{sae_name_A}_{sae_name_B}_multL_scores.pkl', 'wb') as f:
                    pickle.dump(model_layer_to_dictscores, f)
            else:
                with open(f'{model_name_A}_{model_name_B}_multL_scores.pkl', 'wb') as f:
                    pickle.dump(model_layer_to_dictscores, f)
    if compare_SAEs_bool:
        with open(f'{sae_name_A}_{sae_name_B}_multL_scores.pkl', 'wb') as f:
            pickle.dump(model_layer_to_dictscores, f)
    else:
        with open(f'{model_name_A}_{model_name_B}_multL_scores.pkl', 'wb') as f:
            pickle.dump(model_layer_to_dictscores, f)

if __name__ == "__main__":
    main()
