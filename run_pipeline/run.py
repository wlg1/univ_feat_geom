### for pythia model comparisons

import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from typing import NamedTuple, Optional, Callable, Union, List, Tuple
# from jaxtyping import Float, Int
from collections import Counter
import seaborn as sns
import pandas as pd

import einops
import torch
from torch import Tensor, nn
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.config import SaeConfig
from sparsify.utils import decoder_impl
from sparsify import Sae

device = "cuda" if torch.cuda.is_available() else "cpu"

## load local fns
from correlation_fns import *
from sim_fns import *
from get_rand_fns import *
from interpret_fns import *
from get_actv_fns import *
from run_expm_fns import *
from plot_fns import *

import argparse

# import pdb
# pdb.set_trace()

"""
example run:
python run.py --batch_size 400 --max_length 400 --num_rand_runs 1 --oneToOne_bool --model_A_endLayer 6 --model_B_endLayer 12 --layer_step_size 2

python run.py --batch_size 100 --max_length 100 --num_rand_runs 1 --oneToOne_bool --model_A_endLayer 4 --model_B_endLayer 4 --layer_step_size 2
"""

### script args
# batch_size = 400
# max_length = 400
# num_rand_runs = 1
# oneToOne_bool = False
# model_A_endLayer = 6
# model_B_endLayer = 12
# layer_step_size = 2

def main():
    parser = argparse.ArgumentParser(description="Run pythia model comparisons")
    parser.add_argument("--batch_size", type=int, default=400, help="Batch size")
    parser.add_argument("--max_length", type=int, default=400, help="Maximum sequence length")
    parser.add_argument("--num_rand_runs", type=int, default=1, help="Number of random runs")
    parser.add_argument("--oneToOne_bool", action="store_true", help="Use one-to-one mapping flag")
    parser.add_argument("--model_A_endLayer", type=int, default=6, help="Model A end layer")
    parser.add_argument("--model_B_endLayer", type=int, default=12, help="Model B end layer")
    parser.add_argument("--layer_step_size", type=int, default=12, help="Layer step size")
    
    args = parser.parse_args()
    
    # Assign the arguments to variables
    batch_size = args.batch_size
    max_length = args.max_length
    num_rand_runs = args.num_rand_runs
    oneToOne_bool = args.oneToOne_bool
    model_A_endLayer = args.model_A_endLayer
    model_B_endLayer = args.model_B_endLayer
    layer_step_size = args.layer_step_size

    ### load models

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    model_2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.pad_token = tokenizer.eos_token

    ### load data
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

        # Tokenize the batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        return batch, inputs

    batch, inputs = get_next_batch(dataset, batch_size=batch_size, max_length=max_length)

    ### store sae actvs
    sae_name = "EleutherAI/sae-pythia-70m-32k"
    saeActvs_by_layer_1 = {}
    for layer_id in range(1, model_A_endLayer, layer_step_size): # step = layer_step_size
        print("Model A Layer: " + str(layer_id))
        with torch.inference_mode():
            weight_matrix, reshaped_activations, feature_acts_model =  get_sae_actvs(model, sae_name, inputs, layer_id, batch_size=32)
            saeActvs_by_layer_1[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

    sae_name = "EleutherAI/sae-pythia-160m-32k"
    saeActvs_by_layer_2 = {}
    for layer_id in range(1, model_B_endLayer, layer_step_size): # step = layer_step_size
        print("Model B Layer: " + str(layer_id))
        with torch.inference_mode():
            weight_matrix, reshaped_activations, feature_acts_model =  get_sae_actvs(model_2, sae_name, inputs, layer_id, batch_size=32)
            saeActvs_by_layer_2[layer_id] = (weight_matrix, reshaped_activations, feature_acts_model)

    ### run
    model_A_layers = list(range(1, model_A_endLayer, layer_step_size))
    model_B_layers = list(range(1, model_B_endLayer, layer_step_size)) 
    # layer_start = 1
    # layer_end = len(model_2.gpt_neox.layers)
    model_layer_to_dictscores = {}
    for layer_id in model_A_layers:
        print("Model A Layer: " + str(layer_id))
        model_layer_to_dictscores[layer_id] = {}
        for layer_id_2 in model_B_layers: # in range(layer_start, layer_end): # 0, 12
            print("Model B Layer: " + str(layer_id_2))

            model_layer_to_dictscores[layer_id][layer_id_2] = run_expm(inputs, tokenizer, layer_id, 
                                                        saeActvs_by_layer_1[layer_id],
                                                        saeActvs_by_layer_2[layer_id_2], 
                                                        num_rand_runs=num_rand_runs, oneToOne_bool=oneToOne_bool)
            
            for key, value in model_layer_to_dictscores[layer_id][layer_id_2].items():
                print(key + ": " + str(value))
            print("\n")

    with open(f'pythia70m_pythia160m_multL_scores.pkl', 'wb') as f:
        pickle.dump(model_layer_to_dictscores, f)


if __name__ == "__main__":
    main()