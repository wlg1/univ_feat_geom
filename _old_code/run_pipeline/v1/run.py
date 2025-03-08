### for pythia model comparisons

import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from fnmatch import fnmatch
from pathlib import Path
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

# import pdb
# pdb.set_trace()

### load data

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset
dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

batch_size = 100
maxseqlen = 300

def get_next_batch(dataset_iter, batch_size=100):
    batch = []
    for _ in range(batch_size):
        try:
            sample = next(dataset_iter)
            batch.append(sample['text'])
        except StopIteration:
            break
    return batch

dataset_iter = iter(dataset)
batch = get_next_batch(dataset_iter, batch_size)

# Tokenize the batch
inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=maxseqlen)

### load models

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
model_2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")

with torch.inference_mode():
    # outputs = model(**inputs, output_hidden_states=True)
    outputs_2 = model_2(**inputs, output_hidden_states=True)

### run
model_A_layers = list(range(1, 6))
layer_start = 1
layer_end = len(model_2.gpt_neox.layers)
modA_layer_to_dictscores = {}
for layer_id in model_A_layers:
    print("Model A Layer: " + str(layer_id))
    with torch.inference_mode():
        outputs = get_llm_actvs_batch(model, inputs, layer_id, batch_size=100, maxseqlen=300)

    modA_layer_to_dictscores[layer_id] = run_expm(inputs, tokenizer, layer_id, outputs, outputs_2,
                                                  layer_start, layer_end,
                                                  num_runs=100, oneToOne_bool=False)
    
with open(f'pythia70m_pythia160m_multL_scores.pkl', 'wb') as f:
    pickle.dump(modA_layer_to_dictscores, f)