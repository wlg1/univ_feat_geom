# -*- coding: utf-8 -*-
"""steering_gemma2_nums_months_v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1b7r3SMbkrX5_V9BezSIx_GR2Jx734M90

# setup
"""

# import pdb

import pickle
import numpy as np

import torch
import matplotlib.pyplot as plt

from torch import nn, Tensor
# from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

# from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

"""# load labels"""

import json
with open('gemma-2-2b-20-gemmascope-res-16k-explanations.json', 'rb') as f:
    feat_labels_allData = json.load(f)

feat_labels_lst = [0 for i in range(feat_labels_allData['explanationsCount'])]
feat_labels_dict = {}
for f_dict in feat_labels_allData['explanations']:
    feat_labels_lst[int(f_dict['index'])] = f_dict['description']
    feat_labels_dict[int(f_dict['index'])] = f_dict['description']
    if int(f_dict['index']) == 0:
        print(f_dict['description'])

len(feat_labels_dict)

"""## search for features"""

def find_indices_with_keyword(f_dict, keyword):
    """
    Find all indices of fList which contain the keyword in the string at those indices.

    Args:
    fList (list of str): List of strings to search within.
    keyword (str): Keyword to search for within the strings of fList.

    Returns:
    list of int: List of indices where the keyword is found within the strings of fList.
    """
    filt_dict = {}
    for index, string in f_dict.items():
        # split_list = string.split(',')
        # no_space_list = [i.replace(' ', '').lower() for i in split_list]
        # if keyword in no_space_list:
        if keyword in string:
            filt_dict[index] = string
    return filt_dict

keyword = "number"
number_feats = find_indices_with_keyword(feat_labels_dict, keyword)

keyword = "month"
month_feats = find_indices_with_keyword(feat_labels_dict, keyword)

set(number_feats).intersection(month_feats)

number_feats[5769]

for common_feat in set(number_feats).intersection(month_feats):
    del number_feats[common_feat]

set(number_feats).infor sampID, samp_common_feats in enumerate(common_feats_per_row):
    print('samp: ', sampID)
    for f_ind in samp_common_feats:
        print(f_ind, feat_labels_lst[f_ind])tersection(month_feats)

"""# load model"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import numpy as np
import torch

from huggingface_hub import hf_hub_download, notebook_login
notebook_login()

torch.set_grad_enabled(False) # avoid blowing up mem

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map='auto',
)

tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b")

"""# load transformerlens"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install transformer_lens

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# from transformer_lens import HookedTransformer
# 
# # uses a lot of memory, use A100
# model_2 = HookedTransformer.from_pretrained(
#     "gemma-2-2b"
# )

from transformer_lens.hook_points import HookPoint
from functools import partial
from jaxtyping import Float, Int

"""# test prompts"""

prompt = "thirteen fourteen fifteen sixteen "
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=1)
print(tokenizer.decode(outputs[0, -1]))

"""# load sae"""

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    force_download=False,
)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

import torch.nn as nn
class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon

sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)

sae.cuda()

"""## load sae weights

"""

weight_matrix = sae.W_dec.detach().cpu().numpy()
weight_matrix.shape

"""## get actv fns"""

prompt = "January February March April"
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0]
    return outputs
  handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
  _ = model.forward(inputs)
  handle.remove()
  return target_act

target_act = gather_residual_activations(model, 20, inputs)

"""Now, we can run our SAE on the saved activations."""

sae_acts = sae.encode(target_act.to(torch.float32))
# recon = sae.decode(sae_acts)

sae_acts.shape

"""# steering fns

Steering sae fns
"""

def steer_by_sae_actvs(prompt, steer_vec, multp):
    tokens = model_2.to_tokens(prompt).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    target_act = gather_residual_activations(model, 20, inputs)
    sae_acts_3 = sae.encode(target_act.to(torch.float32))

    sae_acts_3  += multp * steer_vec # in-place
    recon = sae.decode(sae_acts_3)
    recon.shape

    # replace LLM actvs in that layer with decoder output

    layer_name = 'blocks.20.hook_resid_post'

    def patch_layer(
        orig_actvs: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        LLM_patch: Float[Tensor, "batch pos d_model"],
        layer_to_patch: int,
    ) -> Float[Tensor, "batch pos d_model"]:
        if layer_to_patch == hook.layer():
            orig_actvs[:, :, :] = LLM_patch
        return orig_actvs

    hook_fn = partial(
            patch_layer,
            LLM_patch= recon,
            layer_to_patch = 20
        )

    # if you use run_with_cache, you need to add_hook before
    # if you use run_with_hooks, you dont need add_hook, just add it in fwd_hooks arg
    # no need to reset hoooks after since run_with_hooks isn't permanent like add_hook with perm arg

    # rerun clean inputs on ablated model
    ablated_logits = model_2.run_with_hooks(tokens,
                        fwd_hooks=[
                            (layer_name, hook_fn),
                        ]
                    )

    next_token = ablated_logits[0, -1].argmax(dim=-1)
    next_char = model_2.to_string(next_token)
    print(next_char)

def steer_by_sae_lastTok(prompt, steer_vec, multp):
    tokens = model_2.to_tokens(prompt).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    target_act = gather_residual_activations(model, 20, inputs)
    sae_acts_3 = sae.encode(target_act.to(torch.float32))

    sae_acts_3[:, -1, :]  += multp * steer_vec[:, -1, :] # in-place
    recon = sae.decode(sae_acts_3)
    recon.shape

    # replace LLM actvs in that layer with decoder output

    layer_name = 'blocks.20.hook_resid_post'

    def patch_layer(
        orig_actvs: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        LLM_patch: Float[Tensor, "batch pos d_model"],
        layer_to_patch: int,
    ) -> Float[Tensor, "batch pos d_model"]:
        if layer_to_patch == hook.layer():
            orig_actvs[:, :, :] = LLM_patch
        return orig_actvs

    hook_fn = partial(
            patch_layer,
            LLM_patch= recon,
            layer_to_patch = 20
        )

    # if you use run_with_cache, you need to add_hook before
    # if you use run_with_hooks, you dont need add_hook, just add it in fwd_hooks arg
    # no need to reset hoooks after since run_with_hooks isn't permanent like add_hook with perm arg

    # rerun clean inputs on ablated model
    ablated_logits = model_2.run_with_hooks(tokens,
                        fwd_hooks=[
                            (layer_name, hook_fn),
                        ]
                    )

    next_token = ablated_logits[0, -1].argmax(dim=-1)
    next_char = model_2.to_string(next_token)
    print(next_char)

"""LLM steering fns"""

def steer_by_LLM_lastTok(prompt, steer_vec, multp):
    tokens = model_2.to_tokens(prompt).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    target_act = gather_residual_activations(model, 20, inputs)

    target_act[:, -1, :]  += multp * steer_vec[:, -1, :] # in-place

    # replace LLM actvs in that layer with decoder output

    layer_name = 'blocks.20.hook_resid_post'

    def patch_layer(
        orig_actvs: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        LLM_patch: Float[Tensor, "batch pos d_model"],
        layer_to_patch: int,
    ) -> Float[Tensor, "batch pos d_model"]:
        if layer_to_patch == hook.layer():
            orig_actvs[:, :, :] = LLM_patch
        return orig_actvs

    hook_fn = partial(
            patch_layer,
            LLM_patch= target_act,
            layer_to_patch = 20
        )

    # if you use run_with_cache, you need to add_hook before
    # if you use run_with_hooks, you dont need add_hook, just add it in fwd_hooks arg
    # no need to reset hoooks after since run_with_hooks isn't permanent like add_hook with perm arg

    # rerun clean inputs on ablated model
    ablated_logits = model_2.run_with_hooks(tokens,
                        fwd_hooks=[
                            (layer_name, hook_fn),
                        ]
                    )

    next_token = ablated_logits[0, -1].argmax(dim=-1)
    next_char = model_2.to_string(next_token)
    print(next_char)

def steer_by_LLM_decompSAE(prompt, patch, multp):
    # replace LLM actvs in that layer with decoder output

    layer_name = 'blocks.20.hook_resid_post'

    def patch_layer(
        orig_actvs: Float[Tensor, "batch pos d_model"],
        hook: HookPoint,
        LLM_patch: Float[Tensor, "batch pos d_model"],
        layer_to_patch: int,
    ) -> Float[Tensor, "batch pos d_model"]:
        if layer_to_patch == hook.layer():
            orig_actvs[:, :, :] = LLM_patch
        return orig_actvs

    hook_fn = partial(
            patch_layer,
            LLM_patch= patch,
            layer_to_patch = 20
        )

    # if you use run_with_cache, you need to add_hook before
    # if you use run_with_hooks, you dont need add_hook, just add it in fwd_hooks arg
    # no need to reset hoooks after since run_with_hooks isn't permanent like add_hook with perm arg

    # rerun clean inputs on ablated model
    ablated_logits = model_2.run_with_hooks(tokens,
                        fwd_hooks=[
                            (layer_name, hook_fn),
                        ]
                    )

    next_token = ablated_logits[0, -1].argmax(dim=-1)
    next_char = model_2.to_string(next_token)
    print(next_char)

"""# steer by SAE using members not seqs

## get sv without subtraction- numerals
"""

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
numword_samps = []
for i in range(0, 10):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
sae_acts_1 = sae.encode(target_act.to(torch.float32))

# nw_sv = sae_acts_1
mean_num_sv = sae_acts_1.mean(dim=0)
mean_num_sv = mean_num_sv.unsqueeze(0)
mean_num_sv.shape

"""### steer"""

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]}" # {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    months_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months_samps = []
for i in range(0, 5):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    months_samps.append(samp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
tomorrowIs_samps = []
for i in range(0, 6):
    samp = f"Today is {words[i]}, tomorrow is"
    tomorrowIs_samps.append(samp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 1

for prompt in tomorrowIs_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 3

for prompt in tomorrowIs_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

#
prompt = "This month is July, next month is"
steer_vec = mean_num_sv
multp = 1
steer_by_sae_lastTok(prompt, steer_vec, multp)

#
prompt = "four three two"
steer_vec = mean_num_sv
multp = 1
steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve']
SPnumword_samps = []
for i in range(0, 6):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    SPnumword_samps.append(samp)

steer_vec = mean_num_sv
multp = 0

for prompt in SPnumword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = mean_num_sv
multp = 1

for prompt in SPnumword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

"""### nonseq prompts"""

nonseq_prompts = [ "aa a a", "I AM A CAT", "Bob and Mary went to store", "my favorite colour is"]
steer_vec = mean_num_sv
multp = 3

for prompt in nonseq_prompts:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

nonseq_prompts = [ "aa a a", "I AM A CAT", "Bob and Mary went to store", "my favorite colour is"]
steer_vec = mean_num_sv
multp = 10

for prompt in nonseq_prompts:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

"""we score using the correct expected answer

## get sv without subtraction- numerals (no 10)
"""

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
sae_acts_1 = sae.encode(target_act.to(torch.float32))

# nw_sv = sae_acts_1
mean_num_sv = sae_acts_1.mean(dim=0)
mean_num_sv = mean_num_sv.unsqueeze(0)
mean_num_sv.shape

"""### steer"""

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]}" # {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    months_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months_samps = []
for i in range(0, 5):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    months_samps.append(samp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
tomorrowIs_samps = []
for i in range(0, 6):
    samp = f"Today is {words[i]}, tomorrow is"
    tomorrowIs_samps.append(samp)

# assume monday is the first day, aim to get 4 if ends on Wedn
steer_vec = mean_num_sv
multp = 1

for prompt in tomorrowIs_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
nextMonthIs_samps = []
for i in range(0, 6):
    samp = f"This month is {words[i]}, next month is"
    nextMonthIs_samps.append(samp)

#
steer_vec = mean_num_sv
multp = 1

for prompt in nextMonthIs_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

#
steer_vec = mean_num_sv
multp = 3

for prompt in nextMonthIs_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'][::-1]
backw_numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    backw_numword_samps.append(samp)

steer_vec = mean_num_sv
multp = 1

for prompt in backw_numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = mean_num_sv
multp = 3

for prompt in backw_numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve']
SPnumword_samps = []
for i in range(0, 6):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    SPnumword_samps.append(samp)

steer_vec = mean_num_sv
multp = 0

for prompt in SPnumword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = mean_num_sv
multp = 1

for prompt in SPnumword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

"""### nonseq prompts"""

nonseq_prompts = [ "aa a a", "I AM A CAT", "Bob and Mary went to store", "my favorite colour is"]
steer_vec = mean_num_sv
multp = 3

for prompt in nonseq_prompts:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

nonseq_prompts = [ "aa a a", "I AM A CAT", "Bob and Mary went to store", "my favorite colour is"]
steer_vec = mean_num_sv
multp = 10

for prompt in nonseq_prompts:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

"""we score using the correct expected answer

## get nw sv without subtraction
"""

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
numword_samps = []
for i in range(0, 10):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
sae_acts_1 = sae.encode(target_act.to(torch.float32))

mean_nw_sv = sae_acts_1.mean(dim=0)
mean_nw_sv = mean_nw_sv.unsqueeze(0)
mean_nw_sv.shape

"""### steer"""

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
numerals_samps = []
for i in range(0, 6):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    numerals_samps.append(samp)

steer_vec = mean_nw_sv
multp = 1

for prompt in numerals_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    months_samps.append(samp)

steer_vec = mean_nw_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = mean_nw_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

"""## get sv as nw-months"""

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October'] # , 'November', 'December'
month_samps = []
for i in range(0, 10):
    samp = f"{words[i]}"
    month_samps.append(samp)
inputs = tokenizer(month_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
sae_acts_2 = sae.encode(target_act.to(torch.float32))

"""pair each num prompt with its months analogue, and subtract"""

months_minus_nw_nonseq = sae_acts_2 - sae_acts_1
mean_months_minus_nw_nonseq = months_minus_nw_nonseq.mean(dim=0)
mean_months_minus_nw_nonseq = mean_months_minus_nw_nonseq.unsqueeze(0)
mean_months_minus_nw_nonseq.shape

"""### steer"""

prompt = "four five six"
steer_vec = mean_months_minus_nw_nonseq
multp = 4
steer_by_sae_lastTok(prompt, steer_vec, multp)

prompt = "four"
steer_vec = mean_months_minus_nw_nonseq
multp = 1
steer_by_sae_lastTok(prompt, steer_vec, multp)

prompt = "four"
steer_vec = mean_months_minus_nw_nonseq
multp = 5
steer_by_sae_lastTok(prompt, steer_vec, multp)

prompt = "one"
steer_vec = mean_months_minus_nw_nonseq
multp = 10
steer_by_sae_lastTok(prompt, steer_vec, multp)

"""## mod 10"""

#
prompt = "thirteen fourteen fifteen sixteen "
steer_vec = mean_num_sv
multp = 0
steer_by_sae_lastTok(prompt, steer_vec, multp)

#
prompt = "seven eight nine ten eleven"
steer_vec = mean_num_sv
multp = 0
steer_by_sae_lastTok(prompt, steer_vec, multp)

#
prompt = "seven eight nine ten eleven"
steer_vec = mean_num_sv
multp = 1.5
steer_by_sae_lastTok(prompt, steer_vec, multp)

#
prompt = "six seven eight nine ten eleven twelve"
steer_vec = mean_num_sv
multp = 0
steer_by_sae_lastTok(prompt, steer_vec, multp)

"""# steer by LLM actvs"""

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
target_act.shape

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
numword_samps = []
for i in range(0, 10):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
# sae_acts_1 = sae.encode(target_act.to(torch.float32))

target_act.shape

LLM_mean_num_sv = target_act.mean(dim=0)
LLM_mean_num_sv = LLM_mean_num_sv.unsqueeze(0)
LLM_mean_num_sv.shape

"""## steer"""

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = LLM_mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    months_samps.append(samp)

steer_vec = LLM_mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 10

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months_samps = []
for i in range(0, 5):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    months_samps.append(samp)

#unsteered
steer_vec = LLM_mean_num_sv
multp = 0

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 10

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

"""# Decompose the LLM steering vec"""

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)

LLM_mean_num_sv = target_act.mean(dim=0)
LLM_mean_num_sv = LLM_mean_num_sv.unsqueeze(0)
LLM_mean_num_sv.shape

# decompose LLM steering vec
sae_LLM_mean_num_sv = sae.encode(LLM_mean_num_sv.to(torch.float32))

feat_k = 100
one_top_acts_values, one_top_acts_indices = sae_LLM_mean_num_sv[0, -1, :].topk(feat_k, dim=-1)

rank = 1
for val, ind in zip(one_top_acts_values, one_top_acts_indices):
    print(f"Rank {rank}", round(val.item(), 2), ind.item(), feat_labels_lst[ind])
    rank += 1

"""These are nonzero:
- Rank 1 150.1 13528 numeric values indicating measurements or quantities
- Rank 2 85.52 11527 the start of a document
- Rank 3 34.94 9768 terms related to control and authority, particularly in political or systemic contexts
- Rank 4 34.91 8684  technical jargon and programming-related terms
- Rank 5 33.51 140  instances of the word "in"
- Rank 6 30.87 833 the numerical values indicating measurements or assessments
- Rank 7 29.39 11174  numeric identifiers or values in a structured format
- Rank 8 21.74 2437  patterns related to numerical information, particularly involving the number four
- Rank 9 21.66 6305 mathematical expressions and notations used in equations and proofs
- Rank 10 15.28 3019  elements related to operational or procedural contexts in a structured format
- Rank 11 10.27 11795 the phrase "The" at the start of sentences
- Rank 12 9.79 745  sequences of numerical values
- Rank 13 9.0 3435  numeric values, particularly related to technology specifications or measurements
- Rank 14 8.34 1344 lists and resources for educational purposes
- Rank 15 7.86 16175 numeric identifiers or codes, particularly in a structured format or specification

Try ablating the following:
- Rank 2 85.52 11527 the start of a document
- Rank 3 34.94 9768 terms related to control and authority, particularly in political or systemic contexts
- Rank 4 34.91 8684  technical jargon and programming-related terms
- Rank 5 33.51 140  instances of the word "in"
- Rank 10 15.28 3019  elements related to operational or procedural contexts in a structured format
- Rank 11 10.27 11795 the phrase "The" at the start of sentences

## ablate then recon to steer
"""

def ablate_SAE_lastTok(prompt, steer_vec, multp):
    tokens = model_2.to_tokens(prompt).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    target_act = gather_residual_activations(model, 20, inputs)
    sae_acts_3 = sae.encode(target_act.to(torch.float32))

    sae_acts_3[:, -1, :]  += multp * steer_vec[:, -1, :] # in-place
    recon = sae.decode(sae_acts_3)
    recon.shape

abl_sae_LLM_mean_num_sv = torch.clone(sae_LLM_mean_num_sv)

feats_to_ablate = [11527, 9768, 8684, 140, 3019, 11795]

for fInd in feats_to_ablate:
    abl_sae_LLM_mean_num_sv[:, -1, fInd] = 0

recon_abl_sae_LLM_mean_num_sv = sae.decode(abl_sae_LLM_mean_num_sv)
recon_abl_sae_LLM_mean_num_sv.shape

"""## steer ablated recon in LLM space"""

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = recon_abl_sae_LLM_mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = recon_abl_sae_LLM_mean_num_sv
multp = 3

for prompt in numword_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    months_samps.append(samp)

steer_vec = recon_abl_sae_LLM_mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = recon_abl_sae_LLM_mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 10

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months_samps = []
for i in range(0, 5):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    months_samps.append(samp)

#unsteered
steer_vec = LLM_mean_num_sv
multp = 0

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

steer_vec = LLM_mean_num_sv
multp = 10

for prompt in months_samps:
    print(prompt)
    steer_by_LLM_lastTok(prompt, steer_vec, multp)

"""## steer ablated recon in SAE space"""

words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    numword_samps.append(samp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 1

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 3

for prompt in numword_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months_samps = []
for i in range(0, 9):
    samp = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
    months_samps.append(samp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months_samps = []
for i in range(0, 5):
    samp = f"{words[i]} {words[i+1]} {words[i+2]}" # {words[i+3]}"
    months_samps.append(samp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 1

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 3

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

steer_vec = abl_sae_LLM_mean_num_sv
multp = 10

for prompt in months_samps:
    print(prompt)
    steer_by_sae_lastTok(prompt, steer_vec, multp)

"""# Ablate directly from SAE steering vec"""

words = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
numword_samps = []
for i in range(0, 9):
    samp = f"{words[i]}"
    numword_samps.append(samp)
inputs = tokenizer(numword_samps, return_tensors="pt", padding=True, truncation=True, max_length=300)['input_ids'].to("cuda")
target_act = gather_residual_activations(model, 20, inputs)
sae_acts_1 = sae.encode(target_act.to(torch.float32))

mean_num_sv = sae_acts_1.mean(dim=0)
mean_num_sv = mean_num_sv.unsqueeze(0)
mean_num_sv.shape

feat_k = 100 # 50
one_top_acts_values, one_top_acts_indices = mean_num_sv[0, -1, :].topk(feat_k, dim=-1)

rank = 1
for val, ind in zip(one_top_acts_values, one_top_acts_indices):
    print(f"Rank {rank}", round(val.item(), 2), ind.item(), feat_labels_lst[ind])
    rank += 1
