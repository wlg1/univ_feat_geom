#!/usr/bin/env python
# coding: utf-8

# This notebook obtains the activations of GPT-2 MLP0 and passes tokens belonging to ordered sequences through them.
# 
# We use this code instead of SAElens as the code for the SAEs is more explicitly shown in here.

# 
# 
# > we perform a case study on the attention head (L12H0) with the maximal
# successor score in Pythia-1.4B
# 
# 

# # Setup

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install transformer_lens\n!pip install datasets\n!pip install zstandard\n')


# In[ ]:


from transformer_lens import utils, HookedTransformer, ActivationCache
from dataclasses import dataclass
import torch as t
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
import einops
from datasets import load_dataset

from tqdm import tqdm
from rich.table import Table
from rich import print as rprint


# ## Load Model

# In[ ]:


device = t.device("cuda" if t.cuda.is_available() else "cpu")


# In[ ]:


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    # "gpt2-xl",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)


# # Autoencoder Training

# ## Class Setup

# In[ ]:


@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False
    weight_normalize_eps: float = 1e-8


# In[ ]:


def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


# In[ ]:


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig, h):
        super().__init__()
        self.cfg = cfg

        self.model_h = h

        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))

        self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))

        self.to(device)

    def normalize_and_return_W_dec(self) -> Float[Tensor, "n_instances n_hidden_ae n_input_ae"]:
        '''
        If self.cfg.tied_weights = True, we return the normalized & transposed encoder weights.
        If self.cfg.tied_weights = False, we normalize the decoder weights in-place, and return them.

        Normalization should be over the `n_input_ae` dimension, i.e. each feature should have a noramlized decoder weight.
        '''
        if self.cfg.tied_weights:
            return self.W_enc.transpose(-1, -2) / (self.W_enc.transpose(-1, -2).norm(dim=1, keepdim=True) + self.cfg.weight_normalize_eps)
        else:
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=2, keepdim=True) + self.cfg.weight_normalize_eps)
            return self.W_dec

    def forward(self, h: Float[Tensor, "batch_size n_instances n_input_ae"]):

        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, self.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean(-1) # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @t.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size n_instances n_input_ae"],
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> Tuple[List[List[str]], str]:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        pass # See below for a solution to this function

    def optimize(
        self,
        # model: Model,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        '''
        Optimizes the autoencoder using the given hyperparameters.

        This function should take a trained model as input.
        '''
        if neuron_resample_window is not None:
            assert (dead_neuron_window is not None) and (dead_neuron_window < neuron_resample_window)

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"W_enc": [], "W_dec": [], "colors": [], "titles": [], "frac_active": []}
        colors = None
        title = "no resampling yet"

        for step in progress_bar:

            # Resample dead neurons
            # if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
            #     # Get the fraction of neurons active in the previous window
            #     frac_active_in_window = t.stack(frac_active_list[-neuron_resample_window:], dim=0)
            #     # Compute batch of hidden activations which we'll use in resampling
            #     batch = model.generate_batch(batch_size)
            #     h = einops.einsum(
            #         batch, model.W,
            #         "batch_size instances features, instances hidden features -> batch_size instances hidden"
            #     )
            #     # Resample
            #     colors, title = self.resample_neurons(h, frac_active_in_window, neuron_resample_scale)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            ### MODIFY THIS to use h,  activations from transformerlens ###
            # Get a batch of hidden activations from the model
            # with t.inference_mode():
                # features = model.generate_batch(batch_size)
                # h = einops.einsum(
                #     features, model.W,
                #     "... instances features, instances hidden features -> ... instances hidden"
                # )

            h = self.model_h

            # Optimize
            l1_loss, l2_loss, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate the mean sparsities over batch dim for each (instance, feature)
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(l1_loss=self.cfg.l1_coeff * l1_loss.mean(0).sum().item(), l2_loss=l2_loss.mean(0).sum().item(), lr=step_lr)
                data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                data_log["W_dec"].append(self.normalize_and_return_W_dec().detach().cpu().clone())
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}: {title}")
                data_log["frac_active"].append(frac_active.detach().cpu().clone())

        return data_log


# Return a dictionary `data_log` containing data which is useful for visualizing the training process

# ## Training data

# > We train the SAE using number tokens from 0 to 500, both with and without a space (‘123’ and
# ‘ 123’), alongside other tasks, such as number words, cardinal words, days, months, etc. 90% of
# these tokens go into the train set, and the remaining 10% to the validation set. Even with the other
# tasks, the dataset is dominated by numbers, but creating a more balanced dataset would give us less
# data to work with, and without enough data, the SAE fails to generalize to the validation set. Hence,
# we only concern ourselves with the features that the SAE learns for number tokens, and we then
# separately check whether these features generalize to the other tasks on the basis of logits, rather
# than SAE activations.
# 
# 

# In[ ]:


input_as_str = [str(i) for i in range(500)]


# Future code will do this more efficient (not passing in batch all at once to get h)

# In[ ]:


# dataset = load_dataset("stas/openwebtext-10k", split='train', streaming=True)
# # dataset = load_dataset("EleutherAI/pile", split='train', streaming=True)


# In[ ]:


# total_len = 0
# i = 0
# for sample in dataset:
#     total_len += len(sample["text"])
#     i += 1
#     # if i == 1000:
#     #     break
# print(total_len / i)


# In[ ]:


# strMaxLen = 100 # 100
# batchLen = 100 # 1000
# batch_input = []
# for sample in dataset:
#     input_sample = sample["text"][:strMaxLen]
#     batch_input.append(input_sample)
#     if len(batch_input) == batchLen:
#         break
# print(len(batch_input))
# # print(input_sample)


# ## Get activations to train SAE

# In[ ]:


layer_name = 'blocks.0.mlp.hook_post'


# In[ ]:


# https://neelnanda-io.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html

tokens = model.to_tokens(input_as_str)
tokens.shape


# Seq Len is number of tokens, not string max len

# In[ ]:


# h_store = t.zeros(model_cache['blocks.5.mlp.hook_post'].shape, device=model.cfg.device)
seqLen = tokens.shape[1]
h_store = t.zeros((len(input_as_str), seqLen, model.cfg.d_mlp), device=model.cfg.device)


# In[ ]:


h_store.shape


# Use hook fn to avoid storing all activations

# In[ ]:


def store_h_hook(
    pattern: Float[Tensor, "batch seqlen dmlp"],
    # hook: HookPoint,
    hook
):
    # Store the result.
    # h_store = pattern  # this won't work b/c replaces entire thing, so won't be stored
    # h_store.append(1) # if h_store = [], this will work
    h_store[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[ ]:


model.run_with_hooks(
    tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# h_store  # check actvs are stored


# ## Train SAE

# In[ ]:


# convert to h dim: "batch_size * seq_len, n_instances, n_input_ae"
print(h_store.shape)
h_store = h_store.reshape(h_store.shape[0] * h_store.shape[1], 3072)
h_store = h_store.unsqueeze(1)
print(h_store.shape)


# In[ ]:


# h_store has "grad_fn=<UnsqueezeBackward0>)", so get rid of it
h = h_store.detach()  # Detaches values from the computation graph
# h


# >

# In[ ]:


ae_cfg = AutoEncoderConfig(
    n_instances = 2, # 8
    n_input_ae = h.shape[-1],  # model's n_hidden
    n_hidden_ae = 2 * h.shape[-1],  # require n_hidden_ae >= n_features. can use R * n_input_ae
    l1_coeff = 0.5,
)

autoencoder = AutoEncoder(ae_cfg, h)

data_log = autoencoder.optimize(
    steps = 100, # 10_000
    log_freq = 200,
)


# # Find features that actv highest for sample X

# Most important means highest change in output after ablating. But here, we look for highest activations on these tokens. However, this doesn't mean much because certain features may fire highly for all numbers in general! So use the paper's definition of 'most important'

# ## Test on features from class 3

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('3')]
mod_10_class_3


# In[ ]:


all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


h_store = t.zeros((10, 2, model.cfg.d_mlp), device=model.cfg.device)

model.run_with_hooks(
    all_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(h_store, "batch seq d_mlp -> (batch seq) instances d_mlp", instances=2)
post_reshaped.shape


# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
feat_k = 5
top_acts_values, top_acts_indices = acts.topk(feat_k, dim=-1)


# In[ ]:


top_acts_indices


# In[ ]:


all_tokens = model.to_tokens(input_as_str, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
for feature_idx in top_acts_indices[0][0]:
    feature_idx = feature_idx.item()
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(all_tokens, model, autoencoder, feature_idx,
                                                            autoencoder_B=False, k=samp_m, layer_name=layer_name)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, all_tokens)


# ## Test on features from class 6

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('6')]
mod_10_class_3


# In[ ]:


# all_tokens = model.to_tokens(input_as_str, prepend_bos=True)

all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


h_store = t.zeros((10, 2, model.cfg.d_mlp), device=model.cfg.device)

model.run_with_hooks(
    all_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(h_store, "batch seq d_mlp -> (batch seq) instances d_mlp", instances=2)
post_reshaped.shape


# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
feat_k = 5
top_acts_values, top_acts_indices = acts.topk(feat_k, dim=-1)


# In[ ]:


top_acts_indices


# In[ ]:


all_tokens = model.to_tokens(input_as_str, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5
for feature_idx in top_acts_indices[0][0]:
    feature_idx = feature_idx.item()
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(all_tokens, model, autoencoder, feature_idx,
                                                            autoencoder_B=False, k=samp_m, layer_name=layer_name)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, all_tokens)


# # Get top samples for a feature

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('3')]
mod_10_class_3


# In[ ]:


# all_tokens = model.to_tokens(input_as_str, prepend_bos=True)

all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


@t.inference_mode()
def highest_activating_tokens(
    tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    feature_idx: int,
    autoencoder_B: bool = False,
    k: int = 10,
    layer_name: str = 'blocks.5.mlp.hook_post',
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    batch_size, seq_len = tokens.shape
    instance_idx = 1 if autoencoder_B else 0

    # Get the LLM model's post activations from the clean run
    cache = model.run_with_cache(tokens, names_filter=[layer_name])[1]
    post = cache[layer_name]
    post_reshaped = einops.rearrange(post, "batch seq d_mlp -> (batch seq) d_mlp")

    # Compute SAE activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    h_cent = post_reshaped - autoencoder.b_dec[instance_idx]
    acts = einops.einsum(
        h_cent, autoencoder.W_enc[instance_idx, :, feature_idx],
        "batch_size n_input_ae, n_input_ae -> batch_size"
    )

    # Get the top k SAE largest activations for that SAE feature
    top_acts_values, top_acts_indices = acts.topk(k)

    # Convert the indices into (batch, seq) indices
    top_acts_batch = top_acts_indices // seq_len
    top_acts_seq = top_acts_indices % seq_len

    return t.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values


def display_top_sequences(top_acts_indices, top_acts_values, tokens):
    table = Table("Sequence", "Activation", title="Tokens which most activate feature " + str(feature_idx))
    # indices is that highest token in (sampNum, pos) pair
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # Get the sequence as a string (with some padding on either side of our sequence)
        seq = ""
        # around the token's pos as center, loop thru window of 10, or bounds of sequence
        for i in range(max(seq_idx-5, 0), min(seq_idx+5, all_tokens.shape[1])):
            # the curr token in the loop thru the window of the seq
            new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n")
            # Highlight the token with the high activation
            if i == seq_idx: new_str_token = f"[b u dark_orange]{new_str_token}[/]"
            # add all tokens in len-10 window to the row to display
            seq += new_str_token
        # Print the sequence, and the activation value
        table.add_row(seq, f'{value:.2f}')
    rprint(table)

tokens = all_tokens[:200]
feature_idx = 0
k = 5 # all_tokens.shape[0]
top_acts_indices, top_acts_values = highest_activating_tokens(tokens, model, autoencoder, feature_idx, autoencoder_B=False, k=k, layer_name=layer_name)
display_top_sequences(top_acts_indices, top_acts_values, tokens)


# In[ ]:


autoencoder.W_enc.shape


# In[ ]:


# for feature_idx in range(model.cfg.d_mlp*2):
for feature_idx in range(3):
    top_acts_indices, top_acts_values = highest_activating_tokens(tokens, model, autoencoder, feature_idx, autoencoder_B=False, k=k, layer_name=layer_name)
    display_top_sequences(top_acts_indices, top_acts_values, tokens)

for feature_idx in range(model.cfg.d_mlp*2 -3, model.cfg.d_mlp*2):
    top_acts_indices, top_acts_values = highest_activating_tokens(tokens, model, autoencoder, feature_idx, autoencoder_B=False, k=k, layer_name=layer_name)
    display_top_sequences(top_acts_indices, top_acts_values, tokens)


# # Find most impt features

# Most important: highest change in output probability after ablation

# ## Ablate a SAE feature, then reconstruct

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('3')]
mod_10_class_3


# In[ ]:


all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


h_store = t.zeros((10, 2, model.cfg.d_mlp), device=model.cfg.device)

model.run_with_hooks(
    all_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(h_store, "batch seq d_mlp -> (batch seq) instances d_mlp", instances=2)
post_reshaped.shape


# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# ablate a feature by setting it to 0
acts[:, :, 0] = 0
# acts[:, :, 0]


# In[ ]:


# reconstruct the output

# _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped) # this doesn't use the ablated actvs

h_reconstructed = einops.einsum(
            acts, autoencoder.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + autoencoder.b_dec


# In[ ]:


h_reconstructed.shape


# This is the output of two SAEs. We only need one, so `h_reconstructed[:, 0, :]`. Then, we can rearrange it to LLM dims (before, could not do this with two SAEs).

# In[ ]:


h_reconstructed_1 = h_reconstructed[:, 0, :]
h_reconstructed_1.shape


# In[ ]:


LLM_patch = einops.rearrange(h_reconstructed_1, "(batch seq) d_mlp -> batch seq d_mlp", batch=10)
LLM_patch.shape


# ## Replace LLM actvs with decoder output

# In[ ]:


# replace LLM actvs in that layer with decoder output

from transformer_lens.hook_points import HookPoint
from functools import partial

layer_name = 'blocks.0.mlp.hook_post'

def patch_mlp_vectors(
    orig_MLP_vector: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    LLM_patch: Float[Tensor, "batch pos d_model"],
    layer_to_patch: int,
) -> Float[Tensor, "batch pos d_model"]:
    if layer_to_patch == hook.layer():
        orig_MLP_vector[:, :, :] = LLM_patch
    return orig_MLP_vector

hook_fn = partial(
        patch_mlp_vectors,
        LLM_patch=LLM_patch,
        layer_to_patch = 0
    )

# if you use run_with_cache, you need to add_hook before
# if you use run_with_hooks, you dont need add_hook, just add it in fwd_hooks arg

# rerun clean inputs on ablated model
ablated_logits = model.run_with_hooks(all_tokens,
                    fwd_hooks=[
                        (layer_name, hook_fn),
                    ]
                )


# # Find preds after ablation

# In[ ]:


model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

orig_logits = model(all_tokens)


# In[ ]:


next_token = orig_logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# In[ ]:


next_token = ablated_logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# NOTE: successor head can turn ‘3’ into ‘4’, but the model itself will turn ‘3’ into ‘.’ if there is no sequence in the prompt.

# In[ ]:


example_prompt = "3"
example_answer = " 4"
# need prepend_bos=False to prev adding EOS token in front
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=False)


# ## Find highest logit diff

# See if '3' predict '4' after putting it through the successor head now. Find the correct ID of '4'

# In[ ]:


corrTok_ID = model.to_tokens('4', prepend_bos=False).item()
corrTok_ID


# In[ ]:


ablated_logits_corrTok: Float[Tensor, "batch"] = ablated_logits[:, -1, corrTok_ID]
ablated_logits_corrTok.shape


# In[ ]:


orig_logits_corrTok: Float[Tensor, "batch"] = orig_logits[:, -1, corrTok_ID]
orig_logits_corrTok.shape


# In[ ]:


answer_logit_diff = orig_logits_corrTok - ablated_logits_corrTok
answer_logit_diff


# In[ ]:


ablated_logits_corrTok


# In[ ]:


# def get_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], dataset: Dataset, per_prompt=False):
#     '''
#     '''
#     corr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.corr_tokenIDs]
#     incorr_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.incorr_tokenIDs]
#     answer_logit_diff = corr_logits - incorr_logits
#     return answer_logit_diff if per_prompt else answer_logit_diff.mean()


# In[ ]:


# heads_not_ablate = []  # ablate all heads but not MLPs
# mlps_not_ablate = []  # ablate all MLPs

# # ablate

# model.reset_hooks(including_permanent=True)  #must do this after running with mean ablation hook

# model = add_ablation_hook_MLP_head(model, dataset_2, heads_not_ablate, mlps_not_ablate)
# logits_minimal = model(dataset.toks)

# # find logit diff
# new_score = get_logit_diff(logits_minimal, dataset)


# # OV Scores with just successor head

# ## Unablated

# In[ ]:


import torch
layer, head = 9, 1
input_text = '3'

cache = {}
model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
model(input_text)
# z_0 = model.blocks[1].attn.ln1(cache["blocks.0.hook_resid_post"])
z_0 = model.blocks[1].attn.hook_z(cache["blocks.0.hook_resid_post"])

v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

o = torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
logits = model.unembed(model.ln_final(o))


# In[ ]:


# pred_tokens = [
#                 model.tokenizer.decode(token)
#                 for token in torch.topk(
#                     logits[seq_idx, dataset.word_idx[word][seq_idx]], k
#                 ).indices
#             ]


# In[ ]:


next_token = logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# ## Ablated

# In[ ]:


cache["blocks.0.hook_resid_post"].shape


# In[ ]:


LLM_patch.shape


# In[ ]:


print(model.cfg.d_mlp)
model.cfg.d_model


# OV scores may not use the same approach as successor heads.
# 
# 'blocks.0.mlp.hook_post'
# 
# vs
# 
# "blocks.0.hook_resid_post"

# # Re-train SAE on blocks.0.hook_resid_post activations

# ## Get activations to train SAE

# In[ ]:


layer_name = 'blocks.0.hook_resid_post'


# In[ ]:


# https://neelnanda-io.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html

tokens = model.to_tokens(input_as_str)
tokens.shape


# Seq Len is number of tokens, not string max len

# In[ ]:


# h_store = t.zeros(model_cache['blocks.5.mlp.hook_post'].shape, device=model.cfg.device)
seqLen = tokens.shape[1]
h_store = t.zeros((len(input_as_str), seqLen, model.cfg.d_model), device=model.cfg.device)


# In[ ]:


h_store.shape


# Use hook fn to avoid storing all activations

# In[ ]:


def store_h_hook(
    pattern: Float[Tensor, "batch seqlen d_model"],
    # hook: HookPoint,
    hook
):
    # Store the result.
    # h_store = pattern  # this won't work b/c replaces entire thing, so won't be stored
    # h_store.append(1) # if h_store = [], this will work
    h_store[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[ ]:


model.run_with_hooks(
    tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# h_store  # check actvs are stored


# ## Train SAE

# In[ ]:


# convert to h dim: "batch_size * seq_len, n_instances, n_input_ae"
print(h_store.shape)
h_store = h_store.reshape(h_store.shape[0] * h_store.shape[1], model.cfg.d_model)
h_store = h_store.unsqueeze(1)
print(h_store.shape)


# In[ ]:


# h_store has "grad_fn=<UnsqueezeBackward0>)", so get rid of it
h = h_store.detach()  # Detaches values from the computation graph
# h


# >

# In[ ]:


ae_cfg = AutoEncoderConfig(
    n_instances = 2, # 8
    n_input_ae = h.shape[-1],  # model's n_hidden
    n_hidden_ae = 2 * h.shape[-1],  # require n_hidden_ae >= n_features. can use R * n_input_ae
    l1_coeff = 0.5,
)

autoencoder = AutoEncoder(ae_cfg, h)

data_log = autoencoder.optimize(
    steps = 100, # 10_000
    log_freq = 200,
)


# ## Ablate a SAE feature, then reconstruct

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('3')]
mod_10_class_3


# In[ ]:


all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


h_store = t.zeros((10, 2, model.cfg.d_model), device=model.cfg.device)

model.run_with_hooks(
    all_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(h_store, "batch seq d_model -> (batch seq) instances d_model", instances=2)
post_reshaped.shape


# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# ablate a feature (idx = 0) by setting it to 0
acts[:, :, 0] = 0
# acts[:, :, 0]


# In[ ]:


# reconstruct the output

# _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped) # this doesn't use the ablated actvs

h_reconstructed = einops.einsum(
            acts, autoencoder.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + autoencoder.b_dec


# In[ ]:


h_reconstructed.shape


# This is the output of two SAEs. We only need one, so `h_reconstructed[:, 0, :]`. Then, we can rearrange it to LLM dims (before, could not do this with two SAEs).

# In[ ]:


h_reconstructed_1 = h_reconstructed[:, 0, :]
h_reconstructed_1.shape


# In[ ]:


LLM_patch = einops.rearrange(h_reconstructed_1, "(batch seq) d_model -> batch seq d_model", batch=10)
LLM_patch.shape


# ## Save model and actvs

# In[ ]:


from google.colab import files

# Save the model's state dictionary
model_path = 'autoencoder.pth'
torch.save(autoencoder.state_dict(), model_path)

# Download the model file
files.download(model_path)


# In[ ]:


# Save the tensor
tensor_path = 'LLM_patch.pt'
torch.save(LLM_patch, tensor_path)

# Download the tensor file
files.download(tensor_path)


# ### load sae

# Must run "Get activations to train SAE" of this section before loading to get h_store

# In[ ]:


# convert to h dim: "batch_size * seq_len, n_instances, n_input_ae"
h_store = h_store.reshape(h_store.shape[0] * h_store.shape[1], model.cfg.d_model)
h_store = h_store.unsqueeze(1)
h = h_store.detach()  # Detaches values from the computation graph


# In[ ]:


ae_cfg = AutoEncoderConfig(
    n_instances = 2, # 8
    n_input_ae = model.cfg.d_model,  # model's n_hidden
    n_hidden_ae = 2 * model.cfg.d_model,  # require n_hidden_ae >= n_features. can use R * n_input_ae
    l1_coeff = 0.5,
)

autoencoder_testLoad = AutoEncoder(ae_cfg, h)

# Load the model's state dictionary
model_path = 'autoencoder.pth'
autoencoder_testLoad.load_state_dict(t.load(model_path))


# ### test loading

# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder_testLoad.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# ablate a feature (idx = 0) by setting it to 0
acts[:, :, 0] = 0
# acts[:, :, 0]


# In[ ]:


# reconstruct the output

# _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped) # this doesn't use the ablated actvs

h_reconstructed = einops.einsum(
            acts, autoencoder.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + autoencoder.b_dec


# In[ ]:


h_reconstructed.shape


# This is the output of two SAEs. We only need one, so `h_reconstructed[:, 0, :]`. Then, we can rearrange it to LLM dims (before, could not do this with two SAEs).

# In[ ]:


h_reconstructed_1 = h_reconstructed[:, 0, :]
h_reconstructed_1.shape


# In[ ]:


LLM_patch_testLoad = einops.rearrange(h_reconstructed_1, "(batch seq) d_model -> batch seq d_model", batch=10)
LLM_patch_testLoad.shape


# In[ ]:


torch.equal(LLM_patch, LLM_patch_testLoad)


# In[ ]:


tensor_path = 'LLM_patch.pt'
LLM_patch_testLoad_2 = torch.load(tensor_path)

torch.equal(LLM_patch, LLM_patch_testLoad)


# # OV Scores with just successor head (0.hook_resid)

# ## Ablated

# In[ ]:


import torch
layer, head = 9, 1
input_text = '3'

z_0 = model.blocks[1].attn.hook_z(LLM_patch)

v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

o = torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
ablated_logits = model.unembed(model.ln_final(o))


# In[ ]:


next_token = ablated_logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# # loop through features to ablate

# In[ ]:


autoencoder = autoencoder_testLoad


# ## get feature actvs

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('3')]
mod_10_class_3


# In[ ]:


all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


h_store = t.zeros((10, 2, model.cfg.d_model), device=model.cfg.device)

model.run_with_hooks(
    all_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(h_store, "batch seq d_model -> (batch seq) instances d_model", instances=2)
post_reshaped.shape


# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# Count the number of 0s in the tensor
num_zeros = (acts == 0).sum().item()

# Count the number of nonzeroes in the tensor
num_ones = (acts > 0).sum().item()

# Calculate the percentage of 1s over 0s
if num_zeros > 0:
    percentage_ones_over_zeros = (num_ones / num_zeros) * 100
else:
    percentage_ones_over_zeros = float('inf')  # Handle division by zero

print(f"Number of 0s: {num_zeros}")
print(f"Number of nonzeroes: {num_ones}")
print(f"Percentage of 1s over 0s: {percentage_ones_over_zeros:.2f}%")


# In[ ]:


post_reshaped == output_tuple[-1]


# In[ ]:


# reconstruct the output

# _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped) # this doesn't use the ablated actvs

h_reconstructed = einops.einsum(
            acts, autoencoder.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + autoencoder.b_dec


# In[ ]:


autoencoder.b_dec


# In[ ]:


h_reconstructed


# This is the output of two SAEs. We only need one, so `h_reconstructed[:, 0, :]`. Then, we can rearrange it to LLM dims (before, could not do this with two SAEs).

# In[ ]:


h_reconstructed_1 = h_reconstructed[:, 0, :]
h_reconstructed_1.shape


# In[ ]:


LLM_patch = einops.rearrange(h_reconstructed_1, "(batch seq) d_model -> batch seq d_model", batch=10)
LLM_patch.shape


# In[ ]:


import torch
layer, head = 9, 1
input_text = '3'

z_0 = model.blocks[1].attn.hook_z(LLM_patch)

v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

o = torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
ablated_logits = model.unembed(model.ln_final(o))


# In[ ]:


next_token = ablated_logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# ## Ablate a SAE feature, then reconstruct

# In[ ]:


acts[:, :, 0]


# In[ ]:


# ablate a feature (idx) by setting it to 0

acts_clone = acts.clone().detach()

acts_clone[:, :, 0] = 0
acts[:, :, 0]


# In[ ]:


# reconstruct the output

# _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped) # this doesn't use the ablated actvs

h_reconstructed = einops.einsum(
            acts, autoencoder.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + autoencoder.b_dec


# In[ ]:


h_reconstructed.shape


# This is the output of two SAEs. We only need one, so `h_reconstructed[:, 0, :]`. Then, we can rearrange it to LLM dims (before, could not do this with two SAEs).

# In[ ]:


h_reconstructed_1 = h_reconstructed[:, 0, :]
h_reconstructed_1.shape


# In[ ]:


LLM_patch = einops.rearrange(h_reconstructed_1, "(batch seq) d_model -> batch seq d_model", batch=10)
LLM_patch.shape


# # Train SAE more steps

# In[ ]:


# convert to h dim: "batch_size * seq_len, n_instances, n_input_ae"
print(h_store.shape)
h_store = h_store.reshape(h_store.shape[0] * h_store.shape[1], model.cfg.d_model)
h_store = h_store.unsqueeze(1)
print(h_store.shape)


# In[ ]:


# h_store has "grad_fn=<UnsqueezeBackward0>)", so get rid of it
h = h_store.detach()  # Detaches values from the computation graph
# h


# >

# In[ ]:


ae_cfg = AutoEncoderConfig(
    n_instances = 2, # 8
    n_input_ae = h.shape[-1],  # model's n_hidden
    n_hidden_ae = 2 * h.shape[-1],  # require n_hidden_ae >= n_features. can use R * n_input_ae
    l1_coeff = 0.5,
)

autoencoder = AutoEncoder(ae_cfg, h)

data_log = autoencoder.optimize(
    steps = 10000, # 10_000
    log_freq = 200,
)


# ## get feature actvs

# In[ ]:


mod_10_class_3 = [str(i) for i in range(101) if str(i).endswith('3')]
mod_10_class_3


# In[ ]:


all_tokens = model.to_tokens(mod_10_class_3, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


h_store = t.zeros((10, 2, model.cfg.d_model), device=model.cfg.device)

model.run_with_hooks(
    all_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(h_store, "batch seq d_model -> (batch seq) instances d_model", instances=2)
post_reshaped.shape


# In[ ]:


# use a fwd pass to compute ALL feature actvs for ALL this steering vec
output_tuple = autoencoder.forward(post_reshaped)
acts = output_tuple[3]
acts.shape


# In[ ]:


# Count the number of 0s in the tensor
num_zeros = (acts == 0).sum().item()

# Count the number of nonzeroes in the tensor
num_ones = (acts > 0).sum().item()

# Calculate the percentage of 1s over 0s
if num_zeros > 0:
    percentage_ones_over_zeros = (num_ones / num_zeros) * 100
else:
    percentage_ones_over_zeros = float('inf')  # Handle division by zero

print(f"Number of 0s: {num_zeros}")
print(f"Number of nonzeroes: {num_ones}")
print(f"Percentage of 1s over 0s: {percentage_ones_over_zeros:.2f}%")


# In[ ]:


# reconstruct the output

# _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped) # this doesn't use the ablated actvs

h_reconstructed = einops.einsum(
            acts, autoencoder.normalize_and_return_W_dec(),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + autoencoder.b_dec


# This is the output of two SAEs. We only need one, so `h_reconstructed[:, 0, :]`. Then, we can rearrange it to LLM dims (before, could not do this with two SAEs).

# In[ ]:


h_reconstructed_1 = h_reconstructed[:, 0, :]
h_reconstructed_1.shape


# In[ ]:


LLM_patch = einops.rearrange(h_reconstructed_1, "(batch seq) d_model -> batch seq d_model", batch=10)
LLM_patch.shape


# In[ ]:


import torch
layer, head = 9, 1
input_text = '3'

z_0 = model.blocks[1].attn.hook_z(LLM_patch)

v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

o = torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
ablated_logits = model.unembed(model.ln_final(o))


# In[ ]:


next_token = ablated_logits[0, -1].argmax(dim=-1)
next_char = model.to_string(next_token)
next_char


# 10000 steps: sometimes is all 0, sometimes there are 5 VALUES out of 61435 (0.01%) that are non zero (these aren’t counting rows). This predicts ‘must’. Reconstruction is bad- because the ablation didn’t do much (first row likely all 0s anyways) it’s the reconstruction that messes up.

# In[ ]:


from google.colab import files
model_path = 'autoencoder_2.pth'
torch.save(autoencoder.state_dict(), model_path)
files.download(model_path)

