#!/usr/bin/env python
# coding: utf-8

# This notebook obtains the activations of GPT-2 using data from the Pile and trains an SAE on them. It then takes activation differences of GPT-2 to obtain a steering vector and decomposes this steering vector.
# 
# The code is not efficient as it is for brainstorming purposes only to get a sense of how to code the more sophisticated experiments in this project.
# 
# For testing purposes, we start with small datasets and SAEs. Next, we will test this on more data and larger models by finding more efficient ways to deal with out-of-memory issues

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


# # Load Model

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

# ## Load training data

# Future code will do this more efficient (not passing in batch all at once to get h)

# In[ ]:


dataset = load_dataset("stas/openwebtext-10k", split='train', streaming=True)
# dataset = load_dataset("EleutherAI/pile", split='train', streaming=True)


# In[ ]:


total_len = 0
i = 0
for sample in dataset:
    total_len += len(sample["text"])
    i += 1
    # if i == 1000:
    #     break
print(total_len / i)


# In[ ]:


strMaxLen = 100 # 100
batchLen = 100 # 1000
batch_input = []
for sample in dataset:
    input_sample = sample["text"][:strMaxLen]
    batch_input.append(input_sample)
    if len(batch_input) == batchLen:
        break
print(len(batch_input))
# print(input_sample)


# ## Get activations to train SAE

# In[ ]:


layer_name = 'blocks.5.mlp.hook_post'


# In[ ]:


# https://neelnanda-io.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html

tokens = model.to_tokens(batch_input)
tokens.shape


# Seq Len is number of tokens, not string max len

# In[ ]:


# h_store = t.zeros(model_cache['blocks.5.mlp.hook_post'].shape, device=model.cfg.device)
seqLen = tokens.shape[1]
h_store = t.zeros((len(batch_input), seqLen, model.cfg.d_mlp), device=model.cfg.device)


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

# ### one sample

# In[ ]:


# input_text = "I think you're"
# logits, model_cache = model.run_with_cache(input_text, remove_batch_dim=True)
# h = model_cache['blocks.5.mlp.hook_post']  # (batch size, seqLen, n_hidden)

# # convert to h dim: "batch_size * seq_len n_instances n_input_ae"
# print(h.shape)
# h = h.unsqueeze(1)
# print(h.shape)


# In[ ]:


# ae_cfg = AutoEncoderConfig(
#     n_instances = 1, # 8
#     n_input_ae = h.shape[-1],  # model's n_hidden
#     n_hidden_ae = 2 * h.shape[-1],  # require n_hidden_ae >= n_features
#     l1_coeff = 0.5,
# )

# autoencoder = AutoEncoder(ae_cfg, h)

# data_log = autoencoder.optimize(
#     steps = 1000,
#     log_freq = 200,
# )


# ### on more samples and instances

# In[ ]:


# # pass multiple inputs
# batch_input = ["deception", "anger"]
# logits, model_cache_2 = model.run_with_cache(batch_input, remove_batch_dim=False)
# h = model_cache_2['blocks.5.mlp.hook_post']
# h.shape  # (batch size, seqLen, n_hidden)

# # convert to h dim: "batch_size * seq_len, n_instances, n_input_ae"
# print(h.shape)
# h = h.reshape(6, 3072)
# h = h.unsqueeze(1)
# print(h.shape)


# In[ ]:


# ae_cfg = AutoEncoderConfig(
#     n_instances = 2, # 8
#     n_input_ae = h.shape[-1],  # model's n_hidden
#     n_hidden_ae = 2 * h.shape[-1],  # require n_hidden_ae >= n_features. can use R * n_input_ae
#     l1_coeff = 0.5,
# )

# autoencoder = AutoEncoder(ae_cfg, h)

# data_log = autoencoder.optimize(
#     steps = 1000, # 10_000
#     log_freq = 200,
# )


# ### on larger dataset

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


# In[ ]:


ae_cfg = AutoEncoderConfig(
    n_instances = 2, # 8
    n_input_ae = h.shape[-1],  # model's n_hidden
    n_hidden_ae = 2 * h.shape[-1],  # require n_hidden_ae >= n_features. can use R * n_input_ae
    l1_coeff = 0.5,
)

autoencoder = AutoEncoder(ae_cfg, h)

data_log = autoencoder.optimize(
    steps = 1000, # 10_000
    log_freq = 200,
)


# ## Reconstruction loss

# In[ ]:


# batch_input = ["deception", "anger"]
all_tokens = model.to_tokens(batch_input, prepend_bos=True)
all_tokens = all_tokens.to(device)
all_tokens.shape


# In[ ]:


@t.no_grad()
def get_reconstruction_loss(
    tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    layer_name: str = 'blocks.5.mlp.hook_post',
) -> Tuple[float, float]:
    '''
    Returns the reconstruction loss of each autoencoder instance on the given batch of tokens (i.e.
    the L2 loss between the activations and the autoencoder's reconstructions, averaged over all tokens).
    '''
    batch_size, seq_len = tokens.shape

    # layer_name = "blocks.5.mlp.hook_post"

    logits, cache = model.run_with_cache(tokens, names_filter = [layer_name])
    post = cache[layer_name]
    assert post.shape == (batch_size, seq_len, model.cfg.d_mlp)

    post_reshaped = einops.repeat(post, "batch seq d_mlp -> (batch seq) instances d_mlp", instances=2)
    # assert post_reshaped.shape == (batch_size * seq_len, 2, model.cfg.d_mlp)

    _, l2_loss, _, _, post_reconstructed = autoencoder.forward(post_reshaped)
    # assert l2_loss.shape == (batch_size * seq_len, 2) # shape is [datapoints n_instances=2]
    # assert post_reconstructed.shape == (batch_size * seq_len, 2, model.cfg.d_mlp) # shape is [datapoints n_instances=2 d_mlp]

    # Print out the avg L2 norm of activations
    print("Avg L2 norm of acts: ", einops.reduce(post_reshaped.pow(2), "batch inst d_mlp -> inst", "mean").tolist())
    # Print out the cosine similarity between original neuron activations & reconstructions (averaged over neurons)
    print("Avg cos sim of neuron reconstructions: ", t.cosine_similarity(post_reconstructed, post_reshaped, dim=0).mean(-1).tolist())

    return l2_loss.mean(0).tolist()


reconstruction_loss = get_reconstruction_loss(all_tokens[:10], model, autoencoder, layer_name)
print(reconstruction_loss)


# # Get top samples for a feature

# In[ ]:


# batch_input = ["deception", "anger"]
all_tokens = model.to_tokens(batch_input, prepend_bos=True)
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


# # Find features that actv highest for sample X

# In[ ]:





# ## Test on features from class X

# # Steering Vector decomposition

# In[ ]:


# do this b/c anger is one token, calm is 2, so this pads anger with 50256
batch_input = ["anger", "calm"]
tokens = model.to_tokens(batch_input)
tokens


# In[ ]:


seqLen = tokens.shape[1]
h_store = t.zeros((1, seqLen, model.cfg.d_mlp), device=model.cfg.device)

model.run_with_hooks(
    tokens[0],
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)

neg_h = t.clone(h_store)


# In[ ]:


seqLen = tokens.shape[1]
h_store = t.zeros((1, seqLen, model.cfg.d_mlp), device=model.cfg.device)

model.run_with_hooks(
    tokens[1],
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)

pos_h = t.clone(h_store)


# In[ ]:


steer_vec = neg_h - pos_h
steer_vec.shape


# In[ ]:


# get LLM activs for steering vec
post_reshaped = einops.repeat(steer_vec, "batch seq d_mlp -> (batch seq) instances d_mlp", instances=2)
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


all_tokens = model.to_tokens(batch_input, prepend_bos=True)
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

