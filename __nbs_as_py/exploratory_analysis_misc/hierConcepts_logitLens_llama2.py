#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


get_ipython().run_cell_magic('capture', '', '%pip install git+https://github.com/neelnanda-io/TransformerLens.git\n')


# In[ ]:


import torch
import numpy as np
from pathlib import Path


# In[ ]:


from transformers import LlamaForCausalLM, LlamaTokenizer
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer


# # Load model

# In[ ]:


get_ipython().system('huggingface-cli login')


# In[ ]:


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True)


# In[ ]:


model = HookedTransformer.from_pretrained(
    LLAMA_2_7B_CHAT_PATH,
    hf_model = hf_model,
    tokenizer = tokenizer,
    device = "cpu",
    fold_ln = False,
    center_writing_weights = False,
    center_unembed = False,
).to("cuda" if torch.cuda.is_available() else "cpu")

del hf_model


# In[ ]:


# Get list of arguments to pass to `generate` (specifically these are the ones relating to sampling)
# generate_kwargs = dict(
#     do_sample = False, # deterministic output so we can compare it to the HF model
#     top_p = 1.0, # suppresses annoying output errors
#     temperature = 1.0, # suppresses annoying output errors
# )

# prompt = "The capital of Germany is"

# output = model.generate(prompt, max_new_tokens=20, **generate_kwargs)

# print(output)


# # Father vs Mother

# ## test prompts

# In[ ]:


example_prompt = "The word for a male parent is "
example_answer = "father"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


example_prompt = "The word for a female parent is "
example_answer = "mother"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# ## Unembed actvs

# In[ ]:


prompts = ["The word for a male parent is ",
           "The word for a female parent is "]
tokens = model.to_tokens(prompts, prepend_bos=True)

model.reset_hooks(including_permanent=True)
original_logits, cache = model.run_with_cache(tokens)
original_logits.shape


# In[ ]:


# last_token_logits = original_logits[:, -1, :]
# values, indices = torch.topk(last_token_logits, 5, dim = -1)
# for token_id in indices[0]:
#     print(model.tokenizer.decode(token_id.item()))


# In[ ]:


# del(orignal_logits)
# del(last_token_logits)
# del values
# del indices


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][0, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][1, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][:, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices[0]:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


list(cache)[0:23]


# In[ ]:


list(cache)[-23:]


# In[ ]:


for layer_name in list(cache)[1:21]:
    print( '.'.join(layer_name.split('.')[2:]) )


# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_mlp_out'][:, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for samp_num in range(len(indices)):
        for token_id in indices[samp_num]:
            print('layer', layer, 'samp', samp_num, model.tokenizer.decode(token_id.item()))


# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for samp_num in range(len(indices)):
        for token_id in indices[samp_num]:
            print('layer', layer, 'samp', samp_num, model.tokenizer.decode(token_id.item()))


# In[ ]:


del last_token_actvs
del unembed_last_token_actvs
del(values)
del(indices)


# In[ ]:


# del cache


# ## Unembed their activation differences

# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][0, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][1, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][1, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][0, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# # Opposites

# ## test prompts

# In[ ]:


example_prompt = "The opposite of left is right. The opposite of tall is"
example_answer = " short"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


example_prompt = "The opposite of left is right. The opposite of tall is"
example_answer = "short"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=False)


# ## Unembed actvs

# In[ ]:


prompts = ["The opposite of left is right. The opposite of tall is",
           "The opposite of left is right. The opposite of short is"]
tokens = model.to_tokens(prompts, prepend_bos=True)

model.reset_hooks(including_permanent=True)
original_logits, cache = model.run_with_cache(tokens)
original_logits.shape


# In[ ]:


# last_token_logits = original_logits[:, -1, :]
# values, indices = torch.topk(last_token_logits, 5, dim = -1)
# for token_id in indices[0]:
#     print(model.tokenizer.decode(token_id.item()))


# In[ ]:


# del(orignal_logits)
# del(last_token_logits)
# del values
# del indices


# In[ ]:


cache


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][0, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][1, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][:, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices[0]:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


# del last_token_actvs
# del unembed_last_token_actvs
# del(values)
# del(indices)


# In[ ]:


# del cache


# In[ ]:


# import sys

# def get_size(obj, seen=None):
#     """Recursively finds size of objects in bytes"""
#     size = sys.getsizeof(obj)
#     if seen is None:
#         seen = set()
#     obj_id = id(obj)
#     if obj_id in seen:
#         return 0
#     seen.add(obj_id)
#     if isinstance(obj, dict):
#         size += sum([get_size(v, seen) for v in obj.values()])
#         size += sum([get_size(k, seen) for k in obj.keys()])
#     elif hasattr(obj, '__dict__'):
#         size += get_size(obj.__dict__, seen)
#     elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
#         size += sum([get_size(i, seen) for i in obj])
#     return size

# for name, obj in sorted(globals().items(), key=lambda x: get_size(x[1]), reverse=True):
#     print(f"{name}: {get_size(obj) / (1024**2):.2f} MB")


# # Animal vs cat

# ## test prompts

# In[ ]:


example_prompt = "A fern is a plant. A rat is an "
example_answer = "animal"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


example_prompt = "A canine animal is called a "
example_answer = "dog"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


example_prompt = "A feline animal is called a "
example_answer = "cat"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# ## Unembed actvs

# In[ ]:


prompts = ["A fern is a plant. A rat is an ",
           "A feline animal is called a "]
tokens = model.to_tokens(prompts, prepend_bos=True)

model.reset_hooks(including_permanent=True)
original_logits, cache = model.run_with_cache(tokens)
original_logits.shape


# In[ ]:


last_token_logits = original_logits[:, -1, :]
values, indices = torch.topk(last_token_logits, 5, dim = -1)
for token_id in indices[1]:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


# del(orignal_logits)
# del(last_token_logits)
# del values
# del indices


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][0, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][1, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][:, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices[0]:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


for layer_name in list(cache)[1:21]:
    print( '.'.join(layer_name.split('.')[2:]) )


# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for samp_num in range(len(indices)):
        for token_id in indices[samp_num]:
            print('layer', layer, 'samp', samp_num, model.tokenizer.decode(token_id.item()))


# In[ ]:


del last_token_actvs
del unembed_last_token_actvs
del(values)
del(indices)


# In[ ]:


# del cache


# ## Unembed their activation differences

# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][0, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][1, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][1, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][0, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# # animal vs dog vs cat

# ## test prompts

# In[ ]:


example_prompt = "A rat is an "
example_answer = "animal"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


example_prompt = "Fern is a plant. Rat is an "
example_answer = "animal"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


example_prompt = "Fern is plant. Rat is an "
example_answer = "animal"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[ ]:


len(model.tokenizer.tokenize('Fern is a plant. Rat is an '))


# In[ ]:


len(model.tokenizer.tokenize('Fern is plant. Rat is an '))


# In[ ]:


len(model.tokenizer.tokenize('A feline animal is called a '))


# ## Unembed actvs

# In[ ]:


prompts = ["Fern is plant. Rat is an ",
           "A feline animal is called a ",
           "A canine animal is called a "]
tokens = model.to_tokens(prompts, prepend_bos=True)

model.reset_hooks(including_permanent=True)
original_logits, cache = model.run_with_cache(tokens)
original_logits.shape


# In[ ]:


last_token_logits = original_logits[:, -1, :]
values, indices = torch.topk(last_token_logits, 5, dim = -1)
for token_id in indices[1]:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


# del(orignal_logits)
# del(last_token_logits)
# del values
# del indices


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][0, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][1, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


last_token_actvs = cache['ln_final.hook_normalized'][:, -1, :]
last_token_actvs = last_token_actvs.unsqueeze(0)
unembed_last_token_actvs = model.unembed(last_token_actvs)
unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
values, indices = torch.topk(unembed_last_token_actvs, 5, dim = -1)
for token_id in indices[0]:
    print(model.tokenizer.decode(token_id.item()))


# In[ ]:


for layer_name in list(cache)[1:21]:
    print( '.'.join(layer_name.split('.')[2:]) )


# In[ ]:


for layer in range(32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for samp_num in range(len(indices)):
        for token_id in indices[samp_num]:
            print('layer', layer, 'samp', samp_num, model.tokenizer.decode(token_id.item()))


# In[ ]:


del last_token_actvs
del unembed_last_token_actvs
del(values)
del(indices)


# In[ ]:


# del cache


# ## Unembed their activation differences

# In[ ]:


# animal - cat
for layer in range(15, 32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][0, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][1, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:


# animal - dog
for layer in range(15, 32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][0, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][2, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:


# cat- animal
for layer in range(15, 32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][1, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][0, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:


# cat - dog
for layer in range(15, 32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][1, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][2, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:


# dog - cat
for layer in range(15, 32):
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][:, -1, :]
    last_token_actvs = cache[f'blocks.{layer}.hook_resid_post'][2, -1, :] - cache[f'blocks.{layer}.hook_resid_post'][1, -1, :]
    last_token_actvs = last_token_actvs.unsqueeze(0).unsqueeze(0)
    unembed_last_token_actvs = model.unembed(last_token_actvs)
    unembed_last_token_actvs = unembed_last_token_actvs.squeeze()
    values, indices = torch.topk(unembed_last_token_actvs, 1, dim = -1)
    for token_id in indices:
        print('layer', layer, model.tokenizer.decode(token_id.item()))


# In[ ]:




