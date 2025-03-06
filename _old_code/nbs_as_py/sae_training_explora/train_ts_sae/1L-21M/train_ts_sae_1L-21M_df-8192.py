#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


from google.colab import drive
import shutil

drive.mount('/content/drive')


# In[2]:


try:
    #import google.colab # type: ignore
    #from google.colab import output
    get_ipython().run_line_magic('pip', 'install sae-lens transformer-lens circuitsvis')
except:
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# In[3]:


import torch
import os

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # Training an SAE
# 
# Now we're ready to train out SAE. We'll make a runner config, instantiate the runner and the rest is taken care of for us!
# 
# During training, you use weights and biases to check key metrics which indicate how well we are able to optimize the variables we care about.
# 
# To get a better sense of which variables to look at, you can read my (Joseph's) post [here](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream) and especially look at my weights and biases report [here](https://links-cdn.wandb.ai/wandb-public-images/links/jbloom/uue9i416.html).
# 
# A few tips:
# - Feel free to reorganize your wandb dashboard to put L0, CE_Loss_score, explained variance and other key metrics in one section at the top.
# - Make a [run comparer](https://docs.wandb.ai/guides/app/features/panels/run-comparer) when tuning hyperparameters.
# - You can download the resulting sparse autoencoder / sparsity estimate from wandb and upload them to huggingface if you want to share your SAE with other.
#     - cfg.json (training config)
#     - sae_weight.safetensors (model weights)
#     - sparsity.safetensors (sparsity estimate)

# 
# I've tuned the hyperparameters below for a decent SAE which achieves 86% CE Loss recovered and an L0 of ~85, and runs in about 2 hours on an M3 Max. You can get an SAE that looks better faster if you only consider L0 and CE loss but it will likely have more dense features and more dead features. Here's a link to my output with two runs with two different L1's: https://wandb.ai/jbloom/sae_lens_tutorial .

# In[4]:


model_name = "tiny-stories-1L-21M"
layer_name = "blocks.0.hook_mlp_out"
hook_layer = 0
d_in = 1024
expa_fac = 8
wandb_project = "sae_" + model_name+"_MLP" + str(hook_layer) + "_df-" + str(d_in * expa_fac)


# In[5]:


total_training_steps = 30_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name=model_name,  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name=layer_name,  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    hook_layer=hook_layer,  # Only one layer in the model.
    d_in=d_in,  # the width of the mlp output.
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    is_dataset_tokenized=True,
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor= expa_fac,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
    apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",
    # Training Parameters
    lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project=wandb_project,
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)
# look at the next cell to see some instruction for what to do while this is running.
sparse_autoencoder = SAETrainingRunner(cfg).run()


# ## save

# In[6]:


# Save the trained sparse autoencoder model
save_path = wandb_project + '.pth'
torch.save(sparse_autoencoder.state_dict(), save_path)
print(f'Model saved to {save_path}')


# In[7]:


# faster

drive_save_path = '/content/drive/MyDrive/'+wandb_project+'.pth'
shutil.copy(save_path, drive_save_path)
print(f'Model copied to {drive_save_path}')


# In[8]:


from google.colab import files
files.download(save_path)

