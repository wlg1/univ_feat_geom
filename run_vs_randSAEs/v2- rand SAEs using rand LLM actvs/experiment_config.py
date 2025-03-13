# taken from: https://github.com/ThomasHeap/random_sae

import os
from pathlib import Path

class Config:
    # Paths
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "cache"
    saved_models_dir = base_dir / "saved_models"
    saved_latents_dir = base_dir / "saved_latents"
    saved_eval_dir = base_dir / "saved_eval"

    # Environment variables
    os.environ['HF_HOME'] = str(cache_dir)

    # Model settings
    # model_name = "EleutherAI/pythia-70m-deduped"
    model_name = "EleutherAI/pythia-70m"
    use_step0 = False
    rerandomize = True  # Whether to use rerandomization
    rerandomize_embeddings = False  # Whether to rerandomize embeddings too
    rerandomize_layer_norm = False  # Whether to rerandomize layer norm parameters
    
    # Random training settings
    use_random_control = False  # Flag to toggle random control
    noise_std = 1.0  # Standard deviation for random noise in random mode

    # # Dataset settings
    # dataset = "togethercomputer/RedPajama-Data-1T-Sample"  # Full path format
    # dataset_name = 'plain_text'  # Optional configuration name
    # train_dataset_split = "train[:50%]"
    # test_dataset_split = "train[55%:60%]"
    # text_key = "text"
    # max_tokens = 300_000_000  # 100 million tokens
    
    # Dataset settings
    dataset = "togethercomputer/RedPajama-Data-1T-Sample"  # Full path format
    dataset_name = 'plain_text'  # Optional configuration name
    train_dataset_split = "train[:12%]"
    test_dataset_split = "train[12%:20%]"
    text_key = "text"
    max_tokens = 100_000_000  # 100 million tokens

    # Training settings
    batch_size = 4
    expansion_factor = 64
    normalize_decoder = True
    num_latents = 0
    k = 32
    multi_topk = False
    layer_stride = 1
    dont_eval = False
    
    # Eval setting
    eval_batch_size_prompts = 2
    n_eval_reconstruction_batches = 1000
    n_eval_sparsity_variance_batches = 1000
    
    # Feature extraction settings
    min_examples = 200
    max_examples = 10000
    n_splits = 5

    # Experiment settings
    n_find_max = 1000
    n_examples_train = 40
    n_examples_test = 100
    n_quantiles = 10
    example_ctx_len = 32
    n_random = 100
    train_type = "random"
    test_type = "quantiles"

    # Cache settings
    cache_batch_size = 8
    cache_ctx_len = 256
    cache_n_tokens = 10_000_000

    # Autocorrelation settings
    max_lag_ratio = 0.5

    # Explanation generator settings
    num_parallel_latents = 5
    offline_explainer = False
    
    random_seed = 42
    
    
    use_embedding_sae = False  

    @property
    def device_map(self):
        """Get the device mapping configuration"""
        return {"": "cuda"}

    @property
    def torch_dtype(self):
        """Get the torch dtype to use"""
        return "bfloat16"
    
    @property
    def save_directory(self):
        """Get the save directory using the automatically generated run name"""
        return self.saved_models_dir / self.run_name
    
    @property
    def eval_directory(self):
        """Get the evaluation directory using the automatically generated run name"""
        return self.saved_eval_dir / self.run_name
    
    @property
    def latents_directory(self):
        """Get the directory for saving latent features"""
        return self.saved_latents_dir / f"latents_{self.run_name}"
    
    @property
    def dataset_short_name(self):
        """Get the dataset name without organization prefix"""
        # Split on '/' and get the last part
        base_name = self.dataset.split('/')[-1].lower()
        # If there's a config, append it
        if self.dataset_name:
            base_name = f"{base_name}_{self.dataset_name}"
        return base_name

    @property
    def model_short_name(self):
        """Get the model name without organization prefix"""
        return self.model_name.split('/')[-1].lower()

    @property
    def tokenized_dataset_path(self):
        """Automatically generate path for tokenized dataset"""
        sanitized_name = self.dataset_short_name.replace('-', '_')
        return self.cache_dir / f"{self.model_name.split('/')[0]}" / "tokenized" / f"{sanitized_name}"

    @property
    def run_name(self):
        """Automatically generate run name based on settings"""
        if self.rerandomize:
            init_strategy = "rerandomised"
            if self.rerandomize_embeddings:
                init_strategy += "_embeddings"
        elif self.use_step0:
            init_strategy = "step0"
        elif self.use_random_control:
            init_strategy = "random_control"
        else:
            init_strategy = "trained"
        
        token_count = self.max_tokens // 1_000_000
        return f"{self.model_short_name}_{self.expansion_factor}_k{self.k}/{self.dataset_short_name}_{token_count}M_{init_strategy}"

    def get_dataset_args(self):
        """Get the correct arguments for loading the dataset"""
        if self.dataset_name is not None:
            return {
                "path": self.dataset,
                "name": self.dataset_name,
                "split": self.train_dataset_split,
            }
        else:
            return {
                "path": self.dataset,
                "split": self.train_dataset_split,
            }

config = Config()