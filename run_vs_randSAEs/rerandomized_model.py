# taken from: https://github.com/ThomasHeap/random_sae

import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel

class RerandomizedModel:
    """Wrapper for models that rerandomizes parameters while preserving distribution statistics"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        rerandomize_embeddings: bool = False,
        rerandomize_layer_norm: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize rerandomized model wrapper
        
        Args:
            model: Base model to rerandomize
            rerandomize_embeddings: Whether to rerandomize embedding layers
            rerandomize_layer_norm: Whether to rerandomize layer normalization parameters
            seed: Random seed for reproducibility
        """
        self.model = model
        if seed is not None:
            torch.manual_seed(seed)
            
        # Temporarily store embeddings if we're preserving them
        original_embeddings = {}
        if not rerandomize_embeddings:
            for name, param in model.named_parameters():
                if "embed" in name.lower():
                    original_embeddings[name] = param.data.clone()
        
        # Rerandomize parameters
        for name, param in self.model.named_parameters():
            # Skip embeddings if we're preserving them
            if not rerandomize_embeddings and "embed" in name.lower():
                continue
                
            # Handle layer norm parameters
            is_layer_norm = any(norm_type in name.lower() 
                              for norm_type in ['layernorm', 'layer_norm', 'ln_'])
            if is_layer_norm and not rerandomize_layer_norm:
                continue
                
            # Calculate mean and std of the parameter
            mean = param.data.mean()
            std = param.data.std()
            
            # Generate new random values with same shape and statistics
            new_values = torch.randn_like(param.data) * std + mean
            param.data.copy_(new_values)
        
        # Restore embeddings if needed, then clear the temporary storage
        if not rerandomize_embeddings:
            for name, orig_data in original_embeddings.items():
                param = dict(self.model.named_parameters())[name]
                param.data.copy_(orig_data)
            del original_embeddings  # Clean up temporary storage
    
    def __getattr__(self, name):
        """Delegate attribute access to base model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)