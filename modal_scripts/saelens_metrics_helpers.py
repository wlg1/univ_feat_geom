"""saelens_metrics_helpers
This utils file abstracts away a few things so that the file I send to modal just does the comparisons.

In each function for getting individual saes, also return weights, activations, etc.


"""


from simSAE_more_metrics_nb_utils_as_py import *
import torch
from datasets import load_dataset
from sae_lens import LanguageModelSAERunnerConfig, SAEConfig, SAE

from transformer_lens import HookedTransformer
from jaxtyping import Float

def get_token_tensor(dataset, tokenizer, batch_size, max_seq_len):
    dataset_iter = iter(dataset)
    tokens_list = []
    for _ in range(batch_size):
        try:
            example = next(dataset_iter)
            text = example['text']
            tokens = tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True,
                                      max_length=max_seq_len)
            # tokens = tokens['encoding']
            tokens_list.append(tokens)
        except StopIteration:
            print("End of dataset")
            break

    if tokens_list:
        tokens_tensor = torch.cat(tokens_list, dim=0).squeeze(1)

        return tokens_tensor
    else:
        return None

def get_token_tensor_in_chunks(dataset_iter, tokenizer, batch_size, max_seq_len):
    # dataset_iter = iter(dataset)
    tokens_list = []
    for _ in range(batch_size):
        try:
            example = next(dataset_iter)
            text = example['text']
            tokens = tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True,
                                      max_length=max_seq_len)
            # tokens = tokens['encoding']
            tokens_list.append(tokens)
        except StopIteration:
            print("End of dataset")
            break

    if tokens_list:
        tokens_tensor = torch.cat(tokens_list, dim=0).squeeze(1)

        return tokens_tensor
    else:
        return None

def highest_activating_tokens_saelens(
    feature_acts,
    feature_idx: int,
    k: int = 10,  # num batch_seq samples
    batch_tokens=None
): # -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    batch_size, seq_len = batch_tokens.shape

    # Get the top k largest activations for only targeted feature
    # need to flatten (batch,seq) into batch*seq first because it's ANY batch_seq, even if in same batch or same pos
    flattened_feature_acts = feature_acts[:, :, feature_idx].reshape(-1)

    top_acts_values, top_acts_indices = flattened_feature_acts.topk(k)
    # top_acts_values should be 1D
    # top_acts_indices should be also be 1D. Now, turn it back to 2D
    # Convert the indices into (batch, seq) indices
    top_acts_batch = top_acts_indices // seq_len
    top_acts_seq = top_acts_indices % seq_len

    return torch.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values

def store_top_toks_saelens(top_acts_indices, top_acts_values, batch_tokens, tokenizer):
    feat_samps = []
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        new_str_token = tokenizer.decode(batch_tokens[batch_idx, seq_idx]).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
        feat_samps.append(new_str_token)
    return feat_samps

