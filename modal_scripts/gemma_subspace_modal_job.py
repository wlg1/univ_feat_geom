import numpy as np
from numpy import dtype
from datetime import datetime

from sympy.crypto import rsa_public_key

from saelens_metrics_helpers import *
from simSAE_more_metrics_nb_utils_as_py import *
import modal

# modal set up here
app = modal.App('gemma_metrics_and_data')
vol = modal.Volume.from_name('saes', create_if_missing=True)
SAES_DIR='/saes'

image = (
    modal.Image
    .debian_slim()
    .apt_install("git")
    .pip_install(
        'numpy==1.24.4',
        "transformers",
        "torch",
        "datasets",
        "git+https://github.com/EleutherAI/sae.git",
        "wandb",
        'matplotlib',
        'scikit-learn',
        'safetensors',
        'huggingface_hub',
        'typing',
        'jaxtyping',
        'natsort',
        'einops',
        'sae-lens',
        'datetime',
    )
)


@app.function(gpu='A100', image=image,
              secrets=[modal.Secret.from_name('my-wandb-secret'),
                       modal.Secret.from_name('my-huggingface-secret')],
              volumes={SAES_DIR: vol},
              timeout=7200)
def generate_data_gemma_2_2b_mlp():
    gemma_2_2b = HookedTransformer.from_pretrained('gemma-2-2b')

    print('loaded gemma 2 2b')

    torch.set_grad_enabled(False)

    batch_size = 150
    max_seq_len = 150

    file_path = f'{SAES_DIR}/gemma_2_2b_activations/'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # example code from an saelens tutorial notebook here: https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb#scrollTo=Qis1IirEl3mm
    # _, cache = model.run_with_cache(
    #     tokens,
    #     stop_at_layer=sae.cfg.hook_layer + 1,
    #     names_filter=[sae.cfg.hook_name]
    # )
    # sae_in = cache[sae.cfg.hook_name]
    # feature_acts = sae.encode(sae_in).squeeze()

    # running with names_filter should only save the cache that we want. this is very helpful for memory.

    # run over all saes to generate. just filter later when doing metrics.
    gemma_scope_2b_pt_mlp_canonical_ids = [
        f'layer_{i}/width_16k/canonical' for i in range(0, 26)
    ]

    gemma_scope_2b_pt_mlp_canonical_saes = []
    gemma_scope_2b_pt_mlp_canonical_saes_hook_names = []
    gemma_scope_saes = {}
    for sae_id in gemma_scope_2b_pt_mlp_canonical_ids:
        sae, cfg, device = SAE.from_pretrained(release='gemma-scope-2b-pt-mlp-canonical',
                                               sae_id=sae_id,
                                               device='cuda')
        # cfg above not same as sae cfg. its overall cfg. easy to trip up there

        gemma_scope_2b_pt_mlp_canonical_saes.append(sae)
        gemma_scope_2b_pt_mlp_canonical_saes_hook_names.append(sae.cfg.hook_name)
        gemma_scope_saes[sae_id] = (sae, sae.cfg.hook_name)


    # test running with batched inputs
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)

    practical_batch_size = 10
    # loop through in chunks of practical_batch_size until hitting declared batch size
    for i in range((batch_size // practical_batch_size)+1):

        for key in gemma_scope_saes.keys():

            layer = key.split('/')[0]

            # if files for both sae and llm activs already exist, skip
            if os.path.exists(f'{SAES_DIR}/gemma_2_2b_activations/gemma_2_2b_{layer}-sae-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy') and \
                os.path.exists(f'{SAES_DIR}/gemma_2_2b_activations/gemma_2_2b_{layer}-llm-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy'):
                print(f'files for {layer} and batch {i*practical_batch_size}-{(i+1) * practical_batch_size} already exist, skipping')
                continue

            names_list = [gemma_scope_saes[key][1]] # second in tuple is the hook name

            tokens_tensor = get_token_tensor_in_chunks(dataset_iter, gemma_2_2b.tokenizer,
                                         batch_size=practical_batch_size, max_seq_len=max_seq_len)
            gemma_2_2b_logits, gemma_2_2b_cache = gemma_2_2b.run_with_cache(tokens_tensor,
                                                                         names_filter=names_list)

            llm_activ = gemma_2_2b_cache[gemma_scope_saes[key][1]]
            sae_activ = gemma_scope_saes[key][0].encode(llm_activ).squeeze().cpu().numpy()
            llm_activ = llm_activ.cpu().numpy()

            # save
            np.save(f'{SAES_DIR}/gemma_2_2b_activations/gemma_2_2b_{layer}-sae-activ_max_seq_len-{max_seq_len}_batch-{i*practical_batch_size}-{(i+1)*practical_batch_size}.npy', sae_activ)
            np.save(f'{SAES_DIR}/gemma_2_2b_activations/gemma_2_2b_{layer}-llm-activ_max_seq_len-{max_seq_len}_batch-{i*practical_batch_size}-{(i+1)*practical_batch_size}.npy', llm_activ)
            print(f'saved {key} activations for batch {i*practical_batch_size}-{(i+1)*practical_batch_size}')

            # clear all old tensors out
            del tokens_tensor
            del gemma_2_2b_logits
            del gemma_2_2b_cache
            del llm_activ
            del sae_activ


@app.function(gpu='A100', image=image,
              secrets=[modal.Secret.from_name('my-wandb-secret'),
                       modal.Secret.from_name('my-huggingface-secret')],
              volumes={SAES_DIR: vol},
              timeout=7200)
def generate_data_gemma_2_2b_res():
    gemma_2_2b = HookedTransformer.from_pretrained('gemma-2-2b')

    print('loaded gemma 2 2b')

    torch.set_grad_enabled(False)

    batch_size = 150
    max_seq_len = 150

    file_path = f'{SAES_DIR}/gemma_2_2b_activations_residual_stream/'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # example code from an saelens tutorial notebook here: https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb#scrollTo=Qis1IirEl3mm
    # _, cache = model.run_with_cache(
    #     tokens,
    #     stop_at_layer=sae.cfg.hook_layer + 1,
    #     names_filter=[sae.cfg.hook_name]
    # )
    # sae_in = cache[sae.cfg.hook_name]
    # feature_acts = sae.encode(sae_in).squeeze()

    # running with names_filter should only save the cache that we want. this is very helpful for memory.

    # run over all saes to generate. just filter later when doing metrics.
    gemma_scope_2b_pt_res_canonical_ids = [
        f'layer_{i}/width_16k/canonical' for i in range(0, 26)
    ]

    gemma_scope_2b_pt_res_canonical_saes = []
    gemma_scope_2b_pt_res_canonical_saes_hook_names = []
    gemma_scope_saes = {}
    for sae_id in gemma_scope_2b_pt_res_canonical_ids:
        sae, cfg, device = SAE.from_pretrained(release='gemma-scope-2b-pt-res-canonical',
                                               sae_id=sae_id,
                                               device='cuda')
        # cfg above not same as sae cfg. its overall cfg. easy to trip up there

        gemma_scope_2b_pt_res_canonical_saes.append(sae)
        gemma_scope_2b_pt_res_canonical_saes_hook_names.append(sae.cfg.hook_name)
        gemma_scope_saes[sae_id] = (sae, sae.cfg.hook_name)


    # test running with batched inputs
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)

    practical_batch_size = 10
    # loop through in chunks of practical_batch_size until hitting declared batch size
    for i in range((batch_size // practical_batch_size)+1):

        for key in gemma_scope_saes.keys():

            layer = key.split('/')[0]

            # if files for both sae and llm activs already exist, skip
            if os.path.exists(f'{SAES_DIR}/gemma_2_2b_activations_residual_stream/gemma_2_2b_{layer}-sae-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy') and \
                os.path.exists(f'{SAES_DIR}/gemma_2_2b_activations_residual_stream/gemma_2_2b_{layer}-llm-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy'):
                print(f'files for {layer} and batch {i*practical_batch_size}-{(i+1) * practical_batch_size} already exist, skipping')
                continue

            names_list = [gemma_scope_saes[key][1]] # second in tuple is the hook name

            tokens_tensor = get_token_tensor_in_chunks(dataset_iter, gemma_2_2b.tokenizer,
                                         batch_size=practical_batch_size, max_seq_len=max_seq_len)
            gemma_2_2b_logits, gemma_2_2b_cache = gemma_2_2b.run_with_cache(tokens_tensor,
                                                                         names_filter=names_list)

            llm_activ = gemma_2_2b_cache[gemma_scope_saes[key][1]]
            sae_activ = gemma_scope_saes[key][0].encode(llm_activ).squeeze().cpu().numpy()
            llm_activ = llm_activ.cpu().numpy()

            # save
            np.save(f'{SAES_DIR}/gemma_2_2b_activations_residual_stream/gemma_2_2b_{layer}-sae-activ_max_seq_len-{max_seq_len}_batch-{i*practical_batch_size}-{(i+1)*practical_batch_size}.npy', sae_activ)
            np.save(f'{SAES_DIR}/gemma_2_2b_activations_residual_stream/gemma_2_2b_{layer}-llm-activ_max_seq_len-{max_seq_len}_batch-{i*practical_batch_size}-{(i+1)*practical_batch_size}.npy', llm_activ)
            print(f'saved {key} activations for batch {i*practical_batch_size}-{(i+1)*practical_batch_size}')

            # clear all old tensors out
            del tokens_tensor
            del gemma_2_2b_logits
            del gemma_2_2b_cache
            del llm_activ
            del sae_activ



@app.function(gpu='H100', image=image,
              secrets=[modal.Secret.from_name('my-wandb-secret'),
                       modal.Secret.from_name('my-huggingface-secret')],
              volumes={SAES_DIR: vol},
              timeout=7200)
def generate_data_gemma_1_2b_res():

    gemma_1_2b = HookedTransformer.from_pretrained('gemma-2b') #note, this is gemma 1 2b. note gemma 2 2b (don't mix up)

    print('loaded gemma 1 2b')

    torch.set_grad_enabled(False)

    batch_size = 150
    max_seq_len = 150

    file_path = f'{SAES_DIR}/gemma_1_2b_activations_residual_stream/'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    gemma_2b_jb_saes_ids = [
        'blocks.0.hook_resid.post',
        'blocks.6.hook_resid.post',
        'blocks.10.hook_resid.post',
        'blocks.12.hook_resid.post',
        'blocks.17.hook_resid.post',
    ]

    gemma_2b_jb_saes_hook_names = []
    gemma_2b_jb_saes = {}

    for sae_id in gemma_2b_jb_saes_ids:
        sae, cfg, device = SAE.from_pretrained(release='gemma-2b-res-jb',
                                               sae_id=sae_id,
                                               device='cuda')
        # cfg above not same as sae cfg. its overall cfg. easy to trip up there

        gemma_2b_jb_saes_hook_names.append(sae.cfg.hook_name)
        gemma_2b_jb_saes[sae_id] = (sae, sae.cfg.hook_name)

    print('loaded all gemma 1 2b saes')

    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)

    practical_batch_size = 10
    # loop through in chunks of practical_batch_size until hitting declared batch size
    for i in range((batch_size // practical_batch_size) + 1):

        for key in gemma_2b_jb_saes.keys():

            layer = key.split('/')[0]

            # if files for both sae and llm activs already exist, skip
            if os.path.exists(f'{SAES_DIR}/gemma_1_2b_activations_residual_stream/gemma_1_2b_{layer}-sae-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy') and \
                os.path.exists(f'{SAES_DIR}/gemma_1_2b_activations_residual_stream/gemma_1_2b_{layer}-llm-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy'):
                print(f'files for {layer} and batch {i*practical_batch_size}-{(i+1) * practical_batch_size} already exist, skipping')
                continue

            names_list = [gemma_2b_jb_saes[key][1]]  # second in tuple is the hook name

            tokens_tensor = get_token_tensor_in_chunks(dataset_iter, gemma_1_2b.tokenizer,
                                                       batch_size=practical_batch_size, max_seq_len=max_seq_len)
            gemma_1_2b_logits, gemma_1_2b_cache = gemma_1_2b.run_with_cache(tokens_tensor,
                                                                            names_filter=names_list)

            llm_activ = gemma_1_2b_cache[gemma_2b_jb_saes[key][1]]
            sae_activ = gemma_2b_jb_saes[key][0].encode(llm_activ).squeeze().cpu().numpy()
            llm_activ = llm_activ.cpu().numpy()

            # save
            np.save(
                f'{SAES_DIR}/gemma_1_2b_activations_residual_stream/gemma_1_2b_{layer}-sae-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy',
                sae_activ)
            np.save(
                f'{SAES_DIR}/gemma_1_2b_activations_residual_stream/gemma_1_2b_{layer}-llm-activ_max_seq_len-{max_seq_len}_batch-{i * practical_batch_size}-{(i + 1) * practical_batch_size}.npy',
                llm_activ)
            print(f'saved {key} activations for batch {i * practical_batch_size}-{(i + 1) * practical_batch_size}')

            del tokens_tensor
            del gemma_1_2b_logits
            del gemma_1_2b_cache
            del llm_activ
            del sae_activ


@app.function(gpu='A10G', image=image,
              secrets=[modal.Secret.from_name('my-wandb-secret'),
                       modal.Secret.from_name('my-huggingface-secret')],
              volumes={SAES_DIR: vol},
              timeout=28800)
def main_sae_sim_metrics_res(run_name: str, base_layer: str, one_to_X: int, target_layer_increments: int = 1):
    """
    Assume here that all weights and activations are stored in the given modal volume. otherwise run the above.

    Bullet list:
    1. Load in activations for sae 1 and sae 2. done
        This needs to be done with some regex in my modal volume because things are stored in increments of ten. will total to 1000
    2. Flatten / reshape the activations. done
    3. Get batched correlations. done
    4. get meta metrics, such as mean actv corr. done
    5. run svcca paired and rsa paired. done
    6. run shuffle rand for pvals and means there. done
    7. save json. done

    Changing to rely on cfg dicts to account for some run errors.
    """

    cfg = {
        'max_seq_len': 150,
        'batch_size': 150,
        'num_runs': 10, # for test
        'base_layer': None, # this should be a layer in gemma 2 2b for test
        'pure_layer_for_file_name': None,
        'base_model': 'gemma_2_2b',
        'target_model': 'gemma_1_2b',
        'target_layer_increments': target_layer_increments, # 1 for every layer, or 2 / 3/ 5 for increments as such
        'one_to_X': None, # this is many to 1, or many to ten, etc
    }

    assert cfg['base_model'] == 'gemma_2_2b' and cfg['target_model'] == 'gemma_1_2b', \
        'for now base and target models must be gemma 2 2b and gemma 1 2b respectively'

    cfg['base_layer'] = base_layer
    cfg['pure_layer_for_file_name'] = base_layer.split('/')[0]
    if one_to_X == -1:
        cfg['one_to_X'] = cfg['max_seq_len'] * cfg['batch_size']
    else:
        cfg['one_to_X'] = one_to_X

    file_path = f'{SAES_DIR}/gemma_2_2b_vs_gemma_1_2b_saes/'
    date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    metrics_dict_filename = f'metrics_dict_res_{run_name}_base-model-{cfg["base_model"]}_target-model-{cfg["target_model"]}_base-layer-{cfg["pure_layer_for_file_name"]}_target-layer-increments-{cfg["target_layer_increments"]}_one-to-X-{cfg["one_to_X"]}_max-seq-len-{cfg["max_seq_len"]}_batch-size-{cfg["batch_size"]}_num-runs-{cfg["num_runs"]}.json'


    gemma_2b_jb_saes_ids = [
        'blocks.0.hook_resid.post',
        'blocks.6.hook_resid.post',
        'blocks.10.hook_resid.post',
        'blocks.12.hook_resid.post',
        'blocks.17.hook_resid.post',
    ]

    gemma_scope_2b_pt_res_canonical_ids = [
        cfg['base_layer'] # 'layer_12/width_16k/canonical'
    ]


    metrics_dict = {'cfg': cfg,
                    'mean_activ_corr': {}, 'num_feat_kept': {}, 'mean_activ_corr_filt': {}, # meta level
                    'svcca_paired': {}, 'svcca_rand_mean': {}, 'svcca_rand_pval': {}, # pairwise level
                    'rsa_paired': {}, 'rsa_rand_mean': {}, 'rsa_rand_pval': {}} # pairwise level

    # in case of bug, load in the saved metrics to avoid recalculation
    if os.path.exists(f'{SAES_DIR}/gemma_2_2b_vs_gemma_1_2b_saes/{metrics_dict_filename}'):
        with open(f'{SAES_DIR}/gemma_2_2b_vs_gemma_1_2b_saes/{metrics_dict_filename}', 'r') as f:
            metrics_dict = json.load(f)

            print(f'loaded metrics dict from {metrics_dict_filename}')

    print(metrics_dict)


    def load_full_batch_activs(sae_id, model, batch_size=cfg['batch_size']):
        layer = sae_id.split('/')[0]
        activs = []
        for i in range(batch_size // 10):
            activs.append(np.load(f'{SAES_DIR}/{model}_activations_residual_stream/{model}_{layer}-sae-activ_max_seq_len-150_batch-{i*10}-{(i+1)*10}.npy'))

        return np.concatenate(activs, axis=0)

    def save_metrics_dict(metrics_dict):
        # convert all floats etc to str representations for dumping to json
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        metrics_dict[key][sub_key] = str(sub_value)

        with open(f'{SAES_DIR}/gemma_2_2b_vs_gemma_1_2b_saes/{metrics_dict_filename}', 'w') as f:
            json.dump(metrics_dict, f)
            print(f'saved metrics dict to {metrics_dict_filename}')

    # loop over all sae pairs
    for gemma_2_2b_sae in gemma_scope_2b_pt_res_canonical_ids:

        full_batch_sae_activs_2b = load_full_batch_activs(gemma_2_2b_sae, 'gemma_2_2b')
        first_dim_reshaped = full_batch_sae_activs_2b.shape[0] * full_batch_sae_activs_2b.shape[1]

        full_batch_sae_activs_2b = full_batch_sae_activs_2b.reshape(first_dim_reshaped, full_batch_sae_activs_2b.shape[-1])

        sae2, cfg2, device2 = SAE.from_pretrained(release='gemma-scope-2b-pt-res-canonical',
                                               sae_id=gemma_2_2b_sae,
                                               device='cuda')


        gemma_2_2b_sae_w_dec = sae2.W_dec.cpu().detach().numpy()

        print(f'loaded gemma_2_2b {gemma_2_2b_sae} and weights')
        print(f'shape of gemma_2_2b_sae_w_dec {gemma_2_2b_sae_w_dec.shape}')
        print(f'shape of full_batch_sae_activs_2b {full_batch_sae_activs_2b.shape}')
        print(f'number of nonzero activations gemma 2 2b {np.count_nonzero(full_batch_sae_activs_2b)}')

        for gemma_1_2b_sae in gemma_2b_jb_saes_ids:

            # Here need some logic so that we can avoid recalculating the same metrics
            pairwise_label = f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'

            # We don't need to do the expensive correlation code if we only do unpaired rand mean and pval
            # Very likely a more efficient way to do this, test quickly for now

            mean_activ_corr_already = False
            num_feat_kept_already = False
            mean_activ_corr_filt_already = False

            svcca_paired_already = False
            svcca_rand_mean_already = False
            svcca_rand_pval_already = False

            rsa_paired_already = False
            rsa_rand_mean_already = False
            rsa_rand_pval_already = False


            if pairwise_label in metrics_dict['mean_activ_corr_filt']:
                mean_activ_corr_already = True
            if pairwise_label in metrics_dict['num_feat_kept']:
                num_feat_kept_already = True
            if pairwise_label in metrics_dict['mean_activ_corr_filt']:
                mean_activ_corr_filt_already = True
            if pairwise_label in metrics_dict['svcca_paired']:
                svcca_paired_already = True
            if pairwise_label in metrics_dict['svcca_rand_mean']:
                svcca_rand_mean_already = True
            if pairwise_label in metrics_dict['svcca_rand_pval']:
                svcca_rand_pval_already = True
            if pairwise_label in metrics_dict['rsa_paired']:
                rsa_paired_already = True
            if pairwise_label in metrics_dict['rsa_rand_mean']:
                rsa_rand_mean_already = True
            if pairwise_label in metrics_dict['rsa_rand_pval']:
                rsa_rand_pval_already = True




            full_batch_sae_activs_1_2b = load_full_batch_activs(gemma_1_2b_sae, 'gemma_1_2b')
            first_dim_reshaped = full_batch_sae_activs_1_2b.shape[0] * full_batch_sae_activs_1_2b.shape[1]
            full_batch_sae_activs_1_2b = full_batch_sae_activs_1_2b.reshape(first_dim_reshaped, full_batch_sae_activs_1_2b.shape[-1])

            sae1, cfg1, device1 = SAE.from_pretrained(release='gemma-scope-9b-pt-res-canonical',
                                                    sae_id=gemma_1_2b_sae,
                                                    device='cuda')

            gemma_1_2b_sae_w_dec = sae1.W_dec.cpu().detach().numpy()

            print(f'loaded gemma_1_2b {gemma_1_2b_sae} and weights')
            print(f'shape of gemma_1_2b_sae_w_dec {gemma_1_2b_sae_w_dec.shape}')
            print(f'shape of full_batch_sae_activs_9b {full_batch_sae_activs_1_2b.shape}')
            print(f'number of nonzero activations gemma 2 9b {np.count_nonzero(full_batch_sae_activs_1_2b)}')

            if mean_activ_corr_already and num_feat_kept_already and mean_activ_corr_filt_already and svcca_paired_already and rsa_paired_already:
                print(f'skipping {pairwise_label} correlation and paired calculation')
                continue
            else:
                full_batch_sae_activs_2b_tensor = torch.tensor(full_batch_sae_activs_2b, dtype=torch.float32)
                full_batch_sae_activs_1_2b_tensor = torch.tensor(full_batch_sae_activs_1_2b, dtype=torch.float32)
                highest_correlations_indices_AB, highest_correlations_values_AB = batched_correlation(full_batch_sae_activs_2b_tensor, full_batch_sae_activs_1_2b_tensor)

                metrics_dict['mean_activ_corr'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = sum(highest_correlations_values_AB) / len(highest_correlations_values_AB)

                print('mean activ corr', metrics_dict['mean_activ_corr'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])

                sorted_feat_counts = Counter(highest_correlations_indices_AB).most_common()
                kept_mod2_feats = [feat_ID for feat_ID, count in sorted_feat_counts if count <= cfg['one_to_X']]

                filt_corr_ind_A = []
                filt_corr_ind_B = []
                seen = set()

                for ind_B, ind_A in enumerate(highest_correlations_indices_AB):
                    if ind_A in kept_mod2_feats:
                        filt_corr_ind_A.append(ind_A)
                        filt_corr_ind_B.append(ind_B)
                    elif ind_A not in seen:  # only keep one if it's over count X
                        seen.add(ind_A)
                        filt_corr_ind_A.append(ind_A)
                        filt_corr_ind_B.append(ind_B)

                new_highest_correlations_indices_A = []
                new_highest_correlations_indices_B = []
                new_highest_correlations_values = []

                for ind_A, ind_B in zip(filt_corr_ind_A, filt_corr_ind_B):
                    val = highest_correlations_values_AB[ind_B]
                    if val > 0:
                        new_highest_correlations_indices_A.append(ind_A)
                        new_highest_correlations_indices_B.append(ind_B)
                        new_highest_correlations_values.append(val)

                metrics_dict['num_feat_kept'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = len(new_highest_correlations_indices_A)
                metrics_dict['mean_activ_corr_filt'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = sum(new_highest_correlations_values) / len(new_highest_correlations_values)

                print('num feat kept', metrics_dict['num_feat_kept'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])
                print('mean activ corr filt', metrics_dict['mean_activ_corr_filt'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])

                print('shape of filtered weights gemma 2 2b', gemma_2_2b_sae_w_dec[new_highest_correlations_indices_A].shape)
                print('shape of filtered weights gemma 2 9b', gemma_1_2b_sae_w_dec[new_highest_correlations_indices_B].shape)

                num_feats = len(new_highest_correlations_indices_A)

                metrics_dict['svcca_paired'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = svcca(gemma_2_2b_sae_w_dec[new_highest_correlations_indices_A],
                                                                                                                    gemma_1_2b_sae_w_dec[new_highest_correlations_indices_B],
                                                                                                                    'nd')

                print('svcca paired', metrics_dict['svcca_paired'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])



                metrics_dict['rsa_paired'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = representational_similarity_analysis(gemma_2_2b_sae_w_dec[new_highest_correlations_indices_A],
                                                                                                                                                 gemma_1_2b_sae_w_dec[new_highest_correlations_indices_B],
                                                                                                                                                 'nd')

                print('rsa paired', metrics_dict['rsa_paired'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])

            if not svcca_rand_mean_already and not svcca_rand_pval_already:

                rand_scores = shuffle_rand(cfg['num_runs'],
                                           gemma_2_2b_sae_w_dec[new_highest_correlations_indices_A],
                                           gemma_1_2b_sae_w_dec[new_highest_correlations_indices_B],
                                           num_feats,
                                           svcca,
                                           shapereq_bool=True)

                metrics_dict['svcca_rand_mean'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = sum(rand_scores) / len(rand_scores)
                metrics_dict['svcca_rand_pval'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = np.mean(np.array(rand_scores) >= metrics_dict['svcca_paired'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])

                print('svcca rand mean', metrics_dict['svcca_rand_mean'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])
                print('svcca rand pval', metrics_dict['svcca_rand_pval'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])
            else:
                print(f'skipping {pairwise_label} svcca rand mean and pval calculation')

            if not rsa_rand_mean_already and not rsa_rand_pval_already:

                rand_scores = shuffle_rand(cfg['num_runs'],
                                           gemma_2_2b_sae_w_dec[new_highest_correlations_indices_A],
                                           gemma_1_2b_sae_w_dec[new_highest_correlations_indices_B],
                                           num_feats,
                                           representational_similarity_analysis,
                                           shapereq_bool=True)

                metrics_dict['rsa_rand_mean'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = sum(rand_scores) / len(rand_scores)
                metrics_dict['rsa_rand_pval'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'] = np.mean(np.array(rand_scores) >= metrics_dict['rsa_paired'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])

                print('rsa rand mean', metrics_dict['rsa_rand_mean'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])
                print('rsa rand pval', metrics_dict['rsa_rand_pval'][f'gemma_2_2b-{gemma_2_2b_sae}_vs_gemma_1_2b-{gemma_1_2b_sae}'])
            else:
                print(f'skipping {pairwise_label} rsa rand mean and pval calculation')

            save_metrics_dict(metrics_dict)

            del full_batch_sae_activs_1_2b
            del sae1
            del cfg1
            del device1

        del full_batch_sae_activs_2b
        del sae2
        del cfg2
        del device2


@app.local_entrypoint()
def main():
    #run_metrics.remote()
    #generate_data_gemma_2_2b_mlp.remote()
    #generate_data_gemma_1_2b_mlp.remote()

    #generate_data_gemma_2_2b_res.remote()
    # generate_data_gemma_1_2b_res.remote()

    main_sae_sim_metrics_res.remote(run_name='testing_cfg', base_layer='layer_22/width_16k/canonical', one_to_X=-1, target_layer_increments=5)
