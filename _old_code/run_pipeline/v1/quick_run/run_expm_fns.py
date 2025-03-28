from collections import Counter

from correlation_fns import *
from sim_fns import *
from get_rand_fns import *
from interpret_fns import *
from get_actv_fns import *
from run_expm_fns import *
from plot_fns import *

def run_expm(inputs, tokenizer, layer_id, outputs, outputs_2, layer_start, layer_end, 
             num_runs=100, oneToOne_bool=False):
    junk_words = ['.', '\\n', '\n', '', ' ', '-' , '<bos>', ',', '!', '?', '<|endoftext|>', '|bos|']
    layer_to_dictscores = {}

    name = "EleutherAI/sae-pythia-70m-32k"
    weight_matrix_np, reshaped_activations_A, feature_acts_model_A = get_weights_and_acts(name, layer_id, outputs)

    name = "EleutherAI/sae-pythia-160m-32k"
    for layerID_2 in range(layer_start, layer_end): # 0, 12
        dictscores = {}

        weight_matrix_2, reshaped_activations_B, feature_acts_model_B = get_weights_and_acts_byLayer(name, layerID_2, outputs_2)

        """
        `batched_correlation(reshaped_activations_B, reshaped_activations_A)`:
        max_corr_inds contains modA's feats as inds, and modB's feats as vals.
        Use the list with smaller number of features (cols) as the second arg
        """
        max_corr_inds, max_corr_vals = batched_correlation(reshaped_activations_A, reshaped_activations_B)

        # num_unq_pairs = len(list(set(max_corr_inds)))
        # print("% unique: ", num_unq_pairs / len(max_corr_inds))

        dictscores["mean_actv_corr"] = sum(max_corr_vals) / len(max_corr_vals)

        ###########
        # filter
        samp_m = 5

        filt_corr_ind_A = []
        filt_corr_ind_B = []
        for feat_B, feat_A in enumerate(max_corr_inds):
            # if feat_B % 2000 == 0:
            #     print(feat_B)
            ds_top_acts_indices = highest_activating_tokens(feature_acts_model_A, feat_A, samp_m, batch_tokens= inputs['input_ids'])
            top_A_labels = store_top_toks(ds_top_acts_indices, inputs['input_ids'], tokenizer)

            ds_top_acts_indices = highest_activating_tokens(feature_acts_model_B, feat_B, samp_m, batch_tokens= inputs['input_ids'])
            top_B_labels = store_top_toks(ds_top_acts_indices, inputs['input_ids'], tokenizer)

            flag = True
            for junk in junk_words:
                if junk in top_A_labels or junk in top_B_labels:
                    flag = False
                    break
            if flag and len(set(top_A_labels).intersection(set(top_B_labels))) > 0:
                filt_corr_ind_A.append(feat_A)
                filt_corr_ind_B.append(feat_B)

        num_unq_pairs = len(list(set(filt_corr_ind_A)))
        print("% unique: ", num_unq_pairs / reshaped_activations_B.shape[-1])
        print("num feats after rmv kw: ", len(filt_corr_ind_A))

        if oneToOne_bool:
            sorted_feat_counts = Counter(max_corr_inds).most_common()
            kept_modA_feats = [feat_ID for feat_ID, count in sorted_feat_counts if count == 1]

            oneToOne_A = []
            oneToOne_B = []
            seen = set()
            for ind_A, ind_B in zip(filt_corr_ind_A, filt_corr_ind_B):
                if ind_A in kept_modA_feats:
                    oneToOne_A.append(ind_A)
                    oneToOne_B.append(ind_B)
                elif ind_A not in seen:  # only keep one if it's over count X
                    seen.add(ind_A)
                    oneToOne_A.append(ind_A)
                    oneToOne_B.append(ind_B)
            num_unq_pairs = len(list(set(oneToOne_A)))
            print("% unique: ", num_unq_pairs / len(oneToOne_A))
            print("num feats after 1-1: ", len(oneToOne_A))

            filt_corr_ind_A = oneToOne_A
            filt_corr_ind_B = oneToOne_B

        new_max_corr_inds_A = []
        new_max_corr_inds_B = []
        new_max_corr_vals = []

        for ind_A, ind_B in zip(filt_corr_ind_A, filt_corr_ind_B):
        # for ind_A, ind_B in zip(oneToOne_A, oneToOne_B):
            val = max_corr_vals[ind_B]
            if val > 0.1:
                new_max_corr_inds_A.append(ind_A)
                new_max_corr_inds_B.append(ind_B)
                new_max_corr_vals.append(val)

        num_unq_pairs = len(list(set(new_max_corr_inds_A)))
        print("% unique after rmv 0s: ", num_unq_pairs / reshaped_activations_B.shape[-1])
        print("num feats after rmv 0s: ", len(new_max_corr_inds_A))
        dictscores["num_feat_kept"] = len(new_max_corr_inds_A)
        dictscores["num_feat_A_unique"] = len(list(set(new_max_corr_inds_A)))

        dictscores["mean_actv_corr_filt"] = sum(new_max_corr_vals) / len(new_max_corr_vals)

        ###########
        # sim tests

        num_feats = len(new_max_corr_inds_A)

        dictscores["svcca_paired"] = svcca(weight_matrix_np[new_max_corr_inds_A], weight_matrix_2[new_max_corr_inds_B], "nd")
        print('svcca paired done')
        if num_runs > 0:
            rand_scores = shuffle_rand(num_runs, weight_matrix_np[new_max_corr_inds_A],
                                        weight_matrix_2[new_max_corr_inds_B], num_feats,
                                        svcca, shapereq_bool=True)
            dictscores["svcca_rand_mean"] = sum(rand_scores) / len(rand_scores)
            dictscores["svcca_rand_pval"] =  np.mean(np.array(rand_scores) >= dictscores["svcca_paired"])

        # dictscores["rsa_paired"] = representational_similarity_analysis(weight_matrix_np[new_max_corr_inds_A], weight_matrix_2[new_max_corr_inds_B], "nd")
        # print('rsa paired done')
        # rand_scores = shuffle_rand(num_runs, weight_matrix_np[new_max_corr_inds_A],
        #                                             weight_matrix_2[new_max_corr_inds_B], num_feats,
        #                                             representational_similarity_analysis, shapereq_bool=True)
        # dictscores["rsa_rand_mean"] = sum(rand_scores) / len(rand_scores)
        # dictscores["rsa_rand_pval"] =  np.mean(np.array(rand_scores) >= dictscores["rsa_paired"])

        print("Layer: " + str(layerID_2))
        for key, value in dictscores.items():
            print(key + ": " + str(value))
        print("\n")

        layer_to_dictscores[layerID_2] = dictscores
    return layer_to_dictscores