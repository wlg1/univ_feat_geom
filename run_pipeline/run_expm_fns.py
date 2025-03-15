from collections import Counter

from correlation_fns import *
from sim_fns import *
from get_rand_fns import *
from interpret_fns import *
from get_actv_fns import *
from plot_fns import *

def run_expm(inputs, tokenizer, saeActvs_1, saeActvs_2, num_rand_runs=100, 
             oneToOne_bool=False, manyA_1B_bool=True):
    nonconc_words = ['.', '\\n', '\n', '', ' ', '-', ',', '!', '?', '<|endoftext|>' , '<bos>', '|bos|', '<pad>']
    dictscores = {}

    weight_matrix_1, reshaped_activations_A, feature_acts_model_A = saeActvs_1
    weight_matrix_2, reshaped_activations_B, feature_acts_model_B = saeActvs_2

    """
    manyA-1B:
    `batched_correlation(reshaped_activations_A, reshaped_activations_B)`: (vals, inds)
    max_corr_inds contains mod A's feats as vals (many), and mod B's feats as inds (one)
    Use the list with smaller number of features (decoder mat cols) as the second arg
    """
    if manyA_1B_bool:
        max_corr_inds, max_corr_vals = batched_correlation(reshaped_activations_A, reshaped_activations_B)
    else:
        max_corr_inds, max_corr_vals = batched_correlation(reshaped_activations_B, reshaped_activations_A)

    # num_unq_pairs = len(list(set(max_corr_inds)))
    # print("% unique: ", num_unq_pairs / len(max_corr_inds))

    dictscores["mean_actv_corr"] = sum(max_corr_vals) / len(max_corr_vals)
    num_unq_pairs = len(list(set(max_corr_inds)))
    print("% unique: ", num_unq_pairs / reshaped_activations_B.shape[-1])
    dictscores["num_feat_unique_beforeFilt"] = num_unq_pairs

    ########### filter ###########
    ### keyword filtering ###
    samp_m = 5

    filt_corr_ind_A = []
    filt_corr_ind_B = []
    for corr_ind_feat, corr_val_feat in enumerate(max_corr_inds):
        if manyA_1B_bool:
            feat_B, feat_A = corr_ind_feat, corr_val_feat
        else:
            feat_A, feat_B = corr_ind_feat, corr_val_feat
        ds_top_acts_indices = highest_activating_tokens(feature_acts_model_A, feat_A, samp_m, batch_tokens= inputs['input_ids'])
        top_A_labels = store_top_toks(ds_top_acts_indices, inputs['input_ids'], tokenizer)

        ds_top_acts_indices = highest_activating_tokens(feature_acts_model_B, feat_B, samp_m, batch_tokens= inputs['input_ids'])
        top_B_labels = store_top_toks(ds_top_acts_indices, inputs['input_ids'], tokenizer)

        flag = True
        for filt_word in nonconc_words:
            if filt_word in top_A_labels or filt_word in top_B_labels:
                flag = False
                break
        if flag and len(set(top_A_labels).intersection(set(top_B_labels))) > 0:
            filt_corr_ind_A.append(feat_A)
            filt_corr_ind_B.append(feat_B)

    if manyA_1B_bool:
        num_unq_pairs = len(list(set(filt_corr_ind_A)))
        print("% unique after rmv kw: ", num_unq_pairs / reshaped_activations_A.shape[-1])
        print("num feats after rmv kw: ", len(filt_corr_ind_A))
    else:
        num_unq_pairs = len(list(set(filt_corr_ind_B)))
        print("% unique after rmv kw: ", num_unq_pairs / reshaped_activations_B.shape[-1])
        print("num feats after rmv kw: ", len(filt_corr_ind_B))

    ### 1-1 filtering ###
    if oneToOne_bool:
        sorted_feat_counts = Counter(max_corr_inds).most_common()
        kept_modA_feats = [feat_ID for feat_ID, count in sorted_feat_counts if count == 1]

        oneToOne_A = []
        oneToOne_B = []
        seen = set()
        for ind_A, ind_B in zip(filt_corr_ind_A, filt_corr_ind_B):
            if manyA_1B_bool:
                if ind_A in kept_modA_feats:
                    oneToOne_A.append(ind_A)
                    oneToOne_B.append(ind_B)
                elif ind_A not in seen:  # only keep one if it's over count X
                    seen.add(ind_A)
                    oneToOne_A.append(ind_A)
                    oneToOne_B.append(ind_B)
            else:
                if ind_B in kept_modA_feats:
                    oneToOne_A.append(ind_A)
                    oneToOne_B.append(ind_B)
                elif ind_B not in seen:
                    seen.add(ind_B) # only keep one if it's over count X
                    oneToOne_A.append(ind_A)
                    oneToOne_B.append(ind_B)

        num_unq_pairs = len(list(set(oneToOne_A)))
        print("% unique after 1-1: ", num_unq_pairs / len(oneToOne_A))
        print("num feats after 1-1: ", len(oneToOne_A))

        filt_corr_ind_A = oneToOne_A
        filt_corr_ind_B = oneToOne_B

    ### low correlation filtering ###
    new_max_corr_inds_A = []
    new_max_corr_inds_B = []
    new_max_corr_vals = []

    for ind_A, ind_B in zip(filt_corr_ind_A, filt_corr_ind_B):
        if manyA_1B_bool:
            val = max_corr_vals[ind_B]
        else:
            val = max_corr_vals[ind_A]
        if val > 0.1:
            new_max_corr_inds_A.append(ind_A)
            new_max_corr_inds_B.append(ind_B)
            new_max_corr_vals.append(val)

    num_unq_pairs = len(list(set(new_max_corr_inds_A)))
    print("% unique after rmv 0s: ", num_unq_pairs / reshaped_activations_B.shape[-1])
    print("num feats after rmv 0s: ", len(new_max_corr_inds_A))
    dictscores["num_feat_kept"] = len(new_max_corr_inds_A)

    dictscores["mean_actv_corr_filt"] = sum(new_max_corr_vals) / len(new_max_corr_vals)

    ########### repr similarity scores ###########

    num_feats = len(new_max_corr_inds_A)

    dictscores["svcca_paired"] = svcca(weight_matrix_1[new_max_corr_inds_A], weight_matrix_2[new_max_corr_inds_B], "nd")
    print('svcca paired done')
    rand_scores = shuffle_rand(num_rand_runs, weight_matrix_1[new_max_corr_inds_A],
                                weight_matrix_2[new_max_corr_inds_B], num_feats,
                                svcca, shapereq_bool=True)
    dictscores["svcca_rand_mean"] = sum(rand_scores) / len(rand_scores)
    dictscores["svcca_rand_pval"] =  np.mean(np.array(rand_scores) >= dictscores["svcca_paired"])

    dictscores["rsa_paired"] = representational_similarity_analysis(weight_matrix_1[new_max_corr_inds_A], weight_matrix_2[new_max_corr_inds_B], "nd")
    print('rsa paired done')
    rand_scores = shuffle_rand(num_rand_runs, weight_matrix_1[new_max_corr_inds_A],
                                                weight_matrix_2[new_max_corr_inds_B], num_feats,
                                                representational_similarity_analysis, shapereq_bool=True)
    dictscores["rsa_rand_mean"] = sum(rand_scores) / len(rand_scores)
    dictscores["rsa_rand_pval"] =  np.mean(np.array(rand_scores) >= dictscores["rsa_paired"])

    return dictscores