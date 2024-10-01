import json
import matplotlib.pyplot as plt
import numpy as np

"""
They were broken because I was giving them strings, not floats.
"""

def plot_gemma_metrics(metrics_dict, key, base_model_layer):
    """
    metrics_dict: dict
        Dictionary containing the metrics to be plotted
        This just contains the whole thing. Index by key to get the plot we want.

        Pass in a different dict for res vs. mlp

    key: str
        The key in the metrics_dict to plot.

    base_model_layer: str
        The base model and layer to plot against others.
        Ex: gemma_2_2b-layer_3
    """

    list_of_contra_layers = []
    list_of_contra_data = []

    #if key == 'mean_activ_corr_filt':
    plot_dict = metrics_dict[key]

    for k in plot_dict.keys():
        if base_model_layer in k:
            base_layer = k

            # split the string on the vs
            vs_split = base_layer.split('vs')
            paired_layer = vs_split[1].split('/')[0][1:] # strip underscore at beginning with last indexing

            list_of_contra_layers.append(paired_layer)
            list_of_contra_data.append(plot_dict[k])


    # print(list_of_contra_layers)
    # print(list_of_contra_data)

    # round it
    if isinstance(list_of_contra_data[0], str):
        list_of_contra_data = [float(x) for x in list_of_contra_data]
        list_of_contra_data = [round(x, 4) for x in list_of_contra_data]

    # now plot the above in a bar graph
    fig, ax = plt.subplots()
    x = np.arange(len(list_of_contra_layers))

    #ax.bar(list_of_contra_layers, list_of_contra_data)
    ax.bar(x, np.array(list_of_contra_data))

    # Add labels and title
    # ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(list_of_contra_layers, rotation=45, ha='right')
    #ax.set_xticks(list_of_contra_layers)
    ax.set_ylabel(key)
    ax.set_title(f'{key} for {base_model_layer} vs. Other Layers')

    # Show plot
    plt.tight_layout()

    #plt.show()
    plt.savefig(f'gemma_viz/{key}_{base_model_layer}.png')


# Step 1: Read the metrics_dict.json file
with open(INSERT_METRICS_DICT_HERE, 'r') as file:
    metrics_dict = json.load(file)

plot_gemma_metrics(metrics_dict, 'mean_activ_corr_filt', 'gemma_2_2b-layer_22')
plot_gemma_metrics(metrics_dict, 'num_feat_kept', 'gemma_2_2b-layer_22')
plot_gemma_metrics(metrics_dict, 'mean_activ_corr', 'gemma_2_2b-layer_22')
plot_gemma_metrics(metrics_dict, 'svcca_paired', 'gemma_2_2b-layer_22')
plot_gemma_metrics(metrics_dict, 'rsa_paired', 'gemma_2_2b-layer_22')

