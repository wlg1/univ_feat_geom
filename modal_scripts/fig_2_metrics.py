import numpy as np
import torch
from more_metrics_ts_extension.simSAE_more_metrics_nb_utils_as_py import svcca, batched_correlation


def highest_correlation_rows(matrix1, matrix2):
    correlations = np.corrcoef(matrix1, matrix2, rowvar=True)
    num_rows = matrix1.shape[0]

    highest_corr_indices = []
    for i in range(num_rows):
        row_corr = correlations[i, num_rows:num_rows * 2]
        highest_corr_index = np.argmax(row_corr)
        highest_corr_indices.append(highest_corr_index)

    return highest_corr_indices

m1 = np.array([[2.48, 5.24, 1.24],
               [5.95, 0.67, 1.8],
               [0.1, 7.4, 4.1],
               [3.15, 6.48, 9.28]])

m2 = np.array([[0.55, 6.13, 3.56],
               # [6.48, 7.44, 5.36],
               [0.91, 5.89, 4.26],
               [6.48, 7.44, 5.36],
               [5.78, 2.22, 4.54]])

m2_rearraged = np.array([[6.48, 7.44, 5.36],
                         [5.78, 2.22, 4.54],
                         [0.55, 6.13, 3.56],
                         [0.91, 5.89, 4.26]])


new_highest_corr_indices = highest_correlation_rows(m1, m2)
print("Rows with the highest correlation:", new_highest_corr_indices)

svcca_unpaired = svcca(m1, m2, shape='nd')
print(svcca_unpaired)

svcca_paired = svcca(m1, m2_rearraged, shape='nd')
print(svcca_paired)



