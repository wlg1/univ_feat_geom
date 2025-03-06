#!/usr/bin/env python
# coding: utf-8

# # setup

# In[ ]:


get_ipython().run_cell_magic('capture', '', '%pip install sae-lens\n')


# In[ ]:


# import pdb


# In[ ]:


import pickle
import numpy as np

import torch
import matplotlib.pyplot as plt

from sae_lens import SAE

from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

from datasets import load_dataset


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# ## corr fns

# In[ ]:


def batched_correlation(reshaped_activations_A, reshaped_activations_B, batch_size=100):
    # Ensure tensors are on GPU
    if torch.cuda.is_available():
        reshaped_activations_A = reshaped_activations_A.to('cuda')
        reshaped_activations_B = reshaped_activations_B.to('cuda')

    # Normalize columns of A
    mean_A = reshaped_activations_A.mean(dim=0, keepdim=True)
    std_A = reshaped_activations_A.std(dim=0, keepdim=True)
    normalized_A = (reshaped_activations_A - mean_A) / (std_A + 1e-8)  # Avoid division by zero

    # Normalize columns of B
    mean_B = reshaped_activations_B.mean(dim=0, keepdim=True)
    std_B = reshaped_activations_B.std(dim=0, keepdim=True)
    normalized_B = (reshaped_activations_B - mean_B) / (std_B + 1e-8)  # Avoid division by zero

    num_batches = (normalized_B.shape[1] + batch_size - 1) // batch_size
    max_values = []
    max_indices = []

    for batch in range(num_batches):
        start = batch * batch_size
        end = min(start + batch_size, normalized_B.shape[1])
        batch_corr_matrix = torch.matmul(normalized_A.t(), normalized_B[:, start:end]) / normalized_A.shape[0]
        max_val, max_idx = batch_corr_matrix.max(dim=0)
        max_values.append(max_val)
        # max_indices.append(max_idx + start)  # Adjust indices for the batch offset
        max_indices.append(max_idx)  # Adjust indices for the batch offset

        del batch_corr_matrix
        torch.cuda.empty_cache()

    return torch.cat(max_indices), torch.cat(max_values)


# ## sim fns

# In[ ]:


import functools
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch


def to_numpy_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[npt.NDArray]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> npt.NDArray:
        return x if isinstance(x, np.ndarray) else x.numpy()

    return list(map(convert, args))


def to_torch_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[torch.Tensor]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    return list(map(convert, args))


def adjust_dimensionality(
    R: npt.NDArray, Rp: npt.NDArray, strategy="zero_pad"
) -> Tuple[npt.NDArray, npt.NDArray]:
    D = R.shape[1]
    Dp = Rp.shape[1]
    if strategy == "zero_pad":
        if D - Dp == 0:
            return R, Rp
        elif D - Dp > 0:
            return R, np.concatenate((Rp, np.zeros((Rp.shape[0], D - Dp))), axis=1)
        else:
            return np.concatenate((R, np.zeros((R.shape[0], Dp - D))), axis=1), Rp
    else:
        raise NotImplementedError()


def center_columns(R: npt.NDArray) -> npt.NDArray:
    return R - R.mean(axis=0)[None, :]


def normalize_matrix_norm(R: npt.NDArray) -> npt.NDArray:
    return R / np.linalg.norm(R, ord="fro")


def sim_random_baseline(
    rep1: torch.Tensor, rep2: torch.Tensor, sim_func: Callable, n_permutations: int = 10
) -> Dict[str, Any]:
    torch.manual_seed(1234)
    scores = []
    for _ in range(n_permutations):
        perm = torch.randperm(rep1.size(0))

        score = sim_func(rep1[perm, :], rep2)
        score = score if isinstance(score, float) else score["score"]

        scores.append(score)

    return {"baseline_scores": np.array(scores)}


class Pipeline:
    def __init__(
        self,
        preprocess_funcs: List[Callable[[npt.NDArray], npt.NDArray]],
        similarity_func: Callable[[npt.NDArray, npt.NDArray], Dict[str, Any]],
    ) -> None:
        self.preprocess_funcs = preprocess_funcs
        self.similarity_func = similarity_func

    def __call__(self, R: npt.NDArray, Rp: npt.NDArray) -> Dict[str, Any]:
        for preprocess_func in self.preprocess_funcs:
            R = preprocess_func(R)
            Rp = preprocess_func(Rp)
        return self.similarity_func(R, Rp)

    def __str__(self) -> str:
        def func_name(func: Callable) -> str:
            return (
                func.__name__
                if not isinstance(func, functools.partial)
                else func.func.__name__
            )

        def partial_keywords(func: Callable) -> str:
            if not isinstance(func, functools.partial):
                return ""
            else:
                return str(func.keywords)

        return (
            "Pipeline("
            + (
                "+".join(map(func_name, self.preprocess_funcs))
                + "+"
                + func_name(self.similarity_func)
                + partial_keywords(self.similarity_func)
            )
            + ")"
        )


# In[ ]:


from typing import List, Set, Union

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch

# from llmcomp.measures.utils import to_numpy_if_needed


def _jac_sim_i(idx_R: Set[int], idx_Rp: Set[int]) -> float:
    return len(idx_R.intersection(idx_Rp)) / len(idx_R.union(idx_Rp))


def jaccard_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    k: int = 10,
    inner: str = "cosine",
    n_jobs: int = 8,
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    indices_R = nn_array_to_setlist(top_k_neighbors(R, k, inner, n_jobs))
    indices_Rp = nn_array_to_setlist(top_k_neighbors(Rp, k, inner, n_jobs))

    return float(
        np.mean(
            [_jac_sim_i(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]
        )
    )


def top_k_neighbors(
    R: npt.NDArray,
    k: int,
    inner: str,
    n_jobs: int,
) -> npt.NDArray:
    # k+1 nearest neighbors, because we pass in all the data, which means that a point
    # will be the nearest neighbor to itself. We remove this point from the results and
    # report only the k nearest neighbors distinct from the point itself.
    nns = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, metric=inner, n_jobs=n_jobs
    )
    nns.fit(R)
    _, nns = nns.kneighbors(R)
    return nns[:, 1:]


def nn_array_to_setlist(nn: npt.NDArray) -> List[Set[int]]:
    return [set(idx) for idx in nn]


# In[ ]:


import functools
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import get_args
from typing import List
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
# from loguru import logger

log = logging.getLogger(__name__)


SHAPE_TYPE = Literal["nd", "ntd", "nchw"]

ND_SHAPE, NTD_SHAPE, NCHW_SHAPE = get_args(SHAPE_TYPE)[0], get_args(SHAPE_TYPE)[1], get_args(SHAPE_TYPE)[2]


class SimilarityFunction(Protocol):
    def __call__(  # noqa: E704
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
    ) -> float: ...


class RSMSimilarityFunction(Protocol):
    def __call__(  # noqa: E704
        self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE, n_jobs: int
    ) -> float: ...


@dataclass
class BaseSimilarityMeasure(ABC):
    larger_is_more_similar: bool
    is_symmetric: bool

    is_metric: bool | None = None
    invariant_to_affine: bool | None = None
    invariant_to_invertible_linear: bool | None = None
    invariant_to_ortho: bool | None = None
    invariant_to_permutation: bool | None = None
    invariant_to_isotropic_scaling: bool | None = None
    invariant_to_translation: bool | None = None
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class FunctionalSimilarityMeasure(BaseSimilarityMeasure):
    @abstractmethod
    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> float:
        raise NotImplementedError


@dataclass(kw_only=True)
class RepresentationalSimilarityMeasure(BaseSimilarityMeasure):
    sim_func: SimilarityFunction

    def __call__(
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
    ) -> float:
        return self.sim_func(R, Rp, shape)


class RSMSimilarityMeasure(RepresentationalSimilarityMeasure):
    sim_func: RSMSimilarityFunction

    @staticmethod
    def estimate_good_number_of_jobs(R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray) -> int:
        # RSMs in are NxN (or DxD) so the number of jobs should roughly scale quadratically with increase in N (or D).
        # False! As long as sklearn-native metrics are used, they will use parallel implementations regardless of job
        # count. Each job would spawn their own threads, which leads to oversubscription of cores and thus slowdown.
        # This seems to be not fully correct (n_jobs=2 seems to actually use two cores), but using n_jobs=1 seems the
        # fastest.
        return 1

    def __call__(
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
        n_jobs: Optional[int] = None,
    ) -> float:
        if n_jobs is None:
            n_jobs = self.estimate_good_number_of_jobs(R, Rp)
        return self.sim_func(R, Rp, shape, n_jobs=n_jobs)


def to_numpy_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[npt.NDArray]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> npt.NDArray:
        return x if isinstance(x, np.ndarray) else x.numpy()

    return list(map(convert, args))


def to_torch_if_needed(*args: Union[torch.Tensor, npt.NDArray]) -> List[torch.Tensor]:
    def convert(x: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    return list(map(convert, args))


def adjust_dimensionality(R: npt.NDArray, Rp: npt.NDArray, strategy="zero_pad") -> Tuple[npt.NDArray, npt.NDArray]:
    D = R.shape[1]
    Dp = Rp.shape[1]
    if strategy == "zero_pad":
        if D - Dp == 0:
            return R, Rp
        elif D - Dp > 0:
            return R, np.concatenate((Rp, np.zeros((Rp.shape[0], D - Dp))), axis=1)
        else:
            return np.concatenate((R, np.zeros((R.shape[0], Dp - D))), axis=1), Rp
    else:
        raise NotImplementedError()


def center_columns(R: npt.NDArray) -> npt.NDArray:
    return R - R.mean(axis=0)[None, :]


def normalize_matrix_norm(R: npt.NDArray) -> npt.NDArray:
    return R / np.linalg.norm(R, ord="fro")


def normalize_row_norm(R: npt.NDArray) -> npt.NDArray:
    return R / np.linalg.norm(R, ord=2, axis=1, keepdims=True)


def standardize(R: npt.NDArray) -> npt.NDArray:
    return (R - R.mean(axis=0, keepdims=True)) / R.std(axis=0)


def double_center(x: npt.NDArray) -> npt.NDArray:
    return x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()


def align_spatial_dimensions(R: npt.NDArray, Rp: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Aligns spatial representations by resizing them to the smallest spatial dimension.
    Subsequent aligned spatial representations are flattened, with the spatial aligned representations
    moving into the *sample* dimension.
    """
    R_re, Rp_re = resize_wh_reps(R, Rp)
    R_re = rearrange(R_re, "n c h w -> (n h w) c")
    Rp_re = rearrange(Rp_re, "n c h w -> (n h w) c")
    if R_re.shape[0] > 5000:
        logger.info(f"Got {R_re.shape[0]} samples in N after flattening. Subsampling to reduce compute.")
        subsample = R_re.shape[0] // 5000
        R_re = R_re[::subsample]
        Rp_re = Rp_re[::subsample]

    return R_re, Rp_re


def average_pool_downsample(R, resize: bool, new_size: tuple[int, int]):
    if not resize:
        return R  # do nothing
    else:
        is_numpy = isinstance(R, np.ndarray)
        R_torch = torch.from_numpy(R) if is_numpy else R
        R_torch = torch.nn.functional.adaptive_avg_pool2d(R_torch, new_size)
        return R_torch.numpy() if is_numpy else R_torch


def resize_wh_reps(R: npt.NDArray, Rp: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Function for resizing spatial representations that are not the same size.
    Does through fourier transform and resizing.

    Args:
        R: numpy array of shape  [batch_size, height, width, num_channels]
        RP: numpy array of shape [batch_size, height, width, num_channels]

    Returns:
        fft_acts1: numpy array of shape [batch_size, (new) height, (new) width, num_channels]
        fft_acts2: numpy array of shape [batch_size, (new) height, (new) width, num_channels]

    """
    height1, width1 = R.shape[2], R.shape[3]
    height2, width2 = Rp.shape[2], Rp.shape[3]
    if height1 != height2 or width1 != width2:
        height = min(height1, height2)
        width = min(width1, width2)
        new_size = [height, width]
        resize = True
    else:
        height = height1
        width = width1
        new_size = None
        resize = False

    # resize and preprocess with fft
    avg_ds1 = average_pool_downsample(R, resize=resize, new_size=new_size)
    avg_ds2 = average_pool_downsample(Rp, resize=resize, new_size=new_size)
    return avg_ds1, avg_ds2


def fft_resize(images, resize=False, new_size=None):
    """Function for applying DFT and resizing.

    This function takes in an array of images, applies the 2-d fourier transform
    and resizes them according to new_size, keeping the frequencies that overlap
    between the two sizes.

    Args:
              images: a numpy array with shape
                      [batch_size, height, width, num_channels]
              resize: boolean, whether or not to resize
              new_size: a tuple (size, size), with height and width the same

    Returns:
              im_fft_downsampled: a numpy array with shape
                           [batch_size, (new) height, (new) width, num_channels]
    """
    assert len(images.shape) == 4, "expecting images to be" "[batch_size, height, width, num_channels]"
    if resize:
        # FFT --> remove high frequencies --> inverse FFT
        im_complex = images.astype("complex64")
        im_fft = np.fft.fft2(im_complex, axes=(1, 2))
        im_shifted = np.fft.fftshift(im_fft, axes=(1, 2))

        center_width = im_shifted.shape[2] // 2
        center_height = im_shifted.shape[1] // 2
        half_w = new_size[0] // 2
        half_h = new_size[1] // 2
        cropped_fft = im_shifted[
            :, center_height - half_h : center_height + half_h, center_width - half_w : center_width + half_w, :
        ]
        cropped_fft_shifted_back = np.fft.ifft2(cropped_fft, axes=(1, 2))
        return cropped_fft_shifted_back.real
    else:
        return images


class Pipeline:
    def __init__(
        self,
        preprocess_funcs: List[Callable[[npt.NDArray], npt.NDArray]],
        similarity_func: Callable[[npt.NDArray, npt.NDArray, SHAPE_TYPE], float],
    ) -> None:
        self.preprocess_funcs = preprocess_funcs
        self.similarity_func = similarity_func

    def __call__(self, R: npt.NDArray, Rp: npt.NDArray, shape: SHAPE_TYPE) -> float:
        try:
            for preprocess_func in self.preprocess_funcs:
                R = preprocess_func(R)
                Rp = preprocess_func(Rp)
            return self.similarity_func(R, Rp, shape)
        except ValueError as e:
            log.info(f"Pipeline failed: {e}")
            return np.nan

    def __str__(self) -> str:
        def func_name(func: Callable) -> str:
            return func.__name__ if not isinstance(func, functools.partial) else func.func.__name__

        def partial_keywords(func: Callable) -> str:
            if not isinstance(func, functools.partial):
                return ""
            else:
                return str(func.keywords)

        return (
            "Pipeline("
            + (
                "+".join(map(func_name, self.preprocess_funcs))
                + "+"
                + func_name(self.similarity_func)
                + partial_keywords(self.similarity_func)
            )
            + ")"
        )


def flatten(*args: Union[torch.Tensor, npt.NDArray], shape: SHAPE_TYPE) -> List[Union[torch.Tensor, npt.NDArray]]:
    if shape == "ntd":
        return list(map(flatten_nxtxd_to_ntxd, args))
    elif shape == "nd":
        return list(args)
    elif shape == "nchw":
        return list(map(flatten_nxcxhxw_to_nxchw, args))  # Flattening non-trivial for nchw
    else:
        raise ValueError("Unknown shape of representations. Must be one of 'ntd', 'nchw', 'nd'.")


def flatten_nxtxd_to_ntxd(R: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
    R = to_torch_if_needed(R)[0]
    log.debug("Shape before flattening: %s", str(R.shape))
    R = torch.flatten(R, start_dim=0, end_dim=1)
    log.debug("Shape after flattening: %s", str(R.shape))
    return R


def flatten_nxcxhxw_to_nxchw(R: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
    R = to_torch_if_needed(R)[0]
    log.debug("Shape before flattening: %s", str(R.shape))
    R = torch.reshape(R, (R.shape[0], -1))
    log.debug("Shape after flattening: %s", str(R.shape))
    return R


# In[ ]:


from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import scipy.stats
import sklearn.metrics
import torch
# from repsim.measures.utils import flatten
# from repsim.measures.utils import RSMSimilarityMeasure
# from repsim.measures.utils import SHAPE_TYPE
# from repsim.measures.utils import to_numpy_if_needed


def representational_similarity_analysis(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    inner="correlation",
    outer="spearman",
    n_jobs: Optional[int] = None,
) -> float:
    """Representational similarity analysis

    Args:
        R (Union[torch.Tensor, npt.NDArray]): N x D representation
        Rp (Union[torch.Tensor, npt.NDArray]): N x D' representation
        inner (str, optional): inner similarity function for RSM. Must be one of
            scipy.spatial.distance.pdist identifiers . Defaults to "correlation".
        outer (str, optional): outer similarity function that compares RSMs. Defaults to
             "spearman". Must be one of "spearman", "euclidean"

    Returns:
        float: _description_
    """
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    if inner == "correlation":
        # n_jobs only works if metric is in PAIRWISE_DISTANCES as defined in sklearn, i.e., not for correlation.
        # But correlation = 1 - cosine dist of row-centered data, so we use the faster cosine metric and center the data.
        R = R - R.mean(axis=1, keepdims=True)
        S = scipy.spatial.distance.squareform(  # take the lower triangle of RSM
            1 - sklearn.metrics.pairwise_distances(R, metric="cosine", n_jobs=n_jobs),  # type:ignore
            checks=False,
        )
        Rp = Rp - Rp.mean(axis=1, keepdims=True)
        Sp = scipy.spatial.distance.squareform(
            1 - sklearn.metrics.pairwise_distances(Rp, metric="cosine", n_jobs=n_jobs),  # type:ignore
            checks=False,
        )
    elif inner == "euclidean":
        # take the lower triangle of RSM
        S = scipy.spatial.distance.squareform(
            sklearn.metrics.pairwise_distances(R, metric=inner, n_jobs=n_jobs), checks=False
        )
        Sp = scipy.spatial.distance.squareform(
            sklearn.metrics.pairwise_distances(Rp, metric=inner, n_jobs=n_jobs), checks=False
        )
    else:
        raise NotImplementedError(f"{inner=}")

    if outer == "spearman":
        return scipy.stats.spearmanr(S, Sp).statistic  # type:ignore
    elif outer == "euclidean":
        return float(np.linalg.norm(S - Sp, ord=2))
    else:
        raise ValueError(f"Unknown outer similarity function: {outer}")


class RSA(RSMSimilarityMeasure):
    def __init__(self):
        # choice of inner/outer in __call__ if fixed to default values, so these values are always the same
        super().__init__(
            sim_func=representational_similarity_analysis,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )


# In[ ]:


##################################################################################
# Copied from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/cca_core.py
# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The core code for applying Canonical Correlation Analysis to deep networks.

This module contains the core functions to apply canonical correlation analysis
to deep neural networks. The main function is get_cca_similarity, which takes in
two sets of activations, typically the neurons in two layers and their outputs
on all of the datapoints D = [d_1,...,d_m] that have been passed through.

Inputs have shape (num_neurons1, m), (num_neurons2, m). This can be directly
applied used on fully connected networks. For convolutional layers, the 3d block
of neurons can either be flattened entirely, along channels, or alternatively,
the dft_ccas (Discrete Fourier Transform) module can be used.

See:
https://arxiv.org/abs/1706.05806
https://arxiv.org/abs/1806.05759
for full details.

"""
import numpy as np
# from repsim.measures.utils import align_spatial_dimensions

num_cca_trials = 5


def positivedef_matrix_sqrt(array):
    """Stable method for computing matrix square roots, supports complex matrices.

    Args:
              array: A numpy 2d array, can be complex valued that is a positive
                     definite symmetric (or hermitian) matrix

    Returns:
              sqrtarray: The matrix square root of array
    """
    w, v = np.linalg.eigh(array)
    #  A - np.dot(v, np.dot(np.diag(w), v.T))
    wsqrt = np.sqrt(w)
    sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
    return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
    """Takes covariance between X, Y, and removes values of small magnitude.

    Args:
              sigma_xx: 2d numpy array, variance matrix for x
              sigma_xy: 2d numpy array, crossvariance matrix for x,y
              sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                        (conjugate) transpose of sigma_xy
              sigma_yy: 2d numpy array, variance matrix for y
              epsilon : cutoff value for norm below which directions are thrown
                         away

    Returns:
              sigma_xx_crop: 2d array with low x norm directions removed
              sigma_xy_crop: 2d array with low x and y norm directions removed
              sigma_yx_crop: 2d array with low x and y norm directiosn removed
              sigma_yy_crop: 2d array with low y norm directions removed
              x_idxs: indexes of sigma_xx that were removed
              y_idxs: indexes of sigma_yy that were removed
    """

    x_diag = np.abs(np.diagonal(sigma_xx))
    y_diag = np.abs(np.diagonal(sigma_yy))
    x_idxs = x_diag >= epsilon
    y_idxs = y_diag >= epsilon

    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop, x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon, verbose=True):
    """Main cca computation function, takes in variances and crossvariances.

    This function takes in the covariances and cross covariances of X, Y,
    preprocesses them (removing small magnitudes) and outputs the raw results of
    the cca computation, including cca directions in a rotated space, and the
    cca correlation coefficient values.

    Args:
              sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                        variance matrix for x
              sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                        crossvariance matrix for x,y
              sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                        crossvariance matrix for x,y (conj) transpose of sigma_xy
              sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                        variance matrix for y
              epsilon:  small float to help with stabilizing computations
              verbose:  boolean on whether to print intermediate outputs

    Returns:
              [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                            ux and vx are (conj) transposes of each other, being
                            the canonical directions in the X subspace.
                            sx is the set of canonical correlation coefficients-
                            how well corresponding directions in vx, Vy correlate
                            with each other.
              [uy, sy, vy]: Same as above, but for Y space
              invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                            directions back to original space
              invsqrt_yy:   Same as above but for sigma_yy
              x_idxs:       The indexes of the input sigma_xx that were pruned
                            by remove_small
              y_idxs:       Same as above but for sigma_yy
    """

    (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = remove_small(
        sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon
    )

    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]

    if numx == 0 or numy == 0:
        return (
            [0, 0, 0],
            [0, 0, 0],
            np.zeros_like(sigma_xx),
            np.zeros_like(sigma_yy),
            x_idxs,
            y_idxs,
        )

    if verbose:
        print("adding eps to diagonal and taking inverse")
    sigma_xx += epsilon * np.eye(numx)
    sigma_yy += epsilon * np.eye(numy)
    inv_xx = np.linalg.pinv(sigma_xx)
    inv_yy = np.linalg.pinv(sigma_yy)

    if verbose:
        print("taking square root")
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

    if verbose:
        print("dot products...")
    arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

    if verbose:
        print("trying to take final svd")
    u, s, v = np.linalg.svd(arr)

    if verbose:
        print("computed everything!")

    return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
    """Computes threshold index of decreasing nonnegative array by summing.

    This function takes in a decreasing array nonnegative floats, and a
    threshold between 0 and 1. It returns the index i at which the sum of the
    array up to i is threshold*total mass of the array.

    Args:
              array: a 1d numpy array of decreasing, nonnegative floats
              threshold: a number between 0 and 1

    Returns:
              i: index at which np.sum(array[:i]) >= threshold
    """
    assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

    for i in range(len(array)):
        if np.sum(array[:i]) / np.sum(array) >= threshold:
            return i


def create_zero_dict(compute_dirns, dimension):
    """Outputs a zero dict when neuron activation norms too small.

    This function creates a return_dict with appropriately shaped zero entries
    when all neuron activations are very small.

    Args:
              compute_dirns: boolean, whether to have zero vectors for directions
              dimension: int, defines shape of directions

    Returns:
              return_dict: a dict of appropriately shaped zero entries
    """
    return_dict = {}
    return_dict["mean"] = (np.asarray(0), np.asarray(0))
    return_dict["sum"] = (np.asarray(0), np.asarray(0))
    return_dict["cca_coef1"] = np.asarray(0)
    return_dict["cca_coef2"] = np.asarray(0)
    return_dict["idx1"] = 0
    return_dict["idx2"] = 0

    if compute_dirns:
        return_dict["cca_dirns1"] = np.zeros((1, dimension))
        return_dict["cca_dirns2"] = np.zeros((1, dimension))

    return return_dict


def get_cca_similarity(
    acts1,
    acts2,
    epsilon=0.0,
    threshold=0.98,
    compute_coefs=True,
    compute_dirns=False,
    verbose=True,
):
    """The main function for computing cca similarities.

    This function computes the cca similarity between two sets of activations,
    returning a dict with the cca coefficients, a few statistics of the cca
    coefficients, and (optionally) the actual directions.

    Args:
              acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                     datapoints where entry (i,j) is the output of neuron i on
                     datapoint j.
              acts2: (num_neurons2, data_points) same as above, but (potentially)
                     for a different set of neurons. Note that acts1 and acts2
                     can have different numbers of neurons, but must agree on the
                     number of datapoints

              epsilon: small float to help stabilize computations

              threshold: float between 0, 1 used to get rid of trailing zeros in
                         the cca correlation coefficients to output more accurate
                         summary statistics of correlations.


              compute_coefs: boolean value determining whether coefficients
                             over neurons are computed. Needed for computing
                             directions

              compute_dirns: boolean value determining whether actual cca
                             directions are computed. (For very large neurons and
                             datasets, may be better to compute these on the fly
                             instead of store in memory.)

              verbose: Boolean, whether intermediate outputs are printed

    Returns:
              return_dict: A dictionary with outputs from the cca computations.
                           Contains neuron coefficients (combinations of neurons
                           that correspond to cca directions), the cca correlation
                           coefficients (how well aligned directions correlate),
                           x and y idxs (for computing cca directions on the fly
                           if compute_dirns=False), and summary statistics. If
                           compute_dirns=True, the cca directions are also
                           computed.
    """

    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
    # check that acts1, acts2 are transposition
    assert acts1.shape[0] < acts1.shape[1], "input must be number of neurons" "by datapoints"
    return_dict = {}

    # compute covariance with numpy function for extra stability
    numx = acts1.shape[0]
    numy = acts2.shape[0]

    covariance = np.cov(acts1, acts2)
    sigmaxx = covariance[:numx, :numx]
    sigmaxy = covariance[:numx, numx:]
    sigmayx = covariance[numx:, :numx]
    sigmayy = covariance[numx:, numx:]

    # rescale covariance to make cca computation more stable
    xmax = np.max(np.abs(sigmaxx))
    ymax = np.max(np.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= np.sqrt(xmax * ymax)
    sigmayx /= np.sqrt(xmax * ymax)

    ([u, s, v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs) = compute_ccas(
        sigmaxx, sigmaxy, sigmayx, sigmayy, epsilon=epsilon, verbose=verbose
    )

    # if x_idxs or y_idxs is all false, return_dict has zero entries
    if (not np.any(x_idxs)) or (not np.any(y_idxs)):
        return create_zero_dict(compute_dirns, acts1.shape[1])

    if compute_coefs:
        # also compute full coefficients over all neurons
        x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
        y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

        return_dict["coef_x"] = u.T
        return_dict["invsqrt_xx"] = invsqrt_xx
        return_dict["full_coef_x"] = np.zeros((numx, numx))
        np.place(return_dict["full_coef_x"], x_mask, return_dict["coef_x"])
        return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
        np.place(return_dict["full_invsqrt_xx"], x_mask, return_dict["invsqrt_xx"])

        return_dict["coef_y"] = v
        return_dict["invsqrt_yy"] = invsqrt_yy
        return_dict["full_coef_y"] = np.zeros((numy, numy))
        np.place(return_dict["full_coef_y"], y_mask, return_dict["coef_y"])
        return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
        np.place(return_dict["full_invsqrt_yy"], y_mask, return_dict["invsqrt_yy"])

        # compute means
        neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
        neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
        return_dict["neuron_means1"] = neuron_means1
        return_dict["neuron_means2"] = neuron_means2

    if compute_dirns:
        # orthonormal directions that are CCA directions
        cca_dirns1 = (
            np.dot(
                np.dot(return_dict["full_coef_x"], return_dict["full_invsqrt_xx"]),
                (acts1 - neuron_means1),
            )
            + neuron_means1
        )
        cca_dirns2 = (
            np.dot(
                np.dot(return_dict["full_coef_y"], return_dict["full_invsqrt_yy"]),
                (acts2 - neuron_means2),
            )
            + neuron_means2
        )

    # get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold(s, threshold)
    idx2 = sum_threshold(s, threshold)

    return_dict["cca_coef1"] = s
    return_dict["cca_coef2"] = s
    return_dict["x_idxs"] = x_idxs
    return_dict["y_idxs"] = y_idxs
    # summary statistics
    return_dict["mean"] = (np.mean(s[:idx1]), np.mean(s[:idx2]))
    return_dict["sum"] = (np.sum(s), np.sum(s))

    if compute_dirns:
        return_dict["cca_dirns1"] = cca_dirns1
        return_dict["cca_dirns2"] = cca_dirns2

    return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-6, compute_dirns=True):
    """Calls get_cca_similarity multiple times while adding noise.

    This function is very similar to get_cca_similarity, and can be used if
    get_cca_similarity doesn't converge for some pair of inputs. This function
    adds some noise to the activations to help convergence.

    Args:
              acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                     datapoints where entry (i,j) is the output of neuron i on
                     datapoint j.
              acts2: (num_neurons2, data_points) same as above, but (potentially)
                     for a different set of neurons. Note that acts1 and acts2
                     can have different numbers of neurons, but must agree on the
                     number of datapoints

              threshold: float between 0, 1 used to get rid of trailing zeros in
                         the cca correlation coefficients to output more accurate
                         summary statistics of correlations.

              epsilon: small float to help stabilize computations

              compute_dirns: boolean value determining whether actual cca
                             directions are computed. (For very large neurons and
                             datasets, may be better to compute these on the fly
                             instead of store in memory.)

    Returns:
              return_dict: A dictionary with outputs from the cca computations.
                           Contains neuron coefficients (combinations of neurons
                           that correspond to cca directions), the cca correlation
                           coefficients (how well aligned directions correlate),
                           x and y idxs (for computing cca directions on the fly
                           if compute_dirns=False), and summary statistics. If
                           compute_dirns=True, the cca directions are also
                           computed.
    """

    for trial in range(num_cca_trials):
        try:
            return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
        except np.linalg.LinAlgError:
            acts1 = acts1 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
            acts2 = acts2 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
            if trial + 1 == num_cca_trials:
                raise

    return return_dict
    # End of copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/cca_core.py


def top_k_pca_comps(singular_values, threshold=0.99):
    total_variance = np.sum(singular_values**2)
    explained_variance = (singular_values**2) / total_variance
    cumulative_variance = np.cumsum(explained_variance)
    return np.argmax(cumulative_variance >= threshold * total_variance) + 1


def _svcca_original(acts1, acts2):
    # Copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/tutorials/001_Introduction.ipynb
    # Modification: get_cca_similarity is in the same file.
    # Modification: top-k PCA component selection s.t. explained variance > 0.99 total variance
    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    # top-k PCA components only
    k1 = top_k_pca_comps(s1)
    k2 = top_k_pca_comps(s2)

    svacts1 = np.dot(s1[:k1] * np.eye(k1), V1[:k1])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:k2] * np.eye(k2), V2[:k2])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    # End of copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/tutorials/001_Introduction.ipynb
    return np.mean(svcca_results["cca_coef1"])


# Copied from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/pwcca.py
# Modification: get_cca_similarity is in the same file.
def compute_pwcca(acts1, acts2, epsilon=0.0):
    """Computes projection weighting for weighting CCA coefficients

    Args:
         acts1: 2d numpy array, shaped (neurons, num_datapoints)
         acts2: 2d numpy array, shaped (neurons, num_datapoints)

    Returns:
         Original cca coefficient mean and weighted mean

    """
    sresults = get_cca_similarity(
        acts1,
        acts2,
        epsilon=epsilon,
        compute_dirns=False,
        compute_coefs=True,
        verbose=False,
    )
    if np.sum(sresults["x_idxs"]) <= np.sum(sresults["y_idxs"]):
        dirns = (
            np.dot(
                sresults["coef_x"],
                (acts1[sresults["x_idxs"]] - sresults["neuron_means1"][sresults["x_idxs"]]),
            )
            + sresults["neuron_means1"][sresults["x_idxs"]]
        )
        coefs = sresults["cca_coef1"]
        acts = acts1
        idxs = sresults["x_idxs"]
    else:
        dirns = (
            np.dot(
                sresults["coef_y"],
                (acts1[sresults["y_idxs"]] - sresults["neuron_means2"][sresults["y_idxs"]]),
            )
            + sresults["neuron_means2"][sresults["y_idxs"]]
        )
        coefs = sresults["cca_coef2"]
        acts = acts2
        idxs = sresults["y_idxs"]
    P, _ = np.linalg.qr(dirns.T)
    weights = np.sum(np.abs(np.dot(P.T, acts[idxs].T)), axis=1)
    weights = weights / np.sum(weights)

    return np.sum(weights * coefs), weights, coefs
    # End of copy from https://github.com/google/svcca/blob/1f3fbf19bd31bd9b76e728ef75842aa1d9a4cd2b/pwcca.py


##################################################################################

from typing import Union  # noqa:e402

import numpy.typing as npt  # noqa:e402
import torch  # noqa:e402

# from repsim.measures.utils import (
#     SHAPE_TYPE,
#     flatten,
#     resize_wh_reps,
#     to_numpy_if_needed,
#     RepresentationalSimilarityMeasure,
# )  # noqa:e402


def svcca(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    return _svcca_original(R.T, Rp.T)


def pwcca(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    return compute_pwcca(R.T, Rp.T)[0]


class SVCCA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=svcca,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class PWCCA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=pwcca,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=False,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=False,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


# ## get rand

# In[ ]:


def score_rand(num_feats, sim_fn, shapereq_bool=False):
    all_rand_scores = []
    # num_feats = len(uniq_corr_indices_AB_forA)
    for i in range(10):
        rand_modA_feats = np.random.randint(low=0, high=weight_matrix.shape[0], size=num_feats).tolist()
        rand_modB_feats = np.random.randint(low=0, high=weight_matrix.shape[0], size=num_feats).tolist()

        if shapereq_bool:
            score = sim_fn(weight_matrix[rand_modA_feats], weight_matrix[rand_modB_feats], "nd")
        else:
            score = sim_fn(weight_matrix[rand_modA_feats], weight_matrix[rand_modB_feats])
        all_rand_scores.append(score)
    print(sum(all_rand_scores) / len(all_rand_scores))
    # plt.hist(all_rand_scores)
    # plt.show()
    return sum(all_rand_scores) / len(all_rand_scores)


# # load sae

# In[ ]:


layer_name = "blocks.8.hook_resid_pre"

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb",
    sae_id = layer_name,
    device = device
)


# # load labels

# In[ ]:


import json
with open('gpt2-small-8-res-jb-explanations.json', 'rb') as f:
    feat_labels_allData = json.load(f)


# In[ ]:


feat_labels_allData


# In[ ]:


feat_labels_allData['explanationsCount']


# In[ ]:


len(feat_labels_allData['explanations'])


# In[ ]:


feat_labels_lst = [0 for i in range(feat_labels_allData['explanationsCount'])]
feat_labels_dict = {}
for f_dict in feat_labels_allData['explanations']:
    feat_labels_lst[int(f_dict['index'])] = f_dict['description']
    feat_labels_dict[int(f_dict['index'])] = f_dict['description']
    if int(f_dict['index']) == 0:
        print(f_dict['description'])
    if int(f_dict['index']) == 24571:
        print(f_dict['description'])
    if int(f_dict['index']) == 24619:
        print(f_dict['description'])


# In[ ]:


len(feat_labels_dict)


# In[ ]:


feat_labels_lst[0]


# In[ ]:


feat_labels_lst[-10:]


# In[ ]:


feat_labels_lst[24619]


# In[ ]:


feat_labels_lst[818]


# # search for features

# In[ ]:


def find_indices_with_keyword(f_dict, keyword):
    """
    Find all indices of fList which contain the keyword in the string at those indices.

    Args:
    fList (list of str): List of strings to search within.
    keyword (str): Keyword to search for within the strings of fList.

    Returns:
    list of int: List of indices where the keyword is found within the strings of fList.
    """
    filt_dict = {}
    for index, string in f_dict.items():
        # split_list = string.split(',')
        # no_space_list = [i.replace(' ', '').lower() for i in split_list]
        # if keyword in no_space_list:
        if keyword in string:
            filt_dict[index] = string
    return filt_dict


# In[ ]:


keyword = "number"
number_feats = find_indices_with_keyword(feat_labels_dict, keyword)


# In[ ]:


number_feats


# In[ ]:


keyword = "month"
month_feats = find_indices_with_keyword(feat_labels_dict, keyword)


# In[ ]:


month_feats


# In[ ]:


set(number_feats).intersection(month_feats)


# In[ ]:


number_feats[5769]


# In[ ]:


for common_feat in set(number_feats).intersection(month_feats):
    del number_feats[common_feat]


# In[ ]:


set(number_feats).intersection(month_feats)


# # get sae actvs

# ## load model

# In[ ]:


from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-small", device = device)


# In[ ]:


layer_name = "blocks.8.hook_resid_pre"


# ## get data

# In[ ]:


# single tokens only
batch = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"] #, "eleven", "twelve"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
input_tokens['input_ids'].shape


# In[ ]:


batch_tokens = input_tokens['input_ids']


# In[ ]:


batch_tokens


# ## get actv fns

# In[107]:


from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

def get_llm_actvs(batch_tokens):
    h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
    # h_store.shape

    def store_h_hook(
        pattern: Float[Tensor, "batch seqlen d_model"],
        hook
    ):
        h_store[:] = pattern  # this works b/c changes values, not replaces entire thing

    model.run_with_hooks(
        batch_tokens,
        return_type = None,
        fwd_hooks=[
            (layer_name, store_h_hook),
        ]
    )
    return h_store


# In[ ]:


from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

def get_sae_actvs(batch_tokens):
    h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
    # h_store.shape

    def store_h_hook(
        pattern: Float[Tensor, "batch seqlen d_model"],
        hook
    ):
        h_store[:] = pattern  # this works b/c changes values, not replaces entire thing

    model.run_with_hooks(
        batch_tokens,
        return_type = None,
        fwd_hooks=[
            (layer_name, store_h_hook),
        ]
    )

    sae.eval()
    with torch.no_grad():
        return sae.encode(h_store)


# In[ ]:


h_store.shape


# ## get number and months actvs

# In[ ]:


number_feature_acts = get_sae_actvs(batch_tokens)
number_feature_acts.shape


# In[ ]:


# single tokens only
batch = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"] #, "November", "December"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
input_tokens['input_ids'].shape


# In[ ]:


batch_tokens = input_tokens['input_ids']


# In[ ]:


batch_tokens


# In[ ]:


months_feature_acts = get_sae_actvs(batch_tokens)
months_feature_acts.shape


# In[ ]:


first_dim_reshaped = number_feature_acts.shape[0] * number_feature_acts.shape[1]
reshaped_activations_A = number_feature_acts.reshape(first_dim_reshaped, number_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[ ]:


reshaped_activations_A.shape


# In[ ]:


reshaped_activations_B = months_feature_acts.reshape(first_dim_reshaped, months_feature_acts.shape[-1]).cpu()
# del feature_acts_model_B
# torch.cuda.empty_cache()


# In[ ]:


reshaped_activations_B.shape


# ## get one and Jan actvs

# In[ ]:


batch = ["one"] #, "November", "December"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_sae_actvs(batch_tokens)
one_feature_acts.shape


# In[ ]:


# single tokens only
batch = ["January"] #, "November", "December"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
jan_feature_acts = get_sae_actvs(batch_tokens)
jan_feature_acts.shape


# In[ ]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[ ]:


one_reshaped_activations.shape


# In[ ]:


jan_reshaped_activations = jan_feature_acts.reshape(first_dim_reshaped, jan_feature_acts.shape[-1]).cpu()
# del feature_acts_model_B
# torch.cuda.empty_cache()


# ### check num of nonzero features

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations)
print(one_indices[:, 1])


# In[ ]:


jan_indices = torch.nonzero(jan_reshaped_activations)
print(jan_indices[:, 1])


# In[ ]:


common_one_jan_feats = set((one_indices[:, 1]).tolist()).intersection(set((jan_indices[:, 1]).tolist()))
common_one_jan_feats


# In[ ]:


len(common_one_jan_feats)


# In[ ]:


len(set((one_indices[:, 1]).tolist()))


# In[ ]:


len(set((jan_indices[:, 1]).tolist()))


# ### check feats over 100

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations > 100)
one_indices


# In[ ]:


jan_indices = torch.nonzero(jan_reshaped_activations > 100)
jan_indices


# ## get one and two actvs

# In[ ]:


batch = ["one"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_sae_actvs(batch_tokens)
one_feature_acts.shape


# In[ ]:


# single tokens only
batch = ["two"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
two_feature_acts = get_sae_actvs(batch_tokens)
two_feature_acts.shape


# In[ ]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[ ]:


one_reshaped_activations.shape


# In[ ]:


two_reshaped_activations = two_feature_acts.reshape(first_dim_reshaped, two_feature_acts.shape[-1]).cpu()
# del feature_acts_model_B
# torch.cuda.empty_cache()


# ### check num of nonzero features

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations)
print(one_indices[:, 1])


# In[ ]:


two_indices = torch.nonzero(two_reshaped_activations)
print(two_indices[:, 1])


# In[ ]:


common_one_two_feats = set((one_indices[:, 1]).tolist()).intersection(set((two_indices[:, 1]).tolist()))
common_one_two_feats


# In[ ]:


len(common_one_two_feats)


# In[ ]:


len(set((one_indices[:, 1]).tolist()))


# In[ ]:


len(set((two_indices[:, 1]).tolist()))


# ## get one and dog actvs

# In[ ]:


batch = ["one"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_sae_actvs(batch_tokens)
one_feature_acts.shape


# In[ ]:


# single tokens only
batch = ["dog"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
dog_feature_acts = get_sae_actvs(batch_tokens)
dog_feature_acts.shape


# In[ ]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[ ]:


one_reshaped_activations.shape


# In[ ]:


dog_reshaped_activations = dog_feature_acts.reshape(first_dim_reshaped, dog_feature_acts.shape[-1]).cpu()


# ### check num of nonzero features

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations)
print(one_indices[:, 1])


# In[ ]:


dog_indices = torch.nonzero(dog_reshaped_activations)
print(dog_indices[:, 1])


# In[ ]:


common_one_dog_feats = set((one_indices[:, 1]).tolist()).intersection(set((dog_indices[:, 1]).tolist()))
common_one_dog_feats


# In[ ]:


len(common_one_dog_feats)


# In[ ]:


len(set((one_indices[:, 1]).tolist()))


# In[ ]:


len(set((dog_indices[:, 1]).tolist()))


# ## get one and blue actvs

# In[ ]:


batch = ["one"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_sae_actvs(batch_tokens)
one_feature_acts.shape


# In[ ]:


# single tokens only
batch = ["blue"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
blue_feature_acts = get_sae_actvs(batch_tokens)
blue_feature_acts.shape


# In[ ]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[ ]:


one_reshaped_activations.shape


# In[ ]:


blue_reshaped_activations = blue_feature_acts.reshape(first_dim_reshaped, blue_feature_acts.shape[-1]).cpu()


# ### check num of nonzero features

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations)
print(one_indices[:, 1])


# In[ ]:


one_reshaped_activations[:, one_indices]


# In[ ]:


blue_indices = torch.nonzero(blue_reshaped_activations)
print(blue_indices[:, 1])


# In[ ]:


blue_reshaped_activations[:, blue_indices]


# In[ ]:


common_one_blue_feats = set((one_indices[:, 1]).tolist()).intersection(set((blue_indices[:, 1]).tolist()))
common_one_blue_feats


# In[ ]:


len(common_one_blue_feats)


# In[ ]:


len(set((one_indices[:, 1]).tolist()))


# In[ ]:


len(set((blue_indices[:, 1]).tolist()))


# ### check feats over 100

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations > 100)
one_indices


# In[ ]:


blue_indices = torch.nonzero(blue_reshaped_activations > 100)
blue_indices


# ## get seqcont actvs

# In[142]:


batch = ["one two three four"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_sae_actvs(batch_tokens)
one_feature_acts.shape


# In[151]:


# single tokens only
batch = ["January February March April"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
two_feature_acts = get_sae_actvs(batch_tokens)
two_feature_acts.shape


# In[154]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[153]:


one_reshaped_activations.shape


# In[155]:


two_reshaped_activations = two_feature_acts.reshape(first_dim_reshaped, two_feature_acts.shape[-1]).cpu()
# del feature_acts_model_B
# torch.cuda.empty_cache()


# ### check num of nonzero features

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations)
print(one_indices[:, -1])


# In[ ]:


two_indices = torch.nonzero(two_reshaped_activations)
print(two_indices[:, -1])


# In[ ]:


common_one_two_feats = set((one_indices[:, 1]).tolist()).intersection(set((two_indices[:, 1]).tolist()))
common_one_two_feats


# In[ ]:


len(common_one_two_feats)


# In[ ]:


len(set((one_indices[:, 1]).tolist()))


# In[ ]:


len(set((two_indices[:, 1]).tolist()))


# ### check feats over 100

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations > 100)
one_indices


# In[ ]:


two_indices = torch.nonzero(two_reshaped_activations > 100)
two_indices


# ### check top 10% feats in common

# In[157]:


feat_k = 15
one_top_acts_values, one_top_acts_indices = one_reshaped_activations.topk(feat_k, dim=-1)
one_top_acts_indices[-1, :].sort()


# In[158]:


one_top_acts_values[-1, :]


# In[159]:


feat_k = 15
jan_top_acts_values, jan_top_acts_indices = two_reshaped_activations.topk(feat_k, dim=-1)
jan_top_acts_indices[-1, :].sort()


# In[160]:


jan_top_acts_values[-1, :]


# In[162]:


common_one_two_feats = set((one_top_acts_indices[-1, :]).tolist()).intersection(set((jan_top_acts_indices[-1, :]).tolist()))
common_one_two_feats


# ## get animal_seq actvs

# In[163]:


# single tokens only
batch = ["My favorite animal is "]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
two_feature_acts = get_sae_actvs(batch_tokens)
two_feature_acts.shape


# In[164]:


first_dim_reshaped = two_feature_acts.shape[0] * two_feature_acts.shape[1]


# In[165]:


two_reshaped_activations = two_feature_acts.reshape(first_dim_reshaped, two_feature_acts.shape[-1]).cpu()


# ### check num of nonzero features

# In[ ]:


two_indices = torch.nonzero(two_reshaped_activations)
print(two_indices[:, -1])


# In[ ]:


common_one_two_feats = set((one_indices[:, 1]).tolist()).intersection(set((two_indices[:, 1]).tolist()))
common_one_two_feats


# In[ ]:


len(common_one_two_feats)


# In[ ]:


len(set((one_indices[:, 1]).tolist()))


# In[105]:


len(set((two_indices[:, 1]).tolist()))


# ### check feats over 100

# In[ ]:


one_indices = torch.nonzero(one_reshaped_activations > 100)
one_indices


# In[106]:


two_indices = torch.nonzero(two_reshaped_activations > 100)
two_indices


# ### check top 10% feats in common

# In[147]:


feat_k = 15
top_acts_values, top_acts_indices = one_reshaped_activations.topk(feat_k, dim=-1)
top_acts_indices[-1, :].sort()


# In[148]:


top_acts_values[-1, :]


# In[166]:


feat_k = 15
ani_top_acts_values, ani_top_acts_indices = two_reshaped_activations.topk(feat_k, dim=-1)
ani_top_acts_indices[-1, :].sort()


# In[167]:


ani_top_acts_values[-1, :]


# In[168]:


set((one_top_acts_indices[-1, :]).tolist()).intersection(set((ani_top_acts_indices[-1, :]).tolist()))


# # test prompts

# In[170]:


import transformer_lens.utils as utils


# In[203]:


example_prompt = "John is tall. Mary is"
example_answer = " short"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# In[201]:


example_prompt = "John is big. Mary is"
example_answer = " small"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# # tall, big abstraction

# In[204]:


batch = ["John is tall. Mary is"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_sae_actvs(batch_tokens)
one_feature_acts.shape


# In[205]:


# single tokens only
batch = ["John is big. Mary is"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
two_feature_acts = get_sae_actvs(batch_tokens)
two_feature_acts.shape


# In[206]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
one_reshaped_activations.shape


# ### check top 10% feats in common

# In[207]:


feat_k = 15
one_top_acts_values, one_top_acts_indices = one_reshaped_activations.topk(feat_k, dim=-1)
one_top_acts_indices[-1, :].sort()


# In[208]:


feat_k = 15
one_top_acts_values, one_top_acts_indices = one_feature_acts[0, -1, :].topk(feat_k, dim=-1)
one_top_acts_indices.sort().values


# In[209]:


one_top_acts_values


# In[210]:


feat_k = 15
two_top_acts_values, two_top_acts_indices = two_feature_acts[0, -1, :].topk(feat_k, dim=-1)
two_top_acts_indices.sort().values


# In[211]:


two_top_acts_indices


# In[213]:


common_feats = set((one_top_acts_indices).tolist()).intersection(set((two_top_acts_indices).tolist()))
common_feats


# In[214]:


for f_ind in common_feats:
    print(f_ind, feat_labels_lst[f_ind])


# # compare llm actvs

# ## get seqcont vs animal actvs

# In[108]:


batch = ["one two three four"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
one_feature_acts = get_llm_actvs(batch_tokens)
one_feature_acts.shape


# In[109]:


# single tokens only
batch = ["My favorite animal is "]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']
two_feature_acts = get_llm_actvs(batch_tokens)
two_feature_acts.shape


# In[110]:


first_dim_reshaped = one_feature_acts.shape[0] * one_feature_acts.shape[1]
one_reshaped_activations = one_feature_acts.reshape(first_dim_reshaped, one_feature_acts.shape[-1]).cpu()
# del feature_acts_model_A
# torch.cuda.empty_cache()


# In[111]:


one_reshaped_activations.shape


# In[113]:


first_dim_reshaped = two_feature_acts.shape[0] * two_feature_acts.shape[1]
two_reshaped_activations = two_feature_acts.reshape(first_dim_reshaped, two_feature_acts.shape[-1]).cpu()


# ### check num of nonzero features

# In[114]:


one_indices = torch.nonzero(one_reshaped_activations)
print(one_indices[:, -1])


# In[115]:


two_indices = torch.nonzero(two_reshaped_activations)
print(two_indices[:, -1])


# In[117]:


common_one_two_feats = set((one_indices[:, 1]).tolist()).intersection(set((two_indices[:, 1]).tolist()))


# In[118]:


len(common_one_two_feats)


# ### check feats over 100

# In[120]:


one_indices = torch.nonzero(one_reshaped_activations > 100)
one_indices


# In[121]:


two_indices = torch.nonzero(two_reshaped_activations > 100)
two_indices


# ## check top 10% feats in common

# In[138]:


feat_k = 15
top_acts_values, top_acts_indices = one_reshaped_activations.topk(feat_k, dim=-1)
top_acts_indices[-1, :].sort()


# In[139]:


top_acts_values[-1, :]


# In[140]:


feat_k = 15
top_acts_values, top_acts_indices = two_reshaped_activations.topk(feat_k, dim=-1)
top_acts_indices[-1, :].sort()


# In[141]:


top_acts_values[-1, :]


# # load llm weights

# In[ ]:


# from transformers import AutoModelForCausalLM
# model_hf = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


# In[ ]:


# weight_matrix = model_hf.transformer.h[8].mlp.c_proj.weight
# weight_matrix = weight_matrix.detach().cpu().numpy()
# weight_matrix.shape


# # load sae weights
# 

# In[ ]:


weight_matrix = sae.W_dec.detach().cpu().numpy()
weight_matrix.shape


# # retain only subset of features for each domain

# In[ ]:


number_feats_lst = list(number_feats.keys())
len(number_feats_lst)


# In[ ]:


month_feats_lst = list(month_feats.keys())
len(month_feats_lst)


# In[ ]:


weight_matrix[number_feats_lst]


# # get corrs

# `batched_correlation(reshaped_activations_B, reshaped_activations_A)` : highest_correlations_indices_AB contains modA's feats as inds, and modB's feats as vals. Use the list with smaller number of features (cols) as the second arg

# In[ ]:


highest_correlations_indices_AB, highest_correlations_values_AB = batched_correlation(reshaped_activations_A, reshaped_activations_B)
highest_correlations_indices_AB = highest_correlations_indices_AB.detach().cpu().numpy()
highest_correlations_values_AB = highest_correlations_values_AB.detach().cpu().numpy()


# In[ ]:


len(highest_correlations_values_AB)


# In[ ]:


num_unq_pairs = len(list(set(highest_correlations_indices_AB)))
print(num_unq_pairs)
num_unq_pairs / len(highest_correlations_indices_AB)


# In[ ]:


sum(highest_correlations_values_AB) / len(highest_correlations_values_AB)


# ## one and jan

# In[ ]:


highest_correlations_indices_one, highest_correlations_values_one = batched_correlation(one_reshaped_activations.t(), jan_reshaped_activations.t())
highest_correlations_indices_one = highest_correlations_indices_one.detach().cpu().numpy()
highest_correlations_values_one = highest_correlations_values_one.detach().cpu().numpy()


# In[ ]:


highest_correlations_indices_one


# In[ ]:


highest_correlations_values_one


# In[ ]:


num_unq_pairs = len(list(set(highest_correlations_indices_AB)))
print(num_unq_pairs)
num_unq_pairs / len(highest_correlations_indices_AB)


# In[ ]:


sum(highest_correlations_values_AB) / len(highest_correlations_values_AB)


# ## one and dog

# In[ ]:


highest_correlations_indices_one, highest_correlations_values_one = batched_correlation(one_reshaped_activations.t(), dog_reshaped_activations.t())
highest_correlations_indices_one = highest_correlations_indices_one.detach().cpu().numpy()
highest_correlations_values_one = highest_correlations_values_one.detach().cpu().numpy()


# In[ ]:


highest_correlations_indices_one


# In[ ]:


highest_correlations_values_one


# In[ ]:


num_unq_pairs = len(list(set(highest_correlations_indices_AB)))
print(num_unq_pairs)
num_unq_pairs / len(highest_correlations_indices_AB)


# In[ ]:


sum(highest_correlations_values_AB) / len(highest_correlations_values_AB)


# ## get corr only for indices in subset

# In[ ]:





# # svcca

# In[ ]:


svcca(weight_matrix, weight_matrix[highest_correlations_indices_AB], "nd")


# In[ ]:


svcca(weight_matrix[highest_correlations_indices_AB], weight_matrix, "nd")


# In[ ]:


num_feats = weight_matrix.shape[0]
score_rand(num_feats, svcca, shapereq_bool=True)


# ## remove all corr feats less than 0.9

# In[ ]:


new_highest_correlations_indices_AB_A = []
new_highest_correlations_indices_AB = []
new_highest_correlations_values_AB = []

ind_A = 0
for ind_B, val in zip(highest_correlations_indices_AB, highest_correlations_values_AB):
    if val > 0.6:
        new_highest_correlations_indices_AB_A.append(ind_A)
        new_highest_correlations_indices_AB.append(ind_B)
        new_highest_correlations_values_AB.append(val)
    ind_A += 1


# In[ ]:


len(new_highest_correlations_values_AB)


# In[ ]:


len(list(set(new_highest_correlations_indices_AB_A)))


# In[ ]:


len(list(set(new_highest_correlations_indices_AB)))


# In[ ]:


weight_matrix_np[new_highest_correlations_indices_AB_A, :].shape


# In[ ]:


# unpaired
jaccard_similarity(weight_matrix_np[:len(new_highest_correlations_indices_AB_A), :], weight_matrix_2[:len(new_highest_correlations_indices_AB_A), :], k =10)


# In[ ]:


jaccard_similarity(weight_matrix_np[new_highest_correlations_indices_AB_A, :], weight_matrix_2[new_highest_correlations_indices_AB, :], k=10)


# In[ ]:


# unpaired
jaccard_similarity(weight_matrix_np[:len(new_highest_correlations_indices_AB_A), :], weight_matrix_2[:len(new_highest_correlations_indices_AB_A), :], k =4)


# In[ ]:


jaccard_similarity(weight_matrix_np[new_highest_correlations_indices_AB_A, :], weight_matrix_2[new_highest_correlations_indices_AB, :], k=4)


# ### 1-1 only

# In[ ]:


# highest_correlations_indices_AB contains modA's feats as inds, and modB's feats as vals

uniq_corr_indices_AB_forB = []
uniq_corr_indices_AB_forA = []
for ind_A, ind_B in zip(new_highest_correlations_indices_AB_A, new_highest_correlations_indices_AB):
    if ind_B not in uniq_corr_indices_AB_forB:
        uniq_corr_indices_AB_forA.append(ind_A)
        uniq_corr_indices_AB_forB.append(ind_B)


# In[ ]:


len(uniq_corr_indices_AB_forA)


# In[ ]:


# unpaired
jaccard_similarity(weight_matrix_np[:len(uniq_corr_indices_AB_forA)], weight_matrix_2[:len(uniq_corr_indices_AB_forA)])


# In[ ]:


jaccard_similarity(weight_matrix_np[uniq_corr_indices_AB_forA], weight_matrix_2[uniq_corr_indices_AB_forB])


# In[ ]:


all_rand_scores = []
num_feats = len(uniq_corr_indices_AB_forA)
for i in range(10):
    rand_modA_feats = np.random.randint(low=0, high=weight_matrix_np.shape[0], size=num_feats).tolist()
    rand_modB_feats = np.random.randint(low=0, high=weight_matrix_2.shape[0], size=num_feats).tolist()

    score = jaccard_similarity(weight_matrix_np[rand_modA_feats], weight_matrix_2[rand_modB_feats])
    all_rand_scores.append(score)
sum(all_rand_scores) / len(all_rand_scores)


# In[ ]:


plt.hist(all_rand_scores)
plt.show()


# # search by actvs

# ## actv data

# In[ ]:


dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)


# In[ ]:


dataset


# In[ ]:


# Function to get next batch
def get_next_batch(dataset_iter, batch_size=100):
    batch = []
    for _ in range(batch_size):
        try:
            sample = next(dataset_iter)
            batch.append(sample['text'])
        except StopIteration:
            break
    return batch

# Get a batch of 100 samples
dataset_iter = iter(dataset)
batch = get_next_batch(dataset_iter)

# Tokenize the batch
batch_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = batch_tokens['input_ids']


# ## get LLM actvs

# In[ ]:


h_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], model.cfg.d_model), device=model.cfg.device)
h_store.shape


# In[ ]:


from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple

def store_h_hook(
    pattern: Float[Tensor, "batch seqlen d_model"],
    hook
):
    h_store[:] = pattern  # this works b/c changes values, not replaces entire thing


# In[ ]:


model.run_with_hooks(
    batch_tokens,
    return_type = None,
    fwd_hooks=[
        (layer_name, store_h_hook),
    ]
)


# ## get SAE actvs

# In[ ]:


sae.eval()
with torch.no_grad():
    feature_acts = sae.encode(h_store)


# ## top actvs of specific inputs

# In[ ]:


# Get the top k largest activations for feature neurons, not batch seq. use , dim=-1
# if want to get highest batch, use dim=0
feat_k = 15
top_acts_values, top_acts_indices = feature_acts.topk(feat_k, dim=-1)

print(top_acts_indices.shape)
top_acts_values.shape


# In[ ]:


top_acts_indices


# In[ ]:


def highest_activating_tokens(
    feature_acts,
    feature_idx: int,
    k: int = 10,  # num batch_seq samples
    batch_tokens=None
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
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


# In[ ]:


from rich import print as rprint
def display_top_sequences(top_acts_indices, top_acts_values, batch_tokens):
    s = ""
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # s += f'{batch_idx}\n'
        s += f'batchID: {batch_idx}, '
        # Get the sequence as a string (with some padding on either side of our sequence)
        seq_start = max(seq_idx - 5, 0)
        # seq_end = min(seq_idx + 5, all_tokens.shape[1])
        seq_end = min(seq_idx + 5, batch_tokens.shape[1])
        seq = ""
        # Loop over the sequence, adding each token to the string (highlighting the token with the large activations)
        for i in range(seq_start, seq_end):
            # new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            new_str_token = model.to_single_str_token(batch_tokens[batch_idx, i].item()).replace("\n", "\\n").replace("<|BOS|>", "|BOS|")
            if i == seq_idx:
                new_str_token = f"[bold u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # Print the sequence, and the activation value
        s += f'Act = {value:.2f}, Seq = "{seq}"\n'

    rprint(s)


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

for feature_idx in top_acts_indices[0, -1, :]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(feature_acts, feature_idx, 1, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

for feature_idx in one_indices[:, 1]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(one_feature_acts, feature_idx, 1, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:


# get top samp_m tokens for all top feat_k feature neurons
samp_m = 5

batch = ["one"]
input_tokens = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=300)
batch_tokens = input_tokens['input_ids']

for feature_idx in one_indices[:, 1]:
    feature_idx = feature_idx.item()
    print('Feature: ', feature_idx)
    ds_top_acts_indices, ds_top_acts_values = highest_activating_tokens(one_feature_acts, feature_idx, 1, batch_tokens=batch_tokens)
    display_top_sequences(ds_top_acts_indices, ds_top_acts_values, batch_tokens=batch_tokens)


# In[ ]:




