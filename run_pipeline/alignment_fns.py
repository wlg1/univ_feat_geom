"""
Cross-SAE feature alignment and feature-space similarity metrics.

Builds a full activation correlation matrix (same normalization as correlation_fns),
then applies greedy / Hungarian / Sinkhorn OT mapping, and computes CKA, RSA,
and activation-space orthogonal Procrustes disparity on matched features.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.spatial.distance
import scipy.stats
import sklearn.metrics
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

try:
    from run_pipeline.correlation_fns import normalize_byChunks
except ImportError:  # pragma: no cover
    from correlation_fns import normalize_byChunks

try:
    import ot
except ImportError:  # pragma: no cover
    ot = None  # type: ignore


def build_activation_corr_matrix(
    acts_a: torch.Tensor,
    pool_a: np.ndarray,
    acts_b: torch.Tensor,
    pool_b: np.ndarray,
    batch_cols: int = 256,
) -> np.ndarray:
    """
    Full cross-correlation C[i, j] = Pearson corr(acts_a[:, pool_a[i]], acts_b[:, pool_b[j]])
    using the same per-column z-score as batched_correlation.
    """
    a_cols = acts_a[:, torch.as_tensor(pool_a, device=acts_a.device, dtype=torch.long)]
    b_cols = acts_b[:, torch.as_tensor(pool_b, device=acts_b.device, dtype=torch.long)]

    normalized_a = normalize_byChunks(a_cols, chunk_size=10000).float()
    normalized_b = normalize_byChunks(b_cols, chunk_size=10000).float()
    if torch.cuda.is_available():
        normalized_a = normalized_a.cuda()
        normalized_b = normalized_b.cuda()

    n = float(normalized_a.shape[0])
    n_a = normalized_a.shape[1]
    n_b = normalized_b.shape[1]
    c = torch.zeros((n_a, n_b), device=normalized_a.device, dtype=torch.float32)
    for start in range(0, n_b, batch_cols):
        end = min(start + batch_cols, n_b)
        block = torch.matmul(normalized_a.t(), normalized_b[:, start:end]) / n
        c[:, start:end] = block
    return c.cpu().numpy()


def map_greedy_b_to_a(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each B column j maps to argmax_i C[i, j]. Returns a_pool_index[j], corr[j]."""
    j_idx = np.arange(C.shape[1])
    i_idx = np.argmax(C, axis=0)
    corr = C[i_idx, j_idx]
    return i_idx.astype(np.int64), corr.astype(np.float64)


def map_hungarian_b_to_a(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-to-one assignment maximizing sum of correlations.
    C has shape (n_a, n_b); requires n_a == n_b (equal pool sizes).
    """
    n_a, n_b = C.shape
    if n_a != n_b:
        raise ValueError(
            "Hungarian mapping expects equal pool sizes (square correlation matrix). "
            "Use the same --corr-pool-size for both sides."
        )
    # Workers = B (rows), jobs = A (cols); cost[b, a] = -C[a, b]
    cost = (-C).T.astype(np.float64)
    r_ind, c_ind = linear_sum_assignment(cost)
    a_for_b = np.empty(n_b, dtype=np.int64)
    corr = np.empty(n_b, dtype=np.float64)
    for k in range(len(r_ind)):
        b = int(r_ind[k])
        a = int(c_ind[k])
        a_for_b[b] = a
        corr[b] = float(C[a, b])
    return a_for_b, corr


def map_sinkhorn_ot_b_to_a(
    C: np.ndarray,
    reg: float = 0.05,
    num_iter: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Entropic OT with uniform marginals; hard B->A map = argmax_a T[a, b] per column b.
    """
    if ot is None:
        raise ImportError("Install POT for OT mapping: pip install POT")

    n_a, n_b = C.shape
    a = np.ones(n_a, dtype=np.float64) / n_a
    b = np.ones(n_b, dtype=np.float64) / n_b
    m = (1.0 - C).astype(np.float64)
    t = ot.sinkhorn(a, b, m, reg=reg, numItermax=num_iter, stopThr=1e-6)
    a_pick = np.argmax(t, axis=0).astype(np.int64)
    j_idx = np.arange(n_b)
    corr = C[a_pick, j_idx].astype(np.float64)
    return a_pick, corr, t


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA; X is (n, p), Y is (n, q), same n."""
    if Y.shape[0] != X.shape[0]:
        raise ValueError("CKA requires the same number of rows (samples).")
    xc = X - X.mean(axis=0, keepdims=True)
    yc = Y - Y.mean(axis=0, keepdims=True)
    xty = xc.T @ yc
    num = float(np.linalg.norm(xty, ord="fro") ** 2)
    den = float(np.linalg.norm(xc.T @ xc, ord="fro") * np.linalg.norm(yc.T @ yc, ord="fro") + 1e-12)
    return num / den


def rsa_spearman_cosine_rows(R: np.ndarray, Rp: np.ndarray) -> float:
    """RSA Spearman between lower triangles of cosine RDMs (each row = one feature)."""
    r0 = R - R.mean(axis=1, keepdims=True)
    rp0 = Rp - Rp.mean(axis=1, keepdims=True)
    d1 = 1.0 - sklearn.metrics.pairwise_distances(r0, metric="cosine")
    d2 = 1.0 - sklearn.metrics.pairwise_distances(rp0, metric="cosine")
    s1 = scipy.spatial.distance.squareform(d1, checks=False)
    s2 = scipy.spatial.distance.squareform(d2, checks=False)
    stat = scipy.stats.spearmanr(s1, s2).statistic
    return float(stat) if stat is not None and not np.isnan(stat) else float("nan")


def procrustes_relative_rmse(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes: X, Y (n_samples, n_features), matched columns."""
    r, _scale = orthogonal_procrustes(X, Y)
    aligned = X @ r
    diff = np.linalg.norm(aligned - Y, ord="fro")
    yn = np.linalg.norm(Y, ord="fro") + 1e-12
    return float(diff / yn)


def pool_maps_from_indices(
    pool_a: np.ndarray,
    pool_b: np.ndarray,
    a_pool_idx: np.ndarray,
    corr: np.ndarray,
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, int]]:
    """Convert pool-index alignment to global feature id maps."""
    b_to_a: Dict[int, int] = {}
    b_to_corr: Dict[int, float] = {}
    for j in range(len(pool_b)):
        fb = int(pool_b[j])
        ia = int(a_pool_idx[j])
        fa = int(pool_a[ia])
        b_to_a[fb] = fa
        b_to_corr[fb] = float(corr[j])
    a_to_b: Dict[int, int] = {}
    for fb, fa in b_to_a.items():
        a_to_b[fa] = fb
    return b_to_a, b_to_corr, a_to_b


def compute_alignment_and_metrics(
    acts_a_2d: torch.Tensor,
    acts_b_2d: torch.Tensor,
    pool_a: np.ndarray,
    pool_b: np.ndarray,
    C: np.ndarray,
    selected_b: np.ndarray,
    b_to_a: Dict[int, int],
    mapping_method: str,
    ot_reg: float,
    ot_plan: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Similarity metrics on selected B features and their mapped A features."""
    sb = np.sort(selected_b.astype(np.int64))
    mapped_a = np.array([b_to_a[int(fb)] for fb in sb], dtype=np.int64)

    xa = acts_a_2d[:, torch.as_tensor(mapped_a, device=acts_a_2d.device)].float().cpu().numpy()
    xb = acts_b_2d[:, torch.as_tensor(sb, device=acts_b_2d.device)].float().cpu().numpy()

    cka = linear_cka(xa, xb)
    rsa = rsa_spearman_cosine_rows(xa.T, xb.T)
    pr_rel = procrustes_relative_rmse(xa, xb)

    per_pair = np.array([float(np.corrcoef(xa[:, i], xb[:, i])[0, 1]) for i in range(xa.shape[1])])
    mean_matched_pearson = float(np.nanmean(per_pair))

    inv_a = {int(pool_a[i]): i for i in range(len(pool_a))}
    j_idx = np.arange(len(pool_b))
    a_cols = np.array([inv_a[b_to_a[int(pool_b[j])]] for j in j_idx], dtype=np.int64)
    pool_corrs = C[a_cols, j_idx]

    out: Dict[str, Any] = {
        "mapping_method": mapping_method,
        "n_pool_features": int(C.shape[0]),
        "n_selected_b": int(len(sb)),
        "linear_cka_activations": float(cka),
        "rsa_spearman_activation_rdm": float(rsa),
        "procrustes_rel_rmse_activations": float(pr_rel),
        "mean_matched_column_pearson": mean_matched_pearson,
        "mean_abs_pool_corr_at_map": float(np.mean(np.abs(pool_corrs))),
        "mean_pool_corr_at_map": float(np.mean(pool_corrs)),
    }

    if ot_plan is not None:
        out["ot_primal_cost_sinkhorn"] = float(np.sum(ot_plan * (1.0 - C)))
        out["ot_reg"] = float(ot_reg)
    return out


def apply_mapping_method(
    C: np.ndarray,
    pool_a: np.ndarray,
    pool_b: np.ndarray,
    method: str,
    ot_reg: float,
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, int], Optional[np.ndarray]]:
    """Returns (b_to_a, b_to_corr, a_to_b, ot_plan_or_none)."""
    method = method.lower().strip()
    ot_plan: Optional[np.ndarray] = None
    if method == "greedy":
        a_idx, corr = map_greedy_b_to_a(C)
    elif method == "hungarian":
        a_idx, corr = map_hungarian_b_to_a(C)
    elif method == "ot":
        a_idx, corr, ot_plan = map_sinkhorn_ot_b_to_a(C, reg=ot_reg)
    else:
        raise ValueError(f"Unknown mapping method: {method}")
    return (*pool_maps_from_indices(pool_a, pool_b, a_idx, corr), ot_plan)
