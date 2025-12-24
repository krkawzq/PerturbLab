"""Distribution comparison metrics for perturbation analysis.

This module provides metrics to evaluate how well the predicted gene expression
distribution matches the ground truth distribution, including MMD and Wasserstein distance.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as scipy_wasserstein

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "mmd",
    "wasserstein_distance",
    "compute_distribution_metrics",
]


def _to_dense(data: np.ndarray) -> np.ndarray:
    """Convert sparse matrix to dense numpy array if needed."""
    if hasattr(data, "toarray"):
        return data.toarray()
    return data


def mmd(
    pred: np.ndarray,
    true: np.ndarray,
    kernel: Literal["rbf"] = "rbf",
    gamma: float = 1.0,
    per_gene: bool = False,
) -> float | np.ndarray:
    """Compute Maximum Mean Discrepancy (MMD) between predicted and true distributions.

    MMD measures the distance between two probability distributions by comparing
    their mean embeddings in a reproducing kernel Hilbert space (RKHS).

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    kernel
        Kernel type. Currently only 'rbf' (Radial Basis Function) is supported.
    gamma
        RBF kernel parameter (bandwidth). Smaller values give smoother kernels.
    per_gene
        If True, compute MMD for each gene separately and return array.
        If False (default), compute average MMD across all genes.

    Returns
    -------
    float or np.ndarray
        If per_gene=False: Average MMD value across all genes.
        If per_gene=True: Array of MMD values for each gene.
        Lower values indicate better match between distributions.

    Notes
    -----
    MMD is computed using the unbiased estimator:
        MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    where k is the kernel function, x~P (true), y~Q (pred).

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import mmd
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> mmd_score = mmd(pred, true, gamma=1.0)
    >>> print(f"MMD = {mmd_score:.4f}")
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    if kernel != "rbf":
        raise ValueError(f"Unsupported kernel: {kernel}. Only 'rbf' is currently supported.")

    n_genes = pred.shape[1]
    mmd_vals = []

    for g in range(n_genes):
        # Extract gene expression for this gene
        pred_gene = pred[:, g].reshape(-1, 1)
        true_gene = true[:, g].reshape(-1, 1)

        # Compute pairwise squared Euclidean distances
        dist_pred = cdist(pred_gene, pred_gene, metric="sqeuclidean")
        dist_true = cdist(true_gene, true_gene, metric="sqeuclidean")
        dist_cross = cdist(pred_gene, true_gene, metric="sqeuclidean")

        # Apply RBF kernel: k(x,y) = exp(-gamma * ||x-y||²)
        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_true)
        Kxy = np.exp(-gamma * dist_cross)

        # MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        mmd_val = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
        mmd_vals.append(mmd_val)

    mmd_vals = np.array(mmd_vals)

    if per_gene:
        return mmd_vals
    else:
        return float(np.mean(mmd_vals))


def wasserstein_distance(
    pred: np.ndarray,
    true: np.ndarray,
    per_gene: bool = False,
) -> float | np.ndarray:
    """Compute Wasserstein distance (Earth Mover's Distance) between distributions.

    The Wasserstein distance measures the minimum cost of transforming one
    distribution into another, where cost is proportional to the distance moved.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    per_gene
        If True, compute Wasserstein distance for each gene separately.
        If False (default), compute average distance across all genes.

    Returns
    -------
    float or np.ndarray
        If per_gene=False: Average Wasserstein distance across all genes.
        If per_gene=True: Array of Wasserstein distances for each gene.
        Lower values indicate better match between distributions.

    Notes
    -----
    The Wasserstein distance is also known as:
    - Earth Mover's Distance (EMD)
    - Kantorovich-Rubinstein metric
    - Optimal transport distance

    It is particularly useful for comparing distributions that may have different
    supports or shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import wasserstein_distance
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> ws_dist = wasserstein_distance(pred, true)
    >>> print(f"Wasserstein distance = {ws_dist:.4f}")
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    n_genes = pred.shape[1]
    ws_vals = []

    for g in range(n_genes):
        # Extract gene expression for this gene
        pred_gene = pred[:, g].flatten()
        true_gene = true[:, g].flatten()

        # Compute Wasserstein distance for this gene
        ws_val = scipy_wasserstein(pred_gene, true_gene)
        ws_vals.append(ws_val)

    ws_vals = np.array(ws_vals)

    if per_gene:
        return ws_vals
    else:
        return float(np.mean(ws_vals))


def compute_distribution_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    mmd_gamma: float = 1.0,
) -> Dict[str, float]:
    """Compute all distribution comparison metrics at once.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    mmd_gamma
        Gamma parameter for MMD's RBF kernel.

    Returns
    -------
    dict
        Dictionary containing:
        - MMD: Maximum Mean Discrepancy (average across genes)
        - Wasserstein: Wasserstein distance (average across genes)

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import compute_distribution_metrics
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> metrics = compute_distribution_metrics(pred, true)
    >>> print(f"MMD = {metrics['MMD']:.4f}")
    >>> print(f"Wasserstein = {metrics['Wasserstein']:.4f}")
    """
    return {
        "MMD": mmd(pred, true, kernel="rbf", gamma=mmd_gamma, per_gene=False),
        "Wasserstein": wasserstein_distance(pred, true, per_gene=False),
    }
