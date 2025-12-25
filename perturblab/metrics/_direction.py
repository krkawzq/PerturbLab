"""Direction consistency metrics for perturbation analysis.

This module evaluates whether the predicted perturbation effects have the correct
direction (up-regulation vs down-regulation) compared to ground truth.
"""

from __future__ import annotations

import numpy as np

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "delta_direction_accuracy",
    "compute_direction_metrics",
]


def _to_dense(data: np.ndarray) -> np.ndarray:
    """Convert sparse matrix to dense numpy array if needed."""
    if hasattr(data, "toarray"):
        return data.toarray()
    return data


def delta_direction_accuracy(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray,
    per_gene: bool = False,
) -> float | np.ndarray:
    """Compute delta direction agreement accuracy.

    For each gene, checks if the predicted perturbation effect (pred - ctrl)
    has the same sign (positive/negative) as the true perturbation effect
    (true - ctrl). This metric evaluates whether the model correctly predicts
    whether genes are up-regulated or down-regulated.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes].
    per_gene
        If True, return direction accuracy for each gene separately.
        If False (default), return average accuracy across all genes.

    Returns
    -------
    float or np.ndarray
        If per_gene=False: Average direction consistency accuracy (range [0, 1]).
        If per_gene=True: Boolean array indicating direction consistency for each gene.
        Higher values indicate better direction prediction.

    Notes
    -----
    This metric is particularly useful when:
    - The exact magnitude of change is less important than the direction
    - Evaluating whether a model captures biological regulation patterns
    - Comparing models that may have different scaling but correct trends

    A value of 1.0 means perfect direction agreement for all genes.
    A value of 0.5 would be expected from random guessing.

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import delta_direction_accuracy
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> ctrl = np.random.rand(100, 50)
    >>> acc = delta_direction_accuracy(pred, true, ctrl)
    >>> print(f"Direction accuracy = {acc:.2%}")

    See Also
    --------
    perturblab.metrics.compute_expression_metrics : For magnitude-aware metrics
    """
    pred = _to_dense(pred)
    true = _to_dense(true)
    ctrl = _to_dense(ctrl)

    # Compute mean expression across cells
    pred_mean = np.mean(pred, axis=0)
    true_mean = np.mean(true, axis=0)
    ctrl_mean = np.mean(ctrl, axis=0)

    # Calculate deltas (perturbation effects)
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean

    # Check direction consistency for each gene
    # Both positive or both negative = consistent
    pred_positive = pred_delta > 0
    true_positive = true_delta > 0

    consistent_genes = pred_positive == true_positive

    if per_gene:
        return consistent_genes
    else:
        return float(np.mean(consistent_genes))


def compute_direction_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray,
) -> dict[str, float]:
    """Compute all direction consistency metrics.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes].

    Returns
    -------
    dict
        Dictionary containing:
        - delta_agreement_acc: Direction agreement accuracy
        - n_genes_up_pred: Number of up-regulated genes in prediction
        - n_genes_down_pred: Number of down-regulated genes in prediction
        - n_genes_up_true: Number of up-regulated genes in truth
        - n_genes_down_true: Number of down-regulated genes in truth
        - n_genes_agree: Number of genes with correct direction

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import compute_direction_metrics
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> ctrl = np.random.rand(100, 50)
    >>> metrics = compute_direction_metrics(pred, true, ctrl)
    >>> print(f"Direction accuracy = {metrics['delta_agreement_acc']:.2%}")
    >>> print(f"Genes with correct direction: {metrics['n_genes_agree']}")
    """
    pred = _to_dense(pred)
    true = _to_dense(true)
    ctrl = _to_dense(ctrl)

    # Compute per-gene direction consistency
    consistent_genes = delta_direction_accuracy(pred, true, ctrl, per_gene=True)

    # Compute mean expression across cells
    pred_mean = np.mean(pred, axis=0)
    true_mean = np.mean(true, axis=0)
    ctrl_mean = np.mean(ctrl, axis=0)

    # Calculate deltas
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean

    return {
        "delta_agreement_acc": float(np.mean(consistent_genes)),
        "n_genes_up_pred": int(np.sum(pred_delta > 0)),
        "n_genes_down_pred": int(np.sum(pred_delta < 0)),
        "n_genes_up_true": int(np.sum(true_delta > 0)),
        "n_genes_down_true": int(np.sum(true_delta < 0)),
        "n_genes_agree": int(np.sum(consistent_genes)),
    }
