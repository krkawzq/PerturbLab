"""Expression prediction metrics for perturbation analysis.

This module provides metrics to evaluate how well predicted gene expression
profiles match the ground truth, including both absolute expression and
perturbation effect (delta) metrics.
"""

from __future__ import annotations

from typing import Dict
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "r_squared",
    "pearson_correlation",
    "mse",
    "rmse",
    "mae",
    "cosine_similarity_score",
    "l2_distance",
    "compute_expression_metrics",
]


def _to_dense(data: np.ndarray) -> np.ndarray:
    """Convert sparse matrix to dense numpy array if needed."""
    if hasattr(data, "toarray"):
        return data.toarray()
    return data


def r_squared(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute R² (coefficient of determination) between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes R²
        for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        R² score. Higher is better (max 1.0).

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import r_squared
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> r2 = r_squared(pred, true)
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    # Compute mean across cells
    mean_pred = np.mean(pred, axis=0)
    mean_true = np.mean(true, axis=0)

    if ctrl is not None:
        ctrl = _to_dense(ctrl)
        mean_ctrl = np.mean(ctrl, axis=0)
        mean_pred = mean_pred - mean_ctrl
        mean_true = mean_true - mean_ctrl

    try:
        with catch_warnings():
            simplefilter("ignore")
            return float(r2_score(mean_true, mean_pred))
    except Exception as e:
        logger.warning(f"R² computation failed: {e}")
        return np.nan


def pearson_correlation(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute Pearson correlation coefficient between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes correlation
        for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        Pearson correlation coefficient. Range [-1, 1], higher is better.

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import pearson_correlation
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> corr = pearson_correlation(pred, true)
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    # Compute mean across cells
    mean_pred = np.mean(pred, axis=0)
    mean_true = np.mean(true, axis=0)

    if ctrl is not None:
        ctrl = _to_dense(ctrl)
        mean_ctrl = np.mean(ctrl, axis=0)
        mean_pred = mean_pred - mean_ctrl
        mean_true = mean_true - mean_ctrl

    try:
        with catch_warnings():
            simplefilter("ignore")
            return float(pearsonr(mean_true, mean_pred)[0])
    except Exception as e:
        logger.warning(f"Pearson correlation computation failed: {e}")
        return np.nan


def mse(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute Mean Squared Error between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes MSE
        for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        Mean squared error. Lower is better (min 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import mse
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> error = mse(pred, true)
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    # Compute mean across cells
    mean_pred = np.mean(pred, axis=0)
    mean_true = np.mean(true, axis=0)

    if ctrl is not None:
        ctrl = _to_dense(ctrl)
        mean_ctrl = np.mean(ctrl, axis=0)
        mean_pred = mean_pred - mean_ctrl
        mean_true = mean_true - mean_ctrl

    return float(mean_squared_error(mean_true, mean_pred))


def rmse(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute Root Mean Squared Error between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes RMSE
        for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        Root mean squared error. Lower is better (min 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import rmse
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> error = rmse(pred, true)
    """
    return np.sqrt(mse(pred, true, ctrl))


def mae(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute Mean Absolute Error between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes MAE
        for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        Mean absolute error. Lower is better (min 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import mae
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> error = mae(pred, true)
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    # Compute mean across cells
    mean_pred = np.mean(pred, axis=0)
    mean_true = np.mean(true, axis=0)

    if ctrl is not None:
        ctrl = _to_dense(ctrl)
        mean_ctrl = np.mean(ctrl, axis=0)
        mean_pred = mean_pred - mean_ctrl
        mean_true = mean_true - mean_ctrl

    return float(mean_absolute_error(mean_true, mean_pred))


def cosine_similarity_score(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute cosine similarity between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes cosine
        similarity for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        Cosine similarity. Range [-1, 1], higher is better.

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import cosine_similarity_score
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> sim = cosine_similarity_score(pred, true)
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    # Compute mean across cells
    mean_pred = np.mean(pred, axis=0)
    mean_true = np.mean(true, axis=0)

    if ctrl is not None:
        ctrl = _to_dense(ctrl)
        mean_ctrl = np.mean(ctrl, axis=0)
        mean_pred = mean_pred - mean_ctrl
        mean_true = mean_true - mean_ctrl

    return float(cosine_similarity([mean_true], [mean_pred])[0, 0])


def l2_distance(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray | None = None,
) -> float:
    """Compute L2 (Euclidean) distance between predicted and true expression.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes]. If provided, computes L2
        distance for perturbation effects (delta = treated - control).

    Returns
    -------
    float
        L2 distance. Lower is better (min 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import l2_distance
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> dist = l2_distance(pred, true)
    """
    pred = _to_dense(pred)
    true = _to_dense(true)

    # Compute mean across cells
    mean_pred = np.mean(pred, axis=0)
    mean_true = np.mean(true, axis=0)

    if ctrl is not None:
        ctrl = _to_dense(ctrl)
        mean_ctrl = np.mean(ctrl, axis=0)
        mean_pred = mean_pred - mean_ctrl
        mean_true = mean_true - mean_ctrl

    return float(np.linalg.norm(mean_true - mean_pred))


def compute_expression_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray,
    include_delta: bool = True,
) -> Dict[str, float]:
    """Compute all expression metrics at once.

    Computes both absolute expression metrics and delta (perturbation effect) metrics.

    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes].
    include_delta
        Whether to include delta metrics (treated - control).

    Returns
    -------
    dict
        Dictionary containing all metrics:
        - Absolute metrics: R_squared, Pearson_Correlation, MSE, RMSE, MAE,
          Cosine_Similarity, L2
        - Delta metrics (if include_delta=True): R_squared_delta,
          Pearson_Correlation_delta, MSE_delta, RMSE_delta, MAE_delta,
          Cosine_Similarity_delta, L2_delta

    Examples
    --------
    >>> import numpy as np
    >>> from perturblab.metrics import compute_expression_metrics
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> ctrl = np.random.rand(100, 50)
    >>> metrics = compute_expression_metrics(pred, true, ctrl)
    >>> print(f"R² = {metrics['R_squared']:.3f}")
    >>> print(f"R² delta = {metrics['R_squared_delta']:.3f}")
    """
    metrics = {}

    # Absolute expression metrics
    metrics["R_squared"] = r_squared(pred, true, ctrl=None)
    metrics["Pearson_Correlation"] = pearson_correlation(pred, true, ctrl=None)
    metrics["MSE"] = mse(pred, true, ctrl=None)
    metrics["RMSE"] = rmse(pred, true, ctrl=None)
    metrics["MAE"] = mae(pred, true, ctrl=None)
    metrics["Cosine_Similarity"] = cosine_similarity_score(pred, true, ctrl=None)
    metrics["L2"] = l2_distance(pred, true, ctrl=None)

    # Delta metrics (perturbation effect)
    if include_delta:
        metrics["R_squared_delta"] = r_squared(pred, true, ctrl=ctrl)
        metrics["Pearson_Correlation_delta"] = pearson_correlation(pred, true, ctrl=ctrl)
        metrics["MSE_delta"] = mse(pred, true, ctrl=ctrl)
        metrics["RMSE_delta"] = rmse(pred, true, ctrl=ctrl)
        metrics["MAE_delta"] = mae(pred, true, ctrl=ctrl)
        metrics["Cosine_Similarity_delta"] = cosine_similarity_score(pred, true, ctrl=ctrl)
        metrics["L2_delta"] = l2_distance(pred, true, ctrl=ctrl)

    return metrics
