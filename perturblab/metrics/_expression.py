"""Expression prediction metrics for perturbation analysis.

Structure:
1. Functional Primitives: Pure mathematical/statistical functions (NumPy arrays in, float out).
2. Data Transforms: Functions to handle pseudobulking and delta calculation.
3. High-Level API: User-facing functions that orchestrate transforms and metrics.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    # Functional Primitives
    "r2",
    "pearson",
    "mse",
    "rmse",
    "mae",
    "cosine",
    "l2",
    # Transformations
    "to_dense",
    "aggregate_pseudobulk",
    "compute_delta",
    # High-Level API
    "evaluate_perturbation",
]

def r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Coefficient of Determination ($R^2$)."""
    try:
        with catch_warnings():
            simplefilter("ignore")
            return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan

def pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Pearson correlation coefficient ($r$)."""
    try:
        with catch_warnings():
            simplefilter("ignore")
            val, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            return 0.0 if np.isnan(val) else float(val)
    except Exception:
        return 0.0

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Mean Squared Error ($MSE$)."""
    return float(mean_squared_error(y_true, y_pred))

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Root Mean Squared Error ($RMSE$)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Mean Absolute Error ($MAE$)."""
    return float(mean_absolute_error(y_true, y_pred))

def cosine(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Cosine Similarity."""
    # Reshape to (1, -1) if 1D array to satisfy sklearn requirement
    u = y_pred.reshape(1, -1) if y_pred.ndim == 1 else y_pred
    v = y_true.reshape(1, -1) if y_true.ndim == 1 else y_true
    return float(cosine_similarity(u, v)[0, 0])

def l2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes L2 (Euclidean) Distance."""
    return float(np.linalg.norm(y_true - y_pred))

def to_dense(data: np.ndarray) -> np.ndarray:
    """Ensure data is a dense numpy array."""
    if hasattr(data, "toarray"):
        return data.toarray()
    return np.asarray(data)

def aggregate_pseudobulk(matrix: np.ndarray) -> np.ndarray:
    """Aggregates single-cell matrix to pseudobulk vector (mean expression).
    
    Args:
        matrix: shape (n_cells, n_genes)
    Returns:
        vector: shape (n_genes,)
    """
    dense_mat = to_dense(matrix)
    return np.mean(dense_mat, axis=0)

def compute_delta(
    target: np.ndarray, 
    control: np.ndarray
) -> np.ndarray:
    """Computes perturbation effect (Delta = Target - Control).
    
    Both inputs will be aggregated to pseudobulk first if they are matrices.
    """
    v_target = aggregate_pseudobulk(target) if target.ndim > 1 else target
    v_ctrl = aggregate_pseudobulk(control) if control.ndim > 1 else control
    return v_target - v_ctrl


def prepare_evaluation_vectors(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: Optional[np.ndarray] = None,
    mode: str = 'absolute'
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to prepare aligned vectors based on evaluation mode.
    
    Args:
        mode: 'absolute' (raw expression) or 'delta' (change from control).
    """
    # 1. Base Aggregation (Pseudobulk)
    v_pred = aggregate_pseudobulk(pred)
    v_true = aggregate_pseudobulk(true)

    # 2. Mode Handling
    if mode == 'delta':
        if ctrl is None:
            raise ValueError("Control data required for delta metrics.")
        v_ctrl = aggregate_pseudobulk(ctrl)
        v_pred = v_pred - v_ctrl
        v_true = v_true - v_ctrl
    
    return v_pred, v_true

def evaluate_perturbation(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: Optional[np.ndarray] = None,
    include_delta: bool = True
) -> Dict[str, float]:
    """Main entry point to compute all standard metrics for a perturbation.
    
    Computes metrics for both absolute expression and (optionally) delta expression.
    
    Args:
        pred: Predicted expression (n_cells, n_genes) or aggregated (n_genes,)
        true: True expression (n_cells, n_genes) or aggregated (n_genes,)
        ctrl: Control expression (n_cells, n_genes) or aggregated (n_genes,)
        include_delta: Whether to compute 'delta' metrics (pred - ctrl vs true - ctrl)
        
    Returns:
        Dictionary with keys like 'MSE', 'Pearson', 'MSE_delta', etc.
    """
    metrics = {}
    
    # 1. Compute Absolute Metrics
    v_pred_abs, v_true_abs = prepare_evaluation_vectors(pred, true, mode='absolute')
    
    metric_funcs = {
        "R2": r2,
        "Pearson": pearson,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Cosine": cosine,
        "L2": l2
    }

    for name, func in metric_funcs.items():
        metrics[name] = func(v_pred_abs, v_true_abs)

    # 2. Compute Delta Metrics (Optional)
    if include_delta and ctrl is not None:
        try:
            v_pred_delta, v_true_delta = prepare_evaluation_vectors(pred, true, ctrl, mode='delta')
            for name, func in metric_funcs.items():
                metrics[f"{name}_delta"] = func(v_pred_delta, v_true_delta)
        except Exception as e:
            logger.warning(f"Skipping delta metrics: {e}")

    return metrics
