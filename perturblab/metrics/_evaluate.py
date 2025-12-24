"""Comprehensive evaluation function combining all metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from perturblab.utils import get_logger

from ._expression import compute_expression_metrics
from ._distribution import compute_distribution_metrics
from ._direction import compute_direction_metrics
from ._deg_overlap import compute_deg_overlap_metrics

logger = get_logger()

__all__ = ["evaluate_prediction"]


def evaluate_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray,
    pred_degs: Optional[pd.DataFrame] = None,
    true_degs: Optional[pd.DataFrame] = None,
    include_expression: bool = True,
    include_distribution: bool = True,
    include_direction: bool = True,
    include_deg_overlap: bool = True,
    mmd_gamma: float = 1.0,
    deg_top_n: List[int] = [20, 50, 100, 200],
    deg_p_thresholds: List[float] = [0.05, 0.01],
    deg_fdr_thresholds: List[float] = [0.05, 0.01],
) -> Dict[str, float]:
    """Comprehensive evaluation of perturbation prediction.
    
    Computes all available metrics to evaluate prediction quality, including:
    - Expression accuracy (R², Pearson, MSE, MAE, etc.)
    - Distribution similarity (MMD, Wasserstein)
    - Direction consistency (delta agreement)
    - DEG overlap (if DEG DataFrames provided)
    
    Parameters
    ----------
    pred
        Predicted expression matrix [cells × genes].
    true
        True expression matrix [cells × genes].
    ctrl
        Control expression matrix [cells × genes].
    pred_degs
        Predicted DEG DataFrame (from differential_expression).
        Required if include_deg_overlap=True.
    true_degs
        True DEG DataFrame (from differential_expression).
        Required if include_deg_overlap=True.
    include_expression
        Whether to include expression metrics.
    include_distribution
        Whether to include distribution metrics.
    include_direction
        Whether to include direction consistency metrics.
    include_deg_overlap
        Whether to include DEG overlap metrics.
        Requires pred_degs and true_degs to be provided.
    mmd_gamma
        Gamma parameter for MMD computation.
    deg_top_n
        List of top-N thresholds for DEG overlap.
    deg_p_thresholds
        List of p-value thresholds for DEG overlap.
    deg_fdr_thresholds
        List of FDR thresholds for DEG overlap.
    
    Returns
    -------
    dict
        Dictionary containing all requested metrics.
    
    Examples
    --------
    >>> import perturblab as pl
    >>> import numpy as np
    >>> 
    >>> # Generate example data
    >>> pred = np.random.rand(100, 50)
    >>> true = np.random.rand(100, 50)
    >>> ctrl = np.random.rand(100, 50)
    >>> 
    >>> # Evaluate without DEG overlap
    >>> metrics = pl.metrics.evaluate_prediction(
    ...     pred, true, ctrl,
    ...     include_deg_overlap=False
    ... )
    >>> 
    >>> # Evaluate with DEG overlap (requires DEG DataFrames)
    >>> from anndata import AnnData
    >>> pred_adata = AnnData(pred)
    >>> true_adata = AnnData(true)
    >>> pred_adata.obs['condition'] = 'treated'
    >>> true_adata.obs['condition'] = 'treated'
    >>> 
    >>> pred_degs = pl.data.process.differential_expression(
    ...     pred_adata, groupby_key='condition', reference='ctrl'
    ... )
    >>> true_degs = pl.data.process.differential_expression(
    ...     true_adata, groupby_key='condition', reference='ctrl'
    ... )
    >>> 
    >>> metrics = pl.metrics.evaluate_prediction(
    ...     pred, true, ctrl,
    ...     pred_degs=pred_degs,
    ...     true_degs=true_degs
    ... )
    >>> 
    >>> # Print summary
    >>> print("Evaluation Results:")
    >>> for key, value in metrics.items():
    ...     if isinstance(value, float):
    ...         print(f"  {key}: {value:.4f}")
    
    See Also
    --------
    compute_expression_metrics : Expression accuracy metrics
    compute_distribution_metrics : Distribution similarity metrics
    compute_direction_metrics : Direction consistency metrics
    compute_deg_overlap_metrics : DEG overlap metrics
    """
    all_metrics = {}
    
    # Expression metrics
    if include_expression:
        logger.info("Computing expression metrics...")
        expr_metrics = compute_expression_metrics(
            pred, true, ctrl, include_delta=True
        )
        all_metrics.update(expr_metrics)
    
    # Distribution metrics
    if include_distribution:
        logger.info("Computing distribution metrics...")
        dist_metrics = compute_distribution_metrics(
            pred, true, mmd_gamma=mmd_gamma
        )
        all_metrics.update(dist_metrics)
    
    # Direction metrics
    if include_direction:
        logger.info("Computing direction consistency metrics...")
        dir_metrics = compute_direction_metrics(pred, true, ctrl)
        all_metrics.update(dir_metrics)
    
    # DEG overlap metrics
    if include_deg_overlap:
        if pred_degs is None or true_degs is None:
            logger.warning(
                "DEG overlap requested but pred_degs or true_degs not provided. "
                "Skipping DEG overlap metrics."
            )
        else:
            logger.info("Computing DEG overlap metrics...")
            deg_metrics = compute_deg_overlap_metrics(
                pred_degs, true_degs,
                top_n_list=deg_top_n,
                p_thresholds=deg_p_thresholds,
                fdr_thresholds=deg_fdr_thresholds,
            )
            all_metrics.update(deg_metrics)
    
    logger.info(f"✅ Evaluation complete! Computed {len(all_metrics)} metrics.")
    
    return all_metrics

