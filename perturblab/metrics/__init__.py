"""Evaluation metrics for perturbation prediction models.

This module provides comprehensive metrics to evaluate how well predicted
perturbation effects match ground truth, including expression accuracy,
distribution similarity, direction consistency, and DEG overlap.

Available Metrics
-----------------

Expression Metrics (_expression.py):
    - r2: R² coefficient of determination
    - pearson: Pearson correlation coefficient
    - mse: Mean squared error
    - rmse: Root mean squared error
    - mae: Mean absolute error
    - cosine: Cosine similarity
    - l2: L2 (Euclidean) distance
    - evaluate_perturbation: Compute all expression metrics at once

Distribution Metrics (_distribution.py):
    - mmd: Maximum Mean Discrepancy
    - wasserstein_distance: Wasserstein distance (Earth Mover's Distance)
    - compute_distribution_metrics: All distribution metrics at once

Direction Consistency (_direction.py):
    - delta_direction_accuracy: Gene-level direction agreement
    - compute_direction_metrics: Direction metrics with statistics

DEG Overlap (_deg_overlap.py):
    - deg_overlap_topn: Overlap for top-N genes
    - deg_overlap_pvalue: Overlap for p-value threshold
    - deg_overlap_fdr: Overlap for FDR threshold
    - compute_deg_overlap_metrics: All DEG overlap metrics

Spatial Autocorrelation (_spatial.py, from scanpy):
    - morans_i: Moran's I global spatial autocorrelation
    - gearys_c: Geary's C local spatial autocorrelation
    - compute_spatial_metrics: Both spatial metrics at once

Classification (_classification.py, from scanpy):
    - confusion_matrix: Labeled confusion matrix for clustering/annotation

Comprehensive Evaluation:
    - evaluate_prediction: Compute all metrics at once

Examples
--------
Basic usage:

>>> from perturblab import metrics
>>>
>>> # Compute individual expression metrics
>>> r2_score = metrics.r2(pred_vector, true_vector)
>>> pearson_score = metrics.pearson(pred_vector, true_vector)
>>> mse_score = metrics.mse(pred_vector, true_vector)
>>>
>>> # Compute all expression metrics at once
>>> expr_metrics = metrics.evaluate_perturbation(
...     pred_data,
...     true_data,
...     ctrl_data,
...     include_delta=True
... )
>>> print(f"R² = {expr_metrics['R2']:.3f}")
>>> print(f"R² delta = {expr_metrics['R2_delta']:.3f}")
>>>
>>> # Compute distribution metrics
>>> dist_metrics = metrics.compute_distribution_metrics(pred_data, true_data)
>>> print(f"MMD = {dist_metrics['MMD']:.4f}")
>>>
>>> # Compute direction accuracy
>>> direction_acc = metrics.delta_direction_accuracy(pred_data, true_data, ctrl_data)
>>> print(f"Direction accuracy = {direction_acc:.2%}")
>>>
>>> # Comprehensive evaluation
>>> all_metrics = metrics.evaluate_prediction(
...     pred_data,
...     true_data,
...     ctrl_data
... )
"""

from ._classification import (
    confusion_matrix,
)
from ._deg_overlap import (
    compute_deg_overlap_metrics,
    deg_overlap_fdr,
    deg_overlap_pvalue,
    deg_overlap_topn,
)
from ._direction import (
    compute_direction_metrics,
    delta_direction_accuracy,
)
from ._distribution import (
    compute_distribution_metrics,
    mmd,
    wasserstein_distance,
)
from ._evaluate import evaluate_prediction
from ._expression import (
    cosine,
    evaluate_perturbation,
    l2,
    mae,
    mse,
    pearson,
    r2,
    rmse,
)
from ._spatial import (
    compute_spatial_metrics,
    gearys_c,
    morans_i,
)

__all__ = [
    # Expression metrics (functional primitives)
    "r2",
    "pearson",
    "mse",
    "rmse",
    "mae",
    "cosine",
    "l2",
    "evaluate_perturbation",
    # Distribution metrics
    "mmd",
    "wasserstein_distance",
    "compute_distribution_metrics",
    # Direction metrics
    "delta_direction_accuracy",
    "compute_direction_metrics",
    # DEG overlap metrics
    "deg_overlap_topn",
    "deg_overlap_pvalue",
    "deg_overlap_fdr",
    "compute_deg_overlap_metrics",
    # Spatial metrics
    "morans_i",
    "gearys_c",
    "compute_spatial_metrics",
    # Classification metrics
    "confusion_matrix",
    # Comprehensive evaluation
    "evaluate_prediction",
]
