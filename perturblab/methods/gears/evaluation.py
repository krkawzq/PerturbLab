"""GEARS evaluation metrics and analysis functions.

This module provides evaluation functions for GEARS models, leveraging
PerturbLab's unified metrics module for consistent computation.

It focuses on "Pseudobulk" level metrics, where single-cell gene expression
is averaged across cells within the same perturbation condition before
comparison. This reduces noise and focuses on the mean shift caused by the perturbation.
"""

import numpy as np
import torch

# 假设 perturblab.metrics 提供了基础的数学计算函数
# 如果这些函数未定义，请确保 perturblab.metrics 模块存在或替换为 sklearn/scipy 实现
from perturblab.metrics import mae, mse, pearson, r2
from perturblab.types import PerturbationData
from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "evaluate_predictions",
    "compute_perturbation_metrics",
    "compute_de_metrics",
]


def evaluate_predictions(
    predictions: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
    perturbations: np.ndarray,
    de_indices: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluates GEARS model predictions against ground truth.

    Collects predictions and ground truth, organizing them by perturbation
    category to facilitate downstream metric computation. It handles the
    conversion of tensors to numpy arrays and extracts DE (Differentially Expressed)
    gene subsets if indices are provided.

    Args:
        predictions: Predicted expression values.
            Shape: (n_samples, n_genes).
        ground_truth: True expression values.
            Shape: (n_samples, n_genes).
        perturbations: Perturbation labels for each sample.
            Shape: (n_samples,).
        de_indices: List of arrays, where each array contains the indices of
            differentially expressed genes for the corresponding sample.
            If None, only metrics on all genes are prepared.

    Returns:
        Dictionary containing:
            - 'pert_cat': (np.ndarray) Perturbation categories for each sample.
            - 'pred': (np.ndarray) Full prediction matrix.
            - 'truth': (np.ndarray) Full ground truth matrix.
            - 'pred_de': (np.ndarray) Predictions subset to DE genes (if de_indices provided).
            - 'truth_de': (np.ndarray) Ground truth subset to DE genes (if de_indices provided).

    Examples:
        >>> results = evaluate_predictions(
        ...     predictions=model_output,
        ...     ground_truth=batch.y,
        ...     perturbations=batch.perturbation
        ... )
    """
    # 1. Standardization: Convert PyTorch tensors to NumPy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # 2. Store full matrix results
    results = {
        "pert_cat": perturbations,
        "pred": predictions,
        "truth": ground_truth,
    }

    # 3. Extract DE gene subsets (if requested)
    # This creates a "ragged" structure or a flattened list depending on implementation,
    # here we assume a structure that can be stacked or is processed per-perturbation later.
    if de_indices is not None:
        pred_de = []
        truth_de = []

        for i, indices in enumerate(de_indices):
            # Select specific genes for this sample
            p_val = predictions[i, indices]
            t_val = ground_truth[i, indices]

            # Ensure scalar extraction works for numpy
            pred_de.append(p_val)
            truth_de.append(t_val)

        # Note: If different samples have different numbers of DE genes,
        # simple stacking might fail or create a ragged array.
        # For GEARS metrics, we typically compute means per perturbation group,
        # so keeping them aligned by sample index is key.
        # Here we use object array or similar if lengths differ,
        # but GEARS typically handles this by masking or fixed-size evaluation.
        # For safety in this general function, we assume consistent shapes or handle downstream.
        try:
            results["pred_de"] = np.vstack(pred_de)
            results["truth_de"] = np.vstack(truth_de)
        except ValueError:
            # Fallback for ragged arrays (different DE count per cell)
            results["pred_de"] = np.array(pred_de, dtype=object)
            results["truth_de"] = np.array(truth_de, dtype=object)

    return results


def compute_perturbation_metrics(
    results: dict[str, np.ndarray],
    control_label: str = "ctrl",
) -> dict:
    """Computes expression metrics for each perturbation condition.

    This function aggregates single-cell predictions into "pseudobulk" profiles
    (mean expression across cells in the same perturbation) and then calculates
    metrics comparing the predicted mean profile to the true mean profile.

    **Metrics Explanation**:
    - **MSE (Mean Squared Error)**: Measures the average squared difference between
      predicted and true expression. Lower is better. Indicates overall accuracy.
    - **Pearson (Correlation)**: Measures the linear correlation (trend similarity)
      between the predicted and true gene expression profiles across all genes.
      Range [-1, 1], higher is better. Indicates if the model captures the "shape" of the effect.
    - **R² (Coefficient of Determination)**: Measures the proportion of variance in
      the true expression explained by the model. Range (-inf, 1], higher is better.
    - **MAE (Mean Absolute Error)**: Measures the average magnitude of errors.
      Less sensitive to outliers than MSE. Lower is better.

    Args:
        results: Dictionary output from `evaluate_predictions`.
        control_label: Label used for control samples (usually excluded from DE analysis).

    Returns:
        Dictionary containing:
            - 'per_perturbation': Nested dict with {pert_name: {metric: value}}.
            - 'mse', 'pearson', etc.: Lists containing metric values for all perturbations
              (useful for calculating overall averages).

    """
    metrics = {
        "mse": [],
        "pearson": [],
        "r2": [],
        "mae": [],
        "mse_de": [],
        "pearson_de": [],
    }
    metrics_per_pert = {}

    unique_perts = np.unique(results["pert_cat"])

    for pert in unique_perts:
        metrics_per_pert[pert] = {}
        # Find indices of cells belonging to this perturbation
        p_idx = np.where(results["pert_cat"] == pert)[0]

        # --- Aggregation Step (Pseudobulk) ---
        # We average across cells (axis 0) to get a single vector of gene expression
        # for this perturbation condition.
        pred_mean = results["pred"][p_idx].mean(axis=0)
        truth_mean = results["truth"][p_idx].mean(axis=0)

        # --- Compute Metrics (All Genes) ---
        m_mse = mse(pred_mean, truth_mean)
        m_pearson = pearson(pred_mean, truth_mean)
        m_r2 = r2(pred_mean, truth_mean)
        m_mae = mae(pred_mean, truth_mean)

        metrics_per_pert[pert].update(
            {"mse": m_mse, "pearson": m_pearson, "r2": m_r2, "mae": m_mae}
        )

        # Append to global lists for summary
        metrics["mse"].append(m_mse)
        metrics["pearson"].append(m_pearson)
        metrics["r2"].append(m_r2)
        metrics["mae"].append(m_mae)

        # --- Compute Metrics (DE Genes Only) ---
        # Only computed if 'pred_de' exists and it's not the control condition
        # (since DE is usually defined relative to control).
        if "pred_de" in results and pert != control_label:
            # Note: Depending on how `evaluate_predictions` handled ragged arrays,
            # this mean computation might need adjustment. Assuming dense stacked arrays here.
            pred_de_sub = results["pred_de"][p_idx]
            truth_de_sub = results["truth_de"][p_idx]

            # Check if we have valid data to average
            if pred_de_sub.size > 0:
                pred_de_mean = pred_de_sub.mean(axis=0)
                truth_de_mean = truth_de_sub.mean(axis=0)

                m_mse_de = mse(pred_de_mean, truth_de_mean)
                m_pearson_de = pearson(pred_de_mean, truth_de_mean)

                metrics_per_pert[pert]["mse_de"] = m_mse_de
                metrics_per_pert[pert]["pearson_de"] = m_pearson_de

                metrics["mse_de"].append(m_mse_de)
                metrics["pearson_de"].append(m_pearson_de)

    metrics["per_perturbation"] = metrics_per_pert

    # Log summary statistics
    logger.info(f"Computed metrics for {len(metrics_per_pert)} perturbations")
    logger.info(f"  Overall MSE:     {np.mean(metrics['mse']):.4f}")
    logger.info(f"  Overall Pearson: {np.mean(metrics['pearson']):.4f}")
    if metrics["mse_de"]:
        logger.info(f"  DE Genes MSE:    {np.mean(metrics['mse_de']):.4f}")
        logger.info(f"  DE Genes Pearson:{np.mean(metrics['pearson_de']):.4f}")

    return metrics


def compute_de_metrics(
    data: PerturbationData,
    predictions: np.ndarray,
    de_key: str = "rank_genes_groups",
    n_top_genes: int = 20,
) -> dict:
    """Computes metrics specifically on the top-N Differentially Expressed (DE) genes.

    This function is critical for evaluating whether the model captures the
    specific effects of a perturbation, rather than just the background gene expression.

    Args:
        data: PerturbationData object containing the AnnData and DE results.
            DE results must be pre-calculated in `data.adata.uns[de_key]`.
        predictions: Predicted expression matrix (n_cells, n_genes).
        de_key: Key in `adata.uns` where differential expression results are stored.
            Default: 'rank_genes_groups'.
        n_top_genes: Number of top DE genes to consider for evaluation.

    Returns:
        Dictionary mapping perturbation names to their DE-specific metrics.
        e.g., {'TP53': {'mse': 0.1, 'pearson': 0.85, ...}, ...}
    """
    if de_key not in data.adata.uns:
        raise ValueError(
            f"DE results not found in adata.uns['{de_key}']. "
            f"Please run `data.calculate_de()` or `scanpy.tl.rank_genes_groups` first."
        )

    de_results = data.adata.uns[de_key]
    de_genes = de_results["names"]

    metrics = {}

    # Iterate over all perturbations in the dataset
    for pert in data.unique_perturbations:
        # Skip control and perturbations without DE results
        if pert in data.control_labels:
            continue
        if pert not in de_genes.dtype.names:
            continue

        # 1. Identify Top-N DE Genes
        top_genes = de_genes[pert][:n_top_genes]

        # Map gene names to indices in the expression matrix
        gene_indices = [
            data.adata.var_names.get_loc(g) for g in top_genes if g in data.adata.var_names
        ]

        if not gene_indices:
            logger.warning(f"No valid gene indices found for perturbation {pert}")
            continue

        # 2. Extract Data (Prediction vs Truth) for this perturbation
        pert_mask = data.perturbations == pert

        # Shape: (n_cells_in_pert, n_selected_genes)
        pred_pert = predictions[pert_mask][:, gene_indices]
        truth_pert = data.adata.X[pert_mask][:, gene_indices]

        # Ensure dense array
        if hasattr(truth_pert, "toarray"):
            truth_pert = truth_pert.toarray()

        # 3. Aggregate to Pseudobulk (Mean)
        pred_mean = pred_pert.mean(axis=0)
        truth_mean = truth_pert.mean(axis=0)

        # 4. Compute Metrics
        metrics[pert] = {
            "mse": mse(pred_mean, truth_mean),
            "pearson": pearson(pred_mean, truth_mean),
            "r2": r2(pred_mean, truth_mean),
            "mae": mae(pred_mean, truth_mean),
            "n_genes": len(gene_indices),
        }

    logger.info(f"Computed top-{n_top_genes} DE metrics for {len(metrics)} perturbations")
    return metrics
