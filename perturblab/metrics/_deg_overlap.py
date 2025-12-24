"""Differential Expression Gene (DEG) overlap metrics.

This module evaluates how well predicted DEGs match the true DEGs,
using various ranking and significance thresholds.
"""

from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "deg_overlap_topn",
    "deg_overlap_pvalue",
    "deg_overlap_fdr",
    "compute_deg_overlap_metrics",
]


def deg_overlap_topn(
    pred_degs: pd.DataFrame,
    true_degs: pd.DataFrame,
    top_n: int = 20,
    gene_col: str = "feature",
) -> float:
    """Compute DEG overlap for top-N ranked genes.

    Measures the fraction of top-N true DEGs that are also in the top-N predicted DEGs.
    This metric focuses on the most significant genes.

    Parameters
    ----------
    pred_degs
        Predicted DEG results (from differential_expression).
        Must contain a column with gene names (default: 'feature').
    true_degs
        True DEG results (from differential_expression).
        Must contain a column with gene names (default: 'feature').
    top_n
        Number of top genes to consider.
    gene_col
        Name of the column containing gene names.

    Returns
    -------
    float
        Overlap fraction (range [0, 1]). Higher is better.
        Overlap = |top-N pred ∩ top-N true| / top-N

    Notes
    -----
    The DataFrames should be pre-sorted by significance (e.g., by p-value or
    absolute log fold change). This function takes the first `top_n` genes
    from each DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from perturblab.metrics import deg_overlap_topn
    >>> pred_degs = pd.DataFrame({'feature': ['GeneA', 'GeneB', 'GeneC', ...]})
    >>> true_degs = pd.DataFrame({'feature': ['GeneB', 'GeneC', 'GeneD', ...]})
    >>> overlap = deg_overlap_topn(pred_degs, true_degs, top_n=20)
    >>> print(f"Top-20 overlap = {overlap:.2%}")
    """
    if len(pred_degs) < top_n or len(true_degs) < top_n:
        logger.warning(
            f"Insufficient DEGs: pred={len(pred_degs)}, true={len(true_degs)}, "
            f"requested top_n={top_n}"
        )
        return np.nan

    # Get top-N genes
    pred_top = set(pred_degs.head(top_n)[gene_col])
    true_top = set(true_degs.head(top_n)[gene_col])

    # Compute overlap
    overlap = len(pred_top & true_top)
    return float(overlap / top_n)


def deg_overlap_pvalue(
    pred_degs: pd.DataFrame,
    true_degs: pd.DataFrame,
    p_threshold: float = 0.05,
    gene_col: str = "feature",
    pval_col: str = "p_value",
) -> float:
    """Compute DEG overlap for genes with p-value < threshold.

    Measures the fraction of significant true DEGs that are also significant
    in the predictions.

    Parameters
    ----------
    pred_degs
        Predicted DEG results (from differential_expression).
    true_degs
        True DEG results (from differential_expression).
    p_threshold
        P-value threshold for significance (default: 0.05).
    gene_col
        Name of the column containing gene names.
    pval_col
        Name of the column containing p-values.

    Returns
    -------
    float
        Overlap fraction (range [0, 1]). Higher is better.
        Overlap = |sig pred ∩ sig true| / |sig true|
        Returns NaN if no significant genes in true DEGs.

    Examples
    --------
    >>> import pandas as pd
    >>> from perturblab.metrics import deg_overlap_pvalue
    >>> pred_degs = pd.DataFrame({
    ...     'feature': ['GeneA', 'GeneB', 'GeneC'],
    ...     'p_value': [0.001, 0.01, 0.9]
    ... })
    >>> true_degs = pd.DataFrame({
    ...     'feature': ['GeneB', 'GeneC', 'GeneD'],
    ...     'p_value': [0.001, 0.02, 0.8]
    ... })
    >>> overlap = deg_overlap_pvalue(pred_degs, true_degs, p_threshold=0.05)
    >>> print(f"P<0.05 overlap = {overlap:.2%}")
    """
    # Get significant genes
    pred_sig = set(pred_degs[pred_degs[pval_col] < p_threshold][gene_col])
    true_sig = set(true_degs[true_degs[pval_col] < p_threshold][gene_col])

    if len(true_sig) == 0:
        logger.warning("No significant genes in true DEGs at p<{p_threshold}")
        return np.nan

    # Compute overlap
    overlap = len(pred_sig & true_sig)
    return float(overlap / len(true_sig))


def deg_overlap_fdr(
    pred_degs: pd.DataFrame,
    true_degs: pd.DataFrame,
    fdr_threshold: float = 0.05,
    gene_col: str = "feature",
    fdr_col: str = "fdr",
) -> float:
    """Compute DEG overlap for genes with FDR < threshold.

    Measures the fraction of significant true DEGs (FDR-corrected) that are also
    significant in the predictions.

    Parameters
    ----------
    pred_degs
        Predicted DEG results (from differential_expression).
    true_degs
        True DEG results (from differential_expression).
    fdr_threshold
        FDR threshold for significance (default: 0.05).
    gene_col
        Name of the column containing gene names.
    fdr_col
        Name of the column containing FDR values.

    Returns
    -------
    float
        Overlap fraction (range [0, 1]). Higher is better.
        Overlap = |sig pred ∩ sig true| / |sig true|
        Returns NaN if no significant genes in true DEGs.

    Examples
    --------
    >>> import pandas as pd
    >>> from perturblab.metrics import deg_overlap_fdr
    >>> pred_degs = pd.DataFrame({
    ...     'feature': ['GeneA', 'GeneB', 'GeneC'],
    ...     'fdr': [0.001, 0.01, 0.9]
    ... })
    >>> true_degs = pd.DataFrame({
    ...     'feature': ['GeneB', 'GeneC', 'GeneD'],
    ...     'fdr': [0.001, 0.02, 0.8]
    ... })
    >>> overlap = deg_overlap_fdr(pred_degs, true_degs, fdr_threshold=0.05)
    >>> print(f"FDR<0.05 overlap = {overlap:.2%}")
    """
    # Get significant genes
    pred_sig = set(pred_degs[pred_degs[fdr_col] < fdr_threshold][gene_col])
    true_sig = set(true_degs[true_degs[fdr_col] < fdr_threshold][gene_col])

    if len(true_sig) == 0:
        logger.warning(f"No significant genes in true DEGs at FDR<{fdr_threshold}")
        return np.nan

    # Compute overlap
    overlap = len(pred_sig & true_sig)
    return float(overlap / len(true_sig))


def compute_deg_overlap_metrics(
    pred_degs: pd.DataFrame,
    true_degs: pd.DataFrame,
    top_n_list: List[int] = [20, 50, 100, 200],
    p_thresholds: List[float] = [0.05, 0.01],
    fdr_thresholds: List[float] = [0.05, 0.01],
    gene_col: str = "feature",
    pval_col: str = "p_value",
    fdr_col: str = "fdr",
    sort_by: Literal["p_value", "fdr", "abs_logfc"] = "p_value",
) -> Dict[str, float]:
    """Compute all DEG overlap metrics.

    Computes overlap at multiple Top-N thresholds and significance levels.

    Parameters
    ----------
    pred_degs
        Predicted DEG results (from differential_expression).
    true_degs
        True DEG results (from differential_expression).
    top_n_list
        List of top-N thresholds to evaluate.
    p_thresholds
        List of p-value thresholds to evaluate.
    fdr_thresholds
        List of FDR thresholds to evaluate.
    gene_col
        Name of the column containing gene names.
    pval_col
        Name of the column containing p-values.
    fdr_col
        Name of the column containing FDR values.
    sort_by
        How to sort DEGs before taking top-N:
        - 'p_value': Sort by p-value (ascending)
        - 'fdr': Sort by FDR (ascending)
        - 'abs_logfc': Sort by absolute log fold change (descending)

    Returns
    -------
    dict
        Dictionary containing all overlap metrics:
        - Top{N}_DEG_Overlap: for each N in top_n_list
        - P<{thresh}_DEG_Overlap: for each threshold in p_thresholds
        - FDR<{thresh}_DEG_Overlap: for each threshold in fdr_thresholds

    Examples
    --------
    >>> import pandas as pd
    >>> from perturblab.metrics import compute_deg_overlap_metrics
    >>> pred_degs = pd.DataFrame({
    ...     'feature': ['GeneA', 'GeneB', 'GeneC'],
    ...     'p_value': [0.001, 0.01, 0.9],
    ...     'fdr': [0.01, 0.05, 0.95],
    ...     'log2_fold_change': [2.1, 1.5, 0.1]
    ... })
    >>> true_degs = pd.DataFrame({
    ...     'feature': ['GeneB', 'GeneC', 'GeneD'],
    ...     'p_value': [0.001, 0.02, 0.8],
    ...     'fdr': [0.01, 0.08, 0.90],
    ...     'log2_fold_change': [1.8, 1.2, 0.2]
    ... })
    >>> metrics = compute_deg_overlap_metrics(pred_degs, true_degs)
    >>> print(f"Top-20 overlap = {metrics['Top20_DEG_Overlap']:.2%}")
    """
    # Sort DEGs according to specified criterion
    if sort_by == "p_value":
        pred_sorted = pred_degs.sort_values(pval_col, ascending=True)
        true_sorted = true_degs.sort_values(pval_col, ascending=True)
    elif sort_by == "fdr":
        pred_sorted = pred_degs.sort_values(fdr_col, ascending=True)
        true_sorted = true_degs.sort_values(fdr_col, ascending=True)
    elif sort_by == "abs_logfc":
        # Compute absolute log fold change if not present
        if "abs_logfoldchanges" not in pred_degs.columns:
            pred_degs = pred_degs.copy()
            pred_degs["abs_logfoldchanges"] = pred_degs["log2_fold_change"].abs()
        if "abs_logfoldchanges" not in true_degs.columns:
            true_degs = true_degs.copy()
            true_degs["abs_logfoldchanges"] = true_degs["log2_fold_change"].abs()
        pred_sorted = pred_degs.sort_values("abs_logfoldchanges", ascending=False)
        true_sorted = true_degs.sort_values("abs_logfoldchanges", ascending=False)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    metrics = {}

    # Top-N overlaps
    for n in top_n_list:
        overlap = deg_overlap_topn(pred_sorted, true_sorted, top_n=n, gene_col=gene_col)
        metrics[f"Top{n}_DEG_Overlap"] = overlap

    # P-value thresholds
    for p_thresh in p_thresholds:
        overlap = deg_overlap_pvalue(
            pred_degs,
            true_degs,
            p_threshold=p_thresh,
            gene_col=gene_col,
            pval_col=pval_col,
        )
        # Format threshold for key (remove decimal if integer)
        thresh_str = f"{p_thresh:g}".replace(".", "")
        metrics[f"P<0{thresh_str}_DEG_Overlap"] = overlap

    # FDR thresholds
    for fdr_thresh in fdr_thresholds:
        overlap = deg_overlap_fdr(
            pred_degs,
            true_degs,
            fdr_threshold=fdr_thresh,
            gene_col=gene_col,
            fdr_col=fdr_col,
        )
        # Format threshold for key
        thresh_str = f"{fdr_thresh:g}".replace(".", "")
        metrics[f"FDR<0{thresh_str}_DEG_Overlap"] = overlap

    return metrics
