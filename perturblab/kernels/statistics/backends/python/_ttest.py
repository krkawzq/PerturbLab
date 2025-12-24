"""NumPy/SciPy fallback implementation for t-test.

This is a pure Python implementation that always works but is slower
than C++ backend.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import ttest_ind

__all__ = ["ttest_numpy", "log_fold_change_numpy"]


def ttest_numpy(
    matrix_csc: csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    method: str = "welch",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """T-test using SciPy (fallback implementation).

    Parameters
    ----------
    matrix_csc : csc_matrix
        Expression matrix in CSC format, shape (n_cells, n_genes).
    group_id : np.ndarray
        Group assignments, shape (n_cells,).
        0: reference group, 1 to n_targets: target groups.
    n_targets : int
        Number of target groups.
    method : str, default='welch'
        Test method: 'student' or 'welch'.

    Returns
    -------
    t_stat : np.ndarray
        T-statistics, shape (n_targets, n_genes).
    p_val : np.ndarray
        Two-sided p-values, shape (n_targets, n_genes).
    mean_diff : np.ndarray
        Mean differences (target - reference), shape (n_targets, n_genes).
    log2_fc : np.ndarray
        Log2 fold changes, shape (n_targets, n_genes).
    """
    n_cols = matrix_csc.shape[1]

    # Allocate outputs
    t_stat = np.zeros((n_targets, n_cols), dtype=np.float64)
    p_val = np.ones((n_targets, n_cols), dtype=np.float64)
    mean_diff = np.zeros((n_targets, n_cols), dtype=np.float64)
    log2_fc = np.zeros((n_targets, n_cols), dtype=np.float64)

    ref_mask = group_id == 0
    equal_var = method == "student"

    # Process each target group
    for t in range(n_targets):
        tar_mask = group_id == (t + 1)

        # Process each gene
        for g in range(n_cols):
            ref_vals = matrix_csc[ref_mask, g].toarray().flatten()
            tar_vals = matrix_csc[tar_mask, g].toarray().flatten()

            if len(ref_vals) < 2 or len(tar_vals) < 2:
                continue

            try:
                stat, pval = ttest_ind(tar_vals, ref_vals, equal_var=equal_var)
                t_stat[t, g] = stat
                p_val[t, g] = pval

                ref_mean = np.mean(ref_vals)
                tar_mean = np.mean(tar_vals)
                mean_diff[t, g] = tar_mean - ref_mean
                log2_fc[t, g] = np.log2((tar_mean + 1e-9) / (ref_mean + 1e-9))
            except Exception:
                # Handle edge cases
                p_val[t, g] = 1.0

    return t_stat, p_val, mean_diff, log2_fc


def log_fold_change_numpy(
    matrix_csc: csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    pseudocount: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log fold change using NumPy (fallback implementation).

    Parameters
    ----------
    matrix_csc : csc_matrix
        Expression matrix in CSC format, shape (n_cells, n_genes).
    group_id : np.ndarray
        Group assignments, shape (n_cells,).
        0: reference group, 1 to n_targets: target groups.
    n_targets : int
        Number of target groups.
    pseudocount : float, default=1e-9
        Small value added to avoid log(0).

    Returns
    -------
    mean_diff : np.ndarray
        Mean differences (target - reference), shape (n_targets, n_genes).
    log2_fc : np.ndarray
        Log2 fold changes, shape (n_targets, n_genes).
    """
    n_cols = matrix_csc.shape[1]

    mean_diff = np.zeros((n_targets, n_cols), dtype=np.float64)
    log2_fc = np.zeros((n_targets, n_cols), dtype=np.float64)

    # Compute reference means
    ref_mask = group_id == 0
    ref_data = matrix_csc[ref_mask, :]
    ref_means = np.asarray(ref_data.mean(axis=0)).flatten()

    # Compute for each target group
    for t in range(n_targets):
        tar_mask = group_id == (t + 1)
        tar_data = matrix_csc[tar_mask, :]
        tar_means = np.asarray(tar_data.mean(axis=0)).flatten()

        mean_diff[t, :] = tar_means - ref_means
        log2_fc[t, :] = np.log2((tar_means + pseudocount) / (ref_means + pseudocount))

    return mean_diff, log2_fc
