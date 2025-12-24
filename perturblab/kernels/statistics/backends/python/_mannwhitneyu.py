"""NumPy/SciPy fallback implementation for Mann-Whitney U test.

This is a pure Python implementation that always works but is slower
than C++ or Cython backends.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import mannwhitneyu as scipy_mannwhitneyu

__all__ = ["mannwhitneyu_scipy", "group_mean_numpy"]


def mannwhitneyu_scipy(
    matrix_csc: csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    use_continuity: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mann-Whitney U test using SciPy (fallback implementation).

    Parameters
    ----------
    matrix_csc : csc_matrix
        Expression matrix in CSC format, shape (n_cells, n_genes).
    group_id : np.ndarray
        Group assignments, shape (n_cells,).
        0: reference group, 1 to n_targets: target groups.
    n_targets : int
        Number of target groups.
    use_continuity : bool, default=True
        Whether to use continuity correction.

    Returns
    -------
    U1 : np.ndarray
        U statistic for target groups, shape (n_targets, n_genes).
    U2 : np.ndarray
        U statistic for reference group, shape (n_targets, n_genes).
    P : np.ndarray
        Two-sided p-values, shape (n_targets, n_genes).
    """
    n_cols = matrix_csc.shape[1]

    # Allocate output arrays
    U1 = np.zeros((n_targets, n_cols), dtype=np.float64)
    U2 = np.zeros((n_targets, n_cols), dtype=np.float64)
    P = np.ones((n_targets, n_cols), dtype=np.float64)

    # Get reference group mask
    ref_mask = group_id == 0

    # Process each target group
    for t in range(n_targets):
        tar_mask = group_id == (t + 1)

        # Process each gene
        for g in range(n_cols):
            # Extract values
            ref_vals = matrix_csc[ref_mask, g].toarray().flatten()
            tar_vals = matrix_csc[tar_mask, g].toarray().flatten()

            if len(ref_vals) == 0 or len(tar_vals) == 0:
                continue

            # Call scipy
            try:
                stat, pval = scipy_mannwhitneyu(
                    tar_vals, ref_vals, alternative="two-sided", use_continuity=use_continuity
                )

                # scipy returns U for the first group (target)
                U1[t, g] = stat
                # U2 = n_ref * n_tar - U1
                U2[t, g] = len(ref_vals) * len(tar_vals) - stat
                P[t, g] = pval
            except Exception:
                # Handle edge cases (all values identical, etc.)
                P[t, g] = 1.0

    return U1, U2, P


def group_mean_numpy(
    matrix_csc: csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> np.ndarray:
    """Compute group means using NumPy (fallback implementation).

    Parameters
    ----------
    matrix_csc : csc_matrix
        Expression matrix in CSC format, shape (n_cells, n_genes).
    group_id : np.ndarray
        Group assignments, shape (n_cells,).
    n_groups : int
        Number of groups.
    include_zeros : bool, default=True
        Whether to include zeros in mean calculation.

    Returns
    -------
    means : np.ndarray
        Group means, shape (n_groups, n_genes).
    """
    n_cols = matrix_csc.shape[1]
    means = np.zeros((n_groups, n_cols), dtype=np.float64)

    for g in range(n_groups):
        mask = group_id == g
        if np.sum(mask) == 0:
            continue

        # Extract group data
        group_data = matrix_csc[mask, :]

        # Compute mean
        if include_zeros:
            # Include implicit zeros in the mean
            means[g, :] = np.asarray(group_data.mean(axis=0)).flatten()
        else:
            # Only average non-zero values
            for col in range(n_cols):
                col_data = group_data[:, col].data
                if len(col_data) > 0:
                    means[g, col] = np.mean(col_data)

    return means
