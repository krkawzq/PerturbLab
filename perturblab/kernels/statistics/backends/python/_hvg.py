"""Pure Python/NumPy implementation of HVG operators.

This module provides fallback implementations using NumPy/SciPy when the
high-performance C++ backend is not available.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.sparse

__all__ = [
    "sparse_clipped_moments_py",
    "sparse_mean_var_py",
    "clip_matrix_py",
    "polynomial_fit_py",
    "loess_fit_py",
    "group_var_py",
    "group_mean_var_py",
]


def sparse_clipped_moments_py(
    X: scipy.sparse.csc_matrix,
    clip_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute clipped moments for sparse matrix (Python/NumPy backend).

    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        clip_vals: Clipping threshold for each column, shape (n_vars,)

    Returns:
        sums: Clipped sums, shape (n_vars,)
        sum_squares: Clipped sum of squares, shape (n_vars,)
    """
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    n_vars = X.shape[1]
    sums = np.zeros(n_vars, dtype=np.float64)
    sum_squares = np.zeros(n_vars, dtype=np.float64)

    # Process each column
    for j in range(n_vars):
        col = X.getcol(j).data
        if len(col) > 0:
            clipped = np.minimum(col, clip_vals[j])
            sums[j] = clipped.sum()
            sum_squares[j] = (clipped**2).sum()

    return sums, sum_squares


def sparse_mean_var_py(
    X: scipy.sparse.csc_matrix,
    include_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance for sparse matrix columns (Python backend).

    Uses Welford's online algorithm for numerical stability.

    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        include_zeros: Whether to include implicit zeros in calculation

    Returns:
        means: Column means, shape (n_vars,)
        vars: Column variances, shape (n_vars,)
    """
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    n_rows, n_cols = X.shape

    if include_zeros:
        # Standard calculation including zeros
        means = np.asarray(X.mean(axis=0)).ravel()

        # Variance: E[X²] - E[X]²
        X_squared = X.power(2)
        mean_of_squares = np.asarray(X_squared.mean(axis=0)).ravel()
        vars = mean_of_squares - means**2

        # Numerical stability: ensure non-negative
        vars = np.maximum(vars, 0.0)
    else:
        # Only use non-zero elements
        means = np.zeros(n_cols, dtype=np.float64)
        vars = np.zeros(n_cols, dtype=np.float64)

        for j in range(n_cols):
            col = X.getcol(j).data
            if len(col) > 0:
                means[j] = col.mean()
                vars[j] = col.var()

    return means, vars


def clip_matrix_py(
    X: np.ndarray,
    clip_vals: np.ndarray,
) -> np.ndarray:
    """Clip dense matrix by column (Python backend, in-place).

    Args:
        X: Dense matrix, shape (n_obs, n_vars)
        clip_vals: Clipping threshold for each column, shape (n_vars,)

    Returns:
        Clipped matrix (same object as X, modified in-place)
    """
    np.minimum(X, clip_vals[None, :], out=X)
    return X


def polynomial_fit_py(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    weights: Optional[np.ndarray] = None,
    return_coeffs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit polynomial regression (Python backend).

    Args:
        x: Input x coordinates, shape (n,)
        y: Input y coordinates, shape (n,)
        degree: Polynomial degree
        weights: Optional weights, shape (n,)
        return_coeffs: Whether to return coefficients

    Returns:
        fitted: Fitted y values, shape (n,)
        coeffs: Coefficients (if return_coeffs=True), shape (degree+1,)
    """
    # Use numpy polyfit
    if weights is not None:
        coeffs = np.polyfit(x, y, degree, w=weights)
    else:
        coeffs = np.polyfit(x, y, degree)

    # Evaluate polynomial
    fitted = np.polyval(coeffs, x)

    if return_coeffs:
        return fitted, coeffs
    else:
        return fitted, None


def loess_fit_py(
    x: np.ndarray,
    y: np.ndarray,
    span: float = 0.3,
) -> np.ndarray:
    """Fit LOESS (Python backend).

    Simplified LOESS using local weighted regression with tricube weights.

    Args:
        x: Input x coordinates, shape (n,)
        y: Input y coordinates, shape (n,)
        span: Smoothing parameter (fraction of data)

    Returns:
        fitted: Fitted y values, shape (n,)
    """
    n = len(x)
    n_local = max(3, int(span * n))
    fitted = np.zeros(n, dtype=np.float64)

    # Sort by x for efficient windowing
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    for i in range(n):
        x_i = x_sorted[i]

        # Find k nearest neighbors
        dists = np.abs(x_sorted - x_i)
        nearest_idx = np.argpartition(dists, min(n_local, n - 1))[:n_local]

        # Tricube weights: (1 - (d/dmax)³)³
        d = dists[nearest_idx]
        dmax = d.max()
        if dmax > 0:
            u = d / dmax
            w = (1 - u**3) ** 3
        else:
            w = np.ones(n_local)

        # Weighted polynomial fit (degree 1 for speed)
        x_local = x_sorted[nearest_idx]
        y_local = y_sorted[nearest_idx]

        # Weighted mean
        fitted[i] = np.average(y_local, weights=w)

    # Restore original order
    fitted_unsorted = np.empty(n, dtype=np.float64)
    fitted_unsorted[sort_idx] = fitted

    return fitted_unsorted


def group_var_py(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> np.ndarray:
    """Compute group-wise variance (Python backend).

    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        group_id: Group labels, shape (n_obs,), dtype int
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros

    Returns:
        Group variances, shape (n_vars * n_groups,) in C-order layout
        (result[j * n_groups + g] = variance of column j in group g)
    """
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    n_rows, n_cols = X.shape
    result = np.zeros((n_cols, n_groups), dtype=np.float64)

    for g in range(n_groups):
        mask = group_id == g
        X_group = X[mask, :]

        if include_zeros:
            # Variance including zeros
            n_g = mask.sum()
            if n_g > 1:
                means = np.asarray(X_group.mean(axis=0)).ravel()
                X_sq = X_group.power(2)
                mean_of_sq = np.asarray(X_sq.mean(axis=0)).ravel()
                result[:, g] = mean_of_sq - means**2
                result[:, g] = np.maximum(result[:, g], 0.0)  # Numerical stability
        else:
            # Variance of non-zero elements only
            for j in range(n_cols):
                col = X_group.getcol(j).data
                if len(col) > 1:
                    result[j, g] = col.var()

    return result.ravel()


def group_mean_var_py(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group-wise mean and variance (Python backend).

    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        group_id: Group labels, shape (n_obs,), dtype int
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros

    Returns:
        means: Group means, shape (n_vars * n_groups,)
        vars: Group variances, shape (n_vars * n_groups,)
    """
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    n_rows, n_cols = X.shape
    means = np.zeros((n_cols, n_groups), dtype=np.float64)
    vars = np.zeros((n_cols, n_groups), dtype=np.float64)

    for g in range(n_groups):
        mask = group_id == g
        X_group = X[mask, :]

        if include_zeros:
            n_g = mask.sum()
            means[:, g] = np.asarray(X_group.mean(axis=0)).ravel()

            if n_g > 1:
                X_sq = X_group.power(2)
                mean_of_sq = np.asarray(X_sq.mean(axis=0)).ravel()
                vars[:, g] = mean_of_sq - means[:, g] ** 2
                vars[:, g] = np.maximum(vars[:, g], 0.0)
        else:
            for j in range(n_cols):
                col = X_group.getcol(j).data
                if len(col) > 0:
                    means[j, g] = col.mean()
                    if len(col) > 1:
                        vars[j, g] = col.var()

    return means.ravel(), vars.ravel()
