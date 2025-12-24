# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython implementation of HVG operators.

This module provides Cython-accelerated implementations as a middle-ground
fallback between C++ (fastest) and pure Python (slowest).

Some functions adapted from Scanpy:
https://github.com/scverse/scanpy
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD-3-Clause
"""

import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport fabs, sqrt

cnp.import_array()


ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t INT64_t
ctypedef cnp.int32_t INT32_t


# =============================================================================
# Sparse Clipped Moments (from Scanpy)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_clipped_moments_cy(
    const DTYPE_t[:] data,
    const INT64_t[:] indices,
    const INT64_t[:] indptr,
    const DTYPE_t[:] clip_vals,
):
    """Compute clipped moments for sparse CSC matrix.
    
    Adapted from Scanpy's _sum_and_sum_squares_clipped.
    
    Args:
        data: Non-zero values
        indices: Row indices
        indptr: Column pointers (CSC format)
        clip_vals: Clipping threshold for each column
    
    Returns:
        sums: Clipped sums
        sum_squares: Clipped sum of squares
    """
    cdef:
        Py_ssize_t n_cols = indptr.shape[0] - 1
        Py_ssize_t col, i, start, end
        DTYPE_t val, clipped, clip_val
        cnp.ndarray[DTYPE_t, ndim=1] sums = np.zeros(n_cols, dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=1] sum_squares = np.zeros(n_cols, dtype=np.float64)
    
    for col in range(n_cols):
        clip_val = clip_vals[col]
        start = indptr[col]
        end = indptr[col + 1]
        
        for i in range(start, end):
            val = data[i]
            clipped = val if val < clip_val else clip_val
            sums[col] += clipped
            sum_squares[col] += clipped * clipped
    
    return sums, sum_squares


# =============================================================================
# Sparse Mean/Variance (Welford algorithm)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mean_var_cy(
    const DTYPE_t[:] data,
    const INT64_t[:] indices,
    const INT64_t[:] indptr,
    Py_ssize_t n_rows,
    bint include_zeros=True,
):
    """Compute mean and variance for sparse CSC matrix.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        data: Non-zero values
        indices: Row indices
        indptr: Column pointers
        n_rows: Number of rows
        include_zeros: Whether to include implicit zeros
    
    Returns:
        means: Column means
        vars: Column variances
    """
    cdef:
        Py_ssize_t n_cols = indptr.shape[0] - 1
        Py_ssize_t col, i, start, end, nnz, n
        DTYPE_t val, mean, M2, delta, delta2
        cnp.ndarray[DTYPE_t, ndim=1] means = np.zeros(n_cols, dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=1] vars = np.zeros(n_cols, dtype=np.float64)
    
    for col in range(n_cols):
        start = indptr[col]
        end = indptr[col + 1]
        nnz = end - start
        
        if include_zeros:
            n = n_rows
        else:
            n = nnz
        
        if n == 0:
            continue
        
        # Welford algorithm
        mean = 0.0
        M2 = 0.0
        
        # Process non-zero elements
        for i in range(start, end):
            val = data[i]
            delta = val - mean
            mean += delta / (i - start + 1)
            delta2 = val - mean
            M2 += delta * delta2
        
        # If include_zeros, add zeros contribution
        if include_zeros and nnz < n_rows:
            # Add zeros
            for i in range(nnz, n_rows):
                delta = 0.0 - mean
                mean += delta / (i + 1)
                delta2 = 0.0 - mean
                M2 += delta * delta2
        
        means[col] = mean
        if n > 1:
            vars[col] = M2 / n  # Population variance
        else:
            vars[col] = 0.0
    
    return means, vars


# =============================================================================
# Clip Matrix (Dense)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def clip_matrix_cy(
    DTYPE_t[:, :] X,
    const DTYPE_t[:] clip_vals,
):
    """Clip dense matrix by column (in-place).
    
    Args:
        X: Dense matrix (n_rows, n_cols), modified in-place
        clip_vals: Clipping threshold for each column
    """
    cdef:
        Py_ssize_t n_rows = X.shape[0]
        Py_ssize_t n_cols = X.shape[1]
        Py_ssize_t row, col
        DTYPE_t clip_val, val
    
    for col in range(n_cols):
        clip_val = clip_vals[col]
        for row in range(n_rows):
            val = X[row, col]
            if val > clip_val:
                X[row, col] = clip_val


# =============================================================================
# Polynomial Fit
# =============================================================================

def polynomial_fit_cy(
    const DTYPE_t[:] x,
    const DTYPE_t[:] y,
    int degree,
    const DTYPE_t[:] weights=None,
):
    """Fit polynomial regression.
    
    Args:
        x: Input x coordinates
        y: Input y coordinates
        degree: Polynomial degree
        weights: Optional weights
    
    Returns:
        fitted: Fitted y values
        coeffs: Polynomial coefficients
    """
    # Use NumPy's polyfit (already optimized)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    
    if weights is not None:
        w_np = np.asarray(weights)
        coeffs = np.polyfit(x_np, y_np, degree, w=w_np)
    else:
        coeffs = np.polyfit(x_np, y_np, degree)
    
    fitted = np.polyval(coeffs, x_np)
    
    return fitted, coeffs


# =============================================================================
# LOESS Fit (Simplified)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def loess_fit_cy(
    const DTYPE_t[:] x,
    const DTYPE_t[:] y,
    double span=0.3,
):
    """Fit LOESS (simplified version).
    
    Args:
        x: Input x coordinates
        y: Input y coordinates
        span: Smoothing parameter
    
    Returns:
        fitted: Fitted y values
    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t n_local = max(3, int(span * n))
        Py_ssize_t i, j, k
        DTYPE_t x_i, d, dmax, u, w_sum, y_sum
        cnp.ndarray[DTYPE_t, ndim=1] fitted = np.zeros(n, dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=1] dists = np.zeros(n, dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=1] weights = np.zeros(n_local, dtype=np.float64)
        cnp.ndarray[INT64_t, ndim=1] nearest_idx
    
    # Sort by x
    sort_idx = np.argsort(np.asarray(x))
    x_sorted = np.asarray(x)[sort_idx]
    y_sorted = np.asarray(y)[sort_idx]
    
    for i in range(n):
        x_i = x_sorted[i]
        
        # Compute distances
        for j in range(n):
            dists[j] = fabs(x_sorted[j] - x_i)
        
        # Find k nearest neighbors
        nearest_idx = np.argpartition(dists, min(n_local, n-1))[:n_local]
        
        # Compute tricube weights
        dmax = 0.0
        for k in range(n_local):
            j = nearest_idx[k]
            d = dists[j]
            if d > dmax:
                dmax = d
        
        w_sum = 0.0
        y_sum = 0.0
        
        for k in range(n_local):
            j = nearest_idx[k]
            if dmax > 0:
                u = dists[j] / dmax
                weights[k] = (1.0 - u * u * u) ** 3  # Tricube
            else:
                weights[k] = 1.0
            
            w_sum += weights[k]
            y_sum += weights[k] * y_sorted[j]
        
        if w_sum > 0:
            fitted[i] = y_sum / w_sum
        else:
            fitted[i] = y_sorted[i]
    
    # Restore original order
    fitted_unsorted = np.empty(n, dtype=np.float64)
    fitted_unsorted[sort_idx] = fitted
    
    return fitted_unsorted


# =============================================================================
# Group Operations
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def group_var_cy(
    const DTYPE_t[:] data,
    const INT64_t[:] indices,
    const INT64_t[:] indptr,
    Py_ssize_t n_rows,
    const INT32_t[:] group_id,
    int n_groups,
    bint include_zeros=True,
):
    """Compute group-wise variance.
    
    Args:
        data: Non-zero values
        indices: Row indices
        indptr: Column pointers
        n_rows: Number of rows
        group_id: Group labels for each row
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros
    
    Returns:
        Group variances, shape (n_cols * n_groups,)
    """
    cdef:
        Py_ssize_t n_cols = indptr.shape[0] - 1
        Py_ssize_t col, i, start, end, g, row
        INT32_t gid
        DTYPE_t val, mean, M2, delta, delta2
        cnp.ndarray[DTYPE_t, ndim=2] vars = np.zeros((n_cols, n_groups), dtype=np.float64)
        cnp.ndarray[INT64_t, ndim=1] counts = np.zeros(n_groups, dtype=np.int64)
        cnp.ndarray[DTYPE_t, ndim=1] means = np.zeros(n_groups, dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=1] M2s = np.zeros(n_groups, dtype=np.float64)
    
    for col in range(n_cols):
        # Reset for each column
        for g in range(n_groups):
            counts[g] = 0
            means[g] = 0.0
            M2s[g] = 0.0
        
        start = indptr[col]
        end = indptr[col + 1]
        
        # Count group sizes
        if include_zeros:
            for row in range(n_rows):
                gid = group_id[row]
                counts[gid] += 1
        
        # Process non-zero elements (Welford)
        for i in range(start, end):
            row = indices[i]
            gid = group_id[row]
            val = data[i]
            
            if not include_zeros:
                counts[gid] += 1
            
            delta = val - means[gid]
            means[gid] += delta / counts[gid]
            delta2 = val - means[gid]
            M2s[gid] += delta * delta2
        
        # If include_zeros, add implicit zeros contribution
        if include_zeros:
            # Count how many non-zeros per group
            for g in range(n_groups):
                counts[g] = 0
            
            for row in range(n_rows):
                gid = group_id[row]
                counts[gid] += 1
            
            # Recalculate with zeros
            for g in range(n_groups):
                if counts[g] > 1:
                    vars[col, g] = M2s[g] / counts[g]
                else:
                    vars[col, g] = 0.0
        else:
            for g in range(n_groups):
                if counts[g] > 1:
                    vars[col, g] = M2s[g] / counts[g]
                else:
                    vars[col, g] = 0.0
    
    return vars.ravel()


@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_var_cy(
    const DTYPE_t[:] data,
    const INT64_t[:] indices,
    const INT64_t[:] indptr,
    Py_ssize_t n_rows,
    const INT32_t[:] group_id,
    int n_groups,
    bint include_zeros=True,
):
    """Compute group-wise mean and variance.
    
    Args:
        data: Non-zero values
        indices: Row indices
        indptr: Column pointers
        n_rows: Number of rows
        group_id: Group labels for each row
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros
    
    Returns:
        means: Group means, shape (n_cols * n_groups,)
        vars: Group variances, shape (n_cols * n_groups,)
    """
    cdef:
        Py_ssize_t n_cols = indptr.shape[0] - 1
        Py_ssize_t col, i, start, end, g, row
        INT32_t gid
        DTYPE_t val, delta, delta2
        cnp.ndarray[DTYPE_t, ndim=2] means = np.zeros((n_cols, n_groups), dtype=np.float64)
        cnp.ndarray[DTYPE_t, ndim=2] vars = np.zeros((n_cols, n_groups), dtype=np.float64)
        cnp.ndarray[INT64_t, ndim=1] counts = np.zeros(n_groups, dtype=np.int64)
        cnp.ndarray[DTYPE_t, ndim=1] M2s = np.zeros(n_groups, dtype=np.float64)
    
    for col in range(n_cols):
        # Reset
        for g in range(n_groups):
            counts[g] = 0
            means[col, g] = 0.0
            M2s[g] = 0.0
        
        start = indptr[col]
        end = indptr[col + 1]
        
        # Count group sizes
        if include_zeros:
            for row in range(n_rows):
                gid = group_id[row]
                counts[gid] += 1
        
        # Welford algorithm
        for i in range(start, end):
            row = indices[i]
            gid = group_id[row]
            val = data[i]
            
            if not include_zeros:
                counts[gid] += 1
            
            delta = val - means[col, gid]
            means[col, gid] += delta / counts[gid]
            delta2 = val - means[col, gid]
            M2s[gid] += delta * delta2
        
        # Compute variances
        for g in range(n_groups):
            if counts[g] > 1:
                vars[col, g] = M2s[g] / counts[g]
            else:
                vars[col, g] = 0.0
    
    return means.ravel(), vars.ravel()

