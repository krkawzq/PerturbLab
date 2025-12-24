# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True

"""Cython-accelerated Mann-Whitney U test for single-cell differential expression.

This module provides high-performance implementations of the Mann-Whitney U test
(Wilcoxon rank-sum test) optimized for sparse single-cell expression matrices.

Key features:
    - Sparse CSC/CSR matrix support
    - Multi-threading with OpenMP
    - Tie correction
    - Continuity correction
    - Both exact and asymptotic p-values

Algorithm logic mirrors the C++ implementation from hpdex, but implemented in
Cython for better integration with the Python ecosystem.
"""

import numpy as np

cimport numpy as np
from cython cimport boundscheck, wraparound
from cython.parallel cimport parallel, prange
from libc.math cimport INFINITY, abs, erfc, isfinite, isnan, sqrt
from libc.stdlib cimport free, malloc

# Initialize NumPy C-API
np.import_array()

# Type aliases
ctypedef np.int32_t INT32
ctypedef np.int64_t INT64
ctypedef np.float64_t FLOAT64
ctypedef np.float32_t FLOAT32

# External declarations
cdef extern from "numpy/arrayobject.h":
    void* PyArray_DATA(np.ndarray arr)
    Py_ssize_t PyArray_DIM(np.ndarray arr, int n)

# Constants
cdef double INV_SQRT2 = 0.7071067811865475  # 1.0 / sqrt(2.0)


# =============================================================================
# Helper Functions
# =============================================================================

cdef inline double clip_double(double x, double min_val, double max_val) nogil:
    """Clip value to range [min_val, max_val]."""
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x


cdef inline INT64 max_int64(INT64 a, INT64 b) nogil:
    """Return maximum of two int64 values."""
    return a if a > b else b


cdef inline INT64 min_int64(INT64 a, INT64 b) nogil:
    """Return minimum of two int64 values."""
    return a if a < b else b


# =============================================================================
# Mann-Whitney U Core Algorithm (Float Kernel)
# =============================================================================

@boundscheck(False)
@wraparound(False)
cdef void merge_rank_sum_float(
    FLOAT64* ref_data,
    INT64 n_ref,
    FLOAT64* tar_data,
    INT64 n_tar,
    FLOAT64* U1_out,
    FLOAT64* tie_sum_out,
    INT32* has_tie_out
) nogil:
    """Compute U statistic and tie correction via merge-rank algorithm.
    
    Both ref_data and tar_data must be sorted in ascending order.
    
    Args:
        ref_data: Reference group data (sorted)
        n_ref: Size of reference group
        tar_data: Target group data (sorted)
        n_tar: Size of target group
        U1_out: Output U1 statistic
        tie_sum_out: Output tie sum for correction
        has_tie_out: Output flag for ties
    """
    cdef INT64 i = 0, k = 0
    cdef double running = 1.0
    cdef double rank_sum_t = 0.0
    cdef double tie_sum = 0.0
    cdef INT32 has_ties = 0
    cdef double v
    cdef INT64 cr, ct, c
    
    if n_ref == 0 or n_tar == 0:
        U1_out[0] = 0.0
        tie_sum_out[0] = 0.0
        has_tie_out[0] = 0
        return
    
    # Merge and rank
    while i < n_ref or k < n_tar:
        # Get next value
        if k >= n_tar:
            v = ref_data[i]
        elif i >= n_ref:
            v = tar_data[k]
        elif tar_data[k] <= ref_data[i]:
            v = tar_data[k]
        else:
            v = ref_data[i]
        
        # Count ties
        cr = 0
        ct = 0
        
        while i < n_ref and ref_data[i] == v:
            cr += 1
            i += 1
        
        while k < n_tar and tar_data[k] == v:
            ct += 1
            k += 1
        
        c = cr + ct
        
        # Update tie correction
        if c > 1:
            has_ties = 1
            tie_sum += c * (c * c - 1)
        
        # Update rank sum for target group
        if ct > 0:
            rank_sum_t += ct * (running + 0.5 * (c - 1))
        
        running += c
    
    # Compute U1 statistic
    U1_out[0] = rank_sum_t - 0.5 * n_tar * (n_tar + 1.0)
    tie_sum_out[0] = tie_sum
    has_tie_out[0] = has_ties


@boundscheck(False)
@wraparound(False)
cdef double compute_asymptotic_pvalue(
    double U1,
    double tie_sum,
    INT64 n_tar,
    INT64 n_ref,
    bint tie_correction,
    bint use_continuity
) nogil:
    """Compute asymptotic p-value using normal approximation.
    
    Args:
        U1: U1 statistic
        tie_sum: Tie sum for correction
        n_tar: Target group size
        n_ref: Reference group size
        tie_correction: Whether to apply tie correction
        use_continuity: Whether to apply continuity correction
    
    Returns:
        Two-sided p-value
    """
    cdef INT64 N = n_tar + n_ref
    cdef double mu, sigma2, num, z_abs, p
    
    if N <= 1 or n_tar == 0 or n_ref == 0:
        return 1.0
    
    mu = 0.5 * n_tar * n_ref
    sigma2 = (n_tar * n_ref) * (N + 1.0) / 12.0
    
    # Apply tie correction
    if tie_correction and N > 1:
        sigma2 -= (n_tar * n_ref) * tie_sum / (12.0 * N * (N - 1.0))
    
    if sigma2 <= 0.0 or not isfinite(sigma2):
        return 1.0
    
    num = U1 - mu
    
    # Apply continuity correction
    if use_continuity and num != 0.0:
        if num > 0.0:
            num -= 0.5
        else:
            num += 0.5
    
    z_abs = abs(num) / sqrt(sigma2)
    p = erfc(z_abs * INV_SQRT2)  # Two-sided
    
    # Clip to valid range
    if not isfinite(p) or p > 1.0:
        p = 1.0
    elif p < 0.0:
        p = 0.0
    
    return p


# =============================================================================
# Group Mean Computation
# =============================================================================

def group_mean_csc(
    np.ndarray data,
    np.ndarray indices,
    np.ndarray indptr,
    np.ndarray[INT32, ndim=1] group_id,
    INT32 n_groups,
    bint include_zeros=True,
    int threads=-1
):
    """Compute group-wise means for CSC sparse matrix.
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        group_id: Group assignment for each row (sample)
        n_groups: Total number of groups
        include_zeros: Whether to include zeros in mean calculation
        threads: Number of threads (-1 for all available)
    
    Returns:
        np.ndarray: Shape (n_groups, n_cols) group means
    """
    cdef INT64 n_rows = group_id.shape[0]
    cdef INT64 n_cols = indptr.shape[0] - 1
    cdef INT64 nnz = data.shape[0]
    
    # Convert to appropriate types
    cdef np.ndarray[INT64, ndim=1] indices_i64
    cdef np.ndarray[INT64, ndim=1] indptr_i64
    
    if indices.dtype == np.int64:
        indices_i64 = indices
    else:
        indices_i64 = indices.astype(np.int64)
    
    if indptr.dtype == np.int64:
        indptr_i64 = indptr
    else:
        indptr_i64 = indptr.astype(np.int64)
    
    # Allocate output
    cdef np.ndarray[FLOAT64, ndim=2] means = np.zeros((n_groups, n_cols), dtype=np.float64)
    cdef np.ndarray[INT64, ndim=1] counts = np.zeros(n_groups, dtype=np.int64)
    
    # Count samples per group
    cdef INT64 i
    for i in range(n_rows):
        if group_id[i] >= 0 and group_id[i] < n_groups:
            counts[group_id[i]] += 1
    
    # Dispatch based on data dtype
    if data.dtype == np.float64:
        _group_mean_kernel_f64(
            <FLOAT64*>PyArray_DATA(data),
            <INT64*>PyArray_DATA(indices_i64),
            <INT64*>PyArray_DATA(indptr_i64),
            <INT32*>PyArray_DATA(group_id),
            <FLOAT64*>PyArray_DATA(means),
            <INT64*>PyArray_DATA(counts),
            n_groups, n_cols, nnz, include_zeros, threads
        )
    elif data.dtype == np.float32:
        _group_mean_kernel_f32(
            <FLOAT32*>PyArray_DATA(data),
            <INT64*>PyArray_DATA(indices_i64),
            <INT64*>PyArray_DATA(indptr_i64),
            <INT32*>PyArray_DATA(group_id),
            <FLOAT64*>PyArray_DATA(means),
            <INT64*>PyArray_DATA(counts),
            n_groups, n_cols, nnz, include_zeros, threads
        )
    else:
        # Convert to float64 for other types
        data_f64 = data.astype(np.float64)
        _group_mean_kernel_f64(
            <FLOAT64*>PyArray_DATA(data_f64),
            <INT64*>PyArray_DATA(indices_i64),
            <INT64*>PyArray_DATA(indptr_i64),
            <INT32*>PyArray_DATA(group_id),
            <FLOAT64*>PyArray_DATA(means),
            <INT64*>PyArray_DATA(counts),
            n_groups, n_cols, nnz, include_zeros, threads
        )
    
    return means


@boundscheck(False)
@wraparound(False)
cdef void _group_mean_kernel_f64(
    FLOAT64* data,
    INT64* indices,
    INT64* indptr,
    INT32* group_id,
    FLOAT64* means,
    INT64* counts,
    INT32 n_groups,
    INT64 n_cols,
    INT64 nnz,
    bint include_zeros,
    int threads
) nogil:
    """Kernel for group mean computation (float64 data)."""
    cdef INT64 col, idx, row
    cdef INT32 gid
    cdef INT64 start, end
    cdef double val
    cdef INT64 g
    
    # Process each column in parallel
    for col in prange(n_cols, nogil=True, num_threads=threads if threads > 0 else 1):
        start = indptr[col]
        end = indptr[col + 1]
        
        # Accumulate sums for each group
        for idx in range(start, end):
            row = indices[idx]
            gid = group_id[row]
            
            if gid >= 0 and gid < n_groups:
                val = data[idx]
                means[gid * n_cols + col] += val
        
        # Divide by counts to get means
        for g in range(n_groups):
            if counts[g] > 0:
                means[g * n_cols + col] /= counts[g]


@boundscheck(False)
@wraparound(False)
cdef void _group_mean_kernel_f32(
    FLOAT32* data,
    INT64* indices,
    INT64* indptr,
    INT32* group_id,
    FLOAT64* means,
    INT64* counts,
    INT32 n_groups,
    INT64 n_cols,
    INT64 nnz,
    bint include_zeros,
    int threads
) nogil:
    """Kernel for group mean computation (float32 data)."""
    cdef INT64 col, idx, row
    cdef INT32 gid
    cdef INT64 start, end
    cdef double val
    cdef INT64 g
    
    # Process each column in parallel
    for col in prange(n_cols, nogil=True, num_threads=threads if threads > 0 else 1):
        start = indptr[col]
        end = indptr[col + 1]
        
        # Accumulate sums for each group
        for idx in range(start, end):
            row = indices[idx]
            gid = group_id[row]
            
            if gid >= 0 and gid < n_groups:
                val = <double>data[idx]
                means[gid * n_cols + col] += val
        
        # Divide by counts to get means
        for g in range(n_groups):
            if counts[g] > 0:
                means[g * n_cols + col] /= counts[g]


# =============================================================================
# Mann-Whitney U Test (Sparse CSC)
# =============================================================================

def mannwhitneyu_csc(
    np.ndarray data,
    np.ndarray indices,
    np.ndarray indptr,
    np.ndarray[INT32, ndim=1] group_id,
    INT32 n_targets,
    bint tie_correction=True,
    bint use_continuity=True,
    int threads=-1
):
    """Mann-Whitney U test on sparse CSC matrix.
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        group_id: Group assignment (0=reference, 1..n_targets=targets, -1=ignore)
        n_targets: Number of target groups
        tie_correction: Whether to apply tie correction
        use_continuity: Whether to apply continuity correction
        threads: Number of threads (-1 for all available)
    
    Returns:
        tuple: (U1, U2, P) arrays of shape (n_targets, n_cols)
    """
    cdef INT64 n_rows = group_id.shape[0]
    cdef INT64 n_cols = indptr.shape[0] - 1
    cdef INT64 nnz = data.shape[0]
    
    # Convert indices/indptr to int64
    cdef np.ndarray[INT64, ndim=1] indices_i64
    cdef np.ndarray[INT64, ndim=1] indptr_i64
    
    if indices.dtype == np.int64:
        indices_i64 = indices
    else:
        indices_i64 = indices.astype(np.int64)
    
    if indptr.dtype == np.int64:
        indptr_i64 = indptr
    else:
        indptr_i64 = indptr.astype(np.int64)
    
    # Allocate outputs
    cdef np.ndarray[FLOAT64, ndim=2] U1 = np.zeros((n_targets, n_cols), dtype=np.float64)
    cdef np.ndarray[FLOAT64, ndim=2] U2 = np.zeros((n_targets, n_cols), dtype=np.float64)
    cdef np.ndarray[FLOAT64, ndim=2] P = np.ones((n_targets, n_cols), dtype=np.float64)
    
    # Dispatch based on data dtype
    if data.dtype == np.float64:
        _mannwhitneyu_kernel_f64(
            <FLOAT64*>PyArray_DATA(data),
            <INT64*>PyArray_DATA(indices_i64),
            <INT64*>PyArray_DATA(indptr_i64),
            <INT32*>PyArray_DATA(group_id),
            <FLOAT64*>PyArray_DATA(U1),
            <FLOAT64*>PyArray_DATA(U2),
            <FLOAT64*>PyArray_DATA(P),
            n_rows, n_cols, nnz, n_targets,
            tie_correction, use_continuity, threads
        )
    elif data.dtype == np.float32:
        _mannwhitneyu_kernel_f32(
            <FLOAT32*>PyArray_DATA(data),
            <INT64*>PyArray_DATA(indices_i64),
            <INT64*>PyArray_DATA(indptr_i64),
            <INT32*>PyArray_DATA(group_id),
            <FLOAT64*>PyArray_DATA(U1),
            <FLOAT64*>PyArray_DATA(U2),
            <FLOAT64*>PyArray_DATA(P),
            n_rows, n_cols, nnz, n_targets,
            tie_correction, use_continuity, threads
        )
    else:
        # Convert to float64 for other types
        data_f64 = data.astype(np.float64)
        _mannwhitneyu_kernel_f64(
            <FLOAT64*>PyArray_DATA(data_f64),
            <INT64*>PyArray_DATA(indices_i64),
            <INT64*>PyArray_DATA(indptr_i64),
            <INT32*>PyArray_DATA(group_id),
            <FLOAT64*>PyArray_DATA(U1),
            <FLOAT64*>PyArray_DATA(U2),
            <FLOAT64*>PyArray_DATA(P),
            n_rows, n_cols, nnz, n_targets,
            tie_correction, use_continuity, threads
        )
    
    return U1, U2, P


@boundscheck(False)
@wraparound(False)
cdef void _mannwhitneyu_kernel_f64(
    FLOAT64* data,
    INT64* indices,
    INT64* indptr,
    INT32* group_id,
    FLOAT64* U1_out,
    FLOAT64* U2_out,
    FLOAT64* P_out,
    INT64 n_rows,
    INT64 n_cols,
    INT64 nnz,
    INT32 n_targets,
    bint tie_correction,
    bint use_continuity,
    int threads
) nogil:
    """Mann-Whitney U kernel for float64 data.
    
    This implements the core algorithm:
    1. For each column (gene), extract reference and target group values
    2. Sort the values
    3. Compute U statistic via merge-rank algorithm
    4. Compute p-value using asymptotic approximation
    """
    cdef INT64 col, t
    cdef INT64 start, end, idx, row
    cdef INT32 gid
    cdef INT64 ref_count, tar_count
    cdef FLOAT64* ref_vals
    cdef FLOAT64* tar_vals
    cdef FLOAT64 U1, U2, tie_sum, p
    cdef INT32 has_tie
    cdef INT64 total_U
    
    # Process each column in parallel
    for col in prange(n_cols, nogil=True, num_threads=threads if threads > 0 else 1, schedule='dynamic'):
        start = indptr[col]
        end = indptr[col + 1]
        
        # Extract reference group values
        ref_vals = <FLOAT64*>malloc(n_rows * sizeof(FLOAT64))
        if ref_vals == NULL:
            continue
        
        # Count reference group
        ref_count = 0
        for idx in range(start, end):
            row = indices[idx]
            gid = group_id[row]
            if gid == 0:  # Reference group
                ref_vals[ref_count] = data[idx]
                ref_count = ref_count + 1
        
        # Sort reference values (simple insertion sort for small arrays)
        _insertion_sort_f64(ref_vals, ref_count)
        
        # Process each target group
        for t in range(n_targets):
            # Extract target group values
            tar_vals = <FLOAT64*>malloc(n_rows * sizeof(FLOAT64))
            if tar_vals == NULL:
                free(ref_vals)
                continue
            
            # Count target group
            tar_count = 0
            for idx in range(start, end):
                row = indices[idx]
                gid = group_id[row]
                if gid == (t + 1):  # Target group (1-indexed)
                    tar_vals[tar_count] = data[idx]
                    tar_count = tar_count + 1
            
            # Sort target values
            _insertion_sort_f64(tar_vals, tar_count)
            
            # Compute U statistic
            merge_rank_sum_float(
                ref_vals, ref_count,
                tar_vals, tar_count,
                &U1, &tie_sum, &has_tie
            )
            
            # Compute U2 and p-value
            if ref_count > 0 and tar_count > 0:
                total_U = ref_count * tar_count
                U2 = <FLOAT64>total_U - U1
                
                p = compute_asymptotic_pvalue(
                    U1, tie_sum, tar_count, ref_count,
                    tie_correction, use_continuity
                )
                
                U1_out[t * n_cols + col] = U1
                U2_out[t * n_cols + col] = U2
                P_out[t * n_cols + col] = p
            
            free(tar_vals)
        
        free(ref_vals)


@boundscheck(False)
@wraparound(False)
cdef void _mannwhitneyu_kernel_f32(
    FLOAT32* data,
    INT64* indices,
    INT64* indptr,
    INT32* group_id,
    FLOAT64* U1_out,
    FLOAT64* U2_out,
    FLOAT64* P_out,
    INT64 n_rows,
    INT64 n_cols,
    INT64 nnz,
    INT32 n_targets,
    bint tie_correction,
    bint use_continuity,
    int threads
) nogil:
    """Mann-Whitney U kernel for float32 data (converts to float64 internally)."""
    cdef INT64 col, t
    cdef INT64 start, end, idx, row
    cdef INT32 gid
    cdef FLOAT64* ref_vals
    cdef FLOAT64* tar_vals
    cdef FLOAT64 U1, U2, tie_sum, p
    cdef INT32 has_tie
    cdef INT64 total_U
    cdef INT64 ref_count, tar_count
    
    # Process each column in parallel
    for col in prange(n_cols, nogil=True, num_threads=threads if threads > 0 else 1, schedule='dynamic'):
        start = indptr[col]
        end = indptr[col + 1]
        
        # Extract reference group values
        ref_vals = <FLOAT64*>malloc(n_rows * sizeof(FLOAT64))
        if ref_vals == NULL:
            continue
        
        # Local counter
        ref_count = 0
        for idx in range(start, end):
            row = indices[idx]
            gid = group_id[row]
            if gid == 0:
                ref_vals[ref_count] = <FLOAT64>data[idx]
                ref_count = ref_count + 1
        
        _insertion_sort_f64(ref_vals, ref_count)
        
        # Process each target group
        for t in range(n_targets):
            tar_vals = <FLOAT64*>malloc(n_rows * sizeof(FLOAT64))
            if tar_vals == NULL:
                free(ref_vals)
                continue
            
            # Local counter
            tar_count = 0
            for idx in range(start, end):
                row = indices[idx]
                gid = group_id[row]
                if gid == (t + 1):
                    tar_vals[tar_count] = <FLOAT64>data[idx]
                    tar_count = tar_count + 1
            
            _insertion_sort_f64(tar_vals, tar_count)
            
            merge_rank_sum_float(
                ref_vals, ref_count,
                tar_vals, tar_count,
                &U1, &tie_sum, &has_tie
            )
            
            if ref_count > 0 and tar_count > 0:
                total_U = ref_count * tar_count
                U2 = <FLOAT64>total_U - U1
                
                p = compute_asymptotic_pvalue(
                    U1, tie_sum, tar_count, ref_count,
                    tie_correction, use_continuity
                )
                
                U1_out[t * n_cols + col] = U1
                U2_out[t * n_cols + col] = U2
                P_out[t * n_cols + col] = p
            
            free(tar_vals)
        
        free(ref_vals)


# =============================================================================
# Sorting Utilities
# =============================================================================

@boundscheck(False)
@wraparound(False)
cdef void _insertion_sort_f64(FLOAT64* arr, INT64 n) nogil:
    """In-place insertion sort for float64 array.
    
    Simple but efficient for small arrays (< 100 elements).
    For larger arrays, consider quicksort or mergesort.
    """
    cdef INT64 i, j
    cdef FLOAT64 key
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key

