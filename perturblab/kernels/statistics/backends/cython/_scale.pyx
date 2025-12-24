# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Cython backend for scaling/standardization operations.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport isfinite


def sparse_standardize_csc_cy(
    double[::1] data,
    long[::1] row_indices,
    long[::1] col_ptr,
    Py_ssize_t n_rows,
    Py_ssize_t n_cols,
    double[::1] means,
    double[::1] stds,
    bint zero_center=True,
    double max_value=0.0,
):
    """Standardize sparse CSC matrix by columns (Cython backend, in-place).
    
    Args:
        data: CSC data array (modified in-place)
        row_indices: CSC row indices array
        col_ptr: CSC column pointer array
        n_rows: Number of rows
        n_cols: Number of columns
        means: Column means
        stds: Column standard deviations
        zero_center: If True, subtract mean
        max_value: Maximum absolute value for clipping (0 = no clipping)
    """
    cdef Py_ssize_t j, idx, start, end
    cdef double mean, std, inv_std, val
    
    for j in prange(n_cols, nogil=True, schedule='static'):
        start = col_ptr[j]
        end = col_ptr[j + 1]
        
        mean = means[j]
        std = stds[j]
        
        # Skip if std is zero or invalid
        if std <= 0.0 or not isfinite(std):
            for idx in range(start, end):
                data[idx] = 0.0
            continue
        
        inv_std = 1.0 / std
        
        for idx in range(start, end):
            val = data[idx]
            
            if zero_center:
                val = (val - mean) * inv_std
            else:
                val = val * inv_std
            
            # Apply clipping if specified
            if max_value > 0.0:
                if val > max_value:
                    val = max_value
                elif zero_center and val < -max_value:
                    val = -max_value
            
            data[idx] = val


def sparse_standardize_csr_cy(
    double[::1] data,
    long[::1] col_indices,
    long[::1] row_ptr,
    Py_ssize_t n_rows,
    Py_ssize_t n_cols,
    double[::1] means,
    double[::1] stds,
    bint zero_center=True,
    double max_value=0.0,
):
    """Standardize sparse CSR matrix by columns (Cython backend, in-place).
    
    Args:
        data: CSR data array (modified in-place)
        col_indices: CSR column indices array
        row_ptr: CSR row pointer array
        n_rows: Number of rows
        n_cols: Number of columns
        means: Column means
        stds: Column standard deviations
        zero_center: If True, subtract mean
        max_value: Maximum absolute value for clipping (0 = no clipping)
    """
    cdef Py_ssize_t i, idx, start, end, col_idx
    cdef double mean, std, inv_std, val
    
    for i in prange(n_rows, nogil=True, schedule='static'):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        
        for idx in range(start, end):
            col_idx = col_indices[idx]
            val = data[idx]
            
            mean = means[col_idx]
            std = stds[col_idx]
            
            if std <= 0.0 or not isfinite(std):
                data[idx] = 0.0
                continue
            
            inv_std = 1.0 / std
            
            if zero_center:
                val = (val - mean) * inv_std
            else:
                val = val * inv_std
            
            if max_value > 0.0:
                if val > max_value:
                    val = max_value
                elif zero_center and val < -max_value:
                    val = -max_value
            
            data[idx] = val


def dense_standardize_cy(
    double[:, ::1] data,
    Py_ssize_t n_rows,
    Py_ssize_t n_cols,
    double[::1] means,
    double[::1] stds,
    bint zero_center=True,
    double max_value=0.0,
):
    """Standardize dense matrix by columns (Cython backend, in-place).
    
    Args:
        data: Dense data array (modified in-place)
        n_rows: Number of rows
        n_cols: Number of columns
        means: Column means
        stds: Column standard deviations
        zero_center: If True, subtract mean
        max_value: Maximum absolute value for clipping (0 = no clipping)
    """
    cdef Py_ssize_t i, j
    cdef double mean, std, inv_std, val
    
    for j in prange(n_cols, nogil=True, schedule='static'):
        mean = means[j]
        std = stds[j]
        
        if std <= 0.0 or not isfinite(std):
            for i in range(n_rows):
                data[i, j] = 0.0
            continue
        
        inv_std = 1.0 / std
        
        for i in range(n_rows):
            val = data[i, j]
            
            if zero_center:
                val = (val - mean) * inv_std
            else:
                val = val * inv_std
            
            if max_value > 0.0:
                if val > max_value:
                    val = max_value
                elif zero_center and val < -max_value:
                    val = -max_value
            
            data[i, j] = val

