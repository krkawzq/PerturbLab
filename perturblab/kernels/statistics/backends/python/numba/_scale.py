"""
Numba backend for scaling/standardization operations.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

import numba
import numpy as np
import scipy.sparse
from typing import Tuple


@numba.njit(parallel=True, cache=True)
def sparse_standardize_csc_numba(
    data: np.ndarray,
    row_indices: np.ndarray,
    col_ptr: np.ndarray,
    n_rows: int,
    n_cols: int,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
) -> None:
    """Standardize sparse CSC matrix by columns (Numba backend, in-place).
    
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
    for j in numba.prange(n_cols):
        start = col_ptr[j]
        end = col_ptr[j + 1]
        
        mean = means[j]
        std = stds[j]
        
        # Skip if std is zero or invalid
        if std <= 0.0 or not np.isfinite(std):
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


@numba.njit(parallel=True, cache=True)
def sparse_standardize_csr_numba(
    data: np.ndarray,
    col_indices: np.ndarray,
    row_ptr: np.ndarray,
    n_rows: int,
    n_cols: int,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
) -> None:
    """Standardize sparse CSR matrix by columns (Numba backend, in-place).
    
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
    for i in numba.prange(n_rows):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        
        for idx in range(start, end):
            col_idx = col_indices[idx]
            val = data[idx]
            
            mean = means[col_idx]
            std = stds[col_idx]
            
            if std <= 0.0 or not np.isfinite(std):
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


@numba.njit(parallel=True, cache=True)
def dense_standardize_numba(
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
) -> None:
    """Standardize dense matrix by columns (Numba backend, in-place).
    
    Args:
        data: Dense data array (modified in-place), row-major layout
        n_rows: Number of rows
        n_cols: Number of columns
        means: Column means
        stds: Column standard deviations
        zero_center: If True, subtract mean
        max_value: Maximum absolute value for clipping (0 = no clipping)
    """
    for j in numba.prange(n_cols):
        mean = means[j]
        std = stds[j]
        
        if std <= 0.0 or not np.isfinite(std):
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

