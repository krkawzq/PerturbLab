"""
Numba backend for normalization operations.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

from typing import Tuple

import numba
import numpy as np
import scipy.sparse


@numba.njit(parallel=True, cache=True)
def sparse_row_sum_csr_numba(
    data: np.ndarray,
    indptr: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Compute row sums for CSR sparse matrix (Numba backend).

    Args:
        data: CSR data array
        indptr: CSR index pointer array
        n_rows: Number of rows

    Returns:
        Row sums, shape (n_rows,)
    """
    row_sums = np.zeros(n_rows, dtype=data.dtype)

    for i in numba.prange(n_rows):
        start = indptr[i]
        end = indptr[i + 1]

        # Kahan summation for numerical stability
        sum_val = 0.0
        c = 0.0

        for j in range(start, end):
            y = data[j] - c
            t = sum_val + y
            c = (t - sum_val) - y
            sum_val = t

        row_sums[i] = sum_val

    return row_sums


@numba.njit(parallel=True, cache=True)
def inplace_divide_csr_rows_numba(
    data: np.ndarray,
    indptr: np.ndarray,
    n_rows: int,
    divisors: np.ndarray,
    allow_zero_divisor: bool = False,
) -> None:
    """Divide each row by a scalar in-place (Numba backend).

    Args:
        data: CSR data array (modified in-place)
        indptr: CSR index pointer array
        n_rows: Number of rows
        divisors: Divisors for each row, shape (n_rows,)
        allow_zero_divisor: If False, set row to zero when divisor is zero
    """
    for i in numba.prange(n_rows):
        start = indptr[i]
        end = indptr[i + 1]

        divisor = divisors[i]

        if divisor == 0.0:
            if not allow_zero_divisor:
                for j in range(start, end):
                    data[j] = 0.0
            continue

        inv_divisor = 1.0 / divisor

        for j in range(start, end):
            data[j] *= inv_divisor


def compute_median_nonzero_numba(values: np.ndarray) -> float:
    """Compute median of non-zero values (Numba backend).

    Args:
        values: Array of values

    Returns:
        Median of non-zero values
    """
    nonzero_vals = values[values > 0]
    if len(nonzero_vals) == 0:
        return 0.0
    return float(np.median(nonzero_vals))


@numba.njit(parallel=True, cache=True)
def find_highly_expressed_genes_numba(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    n_rows: int,
    n_cols: int,
    row_sums: np.ndarray,
    max_fraction: float,
) -> np.ndarray:
    """Find highly expressed genes (Numba backend).

    Args:
        data: CSR data array
        indptr: CSR index pointer array
        indices: CSR indices array
        n_rows: Number of rows
        n_cols: Number of columns
        row_sums: Total counts per cell
        max_fraction: Threshold fraction

    Returns:
        Boolean mask, shape (n_cols,). True = highly expressed.
    """
    highly_expressed_count = np.zeros(n_cols, dtype=np.int32)

    for i in numba.prange(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        threshold = row_sums[i] * max_fraction

        for j in range(start, end):
            if data[j] > threshold:
                col_idx = indices[j]
                highly_expressed_count[col_idx] += 1

    gene_mask = highly_expressed_count > 0
    return gene_mask


@numba.njit(parallel=True, cache=True)
def sparse_row_sum_csr_exclude_genes_numba(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    n_rows: int,
    gene_mask: np.ndarray,
) -> np.ndarray:
    """Compute row sums excluding specific genes (Numba backend).

    Args:
        data: CSR data array
        indptr: CSR index pointer array
        indices: CSR indices array
        n_rows: Number of rows
        gene_mask: Boolean mask. True = exclude this gene.

    Returns:
        Row sums, shape (n_rows,)
    """
    row_sums = np.zeros(n_rows, dtype=data.dtype)

    for i in numba.prange(n_rows):
        start = indptr[i]
        end = indptr[i + 1]

        sum_val = 0.0
        c = 0.0  # Kahan summation

        for j in range(start, end):
            col_idx = indices[j]

            # Skip excluded genes
            if gene_mask[col_idx]:
                continue

            y = data[j] - c
            t = sum_val + y
            c = (t - sum_val) - y
            sum_val = t

        row_sums[i] = sum_val

    return row_sums
