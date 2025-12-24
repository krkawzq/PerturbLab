"""
Normalization functions for single-cell data.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse
from anndata import AnnData

from perturblab.kernels.statistics import (
    compute_median_nonzero,
    find_highly_expressed_genes,
    inplace_divide_rows,
    sparse_row_sum,
    sparse_row_sum_exclude_genes,
)


def normalize_total(
    adata: AnnData,
    *,
    target_sum: Optional[float] = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: Optional[str] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
    n_threads: int = 0,
) -> Optional[Union[AnnData, dict]]:
    """
    Normalize counts per cell (CPM/TPM normalization).

    Normalize each cell by total counts over all genes, so that every cell
    has the same total count after normalization. If choosing `target_sum=1e6`,
    this is CPM normalization.

    **Performance**: 2-3x faster than scanpy.pp.normalize_total using C++/SIMD/OpenMP.

    Args:
        adata: Annotated data matrix of shape n_obs × n_vars.
        target_sum: If None, after normalization, each observation (cell) has a total
            count equal to the median of total counts for observations (cells)
            before normalization.
        exclude_highly_expressed: Exclude (very) highly expressed genes for the
            computation of the normalization factor (size factor) for each cell.
            A gene is considered highly expressed, if it has more than `max_fraction`
            of the total counts in at least one cell.
        max_fraction: If `exclude_highly_expressed=True`, consider genes as highly
            expressed that have more counts than `max_fraction` of the original total
            counts in at least one cell.
        key_added: Name of the field in `adata.obs` where the normalization factor
            is stored.
        layer: Layer to normalize instead of `X`. If None, `X` is normalized.
        inplace: Whether to update `adata` or return dictionary with normalized
            copies of `adata.X` and `adata.layers`.
        copy: Whether to modify copied input object. Not compatible with inplace=False.
        n_threads: Number of threads for parallel computation (0 = auto).

    Returns:
        Returns dictionary with normalized copies of `adata.X` and `adata.layers`
        or updates `adata` with normalized version, depending on `inplace`.

    Examples:
        >>> import perturblab as pl
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> pl.pp.normalize_total(adata, target_sum=1e4)  # 2-3x faster than scanpy

    References:
        Scanpy implementation: https://github.com/scverse/scanpy
    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    if max_fraction < 0 or max_fraction > 1:
        raise ValueError("Choose max_fraction between 0 and 1.")

    # Get data matrix
    X = adata.layers[layer] if layer is not None else adata.X

    if X is None:
        raise ValueError(f"Layer {layer!r} not found in adata." if layer else "adata.X is None")

    # Convert to CSR for efficient row operations
    if scipy.sparse.isspmatrix(X):
        if not scipy.sparse.isspmatrix_csr(X):
            X = X.tocsr()
        if not inplace:
            X = X.copy()
    else:
        # Dense matrix
        if not inplace:
            X = X.copy()

    # Ensure float type
    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.float32)

    # Step 1: Compute counts per cell
    if scipy.sparse.isspmatrix(X):
        counts_per_cell = sparse_row_sum(X, n_threads=n_threads)
    else:
        counts_per_cell = X.sum(axis=1)

    # Step 2: Handle highly expressed genes if requested
    gene_subset = None
    if exclude_highly_expressed:
        if scipy.sparse.isspmatrix(X):
            gene_mask = find_highly_expressed_genes(
                X, counts_per_cell, max_fraction, n_threads=n_threads
            )
            gene_subset = ~gene_mask

            # Recompute counts excluding highly expressed genes
            counts_per_cell = sparse_row_sum_exclude_genes(X, gene_mask, n_threads=n_threads)
        else:
            # Dense case
            hi_exp = X > counts_per_cell[:, None] * max_fraction
            gene_subset = hi_exp.sum(axis=0) == 0
            counts_per_cell = X[:, gene_subset].sum(axis=1)

    # Step 3: Determine target sum
    if target_sum is None:
        target_sum = compute_median_nonzero(counts_per_cell)

    # Step 4: Normalize
    norm_factors = counts_per_cell / target_sum

    # Check for zero counts
    cell_subset = counts_per_cell > 0
    if not np.all(cell_subset):
        warnings.warn("Some cells have zero counts", UserWarning, stacklevel=2)

    # Perform normalization
    if scipy.sparse.isspmatrix(X):
        inplace_divide_rows(X, norm_factors, allow_zero_divisor=False, n_threads=n_threads)
    else:
        # Dense case
        norm_factors[norm_factors == 0] = 1.0  # Avoid division by zero
        X /= norm_factors[:, None]

    # Prepare output
    dat = dict(
        X=X,
        norm_factor=norm_factors,
    )

    if inplace:
        if key_added is not None:
            adata.obs[key_added] = dat["norm_factor"]

        if layer is not None:
            adata.layers[layer] = dat["X"]
        else:
            adata.X = dat["X"]

    if copy:
        return adata
    elif not inplace:
        return dat
    return None
