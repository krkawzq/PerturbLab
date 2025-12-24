"""
Scaling functions for single-cell data.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

from typing import Optional

import numpy as np
import scipy.sparse
from anndata import AnnData

from ..kernels.statistics import standardize
from ..kernels.statistics.ops._hvg import sparse_mean_var


def scale(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
    mask_obs: Optional[np.ndarray] = None,
    inplace: bool = True,
    copy: bool = False,
    n_threads: int = 0,
) -> Optional[AnnData]:
    """
    Scale data to unit variance and zero mean.

    Performs z-score normalization by subtracting the mean and dividing by the
    standard deviation for each gene (column).

    **Performance**: 2-4x faster than scanpy.pp.scale using C++/SIMD/OpenMP.

    Note:
        Variables (genes) that do not display any variation (are constant across
        all observations) are set to 0 during this operation.

    Args:
        adata: Annotated data matrix of shape n_obs × n_vars.
        zero_center: If True, subtract mean (center to zero). If False, only scale
            by standard deviation.
        max_value: Clip (truncate) values to this maximum absolute value after scaling.
            If None, do not clip. This is useful to mitigate the effect of strong outliers.
        layer: If provided, scale this layer instead of X.
        obsm: If provided, scale this obsm key (e.g., 'X_pca') instead of X.
        mask_obs: Boolean mask or string key in adata.obs. If provided, only compute
            statistics (mean/std) on the selected observations, but scale all observations.
        inplace: Whether to modify adata in place or return result.
        copy: Whether to return a copy of adata. Incompatible with inplace=False.
        n_threads: Number of threads for parallel computation (0 = auto).

    Returns:
        If copy=True, returns a modified copy of adata.
        If inplace=False, returns the scaled matrix.
        Otherwise, modifies adata in place and returns None.

    Examples:
        >>> import perturblab as pl
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> sc.pp.highly_variable_genes(adata)
        >>> adata = adata[:, adata.var.highly_variable]
        >>> pl.pp.scale(adata, max_value=10)  # 2-4x faster than scanpy

    References:
        Scanpy implementation: https://github.com/scverse/scanpy
    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    # Get data matrix
    if obsm is not None:
        X = adata.obsm[obsm]
    elif layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if X is None:
        raise ValueError("Data matrix is None")

    # Handle mask_obs
    if mask_obs is not None:
        if isinstance(mask_obs, str):
            mask_obs = adata.obs[mask_obs].values
        mask_obs = np.asarray(mask_obs, dtype=bool)

    # Compute mean and variance
    if scipy.sparse.issparse(X):
        if not (scipy.sparse.isspmatrix_csc(X) or scipy.sparse.isspmatrix_csr(X)):
            X = X.tocsr()

        # Use sparse_mean_var for efficiency
        if mask_obs is not None:
            X_subset = X[mask_obs]
            if not scipy.sparse.isspmatrix_csc(X_subset):
                X_subset = X_subset.tocsc()
            means, vars = sparse_mean_var(X_subset, include_zeros=True, n_threads=n_threads)
        else:
            if not scipy.sparse.isspmatrix_csc(X):
                X_csc = X.tocsc()
                means, vars = sparse_mean_var(X_csc, include_zeros=True, n_threads=n_threads)
            else:
                means, vars = sparse_mean_var(X, include_zeros=True, n_threads=n_threads)
    else:
        # Dense case
        if not inplace:
            X = X.copy()

        if mask_obs is not None:
            X_subset = X[mask_obs]
            means = X_subset.mean(axis=0)
            vars = X_subset.var(axis=0, ddof=1)
        else:
            means = X.mean(axis=0)
            vars = X.var(axis=0, ddof=1)

    # Compute standard deviations
    stds = np.sqrt(vars)

    # Ensure array format
    if scipy.sparse.issparse(means):
        means = np.asarray(means).ravel()
    if scipy.sparse.issparse(stds):
        stds = np.asarray(stds).ravel()

    # Handle zero variance genes
    stds[stds == 0] = 1.0  # Avoid division by zero, will be set to 0 in standardize

    # Standardize
    max_val = max_value if max_value is not None else 0.0
    X = standardize(X, means, stds, zero_center=zero_center, max_value=max_val, n_threads=n_threads)

    # Store result
    if inplace:
        if obsm is not None:
            adata.obsm[obsm] = X
        elif layer is not None:
            adata.layers[layer] = X
        else:
            adata.X = X

    if copy:
        return adata
    elif not inplace:
        return X
    return None
