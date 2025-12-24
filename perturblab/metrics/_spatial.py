"""Spatial autocorrelation metrics for gene expression analysis.

This module provides spatial statistics to evaluate whether gene expression
patterns are autocorrelated across cell neighborhoods. These are particularly
useful for spatial transcriptomics and neighborhood-aware analyses.

Migrated and adapted from scanpy.metrics.

Copyright Notice
----------------
The implementation of Moran's I and Geary's C in this module is adapted from
the scanpy package (https://github.com/scverse/scanpy).

Original Copyright:
    Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

Modifications for PerturbLab:
    - Simplified API to accept graph and values directly
    - Enhanced documentation with perturbation-specific examples
    - Consistent logging with PerturbLab style
    - Removed AnnData-specific dispatching
    - Added comprehensive type hints

References
----------
scanpy: https://github.com/scverse/scanpy
Original implementation: scanpy/metrics/_morans_i.py, scanpy/metrics/_gearys_c.py
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import sparse

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "morans_i",
    "gearys_c",
    "compute_spatial_metrics",
]


def _to_csr(matrix) -> sparse.csr_matrix:
    """Convert matrix to CSR format."""
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix)
    return sparse.csr_matrix(matrix)


@numba.njit(cache=True, parallel=True)
def _morans_i_vec(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
    w: float,
) -> float:
    """Compute Moran's I for a single vector (numba optimized)."""
    z = x - x.mean()
    z2ss = (z * z).sum()
    n = len(x)
    inum = 0.0

    for i in numba.prange(n):
        s = slice(g_indptr[i], g_indptr[i + 1])
        i_indices = g_indices[s]
        i_data = g_data[s]
        inum += (i_data * z[i_indices]).sum() * z[i]

    return n / w * inum / z2ss


@numba.njit(cache=True, parallel=True)
def _morans_i_mtx(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Compute Moran's I for multiple vectors (matrix) (numba optimized)."""
    m, n = x.shape
    w = g_data.sum()
    out = np.zeros(m, dtype=np.float64)

    for k in numba.prange(m):
        x_vec = x[k, :]
        out[k] = _morans_i_vec(g_data, g_indices, g_indptr, x_vec, w)

    return out


@numba.njit(cache=True, parallel=True)
def _gearys_c_vec(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
    w: float,
) -> float:
    """Compute Geary's C for a single vector (numba optimized)."""
    n = len(g_indptr) - 1
    x = x.astype(np.float64)
    x_bar = x.mean()

    total = 0.0
    for i in numba.prange(n):
        s = slice(g_indptr[i], g_indptr[i + 1])
        i_indices = g_indices[s]
        i_data = g_data[s]
        total += np.sum(i_data * ((x[i] - x[i_indices]) ** 2))

    numer = (n - 1) * total
    denom = 2 * w * ((x - x_bar) ** 2).sum()

    if denom == 0:
        return np.nan

    return numer / denom


@numba.njit(cache=True, parallel=True)
def _gearys_c_mtx(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Compute Geary's C for multiple vectors (matrix) (numba optimized)."""
    m, n = x.shape
    w = g_data.sum()
    out = np.zeros(m, dtype=np.float64)

    for k in numba.prange(m):
        x_vec = x[k, :]
        out[k] = _gearys_c_vec(g_data, g_indices, g_indptr, x_vec, w)

    return out


def morans_i(
    graph: np.ndarray | sparse.spmatrix,
    values: np.ndarray,
) -> float | np.ndarray:
    """Calculate Moran's I global spatial autocorrelation statistic.

    Moran's I measures whether values on a graph are positively or negatively
    autocorrelated. It is commonly used in spatial transcriptomics to assess
    whether genes show spatially coherent expression patterns.

    .. math::

        I = \\frac{N \\sum_{i,j} w_{i,j} z_i z_j}{S_0 \\sum_i z_i^2}

    where :math:`z_i = x_i - \\bar{x}`, :math:`w_{i,j}` are edge weights,
    :math:`S_0 = \\sum_{i,j} w_{i,j}`, and :math:`N` is the number of observations.

    Parameters
    ----------
    graph
        Sparse adjacency matrix [cells × cells] representing cell neighborhood
        graph. Typically this is the connectivities matrix from neighbor graph
        construction (e.g., from KNN or spatial neighbors).
    values
        Values to compute autocorrelation for:
        - 1D array [cells]: Returns scalar Moran's I
        - 2D array [genes × cells]: Returns array of Moran's I per gene

    Returns
    -------
    float or np.ndarray
        Moran's I statistic. Returns scalar if values is 1D, array if 2D.

        Interpretation:
        - I > 0: Positive autocorrelation (similar values cluster together)
        - I ≈ 0: Random spatial pattern
        - I < 0: Negative autocorrelation (dissimilar values cluster together)
        - Range typically [-1, 1] but can exceed for small samples

    Notes
    -----
    Moran's I is closely related to Geary's C, but they measure slightly
    different aspects:
    - Moran's I is more sensitive to global patterns
    - Geary's C is more sensitive to local patterns

    For hypothesis testing, compute z-score:
    z = (I - E[I]) / sqrt(Var[I]), where E[I] = -1/(N-1) under randomization.

    Examples
    --------
    >>> import perturblab as pl
    >>> import numpy as np
    >>> from scipy import sparse
    >>>
    >>> # Create a simple neighbor graph (ring lattice)
    >>> n = 100
    >>> graph = sparse.lil_matrix((n, n))
    >>> for i in range(n):
    ...     graph[i, (i-1) % n] = 1
    ...     graph[i, (i+1) % n] = 1
    >>> graph = graph.tocsr()
    >>>
    >>> # Spatially autocorrelated pattern
    >>> x_auto = np.sin(np.linspace(0, 4*np.pi, n))
    >>> moran_auto = pl.metrics.morans_i(graph, x_auto)
    >>> print(f"Moran's I (autocorrelated): {moran_auto:.3f}")
    >>>
    >>> # Random pattern
    >>> x_random = np.random.randn(n)
    >>> moran_random = pl.metrics.morans_i(graph, x_random)
    >>> print(f"Moran's I (random): {moran_random:.3f}")
    >>>
    >>> # Multiple genes
    >>> X = np.random.randn(50, n)  # 50 genes × 100 cells
    >>> morans = pl.metrics.morans_i(graph, X)
    >>> print(f"Moran's I per gene: {morans.shape}")

    References
    ----------
    Moran, P.A.P. (1950). "Notes on continuous stochastic phenomena".
    Biometrika 37 (1–2): 17–23.
    """
    # Convert graph to CSR format
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    elif not sparse.isspmatrix_csr(graph):
        graph = sparse.csr_matrix(graph)

    graph = graph.astype(np.float64, copy=False)

    # Validate graph is square
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Graph must be square, got shape {graph.shape}")

    # Handle different value dimensions
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        # Single vector
        if len(values) != graph.shape[0]:
            raise ValueError(
                f"Values length ({len(values)}) must match graph size ({graph.shape[0]})"
            )

        w = graph.data.sum()
        return float(_morans_i_vec(graph.data, graph.indices, graph.indptr, values, w))

    elif values.ndim == 2:
        # Matrix of multiple vectors (genes × cells)
        if values.shape[1] != graph.shape[0]:
            raise ValueError(
                f"Values shape[1] ({values.shape[1]}) must match graph size ({graph.shape[0]})"
            )

        # Filter out constant genes (would cause division by zero)
        gene_std = values.std(axis=1)
        is_constant = gene_std == 0

        if is_constant.any():
            n_constant = is_constant.sum()
            logger.warning(f"{n_constant} genes are constant (std=0), returning NaN for these")

            result = np.full(values.shape[0], np.nan, dtype=np.float64)
            non_constant_idx = ~is_constant

            if non_constant_idx.any():
                result[non_constant_idx] = _morans_i_mtx(
                    graph.data, graph.indices, graph.indptr, values[non_constant_idx]
                )

            return result
        else:
            return _morans_i_mtx(graph.data, graph.indices, graph.indptr, values)

    else:
        raise ValueError(f"Values must be 1D or 2D, got {values.ndim}D")


def gearys_c(
    graph: np.ndarray | sparse.spmatrix,
    values: np.ndarray,
) -> float | np.ndarray:
    """Calculate Geary's C spatial autocorrelation statistic.

    Geary's C measures local spatial autocorrelation by comparing values
    at adjacent locations. It is more sensitive to local patterns than Moran's I.

    .. math::

        C = \\frac{(N-1) \\sum_{i,j} w_{i,j} (x_i - x_j)^2}
                 {2W \\sum_i (x_i - \\bar{x})^2}

    where :math:`w_{i,j}` are edge weights, :math:`W = \\sum_{i,j} w_{i,j}`,
    and :math:`N` is the number of observations.

    Parameters
    ----------
    graph
        Sparse adjacency matrix [cells × cells] representing cell neighborhood
        graph. Typically this is the connectivities matrix from neighbor graph
        construction (e.g., from KNN or spatial neighbors).
    values
        Values to compute autocorrelation for:
        - 1D array [cells]: Returns scalar Geary's C
        - 2D array [genes × cells]: Returns array of Geary's C per gene

    Returns
    -------
    float or np.ndarray
        Geary's C statistic. Returns scalar if values is 1D, array if 2D.

        Interpretation:
        - C < 1: Positive autocorrelation (similar values cluster together)
        - C = 1: Random spatial pattern
        - C > 1: Negative autocorrelation (dissimilar values cluster together)
        - Range is [0, 2], with expected value 1 under randomness

    Notes
    -----
    Geary's C is inversely related to Moran's I:
    - Both measure spatial autocorrelation
    - Geary's C: Based on squared differences (sensitive to local patterns)
    - Moran's I: Based on products (sensitive to global patterns)

    Lower C values indicate stronger positive spatial autocorrelation.

    Examples
    --------
    >>> import perturblab as pl
    >>> import numpy as np
    >>> from scipy import sparse
    >>>
    >>> # Create a simple neighbor graph
    >>> n = 100
    >>> graph = sparse.lil_matrix((n, n))
    >>> for i in range(n):
    ...     graph[i, (i-1) % n] = 1
    ...     graph[i, (i+1) % n] = 1
    >>> graph = graph.tocsr()
    >>>
    >>> # Spatially autocorrelated pattern
    >>> x_auto = np.sin(np.linspace(0, 4*np.pi, n))
    >>> geary_auto = pl.metrics.gearys_c(graph, x_auto)
    >>> print(f"Geary's C (autocorrelated): {geary_auto:.3f}")
    >>>
    >>> # Random pattern
    >>> x_random = np.random.randn(n)
    >>> geary_random = pl.metrics.gearys_c(graph, x_random)
    >>> print(f"Geary's C (random): {geary_random:.3f}")

    References
    ----------
    Geary, R.C. (1954). "The Contiguity Ratio and Statistical Mapping".
    The Incorporated Statistician 5 (3): 115–146.

    DeTomaso et al. (2019). "Functional interpretation of single cell similarity maps".
    Nature Communications 10: 4376. (VISION method)
    """
    # Convert graph to CSR format
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    elif not sparse.isspmatrix_csr(graph):
        graph = sparse.csr_matrix(graph)

    graph = graph.astype(np.float64, copy=False)

    # Validate graph is square
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Graph must be square, got shape {graph.shape}")

    # Handle different value dimensions
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        # Single vector
        if len(values) != graph.shape[0]:
            raise ValueError(
                f"Values length ({len(values)}) must match graph size ({graph.shape[0]})"
            )

        w = graph.data.sum()
        return float(_gearys_c_vec(graph.data, graph.indices, graph.indptr, values, w))

    elif values.ndim == 2:
        # Matrix of multiple vectors (genes × cells)
        if values.shape[1] != graph.shape[0]:
            raise ValueError(
                f"Values shape[1] ({values.shape[1]}) must match graph size ({graph.shape[0]})"
            )

        # Filter out constant genes
        gene_std = values.std(axis=1)
        is_constant = gene_std == 0

        if is_constant.any():
            n_constant = is_constant.sum()
            logger.warning(f"{n_constant} genes are constant (std=0), returning NaN for these")

            result = np.full(values.shape[0], np.nan, dtype=np.float64)
            non_constant_idx = ~is_constant

            if non_constant_idx.any():
                result[non_constant_idx] = _gearys_c_mtx(
                    graph.data, graph.indices, graph.indptr, values[non_constant_idx]
                )

            return result
        else:
            return _gearys_c_mtx(graph.data, graph.indices, graph.indptr, values)

    else:
        raise ValueError(f"Values must be 1D or 2D, got {values.ndim}D")


def compute_spatial_metrics(
    graph: np.ndarray | sparse.spmatrix,
    values: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute both Moran's I and Geary's C spatial autocorrelation metrics.

    Parameters
    ----------
    graph
        Sparse adjacency matrix [cells × cells].
    values
        Values to compute autocorrelation for [cells] or [genes × cells].

    Returns
    -------
    dict
        Dictionary containing:
        - 'morans_i': Moran's I statistic
        - 'gearys_c': Geary's C statistic

    Examples
    --------
    >>> import perturblab as pl
    >>> metrics = pl.metrics.compute_spatial_metrics(graph, expression)
    >>> print(f"Moran's I: {metrics['morans_i']:.3f}")
    >>> print(f"Geary's C: {metrics['gearys_c']:.3f}")
    """
    return {
        "morans_i": morans_i(graph, values),
        "gearys_c": gearys_c(graph, values),
    }
