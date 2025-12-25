"""Cell dataset splitting and sampling algorithms.

This module provides pure algorithmic functions for splitting and sampling
cell datasets, suitable for train/test splits and data sampling operations.

All functions are data-agnostic and can be applied to any array-like data.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import train_test_split

__all__ = [
    "split_cells",
    "sample_cells_simple",
    "sample_cells_weighted",
    "sample_cells_by_group",
    "stratify_split_cells_by_group",
]


def split_cells(
    n_cells: int,
    test_size: float = 0.2,
    stratify_labels: np.ndarray | None = None,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split cell indices into train and test sets.

    Provides a simple wrapper around sklearn's train_test_split with
    graceful handling of stratification failures.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    test_size : float, default=0.2
        Fraction for test set (0.0 to 1.0).
    stratify_labels : np.ndarray, optional
        Optional labels for stratified split (e.g., cell types).
        If stratification fails (e.g., too few samples per class),
        falls back to random split.
    random_state : int, default=42
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle data before split.

    Returns
    -------
    train_indices : np.ndarray
        Indices for training set.
    test_indices : np.ndarray
        Indices for test set.

    Examples
    --------
    >>> # Simple random split
    >>> train_idx, test_idx = split_cells(n_cells=1000, test_size=0.2)
    >>>
    >>> # Stratified split by cell type
    >>> cell_types = np.array(['TypeA', 'TypeB', ...])
    >>> train_idx, test_idx = split_cells(
    ...     n_cells=1000,
    ...     test_size=0.2,
    ...     stratify_labels=cell_types,
    ...     random_state=42
    ... )
    """
    indices = np.arange(n_cells)

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state,
            shuffle=shuffle,
        )
    except ValueError as e:
        warnings.warn(
            f"Stratification failed: {e}. Falling back to random split.",
            UserWarning,
        )
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    return train_idx, test_idx


def sample_cells_simple(
    n_cells: int,
    n: int | None = None,
    frac: float | None = None,
    random_state: int = 42,
    replace: bool = False,
) -> np.ndarray:
    """Simple random sampling of cell indices.

    Randomly samples a subset of cells without considering any grouping.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    n : int, optional
        Number of cells to sample. Mutually exclusive with `frac`.
    frac : float, optional
        Fraction of cells to sample (0.0 to 1.0).
        Mutually exclusive with `n`.
    random_state : int, default=42
        Random seed for reproducibility.
    replace : bool, default=False
        Sample with replacement.

    Returns
    -------
    np.ndarray
        Array of sampled indices.

    Examples
    --------
    >>> # Sample 100 cells
    >>> sampled_idx = sample_cells_simple(n_cells=1000, n=100)
    >>>
    >>> # Sample 10% of cells
    >>> sampled_idx = sample_cells_simple(n_cells=1000, frac=0.1)
    """
    if n is None and frac is None:
        raise ValueError("Must provide either 'n' or 'frac'")
    if n is not None and frac is not None:
        raise ValueError("Cannot provide both 'n' and 'frac'")

    np.random.seed(random_state)
    indices = np.arange(n_cells)

    size = n if n is not None else int(n_cells * frac)
    sel_idx = np.random.choice(indices, size=size, replace=replace)

    return sel_idx


def sample_cells_weighted(
    n_cells: int,
    weights: np.ndarray,
    n: int | None = None,
    frac: float | None = None,
    random_state: int = 42,
    replace: bool = False,
) -> np.ndarray:
    """Weighted sampling of cell indices.

    Samples cells with probabilities proportional to provided weights.
    Useful for importance sampling or biased sampling strategies.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    weights : np.ndarray
        Weight for each cell (shape: n_cells).
        Will be normalized to sum to 1.
    n : int, optional
        Number of cells to sample. Mutually exclusive with `frac`.
    frac : float, optional
        Fraction of cells to sample. Mutually exclusive with `n`.
    random_state : int, default=42
        Random seed for reproducibility.
    replace : bool, default=False
        Sample with replacement.

    Returns
    -------
    np.ndarray
        Array of sampled indices.

    Examples
    --------
    >>> # Sample cells weighted by expression level
    >>> weights = adata.obs['total_counts'].values
    >>> sampled_idx = sample_cells_weighted(
    ...     n_cells=1000,
    ...     weights=weights,
    ...     n=100
    ... )
    """
    if n is None and frac is None:
        raise ValueError("Must provide either 'n' or 'frac'")
    if n is not None and frac is not None:
        raise ValueError("Cannot provide both 'n' and 'frac'")
    if len(weights) != n_cells:
        raise ValueError(f"weights length ({len(weights)}) must match n_cells ({n_cells})")

    np.random.seed(random_state)
    indices = np.arange(n_cells)

    # Normalize weights
    p = weights / weights.sum()

    size = n if n is not None else int(n_cells * frac)
    sel_idx = np.random.choice(indices, size=size, replace=replace, p=p)

    return sel_idx


def sample_cells_by_group(
    group_indices: dict[str, np.ndarray],
    n: int | None = None,
    frac: float | None = None,
    balance: bool = False,
    random_state: int = 42,
    replace: bool = False,
) -> np.ndarray:
    """Sample cells within groups (e.g., cell types, conditions).

    Samples cells separately from each group, with options for balanced
    or proportional sampling.

    Parameters
    ----------
    group_indices : dict[str, np.ndarray]
        Dictionary mapping group names to arrays of cell indices.
    n : int, optional
        Number of cells to sample per group (if balance=True) or
        total cells to sample proportionally (if balance=False).
    frac : float, optional
        Fraction of cells to sample per group.
    balance : bool, default=False
        If True, sample exactly `n` cells from each group.
        If False, sample proportionally to group size.
        Requires `n` to be specified.
    random_state : int, default=42
        Random seed for reproducibility.
    replace : bool, default=False
        Sample with replacement.

    Returns
    -------
    np.ndarray
        Array of sampled indices (shuffled).

    Examples
    --------
    >>> # Balanced sampling: 50 cells per cell type
    >>> group_indices = {
    ...     'TypeA': np.array([0, 1, 2, ...]),
    ...     'TypeB': np.array([100, 101, ...]),
    ... }
    >>> sampled_idx = sample_cells_by_group(
    ...     group_indices=group_indices,
    ...     n=50,
    ...     balance=True
    ... )
    >>>
    >>> # Proportional sampling: 10% from each group
    >>> sampled_idx = sample_cells_by_group(
    ...     group_indices=group_indices,
    ...     frac=0.1
    ... )
    """
    if n is None and frac is None:
        raise ValueError("Must provide either 'n' or 'frac'")
    if balance and n is None:
        raise ValueError("Balance mode requires 'n'")

    np.random.seed(random_state)
    sel_indices = []

    # Calculate total cells for proportional sampling
    total_cells = sum(len(idx) for idx in group_indices.values())

    for group_name, grp_idx in group_indices.items():
        n_grp = len(grp_idx)

        # Determine sample size for this group
        if balance:
            size = n
        else:
            if frac is not None:
                size = int(n_grp * frac)
            else:  # n is not None
                # Proportional to group size
                size = int(n * (n_grp / total_cells))

        # Adjust if size exceeds group size (when not replacing)
        if size > n_grp and not replace:
            size = n_grp

        # Sample from this group
        if size > 0:
            sel = np.random.choice(grp_idx, size=size, replace=replace)
            sel_indices.append(sel)

    # Concatenate and shuffle
    if len(sel_indices) == 0:
        return np.array([], dtype=int)

    final_idx = np.concatenate(sel_indices)
    np.random.shuffle(final_idx)

    return final_idx


def stratify_split_cells_by_group(
    group_indices: dict[str, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split maintaining group proportions.

    Splits each group independently with the same test_size ratio,
    ensuring that the train/test split maintains the original group
    distribution.

    Parameters
    ----------
    group_indices : dict[str, np.ndarray]
        Dictionary mapping group names to arrays of cell indices.
    test_size : float, default=0.2
        Fraction for test set (0.0 to 1.0).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_indices : np.ndarray
        Indices for training set.
    test_indices : np.ndarray
        Indices for test set.

    Examples
    --------
    >>> group_indices = {
    ...     'TypeA': np.array([0, 1, 2, ...]),
    ...     'TypeB': np.array([100, 101, ...]),
    ... }
    >>> train_idx, test_idx = stratify_split_cells_by_group(
    ...     group_indices=group_indices,
    ...     test_size=0.2
    ... )
    """
    np.random.seed(random_state)
    train_indices = []
    test_indices = []

    for group_name, grp_idx in group_indices.items():
        n_grp = len(grp_idx)
        n_test = int(n_grp * test_size)
        n_train = n_grp - n_test

        # Shuffle group indices
        shuffled = grp_idx.copy()
        np.random.shuffle(shuffled)

        # Split
        train_indices.append(shuffled[:n_train])
        test_indices.append(shuffled[n_train:])

    # Concatenate and shuffle
    train_idx = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
    test_idx = np.concatenate(test_indices) if test_indices else np.array([], dtype=int)

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return train_idx, test_idx
