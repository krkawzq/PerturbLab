"""Cell dataset splitting and sampling algorithms.

This module provides pure algorithmic functions for splitting and sampling
cell datasets, decoupled from the CellDataset class.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable
from sklearn.model_selection import train_test_split
import warnings


def split_cells(
    n_cells: int,
    test_size: float = 0.2,
    stratify_labels: Optional[np.ndarray] = None,
    random_state: int = 42,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Split cell indices into train and test sets.
    
    Args:
        n_cells: Total number of cells
        test_size: Fraction for test set
        stratify_labels: Optional labels for stratified split
        random_state: Random seed
        shuffle: Whether to shuffle data
    
    Returns:
        Tuple of (train_indices, test_indices)
    
    Example:
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
            shuffle=shuffle
        )
    except ValueError:
        warnings.warn("Stratification failed. Random split used.")
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
    
    return train_idx, test_idx


def sample_cells_simple(
    n_cells: int,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42,
    replace: bool = False
) -> np.ndarray:
    """Simple random sampling of cell indices.
    
    Args:
        n_cells: Total number of cells
        n: Number of cells to sample
        frac: Fraction of cells to sample
        random_state: Random seed
        replace: Sample with replacement
    
    Returns:
        Array of sampled indices
    
    Example:
        >>> sampled_idx = sample_cells_simple(
        ...     n_cells=1000,
        ...     n=100,
        ...     random_state=42
        ... )
    """
    np.random.seed(random_state)
    indices = np.arange(n_cells)
    
    size = n if n else int(n_cells * (frac or 0.1))
    sel_idx = np.random.choice(indices, size=size, replace=replace)
    
    return sel_idx


def sample_cells_weighted(
    n_cells: int,
    weights: np.ndarray,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42,
    replace: bool = False
) -> np.ndarray:
    """Weighted sampling of cell indices.
    
    Args:
        n_cells: Total number of cells
        weights: Weight for each cell
        n: Number of cells to sample
        frac: Fraction of cells to sample
        random_state: Random seed
        replace: Sample with replacement
    
    Returns:
        Array of sampled indices
    
    Example:
        >>> weights = np.array([...])  # Custom weights
        >>> sampled_idx = sample_cells_weighted(
        ...     n_cells=1000,
        ...     weights=weights,
        ...     n=100,
        ...     random_state=42
        ... )
    """
    np.random.seed(random_state)
    indices = np.arange(n_cells)
    
    # Normalize weights
    p = weights / weights.sum()
    
    size = n if n else int(n_cells * (frac or 0.1))
    sel_idx = np.random.choice(indices, size=size, replace=replace, p=p)
    
    return sel_idx


def sample_cells_by_group(
    group_indices: dict,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    balance: bool = False,
    random_state: int = 42,
    replace: bool = False
) -> np.ndarray:
    """Sample cells within groups (e.g., cell types).
    
    Args:
        group_indices: Dict mapping group names to arrays of indices
        n: Number of cells to sample per group
        frac: Fraction of cells to sample per group
        balance: Force equal number per group (requires n)
        random_state: Random seed
        replace: Sample with replacement
    
    Returns:
        Array of sampled indices (shuffled)
    
    Example:
        >>> # Sample balanced cells from each cell type
        >>> group_indices = {
        ...     'TypeA': np.array([0, 1, 2, ...]),
        ...     'TypeB': np.array([100, 101, ...]),
        ... }
        >>> sampled_idx = sample_cells_by_group(
        ...     group_indices=group_indices,
        ...     n=50,
        ...     balance=True,
        ...     random_state=42
        ... )
    """
    np.random.seed(random_state)
    sel_indices = []
    
    # Calculate total cells for proportional sampling
    total_cells = sum(len(idx) for idx in group_indices.values())
    
    for group_name, grp_idx in group_indices.items():
        n_grp = len(grp_idx)
        
        # Determine sample size for this group
        if balance:
            if not n:
                raise ValueError("Balance mode requires 'n'")
            size = n
        else:
            if frac:
                size = int(n_grp * frac)
            elif n:
                # Proportional to group size
                size = int(n * (n_grp / total_cells))
            else:
                raise ValueError("Provide n or frac")
        
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
    group_indices: dict,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified split maintaining group proportions.
    
    Args:
        group_indices: Dict mapping group names to arrays of indices
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Tuple of (train_indices, test_indices)
    
    Example:
        >>> group_indices = {
        ...     'TypeA': np.array([0, 1, 2, ...]),
        ...     'TypeB': np.array([100, 101, ...]),
        ... }
        >>> train_idx, test_idx = stratify_split_cells_by_group(
        ...     group_indices=group_indices,
        ...     test_size=0.2,
        ...     random_state=42
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

