"""Perturbation splitting strategies for single-cell perturbation experiments.

This module provides various splitting strategies for perturbation datasets,
particularly useful for evaluating model generalization capabilities on:
- Unseen perturbations (simple random split)
- Unseen genes (simulation split)
- Combination perturbations with varying levels of gene overlap

All splitting functions return perturbation names/identifiers, not cell indices.
"""

from __future__ import annotations

from typing import List, Tuple, Set, Optional

import numpy as np

__all__ = [
    "extract_genes",
    "split_perturbations_simple",
    "split_perturbations_simulation",
    "split_perturbations_combo_seen",
    "split_perturbations_no_test",
]


def extract_genes(
    perturbation: str,
    control_labels: Set[str],
) -> Set[str]:
    """Extract genes from perturbation string.
    
    Parses a perturbation identifier (e.g., 'ACTB+GAPDH') to extract
    individual gene names, excluding control labels.
    
    Parameters
    ----------
    perturbation : str
        Perturbation string (e.g., 'ACTB+GAPDH', 'TP53', 'ctrl').
    control_labels : set[str]
        Set of control labels to exclude (e.g., {'ctrl', 'control'}).
    
    Returns
    -------
    set[str]
        Set of gene names in the perturbation.
    
    Examples
    --------
    >>> extract_genes('ACTB+GAPDH', {'ctrl'})
    {'ACTB', 'GAPDH'}
    >>> extract_genes('ctrl', {'ctrl'})
    set()
    """
    return set(
        g.strip()
        for g in perturbation.split('+')
        if g.strip() not in control_labels
    )


def split_perturbations_simple(
    perturbations: List[str],
    test_size: float,
    val_size: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Simple random split of perturbations.
    
    Randomly divides perturbations into train, test, and optionally validation sets.
    This is the simplest split and does not consider gene-level relationships.
    
    Parameters
    ----------
    perturbations : list[str]
        List of perturbation identifiers.
    test_size : float
        Fraction for test set (0.0 to 1.0).
    val_size : float, optional
        Fraction for validation set. If None, no validation set is created.
    seed : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    train_perts : list[str]
        Training perturbations.
    test_perts : list[str]
        Test perturbations.
    val_perts : list[str]
        Validation perturbations (empty list if val_size is None).
    
    Examples
    --------
    >>> perts = ['ACTB', 'GAPDH', 'TP53', 'ACTB+GAPDH']
    >>> train, test, val = split_perturbations_simple(
    ...     perts, test_size=0.2, val_size=0.1
    ... )
    """
    np.random.seed(seed)
    perts = list(perturbations)
    np.random.shuffle(perts)
    
    n_total = len(perts)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size) if val_size else 0
    n_train = n_total - n_test - n_val
    
    train = perts[:n_train]
    val = perts[n_train:n_train + n_val] if n_val > 0 else []
    test = perts[n_train + n_val:]
    
    return train, test, val


def split_perturbations_simulation(
    perturbations: List[str],
    control_labels: Set[str],
    split_type: str = 'simulation',
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    train_gene_fraction: float = 0.7,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Gene-level split for simulation experiments.
    
    This splitting strategy divides genes into "seen" (training) and "unseen" (test) sets,
    then assigns perturbations based on which genes they contain. This tests model
    generalization to entirely new genes.
    
    Parameters
    ----------
    perturbations : list[str]
        List of perturbation identifiers.
    control_labels : set[str]
        Set of control labels to exclude from gene extraction.
    split_type : {'simulation', 'simulation_single'}, default='simulation'
        - 'simulation': Include combination perturbations in split.
        - 'simulation_single': Only use single-gene perturbations.
    test_size : float, default=0.2
        Fraction for test set.
    val_size : float, optional
        Fraction for validation set.
    train_gene_fraction : float, default=0.7
        Fraction of genes to be "seen" during training.
    seed : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    train_perts : list[str]
        Training perturbations (containing only seen genes).
    test_perts : list[str]
        Test perturbations (containing unseen genes).
    val_perts : list[str]
        Validation perturbations.
    
    Examples
    --------
    >>> perts = ['ACTB', 'GAPDH', 'TP53', 'MYC', 'ACTB+GAPDH', 'TP53+MYC']
    >>> train, test, val = split_perturbations_simulation(
    ...     perts,
    ...     control_labels={'ctrl'},
    ...     train_gene_fraction=0.5  # Half genes seen, half unseen
    ... )
    
    Notes
    -----
    - Single perturbations are assigned based on whether their gene is in train_genes
    - Combo perturbations with ≥2 train genes are partially assigned to training
    - This tests generalization to completely new genes
    """
    if split_type not in ('simulation', 'simulation_single'):
        raise ValueError(
            f"split_type must be 'simulation' or 'simulation_single', got {split_type!r}"
        )
    
    np.random.seed(seed)
    
    # Separate singles and combos
    singles = [p for p in perturbations if '+' not in p]
    combos = [p for p in perturbations if '+' in p]
    
    # Extract all unique genes from single perturbations
    all_genes = set()
    for p in singles:
        all_genes.update(extract_genes(p, control_labels))
    all_genes = list(all_genes)
    np.random.shuffle(all_genes)
    
    # Split genes into seen (train) and unseen (test)
    n_train_genes = int(len(all_genes) * train_gene_fraction)
    train_genes = set(all_genes[:n_train_genes])
    test_genes = set(all_genes[n_train_genes:])
    
    # Assign single perturbations based on gene membership
    train_singles = [
        s for s in singles
        if extract_genes(s, control_labels).issubset(train_genes)
    ]
    test_singles = [
        s for s in singles
        if extract_genes(s, control_labels).issubset(test_genes)
    ]
    
    if split_type == 'simulation_single':
        # Only use single perturbations
        valtest_pool = test_singles
        train_perts = train_singles
    else:
        # Include combo perturbations
        # Combos with 2+ train genes go to training (or get split)
        seen2_combos = [
            c for c in combos
            if len(extract_genes(c, control_labels) & train_genes) >= 2
        ]
        # Others (with 0-1 train genes) go to test/val pool
        valtest_combos = [c for c in combos if c not in seen2_combos]
        
        # Split seen2_combos between train and valtest
        np.random.shuffle(seen2_combos)
        n_train_combo = int(len(seen2_combos) * train_gene_fraction)
        train_seen2 = seen2_combos[:n_train_combo]
        valtest_seen2 = seen2_combos[n_train_combo:]
        
        train_perts = train_singles + train_seen2
        valtest_pool = test_singles + valtest_combos + valtest_seen2
    
    # Split valtest_pool into val and test
    np.random.shuffle(valtest_pool)
    if val_size:
        n_val = int(len(valtest_pool) * (val_size / (test_size + val_size)))
    else:
        n_val = 0
    
    val_perts = valtest_pool[:n_val]
    test_perts = valtest_pool[n_val:]
    
    return train_perts, test_perts, val_perts


def split_perturbations_combo_seen(
    perturbations: List[str],
    control_labels: Set[str],
    target_seen_count: int,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    train_gene_fraction: float = 0.7,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split combo perturbations by number of seen genes.
    
    This strategy tests model generalization on combination perturbations where
    0, 1, or 2 genes have been seen during training. This is useful for
    evaluating compositional generalization.
    
    Parameters
    ----------
    perturbations : list[str]
        List of perturbation identifiers.
    control_labels : set[str]
        Set of control labels to exclude.
    target_seen_count : int
        Number of seen genes in test combo perturbations.
        - 0: Both genes unseen (hardest)
        - 1: One gene seen, one unseen
        - 2: Both genes seen (easiest)
    test_size : float, default=0.2
        Fraction for test set.
    val_size : float, optional
        Fraction for validation set.
    train_gene_fraction : float, default=0.7
        Fraction of genes to be "seen" during training.
    seed : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    train_perts : list[str]
        Training perturbations (all single perturbations with seen genes).
    test_perts : list[str]
        Test perturbations (combos with target_seen_count seen genes).
    val_perts : list[str]
        Validation perturbations.
    
    Examples
    --------
    >>> # Test on combos where both genes are unseen (hardest case)
    >>> train, test, val = split_perturbations_combo_seen(
    ...     perturbations,
    ...     control_labels={'ctrl'},
    ...     target_seen_count=0  # Both genes in combo are unseen
    ... )
    >>> 
    >>> # Test on combos where one gene is seen, one unseen
    >>> train, test, val = split_perturbations_combo_seen(
    ...     perturbations,
    ...     control_labels={'ctrl'},
    ...     target_seen_count=1
    ... )
    
    Notes
    -----
    - Training set contains all single perturbations with seen genes
    - Test/validation sets contain only combo perturbations meeting the target_seen_count
    - This evaluates compositional generalization capabilities
    """
    if target_seen_count not in (0, 1, 2):
        raise ValueError(f"target_seen_count must be 0, 1, or 2, got {target_seen_count}")
    
    np.random.seed(seed)
    
    singles = [p for p in perturbations if '+' not in p]
    combos = [p for p in perturbations if '+' in p]
    
    # Extract and split genes
    all_genes = list(set().union(*[
        extract_genes(p, control_labels) for p in perturbations
    ]))
    np.random.shuffle(all_genes)
    
    n_seen = int(len(all_genes) * train_gene_fraction)
    seen_genes = set(all_genes[:n_seen])
    
    # Train: all single perturbations with seen genes
    train_perts = [
        s for s in singles
        if extract_genes(s, control_labels).issubset(seen_genes)
    ]
    
    # Test/Val: combo perturbations with exactly target_seen_count seen genes
    target_combos = []
    for c in combos:
        genes = extract_genes(c, control_labels)
        if len(genes) < 2:
            continue
        n_seen_in_combo = len(genes & seen_genes)
        if n_seen_in_combo == target_seen_count:
            target_combos.append(c)
    
    # Split target_combos into val and test
    np.random.shuffle(target_combos)
    if val_size:
        n_val = int(len(target_combos) * (val_size / (test_size + val_size)))
    else:
        n_val = 0
    
    val_perts = target_combos[:n_val]
    test_perts = target_combos[n_val:]
    
    return train_perts, test_perts, val_perts


def split_perturbations_no_test(
    perturbations: List[str],
    val_size: float,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split with no test set (only train and validation).
    
    Useful when you want to use all data for training but still need
    validation for hyperparameter tuning or early stopping.
    
    Parameters
    ----------
    perturbations : list[str]
        List of perturbation identifiers.
    val_size : float
        Fraction for validation set (0.0 to 1.0).
    seed : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    train_perts : list[str]
        Training perturbations.
    val_perts : list[str]
        Validation perturbations.
    
    Examples
    --------
    >>> train, val = split_perturbations_no_test(
    ...     perturbations,
    ...     val_size=0.1  # 10% for validation, 90% for training
    ... )
    """
    np.random.seed(seed)
    perts = list(perturbations)
    np.random.shuffle(perts)
    
    n_val = int(len(perts) * val_size)
    val_perts = perts[:n_val]
    train_perts = perts[n_val:]
    
    return train_perts, val_perts

