"""Perturbation splitting strategies for single-cell perturbation experiments.

This module provides various splitting strategies for perturbation datasets,
particularly useful for evaluating model generalization capabilities.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional


def extract_genes(perturbation: str, control_labels: Set[str]) -> Set[str]:
    """Extract genes from perturbation string.
    
    Args:
        perturbation: Perturbation string (e.g., 'ACTB+GAPDH')
        control_labels: Set of control labels to exclude
    
    Returns:
        Set of gene names
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
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Simple random split of perturbations.
    
    Args:
        perturbations: List of perturbation strings
        test_size: Fraction for test set
        val_size: Fraction for validation set (optional)
        seed: Random seed
    
    Returns:
        Tuple of (train_perts, test_perts, val_perts)
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
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Gene-level split for simulation experiments.
    
    This splitting strategy divides genes into "seen" (training) and "unseen" (test) sets,
    then assigns perturbations based on which genes they contain. This tests model
    generalization to new genes.
    
    Args:
        perturbations: List of perturbation strings
        control_labels: Set of control labels
        split_type: 'simulation' (include combos) or 'simulation_single' (singles only)
        test_size: Fraction for test set
        val_size: Fraction for validation set
        train_gene_fraction: Fraction of genes to be "seen" (training)
        seed: Random seed
    
    Returns:
        Tuple of (train_perts, test_perts, val_perts)
    
    Example:
        >>> train, test, val = split_perturbations_simulation(
        ...     ['ACTB', 'GAPDH', 'ACTB+GAPDH'],
        ...     control_labels={'ctrl'},
        ...     train_gene_fraction=0.5
        ... )
    """
    np.random.seed(seed)
    
    # Separate singles and combos
    singles = [p for p in perturbations if '+' not in p]
    combos = [p for p in perturbations if '+' in p]
    
    # Extract all unique genes
    all_genes = set()
    for p in singles:
        all_genes.update(extract_genes(p, control_labels))
    all_genes = list(all_genes)
    np.random.shuffle(all_genes)
    
    # Split genes
    n_train_genes = int(len(all_genes) * train_gene_fraction)
    train_genes = set(all_genes[:n_train_genes])
    test_genes = set(all_genes[n_train_genes:])
    
    # Assign singles based on gene membership
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
        # Combos with 2+ train genes go to training (or split)
        seen2_combos = [
            c for c in combos
            if len(extract_genes(c, control_labels) & train_genes) >= 2
        ]
        # Others go to test/val pool
        valtest_combos = [c for c in combos if c not in seen2_combos]
        
        # Split seen2 between train and valtest
        np.random.shuffle(seen2_combos)
        n_train_combo = int(len(seen2_combos) * train_gene_fraction)
        train_seen2 = seen2_combos[:n_train_combo]
        valtest_seen2 = seen2_combos[n_train_combo:]
        
        train_perts = train_singles + train_seen2
        valtest_pool = test_singles + valtest_combos + valtest_seen2
    
    # Split valtest
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
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split combo perturbations by number of seen genes.
    
    This strategy tests model generalization on combo perturbations where
    0, 1, or 2 genes have been seen during training.
    
    Args:
        perturbations: List of perturbation strings
        control_labels: Set of control labels
        target_seen_count: Number of seen genes in test combos (0, 1, or 2)
        test_size: Fraction for test set
        val_size: Fraction for validation set
        train_gene_fraction: Fraction of genes to be "seen"
        seed: Random seed
    
    Returns:
        Tuple of (train_perts, test_perts, val_perts)
    
    Example:
        >>> # Test on combos where both genes are unseen
        >>> train, test, val = split_perturbations_combo_seen(
        ...     perturbations,
        ...     control_labels={'ctrl'},
        ...     target_seen_count=0
        ... )
    """
    np.random.seed(seed)
    
    singles = [p for p in perturbations if '+' not in p]
    combos = [p for p in perturbations if '+' in p]
    
    # Split genes
    all_genes = list(set().union(*[
        extract_genes(p, control_labels) for p in perturbations
    ]))
    np.random.shuffle(all_genes)
    
    n_seen = int(len(all_genes) * train_gene_fraction)
    seen_genes = set(all_genes[:n_seen])
    
    # Train: all singles with seen genes
    train_perts = [
        s for s in singles 
        if extract_genes(s, control_labels).issubset(seen_genes)
    ]
    
    # Test/Val: combos with target_seen_count seen genes
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
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Split with no test set (only train and validation).
    
    Args:
        perturbations: List of perturbation strings
        val_size: Fraction for validation set
        seed: Random seed
    
    Returns:
        Tuple of (train_perts, val_perts)
    """
    np.random.seed(seed)
    perts = list(perturbations)
    np.random.shuffle(perts)
    
    n_val = int(len(perts) * val_size)
    val_perts = perts[:n_val]
    train_perts = perts[n_val:]
    
    return train_perts, val_perts

