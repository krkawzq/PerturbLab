"""Utility tools for perturbation data analysis.

This module provides various utility functions for:
- Data splitting (cell-level and perturbation-level)
- Graph processing (bipartite graph projection)
- GEARS-related utilities

Available Submodules:
    - Cell splitting: Functions for train/test splits of cells
    - Perturbation splitting: Various strategies for perturbation splits
    - GEARS: Bipartite graph projection for gene similarity networks
"""

# Bipartite graph projection tools
from ._bipartite import project_bipartite_graph

# Gene similarity computation tools
from ._gene_similarity import (
    compute_gene_similarity_from_go,
    cosine_similarity_sets,
    jaccard_similarity,
    overlap_coefficient,
)

# Cell splitting tools
from ._split_cell import (
    sample_cells_by_group,
    sample_cells_simple,
    sample_cells_weighted,
    split_cells,
    stratify_split_cells_by_group,
)

# Perturbation splitting tools
from ._split_perturbation import (
    extract_genes,
    split_perturbations_combo_seen,
    split_perturbations_no_test,
    split_perturbations_simple,
    split_perturbations_simulation,
)

__all__ = [
    # Cell splitting
    "split_cells",
    "sample_cells_simple",
    "sample_cells_weighted",
    "sample_cells_by_group",
    "stratify_split_cells_by_group",
    # Perturbation splitting
    "extract_genes",
    "split_perturbations_simple",
    "split_perturbations_simulation",
    "split_perturbations_combo_seen",
    "split_perturbations_no_test",
    # Bipartite graph projection
    "project_bipartite_graph",
    "compute_gene_similarity_from_go",
    "jaccard_similarity",
    "overlap_coefficient",
    "cosine_similarity_sets",
]
