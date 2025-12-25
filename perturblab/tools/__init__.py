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

# Similarity metrics
from ._similarity import (
    pairwise_similarities,
    cosine_similarity_sets,
    cosine_similarity_vectors,
    jaccard_similarity,
    overlap_coefficient,
)

# Gene similarity computation tools
from ._gene_similarity import compute_gene_similarity_from_go

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

# DataFrame and graph conversion tools
from ._df_graph_converting import (
    dataframe_to_weighted_graph,
    weighted_graph_to_dataframe,
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
    # Similarity metrics
    "jaccard_similarity",
    "overlap_coefficient",
    "cosine_similarity_sets",
    "cosine_similarity_vectors",
    "compute_pairwise_similarities",
    # Bipartite graph projection
    "project_bipartite_graph",
    "project_bipartite_graph_df",
    "compute_gene_similarity_from_go",
    "compute_gene_similarity_from_go_df",
    # DataFrame and graph conversion
    "dataframe_to_weighted_graph",
    "weighted_graph_to_dataframe",
]
