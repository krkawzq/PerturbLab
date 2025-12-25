"""GEARS method utilities and workflows.

This module provides complete workflows for using GEARS models, including:
- Graph construction for gene similarity networks
- Data preprocessing and format standardization
- Perturbation string processing

All functions are designed to work with PerturbLab's unified type system
(CellData, PerturbationData, WeightedGraph, etc.) and leverage shared operators
from perturblab.tools.

Usage:
    >>> from perturblab.methods import gears
    >>> from perturblab.types import PerturbationData
    >>>
    >>> # Build gene similarity graph
    >>> graph = gears.build_perturbation_graph(
    ...     gene_vocab=data.genes,
    ...     similarity='jaccard'
    ... )
    >>>
    >>> # Standardize data format
    >>> formatted_data = gears.apply_gears_format(
    ...     data,
    ...     perturb_col='condition',
    ...     control_tag='ctrl'
    ... )
"""

from perturblab.utils import get_logger

logger = get_logger()

from .evaluation import (
    compute_de_metrics,
    compute_perturbation_metrics,
    evaluate_predictions,
)
from .graph import (
    build_coexpression_graph,
    build_go_similarity_graph,
    compute_pearson_correlation,
)
from .loss import GEARSLoss, GEARSUncertaintyLoss, build_loss
from .pipeline import (
    build_graphs,
    build_model,
    build_trainer,
    build_training,
    build_training_components,
    create_loader,
)
from .processing import (
    build_collate_fn,
    extract_genes_from_perturbations,
    filter_perturbations_by_genes,
    format_gears,
)

__all__ = [
    # Graph construction
    "build_go_similarity_graph",
    "build_coexpression_graph",
    "compute_pearson_correlation",
    # Data processing
    "format_gears",
    "extract_genes_from_perturbations",
    "filter_perturbations_by_genes",
    "build_collate_fn",
    # Loss functions
    "build_loss",
    "GEARSLoss",
    "GEARSUncertaintyLoss",
    # Evaluation
    "evaluate_predictions",
    "compute_perturbation_metrics",
    "compute_de_metrics",
    # Pipeline (high-level interface)
    "build_graphs",
    "build_model",
    "build_trainer",
    "create_loader",
    "build_training_components",
    "build_training",
]
