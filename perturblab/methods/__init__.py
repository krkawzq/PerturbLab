"""Methods module for model-specific algorithms and workflows.

This module provides complete pipelines for using each foundation model,
including data preprocessing, training, inference, and evaluation.

Each method submodule follows a standard structure:
    - graph.py: Graph construction utilities (if applicable)
    - processing.py: Data preprocessing and format conversion
    - trainer.py: Training logic (optional, to be added)
    - inference.py: Inference and prediction (to be added)
    - evaluation.py: Evaluation metrics (to be added)

All functions are designed to work with PerturbLab's unified type system:
    - CellData / PerturbationData: Core data containers
    - WeightedGraph / BipartiteGraph / DAG: Graph structures
    - ModelIO subclasses: Typed model inputs/outputs

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
    >>> formatted = gears.apply_gears_format(
    ...     data,
    ...     perturb_col='condition',
    ...     control_tag='ctrl'
    ... )
"""

from . import gears

__all__ = ["gears"]
