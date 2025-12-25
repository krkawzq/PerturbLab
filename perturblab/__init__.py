"""PerturbLab: A unified framework for perturbation analysis and modeling.

PerturbLab provides:
- Unified data structures for single-cell and perturbation data
- Model registry with lazy loading and dependency management
- Pre-trained models (GEARS, scGPT, UCE, etc.)
- Tools for graph construction, similarity computation, and preprocessing
- Comprehensive metrics for evaluation
- Flexible training engines (single-GPU and distributed)
"""

__version__ = "1.0.0"

# Module-level imports for convenience
from . import analysis, data, engine, io, methods, metrics, preprocessing, tools, types, utils

# Core imports
from .core import ModelRegistry

# Model registry instance
from .models import MODELS

# Key types
from .types import (
    DAG,
    BipartiteGraph,
    CellData,
    GeneGraph,
    GeneVocab,
    PerturbationData,
    Vocab,
    WeightedGraph,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "ModelRegistry",
    "MODELS",
    # Types
    "Vocab",
    "GeneVocab",
    "CellData",
    "PerturbationData",
    "WeightedGraph",
    "BipartiteGraph",
    "GeneGraph",
    "DAG",
    # Modules
    "analysis",
    "data",
    "engine",
    "io",
    "metrics",
    "methods",
    "preprocessing",
    "tools",
    "types",
    "utils",
]
