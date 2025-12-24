"""GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations.

This module implements the GEARS (Graph-Enhanced gene Activation and Repression Simulator)
method for predicting cellular responses to genetic perturbations using graph neural networks.

References
----------
.. [1] Roohani et al. (2023). "GEARS: Predicting transcriptional outcomes 
       of novel multi-gene perturbations." Nature Methods.
       https://www.nature.com/articles/s41592-023-01905-6

Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License
"""

from .utils import (
    build_perturbation_graph,
    filter_perturbations_in_go,
    get_perturbation_genes,
    dataframe_to_weighted_graph,
    weighted_graph_to_dataframe,
)

__all__ = [
    # Graph construction
    "build_perturbation_graph",
    # Graph conversion (GEARS-specific)
    "dataframe_to_weighted_graph",
    "weighted_graph_to_dataframe",
    # Utilities
    "filter_perturbations_in_go",
    "get_perturbation_genes",
]

