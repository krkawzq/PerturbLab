"""Input/Output schema definitions for scELMo."""

from __future__ import annotations

from typing import List

from torch import Tensor

from perturblab.core.model_io import ModelIO

__all__ = ["scELMoInput", "scELMoOutput"]


class scELMoInput(ModelIO):
    """Container for scELMo inputs.

    Attributes:
        expression (Tensor):
            Gene expression matrix. Shape: (batch_size, n_genes).
        gene_names (List[str]):
            List of gene symbols corresponding to the columns of `expression`.
            Used to align input genes with the model's internal embedding dictionary.
    """

    expression: Tensor
    gene_names: List[str]


class scELMoOutput(ModelIO):
    """Container for scELMo outputs.

    Attributes:
        cell_embedding (Tensor):
            Aggregated cell-level embeddings. Shape: (batch_size, embedding_dim).
    """

    cell_embedding: Tensor
