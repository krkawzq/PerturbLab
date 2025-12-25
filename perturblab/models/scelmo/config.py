"""Configuration definition for scELMo model."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from perturblab.core.config import Config

__all__ = ["scELMoConfig", "requirements", "dependencies"]

# Required dependencies (mandatory)
# scELMo has no external dependencies beyond PyTorch (which is a core dependency)
requirements = []

# Optional dependencies (for enhanced functionality)
dependencies = []


@dataclass(kw_only=True)
class scELMoConfig(Config):
    """Configuration for scELMo (Single-Cell Embeddings from Language Models).

    Attributes:
        gene_names (List[str]):
            List of gene symbols supported by this model instance.
            This defines the vocabulary for alignment.
        embedding_dim (int):
            Dimension of the gene embeddings (default: 1536 for text-embedding-ada-002).
        aggregation_mode (Literal['wa', 'aa']):
            Method to aggregate gene embeddings into cell embeddings.
            - 'wa': Weighted Average (weighted by gene expression count).
            - 'aa': Arithmetic Average (simple mean of expressed genes).
        api_model (str):
            OpenAI model name used for embedding generation (for metadata tracking).
    Note:
        All fields must be specified as keyword arguments due to dataclass inheritance constraints.
        Example: scELMoConfig(field1=..., field2=...)
    """

    # Required field (using default_factory to allow empty initialization if needed,
    # but practically required for model usage)
    gene_names: List[str] = field(default_factory=list)

    # Optional fields with defaults
    embedding_dim: int = 1536
    aggregation_mode: Literal["wa", "aa"] = "wa"
    api_model: Optional[str] = None
