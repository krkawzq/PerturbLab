"""Configuration for UCE (Universal Cell Embeddings) model."""

from dataclasses import dataclass, field
from perturblab.core.config import Config

__all__ = ["UCEConfig", "dependencies"]

# Required dependencies for UCE model
dependencies = []


@dataclass
class UCEConfig(Config):
    """Configuration for UCE (Universal Cell Embeddings) model.

    UCE is a foundation model for single-cell biology based on the Transformer
    architecture. It is designed to learn universal cell embeddings from gene
    expression data by treating genes as tokens in a sequence.

    References:
        Rosen, Y. et al. (2023). "Universal Cell Embeddings: A Foundation Model
        for Cell Biology." bioRxiv.
        https://doi.org/10.1101/2023.11.28.568918

    Attributes:
        token_dim: Dimension of input gene tokens. This typically includes the
            gene expression value embedding plus any metadata embeddings.
            (Required)
        d_model: The number of expected features in the encoder/decoder inputs.
            Default: 1280.
        nhead: The number of heads in the multiheadattention models.
            Default: 20.
        d_hid: The dimension of the feedforward network model (FFN).
            Default: 5120.
        nlayers: The number of sub-encoder-layers in the encoder.
            Default: 4 (Lightweight version). Note: The full model uses 33 layers.
        output_dim: The dimension of the final cell embedding (CLS token).
            Default: 1280.
        dropout: The dropout value.
            Default: 0.05.
        max_len: The maximum sequence length for positional encoding.
            Default: 1536.
    """

    # Required parameters
    token_dim: int

    # Model architecture defaults
    d_model: int = 1280
    nhead: int = 20
    d_hid: int = 5120
    nlayers: int = 4  # Default to lightweight; use 33 for "UCE-33L"
    output_dim: int = 1280

    # Training/Inference options
    dropout: float = 0.05
    max_len: int = 1536
