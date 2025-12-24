"""scFoundation Configuration.

This module defines the configuration dataclass for scFoundation models.

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License (see forks/scFoundation/LICENSE for details)
"""

from typing import Literal, Optional

from perturblab.core.config import Config

__all__ = ["scFoundationConfig", "requirements", "dependencies"]

# Required dependencies (mandatory)
requirements = []

# Optional dependencies (for enhanced functionality)
# local-attention is optional, used for Performer backend
dependencies = ["local_attention"]


class scFoundationConfig(Config):
    """Configuration options for scFoundation MAE-Autobin model.

    scFoundation is a Transformer-based Masked Autoencoder (MAE) with automatic binning for single-cell RNA-seq data.

    Attributes:
        num_tokens: Size of gene vocabulary.
        max_seq_len: Maximum allowed sequence length (number of genes).
        embed_dim: Dimension of encoder embeddings.
        decoder_embed_dim: Dimension of decoder embeddings.

        bin_num: Number of bins for auto-discretization.
        bin_alpha: Alpha parameter for bin mixing.

        pad_token_id: Integer representing the ID used for padding.
        mask_token_id: Integer representing the ID used for masking.

        transformer_type: Type of Transformer backbone. One of: "performer", "pytorch", "reversible".
        depth: Number of Transformer layers.
        heads: Number of attention heads.
        ff_mult: Multiplier for feedforward hidden size.
        norm_first: If True, use pre-LayerNorm.

        dim_head: Dimension per attention head (Performer-specific).
        local_attn_heads: Number of local attention heads (Performer-specific).
        local_window_size: Local attention window size (Performer-specific).
        causal: If True, use causal attention (Performer-specific).
        reversible: If True, use reversible layers (Performer-specific).
        nb_features: Number of random features for FAVOR+ (Performer-specific).
        feature_redraw_interval: Redraw random features interval (Performer-specific).
        generalized_attention: If True, use generalized attention kernel (Performer-specific).
        kernel_fn: Kernel function for generalized attention, 'relu' or 'gelu' (Performer-specific).
        no_projection: If True, disables projection in attention (Performer-specific).

        tie_embed: If True, tie encoder and decoder embeddings.
        dropout: Dropout rate used in model.
        emb_dropout: Dropout rate on embedding layer.

    Raises:
        ValueError: If invalid parameter values are provided.

    References:
        Wang et al. (2023). "scFoundation: Large-scale Foundation Model for
            Single-cell Transcriptomics." bioRxiv.
    """

    num_tokens: int = 19264
    max_seq_len: int = 3001
    embed_dim: int = 512
    decoder_embed_dim: int = 512

    bin_num: int = 10
    bin_alpha: float = 1.0

    pad_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None

    transformer_type: Literal["performer", "pytorch", "reversible"] = "performer"
    depth: int = 12
    heads: int = 8
    ff_mult: int = 4
    norm_first: bool = True

    dim_head: int = 64
    local_attn_heads: int = 0
    local_window_size: int = 256
    causal: bool = False
    reversible: bool = False
    nb_features: Optional[int] = None
    feature_redraw_interval: int = 1000
    generalized_attention: bool = False
    kernel_fn: Literal["relu", "gelu"] = "relu"
    no_projection: bool = False

    tie_embed: bool = False
    dropout: float = 0.1
    emb_dropout: float = 0.1

    def __post_init__(self):
        """Validates configuration parameters for compatibility.

        Raises:
            ValueError: If embed_dim is not divisible by heads;
                        if transformer_type is not supported;
                        if bin_num is less than 2.
        """
        if self.embed_dim % self.heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by heads ({self.heads})"
            )
        if self.transformer_type not in ("performer", "pytorch", "reversible"):
            raise ValueError(
                f"transformer_type must be one of ['performer', 'pytorch', 'reversible'], "
                f"got {self.transformer_type}"
            )
        if self.bin_num < 2:
            raise ValueError(f"bin_num must be >= 2, got {self.bin_num}")
