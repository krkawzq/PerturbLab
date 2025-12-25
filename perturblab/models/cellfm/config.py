"""CellFM Configuration.

This module defines the configuration dataclass for CellFM models.

CellFM (Cell Foundation Model) uses the Retention mechanism rather than
standard self-attention for efficient long-sequence modeling.

Copyright (c) 2023 CellFM Authors
Licensed under CC BY-NC-ND 4.0
"""

from dataclasses import dataclass

from perturblab.core.config import Config

__all__ = ["CellFMConfig", "requirements", "dependencies"]

# Required dependencies (mandatory)
requirements = []

# Optional dependencies (for enhanced functionality)
dependencies = []


@dataclass(kw_only=True)
class CellFMConfig(Config):
    """Configuration class for CellFM models.

    This class captures configuration parameters for the Cell Foundation Model (CellFM),
        which leverages a Retention mechanism for efficient gene expression modeling.

    Attributes:
        n_genes (int): Number of genes in the vocabulary.
        enc_dims (int): Encoder embedding dimension. Defaults to 1280.
        enc_nlayers (int): Number of encoder retention layers. Defaults to 12.
        enc_num_heads (int): Number of retention heads. Defaults to 10.
        enc_dropout (float): Encoder dropout rate. Defaults to 0.1.
        add_zero (bool): Whether to add explicit zero embedding. Defaults to True.
        pad_zero (bool): Whether to pad with zeros. Defaults to False.
        dropout (float): General dropout rate. Defaults to 0.1.
        lora (int): LoRA rank (0 disables LoRA). Defaults to 0.
        recompute (bool): Whether to use gradient checkpointing. Defaults to False.
        cellwise_use_bias (bool): Whether to use bias in cellwise decoder. Defaults to True.
        ecs (bool): Whether to enable Elastic Cell Similarity loss. Defaults to False.
        ecs_threshold (float): Threshold for ECS loss. Defaults to 0.5.
    Note:
        All fields must be specified as keyword arguments due to dataclass inheritance constraints.
        Example: CellFMConfig(field1=..., field2=...)
    """

    n_genes: int
    enc_dims: int = 1280
    enc_nlayers: int = 12
    enc_num_heads: int = 10
    enc_dropout: float = 0.1

    add_zero: bool = True
    pad_zero: bool = False

    dropout: float = 0.1

    lora: int = 0
    recompute: bool = False
    cellwise_use_bias: bool = True

    ecs: bool = False
    ecs_threshold: float = 0.5

    def __post_init__(self):
        """Validates configuration arguments.

        Raises:
            ValueError: If enc_dims is not divisible by enc_num_heads.
            ValueError: If lora is negative.
        """
        if self.enc_dims % self.enc_num_heads != 0:
            raise ValueError(
                f"enc_dims ({self.enc_dims}) must be divisible by "
                f"enc_num_heads ({self.enc_num_heads})"
            )
        if self.lora < 0:
            raise ValueError(f"lora rank must be >= 0, got {self.lora}")
