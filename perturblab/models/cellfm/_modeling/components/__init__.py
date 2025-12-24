"""CellFM Model Components.

This package contains all building blocks for CellFM models:
- retention.py: Retention mechanism layers and utilities
- module.py: Value encoders/decoders and feedforward networks
- finetune.py: Main FinetuneModel and loss functions
"""

# Core model
from .finetune import FinetuneModel, MaskedMSE, BCELoss

# Encoders and Decoders
from .module import FFN, ValueEncoder, ValueDecoder, CellwiseDecoder

# Retention layers and utilities
from .retention import (
    SiLU,
    Kernel,
    LoraBlock,
    SRMSNorm,
    DropPath,
    MHRetention,
    GatedLinearUnit,
    RetentionLayer,
)

__all__ = [
    # Core Model
    "FinetuneModel",
    "MaskedMSE",
    "BCELoss",
    # Encoders/Decoders
    "FFN",
    "ValueEncoder",
    "ValueDecoder",
    "CellwiseDecoder",
    # Retention Layers
    "SiLU",
    "Kernel",
    "LoraBlock",
    "SRMSNorm",
    "DropPath",
    "MHRetention",
    "GatedLinearUnit",
    "RetentionLayer",
]
