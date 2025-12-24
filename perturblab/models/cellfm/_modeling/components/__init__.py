"""CellFM Model Components.

This package contains all building blocks for CellFM models:
- retention.py: Retention mechanism layers and utilities
- module.py: Value encoders/decoders and feedforward networks
- finetune.py: Main FinetuneModel and loss functions
"""

# Core model
from .finetune import BCELoss, FinetuneModel, MaskedMSE

# Encoders and Decoders
from .module import FFN, CellwiseDecoder, ValueDecoder, ValueEncoder

# Retention layers and utilities
from .retention import (
    DropPath,
    GatedLinearUnit,
    Kernel,
    LoraBlock,
    MHRetention,
    RetentionLayer,
    SiLU,
    SRMSNorm,
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
