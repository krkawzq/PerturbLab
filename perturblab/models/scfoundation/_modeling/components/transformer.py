"""PyTorch Transformer Backend for scFoundation.

This module provides a standard PyTorch Transformer implementation as an
alternative backend to Performer. It is designed to be plug-and-play compatible
with scFoundation's architecture.

Refactored for PerturbLab with:
- Standardized class naming (PascalCase)
- Full type hinting
- Optimized forward pass checks
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["Transformer"]


class Transformer(nn.Module):
    """Standard PyTorch Transformer encoder module.
    
    A clean wrapper around a stack of `nn.TransformerEncoderLayer` with
    final layer normalization. 
    
    NOTE: This implementation manually stacks layers in a ModuleList rather than
    using `nn.TransformerEncoder` to maintain state_dict compatibility with
    original checkpoints (which expect keys like `transformer_encoder.0...` 
    instead of `transformer_encoder.layers.0...`).

    Attributes:
        max_seq_len (int): Maximum sequence length supported.
        depth (int): Number of transformer layers.
        transformer_encoder (nn.ModuleList): The stack of encoder layers.
        norm (nn.LayerNorm): Final layer normalization.
    """
    
    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        ff_mult: int = 4,
        norm_first: bool = False,
    ):
        """Initialize the Transformer backend.

        Args:
            max_seq_len (int): Maximum sequence length.
            dim (int): Hidden dimension (d_model).
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            ff_mult (int, optional): Feedforward dimension multiplier. Defaults to 4.
            norm_first (bool, optional): If True, use Pre-LN. Defaults to False.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth
        
        # We use a ModuleList to replicate the exact structure of the original code
        # for weight compatibility.
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * ff_mult,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        """Forward pass through the transformer.

        Args:
            x (Tensor): Input embeddings.
                Shape: [batch_size, seq_len, dim]
            padding_mask (Tensor): Boolean padding mask.
                Shape: [batch_size, seq_len]
                True indicates a padded position (ignored in attention).

        Returns:
            Tensor: Encoded representations.
                Shape: [batch_size, seq_len, dim]
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum allowed {self.max_seq_len}"
            )

        # Iterate through layers
        # Note: PyTorch TransformerEncoderLayer expects src_key_padding_mask
        # where True = padding/ignore.
        for layer in self.transformer_encoder:
            x = layer(x, src_key_padding_mask=padding_mask)
        
        # Apply final normalization
        x = self.norm(x)

        return x
