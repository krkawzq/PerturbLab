"""scFoundation Embedding Components (Optimized).

This module contains embedding layers for scFoundation models, including
auto-discretization embeddings and positional embeddings.

Refactored for PerturbLab with:
- Improved device handling using register_buffer
- Proper type hinting
- Efficient tensor operations
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["AutoDiscretizationEmbedding", "RandomPositionalEmbedding"]


class AutoDiscretizationEmbedding(nn.Module):
    """Auto-discretization embedding layer with learnable binning.
    
    This layer learns to discretize continuous gene expression values into bins
    using a neural network (soft binning), then embeds these bins.
    
    Architecture:
        Input(1) -> Linear -> LeakyReLU -> Linear -> Residual -> Softmax -> Weighted Sum of Embeddings
    
    Args:
        dim (int): Output embedding dimension.
        bin_num (int): Number of learnable bins.
        bin_alpha (float): Mixing coefficient for the residual connection in the binning network.
        mask_token_id (Optional[float]): Value representing the mask token in input.
        pad_token_id (Optional[float]): Value representing the padding token in input.
    """
    
    def __init__(
        self, 
        dim: int, 
        bin_num: int, 
        bin_alpha: float = 1.0, 
        mask_token_id: Optional[float] = None, 
        pad_token_id: Optional[float] = None
    ):
        super().__init__()
        
        self.dim = dim
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        
        # Binning Network (CRITICAL: attribute names must match original)
        self.mlp = nn.Linear(1, bin_num)
        self.mlp2 = nn.Linear(bin_num, bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        
        # Bin embeddings (CRITICAL: attribute name must be 'emb')
        self.emb = nn.Embedding(bin_num, dim)
        
        # Special token embeddings (CRITICAL: names must match)
        self.emb_mask = nn.Embedding(1, dim)
        self.emb_pad = nn.Embedding(1, dim)
        
        # Pre-register bin indices as buffer for automatic device handling
        self.register_buffer("bin_num_idx", torch.arange(bin_num), persistent=False)
        self.register_buffer("tensor0", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(
        self, 
        x: Tensor, 
        return_weights: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            x (Tensor): Input expression values. Shape: [batch, seq_len, 1].
            return_weights (bool): If True, returns the soft bin weights.

        Returns:
            Tensor: Embedded values. Shape: [batch, seq_len, dim].
            (Optional) Tensor: Soft bin weights. Shape: [batch, seq_len, bin_num].
        """
        # 1. Compute Soft Binning (exactly matching original)
        x = self.mlp(x)  # [B,N,1] -> [B,N,bin_num]
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)  # [B, N, bin_num]
        
        # 2. Get bin embeddings and compute weighted sum
        bin_num_idx = self.bin_num_idx.to(x.device)
        token_emb = self.emb(bin_num_idx)  # [bin_num, dim]
        x = torch.matmul(weight, token_emb)  # [B, N, dim]
        
        # 3. Handle Special Tokens (exactly matching original)
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)
        
        # Mask tokens
        x_mask_idx = (x == self.mask_token_id).nonzero() if self.mask_token_id is not None else torch.empty((0, 2), device=x.device, dtype=torch.long)
        if x_mask_idx.shape[0] > 0:
            mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
            x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(x_mask_idx.shape[0], 1)
        
        # Padding tokens  
        x_pad_idx = (x == self.pad_token_id).nonzero() if self.pad_token_id is not None else torch.empty((0, 2), device=x.device, dtype=torch.long)
        if x_pad_idx.shape[0] > 0:
            pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
            x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(x_pad_idx.shape[0], 1)

        if return_weights:
            return x, weight
        
        return x


class RandomPositionalEmbedding(nn.Module):
    """Learnable absolute positional embedding.
    
    Standard lookup-table based positional embedding.
    
    Args:
        dim (int): Embedding dimension.
        max_seq_len (int): Maximum supported sequence length.
    """
    
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # CRITICAL: attribute name must be 'emb' for weight compatibility
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Generate positional embeddings matching input length.

        Args:
            x (Tensor): Input tensor. Shape: [batch, seq_len, ...].

        Returns:
            Tensor: Positional embeddings. Shape: [seq_len, dim].
        """
        seq_len = x.shape[1]
        
        # Create positions [0, 1, ..., seq_len-1] on the correct device
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        
        return self.emb(positions)
