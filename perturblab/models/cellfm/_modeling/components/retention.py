"""CellFM Retention Mechanism Implementation.

This module implements the Multi-Scale Retention (RetNet) mechanism used in CellFM.
It serves as a drop-in replacement for standard self-attention, offering linear
complexity and decay-based positional encoding.

Components:
    - SRMSNorm: Scaled RMS Normalization.
    - MHRetention: Multi-Head Retention with LoRA support.
    - GatedLinearUnit: SwiGLU-based Feed-Forward Network.
    - RetentionLayer: The main encoder block combining Retention and GLU.

Original source: https://github.com/biomed-AI/CellFM
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["RetentionLayer", "SRMSNorm"]


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation."""

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


class Kernel(nn.Module):
    """Simple ReLU kernel for retention mechanism."""

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class LoraBlock(nn.Module):
    """Low-Rank Adaptation (LoRA) block.

    Attributes:
        A (nn.Linear): Down-projection.
        B (nn.Linear): Up-projection.
    """

    def __init__(self, in_dim: int, out_dim: int, r: int):
        super().__init__()
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.B(self.A(x))


class SRMSNorm(nn.Module):
    """Scaled RMS Normalization.

    A variation of RMSNorm that includes a scaling factor based on embedding dimension.
    Ensures numerical stability by computing norm in float32.
    """

    def __init__(self, emb_dims: int, eps: float = 1e-7):
        super().__init__()
        self.scale = 1.0 / math.sqrt(emb_dims)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 to prevent overflow in mixed precision training
        x_dtype = x.dtype
        x_float = x.float()

        norm = torch.norm(x_float * self.scale, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)

        return (x_float / norm).to(dtype=x_dtype)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.dropout.p == 0.0:
            return x

        # Broadcast dropout mask across sequence/channel dims
        B = x.shape[0]
        mask = torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)
        mask = self.dropout(mask)
        return x * mask


class MHRetention(nn.Module):
    """Multi-Head Retention Mechanism.

    Core component of RetNet. Replaces softmax attention with element-wise
    gating and group norm.
    """

    def __init__(self, emb_dims: int, num_heads: int, lth: Optional[int] = None, lora: int = 0):
        super().__init__()
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.head_dim = emb_dims // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.lora = lora

        # Scaling factor for initialization
        beta = 1.0 if lth is None else (lth * 8) ** -0.25

        # Projections
        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)

        # Init
        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

        # Activation Kernels
        self.kernelQ = Kernel()
        self.kernelK = Kernel()
        self.kernelV = nn.Identity()
        self.kernelU = SiLU()

        # Normalization
        self.pre_norm = SRMSNorm(emb_dims)
        self.inner_norm = SRMSNorm(self.head_dim)

        # LoRA Adapters
        if self.lora > 0:
            self.lora_q = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_k = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        v_pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        seq_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for retention.

        Args:
            x (Tensor): Query input [B, L1, D].
            y (Tensor, optional): Key/Value input. Defaults to x (Self-Retention).
            v_pos (Tensor, optional): Positional scaling factor.
            attn_mask (Tensor, optional): Attention mask.
            seq_mask (Tensor, optional): Sequence mask for queries.

        Returns:
            Tensor: Output [B, L1, D].
        """
        if y is None:
            y = x

        B, L1, D = x.shape

        # Linear Projections
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        u = self.u_proj(x)

        # Apply LoRA if enabled
        if self.lora > 0:
            q = q + self.lora_q(x)
            k = k + self.lora_k(y)
            v = v + self.lora_v(y)
            u = u + self.lora_u(x)

        # Reshape to [B, Heads, Seq, HeadDim]
        # Transpose(1, 2) creates [B, H, L, D_h]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        u = u.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Kernels
        q = self.kernelQ(q) / self.scale
        k = self.kernelK(k) / self.scale
        v = self.kernelV(v)
        u = self.kernelU(u)

        # Apply Masks
        if seq_mask is not None:
            q = q * seq_mask
        if attn_mask is not None:
            # attn_mask typically broadcastable to [B, 1, L, 1] or similar
            k = k * attn_mask
        if v_pos is not None:
            v = v * v_pos

        # Retention Operation: Q @ (K.T @ V)
        # 1. K.T @ V -> [B, H, D_h, D_h] (Invariant state)
        # Note: This implementation assumes parallel retention (matrix mult form)
        # efficient for training, but less memory efficient for long sequences than recurrent form.
        # Original code does: KV = MatMul(K.T, V)
        # Correct dimension logic:
        # K: [B, H, L, D_h] -> Transpose(-1, -2) -> [B, H, D_h, L]
        # V: [B, H, L, D_h]
        # KV: [B, H, D_h, D_h]
        kv = torch.matmul(k.transpose(-1, -2), v)

        # 2. Q @ KV -> [B, H, L, D_h]
        out = torch.matmul(q, kv)

        # 3. Normalization and Gating
        out = self.inner_norm(out) * u

        # Reshape back: [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L1, D)

        # Output Projection
        out = self.o_proj(out)
        if self.lora > 0:
            out = out + self.lora_o(out)

        return out


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (FFN) with Swish/SiLU activation.

    Structure: (U * V) @ O
    Where U = x @ W_u, V = x @ W_v
    """

    def __init__(self, emb_dims: int, lth: Optional[int] = None, lora: int = 0):
        super().__init__()
        beta = 1.0 if lth is None else (lth * 8) ** -0.25

        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.norm = SRMSNorm(emb_dims)
        self.lora = lora

        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

        if self.lora > 0:
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        u = self.u_proj(x_flat)
        v = self.v_proj(x_flat)

        if self.lora > 0:
            u = u + self.lora_u(x_flat)
            v = v + self.lora_v(x_flat)

        # Element-wise gating
        out = u * v
        out = self.o_proj(out)

        if self.lora > 0:
            out = out + self.lora_o(out)

        return out.view(B, L, D)


class RetentionLayer(nn.Module):
    """Single layer of the Retention Network (RetNet).

    Combines Multi-Head Retention and Gated Linear Unit with residual connections
    and LayerNorms.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayers: int,
        dropout: float = 0.0,
        lora: int = 0,
        recompute: bool = False,
    ):
        super().__init__()
        # Alpha scaling for residual connection stability
        self.alpha = (2 * nlayers) ** 0.25

        self.attn = MHRetention(d_model, nhead, nlayers, lora)
        self.ffn = GatedLinearUnit(d_model, nlayers, lora)
        self.dropout = nn.Dropout(p=dropout)

        self.post_norm1 = nn.LayerNorm(d_model)
        self.post_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # 1. Retention Block
        # Residual = x * alpha + Retention(x)
        out = self.dropout(self.attn(x, **kwargs))
        x = self.post_norm1(x * self.alpha + out)

        # 2. FFN Block
        # Residual = x * alpha + FFN(x)
        out = self.dropout(self.ffn(x))
        x = self.post_norm2(x * self.alpha + out)

        return x
