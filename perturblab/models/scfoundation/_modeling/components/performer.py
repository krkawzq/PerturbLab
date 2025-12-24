"""Performer Architecture Implementation for scBERT.

This module implements the Performer mechanism, a linear attention approximation
using positive orthogonal random features (ORFs). It includes specialized
positional embeddings for gene vectors.

Original Source: https://github.com/TencentAILabHealthcare/scBERT
Based on: https://github.com/lucidrains/performer-pytorch
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# External dependencies (assuming installed)
try:
    from local_attention import LocalAttention
except ImportError:
    LocalAttention = None

# Internal dependencies (from your previous refactor)
from .reversible import ReversibleSequence, SequentialSequence

__all__ = [
    "FastAttention",
    "Performer",
    "PerformerModule",
    "Gene2VecPositionalEmbedding",
]


# =============================================================================
# Helpers
# =============================================================================

def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def empty(tensor: Tensor) -> bool:
    return tensor.numel() == 0


def cast_tuple(val: Any) -> Tuple:
    return (val,) if not isinstance(val, tuple) else val


@contextmanager
def null_context():
    yield


def get_module_device(module: nn.Module) -> torch.device:
    """Safely retrieves the device of a module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        # Fallback for modules with no direct parameters (e.g., wrappers)
        for _, v in module.__dict__.items():
            if torch.is_tensor(v):
                return v.device
        return torch.device('cpu')


# =============================================================================
# Kernel Functions (FAVOR+)
# =============================================================================

def softmax_kernel(
    data: Tensor,
    *,
    projection_matrix: Tensor,
    is_query: bool,
    normalize_data: bool = True,
    eps: float = 1e-4,
    device: Optional[torch.device] = None
) -> Tensor:
    """Approximates softmax kernel using orthogonal random features."""
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0
    ratio = (projection_matrix.shape[0] ** -0.5)

    # projection_matrix: shape (j, d)
    # target shape: (b, h, j, d)
    projection = projection_matrix.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                      torch.max(data_dash, dim=-1, keepdim=True).values) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps
        )

    return data_dash.type_as(data)


def generalized_kernel(
    data: Tensor,
    *,
    projection_matrix: Optional[Tensor],
    kernel_fn: nn.Module = nn.ReLU(),
    kernel_epsilon: float = 0.001,
    normalize_data: bool = True,
    device: Optional[torch.device] = None
) -> Tensor:
    """Generalized kernel for non-softmax attention."""
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = projection_matrix.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


@torch.no_grad()
def orthogonal_matrix_chunk(cols: int, device: Optional[torch.device] = None) -> Tensor:
    """Generates a chunk of orthogonal matrix via QR decomposition."""
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q = q.to(device)
    return q.t()


@torch.no_grad()
def gaussian_orthogonal_random_matrix(
    nb_rows: int,
    nb_columns: int,
    scaling: int = 0,
    device: Optional[torch.device] = None
) -> Tensor:
    """Generates Gaussian Orthogonal Random Matrix for FAVOR+."""
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


# =============================================================================
# Attention Implementations
# =============================================================================

def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Standard linear attention (O(N) complexity)."""
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


def causal_linear_attention(q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-6) -> Tensor:
    """Efficient causal linear attention using CUDA kernels if available."""
    try:
        from fast_transformers.causal_product import CausalDotProduct
    except ImportError:
        # Fallback if fast_transformers is not compiled/installed
        return causal_linear_attention_noncuda(q, k, v)

    # Note: Apex dependence removed in favor of PyTorch native AMP
    autocast_enabled = torch.is_autocast_enabled()
    
    # Handle autocast context manually if needed for the kernel
    # In PyTorch 2.0+, many of these kernels might need checking
    cuda_context = null_context if not autocast_enabled else partial(torch.amp.autocast, device_type='cuda', enabled=False)

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))
        out = CausalDotProduct.apply(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out


def causal_linear_attention_noncuda(q: Tensor, k: Tensor, v: Tensor, chunk_size: int = 128) -> Tensor:
    """Pure PyTorch implementation of causal linear attention (slower, memory efficient)."""
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q_chunk, k_chunk, v_chunk in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
        k_cumsum = last_k_cumsum + k_chunk.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q_chunk, k_cumsum.type_as(q_chunk))
        context = torch.einsum('...nd,...ne->...nde', k_chunk, v_chunk)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q_chunk, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim=-2)


class FastAttention(nn.Module):
    """Fast Attention Module implementing FAVOR+."""
    
    def __init__(
        self,
        dim_heads: int,
        nb_features: Optional[int] = None,
        ortho_scaling: int = 0,
        causal: bool = False,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        no_projection: bool = False
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling
        )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

        if causal:
            # Check for CUDA extension availability
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = causal_linear_attention
            except ImportError:
                warnings.warn(
                    "Unable to import CUDA code for auto-regressive Performer. "
                    "Defaulting to slower non-CUDA version."
                )
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device: torch.device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q: Tensor, k: Tensor, v: Tensor, output_attentions: bool = False):
        device = q.device
        
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device
            )
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(
                softmax_kernel,
                projection_matrix=self.projection_matrix,
                device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        
        if output_attentions:
            # Approximate attention weights calculation (expensive)
            v_diag = torch.eye(v.shape[-2], device=device)
            v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0], v.shape[1], 1, 1)
            # Use float16 on CPU to save memory for visualization matrices
            attn_weights = torch.zeros(
                1, q.shape[1], q.shape[2], q.shape[2], 
                device='cpu', dtype=torch.float16
            )
            
            for head_dim in range(q.shape[1]):
                q_h = q[:, head_dim].to(torch.float16)
                k_h = k[:, head_dim].to(torch.float16)
                v_h = v_diag[:, head_dim].to(torch.float16)
                
                weights_h = attn_fn(q_h, k_h, v_h).detach().cpu()
                attn_weights[0, head_dim] = weights_h
                
            attn_weights /= q.shape[1]
            return out, attn_weights
        
        return out


# =============================================================================
# Helper Layers (Normalization & FeedForward)
# =============================================================================

class ReZero(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(x, **kwargs) * self.g


class PreScaleNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)


class PreLayerNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks: int, fn: nn.Module, along_dim: int = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.,
        activation: Optional[Callable] = None,
        glu: bool = False
    ):
        super().__init__()
        activation = default(activation, nn.GELU)
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


# =============================================================================
# Self Attention & Positional Embeddings
# =============================================================================

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        causal: bool = False,
        heads: int = 8,
        dim_head: int = 64,
        local_heads: int = 0,
        local_window_size: int = 256,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        dropout: float = 0.,
        no_projection: bool = False,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        
        self.fast_attention = FastAttention(
            dim_head, nb_features, causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn, no_projection=no_projection
        )

        self.heads = heads
        self.global_heads = heads - local_heads
        
        self.local_attn = None
        if local_heads > 0:
            if LocalAttention is None:
                raise ImportError("local_attention package is required for local_heads > 0")
            self.local_attn = LocalAttention(
                window_size=local_window_size, causal=causal, autopad=True,
                dropout=dropout, look_forward=int(not causal),
                rel_pos_emb_config=(dim_head, local_heads)
            )

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        pos_emb: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        
        b, n = x.shape[0], x.shape[1]
        h = self.heads
        gh = self.global_heads
        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # Split heads: original einops: 'b n (h d) -> b h n d'
        d = q.shape[-1] // h
        q = q.view(b, n, h, d).transpose(1,2)  # (b, h, n, d)
        k = k.view(b, n, h, d).transpose(1,2)
        v = v.view(b, n, h, d).transpose(1,2)

        # Split into global and local: (q, lq), (k, lk), (v, lv)
        q, lq = q[:, :gh], q[:, gh:]
        k, lk = k[:, :gh], k[:, gh:]
        v, lv = v[:, :gh], v[:, gh:]

        attn_outs = []
        attn_weights = None

        # Global Performer Attention
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            if output_attentions:
                out, attn_weights = self.fast_attention(q, k, v, output_attentions=True)
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        # Local Window Attention
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        # Combine
        out = torch.cat(attn_outs, dim=1)
        # Rearrange 'b h n d -> b n (h d)'
        bh, hn, d = out.shape[0], out.shape[2], out.shape[3]
        h_total = out.shape[1]
        out = out.transpose(1,2).contiguous().view(bh, hn, h_total*d)
        out = self.to_out(out)
        
        if output_attentions:
            return self.dropout(out), attn_weights
        
        return self.dropout(out)


# -----------------------------------------------------------------------------
# Positional Embeddings
# -----------------------------------------------------------------------------

def rotate_every_two(x: Tensor) -> Tensor:
    # x: (..., last_dim)
    last_dim = x.shape[-1]
    assert last_dim % 2 == 0
    d = last_dim // 2
    x = x.view(*x.shape[:-1], d, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x_out = torch.stack([-x2, x1], dim=-1)
    return x_out.reshape(*x.shape[:-2], d * 2)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, sinu_pos: Tensor) -> Tuple[Tensor, Tensor]:
    # q, k: (b, h, n, d)
    # sinu_pos: (1, n, d_model)
    # original: rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    b, h, n, d_qk = q.shape
    d_sinu = sinu_pos.shape[-1]
    assert d_sinu == d_qk
    # Reshape sinu_pos: (1, n, d) -> (n, 2, d//2)
    sinu_pos = sinu_pos[0]
    d_half = d_sinu // 2
    sinu_pos = sinu_pos.view(n, 2, d_half)
    sin = sinu_pos[:, 0, :]   # (n, d//2)
    cos = sinu_pos[:, 1, :]   # (n, d//2)

    # repeat 'b n -> b (n j)' with j=2: out shape (b, n * 2)
    def rotary_repeat(tensor, b):
        n, dhalf = tensor.shape
        tensor = tensor.unsqueeze(0).expand(b, n, dhalf)
        tensor = tensor.reshape(b, n * dhalf)
        return tensor

    # Now, cos/sin: (n, d//2) make (b, n, d//2)
    cos = cos.unsqueeze(0).expand(b, n, d_half)
    sin = sin.unsqueeze(0).expand(b, n, d_half)

    # q, k: (b, h, n, d). We split last dim to (d//2, 2) and apply rotary
    def apply_one(x, sin, cos):
        b, h, n, d = x.shape
        dhalf = d // 2
        x_ = x.view(b, h, n, dhalf, 2)
        x1 = x_[..., 0]
        x2 = x_[..., 1]
        sin = sin.unsqueeze(1)  # (b,1,n,d//2)
        cos = cos.unsqueeze(1)
        # result = x * cos + rotate_every_two(x) * sin
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        x_rot = torch.stack([out1, out2], dim=-1)
        return x_rot.reshape(b, h, n, d)
    q_out = apply_one(q, sin, cos)
    k_out = apply_one(k, sin, cos)
    return q_out, k_out


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t).unsqueeze(0)  # (1, n, d)


class Gene2VecPositionalEmbedding(nn.Module):
    """
    Positional embedding initialized with pretrained gene2vec weights.
    
    Refactored to accept weights or path via constructor, avoiding hardcoded paths.
    """
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int, 
        weights_path: Optional[str] = None,
        pretrained_weights: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        if pretrained_weights is not None:
            weights = pretrained_weights
        elif weights_path is not None:
            try:
                weights = np.load(weights_path)
            except FileNotFoundError:
                warnings.warn(f"Gene2Vec weights not found at {weights_path}. Initializing randomly.")
                weights = np.random.rand(max_seq_len - 1, dim)
        else:
            weights = np.random.rand(max_seq_len - 1, dim)

        if weights.shape[1] != dim:
            if weights.shape[1] > dim:
                weights = weights[:, :dim]
            else:
                raise ValueError(f"Gene2Vec dim {weights.shape[1]} < requested dim {dim}")

        padding_row = np.zeros((1, weights.shape[1]))
        weights = np.concatenate((weights, padding_row), axis=0)
        
        weight_tensor = torch.from_numpy(weights).float()
        self.emb = nn.Embedding.from_pretrained(weight_tensor, freeze=False)

    def forward(self, x: Tensor) -> Tensor:
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t).unsqueeze(0)


class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        weights = np.random.rand(max_seq_len - 1, dim)
        padding_row = np.zeros((1, dim))
        weights = np.concatenate((weights, padding_row), axis=0)
        
        weight_tensor = torch.from_numpy(weights).float()
        self.emb = nn.Embedding.from_pretrained(weight_tensor, freeze=False)

    def forward(self, x: Tensor) -> Tensor:
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t).unsqueeze(0)


# =============================================================================
# Performer Model
# =============================================================================

class Performer(nn.Module):
    """
    Performer Transformer Encoder.
    
    Uses efficient linear attention (FAVOR+) to approximate standard self-attention.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        causal: bool = False,
        ff_mult: int = 4,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        reversible: bool = False,
        ff_chunks: int = 1,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        ff_glu: bool = False,
        ff_dropout: float = 0.,
        attn_dropout: float = 0.,
        cross_attend: bool = False,
        no_projection: bool = False,
        auto_check_redraw: bool = True,
        qkv_bias: bool = True
    ):
        super().__init__()
        layers = nn.ModuleList([])
        
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'Local attention heads config length must match depth'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            attn_layer = wrapper_fn(SelfAttention(
                dim, causal=causal, heads=heads, dim_head=dim_head,
                local_heads=local_heads, local_window_size=local_window_size,
                nb_features=nb_features, generalized_attention=generalized_attention,
                kernel_fn=kernel_fn, dropout=attn_dropout,
                no_projection=no_projection, qkv_bias=qkv_bias
            ))
            
            ff_layer = wrapper_fn(Chunk(
                ff_chunks,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu),
                along_dim=1
            ))
            
            layers.append(nn.ModuleList([attn_layer, ff_layer]))

            if not cross_attend:
                continue
                
            cross_attn_layer = wrapper_fn(SelfAttention(
                dim, heads=heads, dim_head=dim_head, nb_features=nb_features,
                generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                dropout=attn_dropout, no_projection=no_projection
            ))
            
            cross_ff_layer = wrapper_fn(Chunk(
                ff_chunks,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu),
                along_dim=1
            ))
            
            layers.append(nn.ModuleList([cross_attn_layer, cross_ff_layer]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)
            fast_attentions = [m for m in self.modules() if isinstance(m, FastAttention)]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x: Tensor, output_attentions: bool = False, **kwargs) -> Tensor:
        if self.auto_check_redraw:
            self.check_redraw_projections()
        return self.net(x, output_attentions=output_attentions, **kwargs)


class PerformerModule(nn.Module):
    """
    Wrapper module combining Performer encoder with final normalization.
    """
    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        causal: bool = False,
        ff_mult: int = 4,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        reversible: bool = False,
        ff_chunks: int = 1,
        ff_glu: bool = False,
        ff_dropout: float = 0.,
        attn_dropout: float = 0.,
        generalized_attention: bool = False,
        kernel_fn: nn.Module = nn.ReLU(),
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        cross_attend: bool = False,
        no_projection: bool = False,
        auto_check_redraw: bool = True,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.performer = Performer(
            dim, depth, heads, dim_head, local_attn_heads, local_window_size,
            causal, ff_mult, nb_features, feature_redraw_interval, reversible,
            ff_chunks, generalized_attention, kernel_fn, use_scalenorm,
            use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend,
            no_projection, auto_check_redraw, qkv_bias
        )
        self.norm = nn.LayerNorm(dim)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x: Tensor, output_attentions: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        b, n, _, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'Sequence length {n} must be <= max sequence length {self.max_seq_len}'

        if output_attentions:
            x, attn_weights = self.performer(x, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            return x, attn_weights
        else:
            x = self.performer(x, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            return x
