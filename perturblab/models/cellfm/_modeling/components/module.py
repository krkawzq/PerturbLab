"""CellFM Core Components Implementation.

This module contains the fundamental building blocks for the CellFM model,
including encoders, decoders, and feed-forward networks.

These components are reimplemented in PyTorch to faithfully mirror the
original MindSpore architecture, ensuring exact weight compatibility.

Original source: https://github.com/biomed-AI/CellFM
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["FFN", "ValueEncoder", "ValueDecoder", "CellwiseDecoder"]


class FFN(nn.Module):
    """Feed-Forward Network with specialized gating/table lookup mechanism.

    This FFN acts as a learnable lookup table or non-linear transformation,
    used primarily in the ValueEncoder.

    Attributes:
        w1 (nn.Linear): First projection layer.
        act1 (nn.LeakyReLU): Activation function.
        w3 (nn.Linear): Second projection/gating layer.
        softmax (nn.Softmax): Softmax for table lookup weighting.
        table (nn.Linear): The value table.
        a (nn.Parameter): Learnable scaling factor.
    """

    def __init__(self, in_dims: int, emb_dims: int, b: int = 256):
        """Initializes the FFN module.

        Args:
            in_dims (int): Input dimension.
            emb_dims (int): Output embedding dimension.
            b (int, optional): Hidden dimension size. Defaults to 256.
        """
        super().__init__()
        # Note: bias=False is crucial for weight compatibility
        self.w1 = nn.Linear(in_dims, b, bias=False)
        self.act1 = nn.LeakyReLU()
        self.w3 = nn.Linear(b, b, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.table = nn.Linear(b, emb_dims, bias=False)

        # Learnable scalar/vector for residual scaling
        self.a = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, in_dims].

        Returns:
            Tensor: Output tensor of shape [batch, seq_len, emb_dims].
        """
        b, l, d = x.shape
        # Flatten for linear layers: [B*L, D]
        v = x.view(-1, d)

        # Projection and Activation
        v = self.act1(self.w1(v))

        # Gating/Residual mechanism: w3(v) + v * a
        # Broadcasting 'a' (1, 1) across batch
        v = self.w3(v) + v * self.a

        # Table lookup mechanism
        v = self.softmax(v)
        v = self.table(v)

        # Reshape back: [B, L, emb_dims]
        return v.view(b, l, -1)


class ValueEncoder(nn.Module):
    """Encoder for continuous gene expression values.

    Encodes expression values into embeddings and handles masking logic.

    Attributes:
        value_enc (FFN): The core FFN encoder.
        mask_emb (nn.Parameter): Learnable embedding for masked values.
    """

    def __init__(self, emb_dims: int):
        """Initializes ValueEncoder.

        Args:
            emb_dims (int): Embedding dimension.
        """
        super().__init__()
        self.value_enc = FFN(1, emb_dims)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, emb_dims))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes expression values.

        Args:
            x (Tensor): Input tensor.
                If shape is [B, L], assumes no explicit mask input (creates ones).
                If shape is [B, L, 2], assumes last dim is [unmask_flag, expression].

        Returns:
            Tuple[Tensor, Tensor]:
                - expr_emb: Encoded embeddings [B, L, emb_dims].
                - unmask: Unmask tensor indicating valid values [B, L, 1].
        """
        if x.dim() == 3:
            # Input contains [mask_flag, expression_value]
            # Chunk along last dimension
            unmask, expr = torch.chunk(x, 2, dim=-1)

            # Encode expression: FFN(expr) * mask
            unmasked = self.value_enc(expr) * unmask

            # Apply learnable mask embedding where unmask == 0
            # mask_emb broadcasts to [B, L, D]
            masked = self.mask_emb * (1 - unmask)

            expr_emb = masked + unmasked
        else:
            # Input is just expression [B, L]
            expr = x.unsqueeze(-1)
            unmask = torch.ones_like(expr)
            expr_emb = self.value_enc(expr)

        return expr_emb, unmask


class ValueDecoder(nn.Module):
    """Gene-wise value decoder.

    Predicts expression values from latent embeddings using an MLP.
    Optionally predicts zero-inflation probabilities.

    Attributes:
        w1 (nn.Linear): First linear layer.
        act (nn.LeakyReLU): Activation.
        w2 (nn.Linear): Second linear layer (output).
        zero_logit (nn.Sequential): Optional branch for zero-probability prediction.
    """

    def __init__(self, emb_dims: int, dropout: float, zero: bool = False):
        """Initializes ValueDecoder.

        Args:
            emb_dims (int): Input embedding dimension.
            dropout (float): Dropout probability (kept for API consistency, though not used in main path).
            zero (bool, optional): Whether to predict zero-probabilities. Defaults to False.
        """
        super().__init__()
        self.zero = zero

        # Main regression path
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=False)
        self.act = nn.LeakyReLU()
        self.w2 = nn.Linear(emb_dims, 1, bias=False)

        # Zero-inflation path
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Decodes embeddings to expression values.

        Args:
            expr_emb (Tensor): Input embeddings [B, L, D].

        Returns:
            Tensor or Tuple:
                - Prediction [B, L].
                - (Optional) Zero probability [B, L] if self.zero is True.
        """
        b, l, d = expr_emb.shape

        # Regression: Linear -> LeakyReLU -> Linear
        # w1(expr_emb): [B, L, D]
        pred = self.w2(self.act(self.w1(expr_emb))).view(b, l)

        if not self.zero:
            return pred

        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob


class CellwiseDecoder(nn.Module):
    """Cell-wise decoder using inner product attention.

    Predicts expression based on interaction between Gene Embeddings (query)
    and Cell Embeddings (key).

    Attributes:
        map (nn.Linear): Projection for gene embeddings.
        zero_logit (nn.Linear): Optional projection for zero-probability prediction.
    """

    def __init__(
        self,
        in_dims: int,
        emb_dims: int | None = None,
        dropout: float = 0.0,
        zero: bool = False,
        use_bias: bool = True,
    ):
        """Initializes CellwiseDecoder.

        Args:
            in_dims (int): Input dimension.
            emb_dims (int, optional): Hidden dimension. Defaults to in_dims.
            dropout (float, optional): Unused, kept for compatibility.
            zero (bool, optional): Whether to predict zero-probabilities.
            use_bias (bool, optional): Whether to use bias in projection. Defaults to True.
        """
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero

        # Project gene embeddings to query space
        self.map = nn.Linear(in_dims, emb_dims, bias=use_bias)

        if zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb: Tensor, gene_emb: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            cell_emb (Tensor): Cell-level embedding [B, D].
            gene_emb (Tensor): Gene-level embeddings [B, L, D] (or broadcastable).

        Returns:
            Tensor or Tuple: Prediction [B, L].
        """
        b = cell_emb.size(0)

        # 1. Prepare Query (Gene)
        # Sigmoid activation on projected gene embeddings
        query = torch.sigmoid(
            self.map(gene_emb)
        )  # [B, L, D] assuming gene_emb broadcasted or expanded

        # 2. Prepare Key (Cell)
        key = cell_emb.view(b, -1, 1)  # [B, D, 1]

        # 3. Inner Product: [B, L, D] @ [B, D, 1] -> [B, L, 1]
        pred = torch.bmm(query, key).squeeze(-1)  # [B, L]

        if not self.zero:
            return pred

        # Zero-inflation path
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).squeeze(-1)

        return pred, zero_prob
