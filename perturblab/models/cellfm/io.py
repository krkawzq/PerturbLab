"""CellFM Input/Output Schemas.

This module defines the input and output data structures for CellFM models.

Copyright (c) 2023 CellFM Authors
Licensed under CC BY-NC-ND 4.0
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from perturblab.core.model_io import ModelIO


@dataclass(kw_only=True)
class CellFMInput(ModelIO):
    """Input schema for CellFM model forward pass.

    CellFM uses a sparse representation where only nonzero gene features are processed.

    Attributes:
        raw_nzdata (torch.Tensor):
            Raw (unnormalized) nonzero gene expression values.
            Shape: [batch_size, max_nonzero_genes].

        dw_nzdata (torch.Tensor):
            Downsampled/normalized nonzero gene expression values.
            Shape: [batch_size, max_nonzero_genes].

        nonz_gene (torch.Tensor):
            Indices of nonzero genes.
            Shape: [batch_size, max_nonzero_genes].

        mask_gene (torch.Tensor):
            Mask indicating positions used for pretraining/loss calculation.
            Shape: [batch_size, max_nonzero_genes].

        zero_idx (torch.Tensor):
            Mask indicating zero/padding positions.
            Shape: [batch_size, max_nonzero_genes + offset].

        ST_feat (Optional[torch.Tensor]):
            Optional statistical features (e.g., total counts).
            Shape: [batch_size, num_stat_features].

        base_mask (Optional[torch.Tensor]):
            Optional base mask for task-specific loss calculation.
    
    Note:
        All fields must be specified as keyword arguments due to dataclass inheritance constraints.
        Example: CellFMInput(raw_nzdata=..., dw_nzdata=..., nonz_gene=..., mask_gene=..., zero_idx=...)
    """

    raw_nzdata: torch.Tensor
    dw_nzdata: torch.Tensor
    nonz_gene: torch.Tensor
    mask_gene: torch.Tensor
    zero_idx: torch.Tensor
    ST_feat: Optional[torch.Tensor] = None
    base_mask: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class CellFMOutput(ModelIO):
    """Output schema for CellFM model forward pass.

    Attributes:
        cls_token (torch.Tensor):
            Cell-level embedding (CLS token).
            Shape: [batch_size, enc_dims].

        loss (Optional[torch.Tensor]):
            Total reconstruction loss (scalar). Present during training.

        loss_breakdown (Optional[Dict[str, torch.Tensor]]):
            Dictionary with individual loss components.

        embeddings (Optional[torch.Tensor]):
            Full sequence embeddings (e.g., [CLS, ST, Expr...]).
            Shape: [batch_size, seq_len, enc_dims].

        gw_pred (Optional[torch.Tensor]):
            Gene-wise predictions.
            Shape: [batch_size, seq_len].

        cw_pred (Optional[torch.Tensor]):
            Cell-wise predictions.
            Shape: [batch_size, seq_len].
    Note:
        All fields must be specified as keyword arguments due to dataclass inheritance constraints.
        Example: CellFMOutput(field1=..., field2=...)
    """

    cls_token: torch.Tensor
    loss: Optional[torch.Tensor] = None
    loss_breakdown: Optional[Dict[str, torch.Tensor]] = None
    embeddings: Optional[torch.Tensor] = None
    gw_pred: Optional[torch.Tensor] = None
    cw_pred: Optional[torch.Tensor] = None
