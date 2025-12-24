"""CellFM: Cell Foundation Model - Main Wrapper.

This module implements the high-level CellFM wrapper that integrates the core
retention-based network with PerturbLab's unified I/O and Config system.

It handles:
1. Input unpacking from CellFMInput.
2. Training/Inference mode switching.
3. Smart weight loading (compatible with original and wrapped checkpoints).

Copyright (c) 2023 CellFM Authors
Licensed under CC BY-NC-ND 4.0
"""

from __future__ import annotations

import logging
from typing import Union

import torch.nn as nn
from torch import Tensor

from perturblab.models.cellfm.config import CellFMConfig
from perturblab.models.cellfm.io import CellFMInput, CellFMOutput

# Import core model logic
# Adjust this import based on where you placed the FinetuneModel class
# (e.g. from .modeling_cellfm import FinetuneModel)
from .components import FinetuneModel

logger = logging.getLogger(__name__)

__all__ = ["CellFMModel"]


class CellFMModel(nn.Module):
    """CellFM main model wrapper.

    This class wraps the core `FinetuneModel` implementation and provides a
    unified interface using PerturbLab's standard schemas.

    Attributes:
        config (CellFMConfig): Model configuration.
        core_model (FinetuneModel): The underlying retention-based network.
    """

    def __init__(self, config: CellFMConfig, device="cuda"):
        """Initialize CellFM model.

        Args:
            config (CellFMConfig): Model configuration object.
            device (str, optional): Device to run on. Defaults to 'cuda'.
        """
        super().__init__()
        # CRITICAL: All attribute names must match original for weight loading
        self.config = config
        self.device = device
        self.n_gene = config.n_genes

        # CRITICAL: Attribute name must be 'net' (not 'core_model')
        self.net = FinetuneModel(config.n_genes, config)
        self.params = list(self.net.parameters())

        # Optimizer and scaler (initialized later if needed)
        self.optimizer = None
        self.scaler = None

    def forward(self, inputs: Union[CellFMInput, dict]) -> CellFMOutput:
        """Forward pass handling both training and inference flows.

        Args:
            inputs (Union[CellFMInput, dict]): Model inputs containing sparse
                expression data and masks.

        Returns:
            CellFMOutput:
                - In training: Contains `loss`, `cls_token`, and `embeddings`.
                - In inference: Contains full predictions (`gw_pred`, `cw_pred`)
                  and breakdown of losses.
        """
        # 1. Unpack Inputs
        if isinstance(inputs, dict):
            inputs = CellFMInput(**inputs)

        raw_nzdata = inputs.raw_nzdata
        dw_nzdata = inputs.dw_nzdata
        ST_feat = inputs.ST_feat
        nonz_gene = inputs.nonz_gene
        mask_gene = inputs.mask_gene
        zero_idx = inputs.zero_idx
        base_mask = inputs.base_mask

        # 2. Training Flow
        if self.training:
            # FinetuneModel.__call__ returns scalar loss during training
            loss = self.net(raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx)

            # For consistency, we often want the embeddings even during training.
            # We explicitly call encode. Note: This doubles the encoding compute.
            # If performance is critical, consider modifying self.net.forward
            # to return embeddings alongside loss.
            embeddings, _ = self.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            cls_token = embeddings[:, 0]

            return CellFMOutput(loss=loss, cls_token=cls_token, embeddings=embeddings)

        # 3. Inference Flow
        else:
            # FinetuneModel.inference returns detailed outputs
            (expr_emb, gw_pred, cw_pred, loss1, nonz_gene_loss, gene_loss, total_loss) = (
                self.net.inference(
                    raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx, base_mask
                )
            )

            # Re-run encode to get the full embedding sequence including CLS and ST tokens
            # (inference method returns trimmed embeddings)
            full_emb, gene_emb = self.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            cls_token = full_emb[:, 0]

            loss_breakdown = {
                "masked_mse": loss1,
                "non_zero_mse": nonz_gene_loss,
                "gene_mse": gene_loss,
                "total_mse": total_loss,
            }

            return CellFMOutput(
                loss=total_loss,
                loss_breakdown=loss_breakdown,
                cls_token=cls_token,
                embeddings=full_emb,
                gene_embeddings=gene_emb,
                gw_pred=gw_pred,
                cw_pred=cw_pred,
            )

    def encode(self, inputs: Union[CellFMInput, dict]) -> Tensor:
        """Utility method to encode cells into embeddings (CLS tokens).

        Args:
            inputs (Union[CellFMInput, dict]): Model inputs.

        Returns:
            Tensor: Cell embeddings (CLS tokens). Shape: [batch, enc_dims].
        """
        if isinstance(inputs, dict):
            inputs = CellFMInput(**inputs)

        emb, _ = self.net.encode(
            inputs.dw_nzdata, inputs.nonz_gene, inputs.ST_feat, inputs.zero_idx
        )
        return emb[:, 0]  # Return CLS token only

    def to(self, device):
        """Move model to device (matching original API)."""
        self.device = device
        self.net = self.net.to(device)
        return super().to(device)
