"""CellFM Model Implementation.

This module implements the CellFM model for single-cell foundation modeling.
It includes the main FinetuneModel and associated loss functions.

The architecture follows the original MindSpore implementation but adapted
for PyTorch, featuring:
- Retention-based Transformer Encoder
- Dual-path Decoding (Gene-wise and Cell-wise)
- Auto-discretization input encoding

Original source: https://github.com/biomed-AI/CellFM
Adapted for PerturbLab.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Internal imports (Assumed to be in the same package structure)
from .module import FFN, CellwiseDecoder, ValueDecoder, ValueEncoder
from .retention import RetentionLayer

__all__ = ["MaskedMSE", "BCELoss", "FinetuneModel"]


class MaskedMSE(nn.Module):
    """Mean Squared Error loss with masking support.

    Computes MSE only on valid (unmasked) elements.

    Attributes:
        tag (str): Optional tag for identifying the loss instance.
    """

    def __init__(self, tag: Optional[str] = None):
        super().__init__()
        self.tag = tag or ""

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Computes masked MSE loss.

        Args:
            pred (Tensor): Predicted values.
            target (Tensor): Target values.
            mask (Optional[Tensor]): Boolean or binary mask (1 for valid, 0 for ignore).

        Returns:
            Tensor: Scalar loss value.
        """
        pred = pred.float()
        target = target.float()

        # Element-wise squared error
        loss = (pred - target) ** 2

        if mask is not None:
            mask = mask.float()
            # Avoid division by zero
            mask_sum = mask.sum()
            loss = (loss * mask).sum() / (mask_sum + 1e-8)
        else:
            loss = loss.mean()

        return loss


class BCELoss(nn.Module):
    """Binary Cross Entropy loss with masking support.

    Custom implementation that handles numerical stability explicitly.

    Attributes:
        tag (str): Optional tag for identifying the loss instance.
        eps (float): Epsilon for numerical stability in log.
    """

    def __init__(self, tag: str = ""):
        super().__init__()
        self.tag = tag
        self.eps = 1e-12

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Computes masked BCE loss.

        Args:
            pred (Tensor): Predicted probabilities [0, 1].
            target (Tensor): Target labels [0, 1].
            mask (Optional[Tensor]): Boolean or binary mask.

        Returns:
            Tensor: Scalar loss value.
        """
        # Reshape to column vectors
        pred = pred.float().reshape(-1, 1)
        target = target.float().reshape(-1, 1)

        # Compute log probabilities manually for stability
        # [1-p, p]
        pred_cat = torch.cat([1 - pred, pred], dim=-1)
        # [1-y, y]
        target_cat = torch.cat([1 - target, target], dim=-1)

        # log(clip(pred))
        log_pred = torch.log(torch.clamp(pred_cat, min=self.eps, max=1.0))

        # Cross entropy: - sum(target * log(pred))
        logit = -torch.sum(log_pred * target_cat, dim=-1)

        if mask is not None:
            mask = mask.float().reshape(-1)
            mask_sum = mask.sum()
            loss = (logit * mask).sum() / (mask_sum + 1e-8)
        else:
            loss = logit.mean()

        return loss


class FinetuneModel(nn.Module):
    """CellFM Main Model for Fine-tuning tasks.

    This model integrates the encoder, decoders, and embedding layers.
    It strictly maintains the attribute names of the original implementation
    to ensure state_dict compatibility.

    Attributes:
        gene_emb (nn.Parameter): Learnable gene embeddings.
        ST_emb (nn.Parameter): Spatial Transcriptomics embeddings.
        cls_token (nn.Parameter): Classification token.
        zero_emb (nn.Parameter): Embedding for zero values.
        value_enc (ValueEncoder): Encoder for expression values.
        ST_enc (FFN): Encoder for ST features.
        encoder (nn.ModuleList): Stack of RetentionLayers.
        value_dec (ValueDecoder): Gene-wise decoder.
        cellwise_dec (CellwiseDecoder): Cell-wise decoder.
    """

    def __init__(self, n_genes: int, cfg: Any):
        """Initializes the FinetuneModel.

        Args:
            n_genes (int): Number of genes in the vocabulary.
            cfg (Any): Configuration object containing model hyperparameters.
                       Must have attributes: enc_nlayers, add_zero, pad_zero, ecs,
                       ecs_threshold, enc_dims, enc_num_heads, enc_dropout, lora,
                       recompute, dropout.
        """
        super().__init__()
        self.depth = cfg.enc_nlayers
        self.if_cls = False
        self.n_genes = n_genes

        # Config flags
        self.add_zero = cfg.add_zero and not cfg.pad_zero
        self.pad_zero = cfg.pad_zero
        self.ecs = cfg.ecs
        self.ecs_threshold = cfg.ecs_threshold

        # ---------------------------------------------------------------------
        # Embeddings (CRITICAL: Names must match for weight loading)
        # ---------------------------------------------------------------------
        # Padding size calculation: align to 8 for hardware efficiency
        pad_size = (8 - (n_genes + 1) % 8) % 8
        total_vocab_size = n_genes + 1 + pad_size

        self.gene_emb = nn.Parameter(torch.empty(total_vocab_size, cfg.enc_dims))
        self.ST_emb = nn.Parameter(torch.empty(1, 2, cfg.enc_dims))
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.enc_dims))
        self.zero_emb = nn.Parameter(torch.zeros(1, 1, cfg.enc_dims))

        # Initialization
        nn.init.xavier_normal_(self.gene_emb)
        nn.init.xavier_normal_(self.ST_emb)
        nn.init.xavier_normal_(self.cls_token)
        # Zero out padding index 0 (assumed)
        with torch.no_grad():
            self.gene_emb[0, :] = 0

        # ---------------------------------------------------------------------
        # Encoders & Decoders
        # ---------------------------------------------------------------------
        self.value_enc = ValueEncoder(cfg.enc_dims)
        self.ST_enc = FFN(1, cfg.enc_dims)

        # Retention Encoder Stack
        self.encoder = nn.ModuleList(
            [
                RetentionLayer(
                    d_model=cfg.enc_dims,
                    nhead=cfg.enc_num_heads,
                    nlayers=cfg.enc_nlayers,
                    dropout=cfg.enc_dropout * i / cfg.enc_nlayers,  # Layer-wise dropout decay
                    lora=cfg.lora,
                    recompute=cfg.recompute,
                )
                for i in range(cfg.enc_nlayers)
            ]
        )

        self.value_dec = ValueDecoder(cfg.enc_dims, dropout=cfg.dropout, zero=self.add_zero)

        # Compatibility check for older configs
        cellwise_use_bias = getattr(cfg, "cellwise_use_bias", True)
        self.cellwise_dec = CellwiseDecoder(
            cfg.enc_dims,
            cfg.enc_dims,
            dropout=cfg.dropout,
            zero=self.add_zero,
            use_bias=cellwise_use_bias,
        )

        # ---------------------------------------------------------------------
        # Loss Functions
        # ---------------------------------------------------------------------
        self.reconstruct1 = MaskedMSE(tag="_ge")  # Gene Expression loss
        self.reconstruct2 = MaskedMSE(
            tag="_ce"
        )  # Cell Expression loss (unused in forward but kept)
        self.bce_loss1 = BCELoss(tag="_ge")
        self.bce_loss2 = BCELoss(tag="_ce")

    @torch.no_grad()
    def embedding_infer(
        self, expr: Tensor, gene: Tensor, ST_feat: Optional[Tensor], zero_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Inference-only embedding generation.

        Args:
            expr (Tensor): Expression values.
            gene (Tensor): Gene indices.
            ST_feat (Optional[Tensor]): Spatial Transcriptomics features.
            zero_idx (Tensor): Zero value indicators.

        Returns:
            Tuple[Tensor, Tensor]: (expression_embeddings, gene_embeddings)
        """
        b, l = gene.shape

        # Lookup Gene Embeddings
        gene_emb = self.gene_emb[gene]

        # Encode Values
        expr_emb, unmask = self.value_enc(expr)

        # Length scaling factor
        # rsqrt(sum(zeros) - 3) approximation
        len_scale = torch.rsqrt(zero_idx.sum(dim=-1, keepdim=True).float() - 3 + 1e-6)
        len_scale = len_scale.view(b, 1, 1, 1).detach()

        # Handle zero padding logic
        if not self.pad_zero:
            # Apply zero embedding where input is zero but unmasked
            zero_unmask = (1 - zero_idx).unsqueeze(-1) * unmask
            expr_emb = zero_unmask * self.zero_emb + (1 - zero_unmask) * expr_emb

        # Combine Gene + Value
        expr_emb = gene_emb + expr_emb

        if ST_feat is None:
            # Add CLS Token
            cls_token = self.cls_token.expand(b, -1, -1)
            expr_emb = torch.cat([cls_token, expr_emb], dim=1)

            # Update zero index for CLS token (assumed valid/1)
            zero_idx = torch.cat([torch.ones((b, 1), device=zero_idx.device), zero_idx], dim=1)

            if self.pad_zero:
                expr_emb = expr_emb * zero_idx.unsqueeze(-1)

            # Construct Attention Mask
            mask_pos = torch.cat(
                [torch.ones((b, 1, 1), device=unmask.device), unmask], dim=1
            ).unsqueeze(1)

            # Encoder Pass - First Half
            for i in range(self.depth // 2):
                expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)

            # Update mask for second half if padding zeros
            mask_pos = zero_idx.view(zero_idx.size(0), 1, -1, 1) if self.pad_zero else None

            # Encoder Pass - Second Half
            for i in range(self.depth // 2, self.depth):
                expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)

            return expr_emb, gene_emb
        else:
            raise NotImplementedError(
                "Inference with ST_feat is not fully supported in embedding_infer yet."
            )

    @torch.no_grad()
    def decode_infer(
        self, cls_token: Tensor, gene_emb: Tensor, expr_emb: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Decodes embeddings to predictions during inference.

        Args:
            cls_token (Tensor): CLS token embedding.
            gene_emb (Tensor): Gene embeddings.
            expr_emb (Tensor): Encoded expression embeddings.

        Returns:
            Tuple[Tensor, Tensor]: (gene_wise_pred, cell_wise_pred)
        """
        # Dual-path decoding
        if self.add_zero:
            # Ignore zero probability outputs during simple inference
            gw_pred, _ = self.value_dec(expr_emb)  # [B, L, 1]
            cw_pred, _ = self.cellwise_dec(cls_token, gene_emb)  # [B, L, 1]
            return gw_pred, cw_pred

        gw_pred = self.value_dec(expr_emb)  # [B, L, 1]
        cw_pred = self.cellwise_dec(cls_token, gene_emb)  # [B, L, 1]
        return gw_pred, cw_pred

    @torch.no_grad()
    def inference(
        self,
        raw_nzdata: Tensor,
        dw_nzdata: Tensor,
        ST_feat: Optional[Tensor],
        nonz_gene: Tensor,
        mask_gene: Tensor,
        zero_idx: Tensor,
        base_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Full inference pipeline computing losses.

        Args:
            raw_nzdata: Raw non-zero data (target).
            dw_nzdata: Downsampled non-zero data (input).
            ST_feat: Spatial features.
            nonz_gene: Non-zero gene indices.
            mask_gene: Masked gene indicators.
            zero_idx: Zero index indicators.
            base_mask: Optional base mask.

        Returns:
            Tuple containing embeddings, predictions, and breakdown of losses.
        """
        if ST_feat is None:
            emb, gene_emb = self.embedding_infer(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            cls_token, expr_emb = emb[:, 0], emb[:, 1:]
        else:
            emb, gene_emb = self.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
            # Layout with ST: [CLS, ST1, ST2, Expr...]
            cls_token, _, expr_emb = emb[:, 0], emb[:, 1:3], emb[:, 3:]

        gw_pred, cw_pred = self.decode_infer(cls_token, gene_emb, expr_emb)

        # Loss Calculation for evaluation
        loss1 = self.reconstruct1(gw_pred, raw_nzdata, mask_gene)
        total_loss = self.reconstruct1(gw_pred, raw_nzdata, None)
        gene_loss = self.reconstruct1(gw_pred, raw_nzdata, base_mask)
        nonz_gene_loss = self.reconstruct1(gw_pred, raw_nzdata, zero_idx)

        return expr_emb, gw_pred, cw_pred, loss1, nonz_gene_loss, gene_loss, total_loss

    def encode(
        self, expr: Tensor, gene: Tensor, ST_feat: Optional[Tensor], zero_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Main encoding logic used during training.

        Args:
            expr: Expression values.
            gene: Gene indices.
            ST_feat: Spatial features (optional).
            zero_idx: Zero indicators.

        Returns:
            Tuple[Tensor, Tensor]: (full_embeddings, gene_embeddings)
        """
        b, l = gene.shape
        gene_emb = self.gene_emb[gene]
        expr_emb, unmask = self.value_enc(expr)

        # Length scale factor
        len_scale = torch.rsqrt(zero_idx.sum(dim=-1, keepdim=True).float() - 3 + 1e-6)
        len_scale = len_scale.view(b, 1, 1, 1).detach()

        # Handle zeros
        if not self.pad_zero:
            zero_unmask = (1 - zero_idx).unsqueeze(-1) * unmask
            expr_emb = zero_unmask * self.zero_emb + (1 - zero_unmask) * expr_emb

        expr_emb = gene_emb + expr_emb

        # Handle Spatial Transcriptomics Features
        if self.training and ST_feat is not None:
            # Train mode: encode features
            st_emb = self.ST_enc(ST_feat.reshape(b, -1, 1))
            st_emb = self.ST_emb + st_emb
        else:
            # Eval/No ST: use learnable parameter
            st_emb = self.ST_emb.expand(b, -1, -1)

        cls_token = self.cls_token.expand(b, -1, -1)

        # Concatenate: [CLS, ST, Expr]
        # ST_emb is [B, 2, D], so we have 3 special tokens total
        expr_emb = torch.cat([cls_token, st_emb, expr_emb], dim=1)

        # Update zero index to include 3 special tokens
        zero_idx = torch.cat([torch.ones((b, 3), device=zero_idx.device), zero_idx], dim=1)

        if self.pad_zero:
            expr_emb = expr_emb * zero_idx.unsqueeze(-1)

        # First half of encoder layers
        mask_pos = torch.cat(
            [torch.ones((b, 3, 1), device=unmask.device), unmask], dim=1
        ).unsqueeze(1)
        for i in range(self.depth // 2):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)

        # Second half of encoder layers (potentially different mask)
        mask_pos = zero_idx.view(zero_idx.size(0), 1, -1, 1) if self.pad_zero else None
        for i in range(self.depth // 2, self.depth):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)

        # Nan handling (simple mean imputation for stability)
        if torch.isnan(expr_emb).any():
            expr_emb = torch.where(torch.isnan(expr_emb), torch.nanmean(expr_emb, dim=0), expr_emb)

        return expr_emb, gene_emb

    def forward(
        self,
        raw_nzdata: Tensor,
        dw_nzdata: Tensor,
        ST_feat: Optional[Tensor],
        nonz_gene: Tensor,
        mask_gene: Tensor,
        zero_idx: Tensor,
        *args,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass for training.

        Args:
            raw_nzdata: Ground truth non-zero data.
            dw_nzdata: Input downsampled/masked data.
            ST_feat: Spatial features.
            nonz_gene: Indices of non-zero genes.
            mask_gene: Mask indicating which genes to predict.
            zero_idx: Zero indicators.

        Returns:
            Tensor: Total loss (if training).
            Tuple[Tensor, Tensor]: (loss1, loss2) (if eval).
        """
        emb, gene_emb = self.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)

        # Split embeddings: CLS (idx 0), ST (idx 1-2), Expr (idx 3+)
        cls_token, st_emb, expr_emb = emb[:, 0], emb[:, 1:3], emb[:, 3:]

        # Decoding
        if self.add_zero:
            gw_pred, z_prob1 = self.value_dec(expr_emb)
            cw_pred, z_prob2 = self.cellwise_dec(cls_token, gene_emb)
        else:
            gw_pred = self.value_dec(expr_emb)
            cw_pred = self.cellwise_dec(cls_token, gene_emb)
            z_prob1 = z_prob2 = None

        # Reconstruction Losses
        mask = mask_gene
        loss1 = self.reconstruct1(gw_pred, raw_nzdata, mask)
        loss2 = self.reconstruct2(cw_pred, raw_nzdata, mask)
        loss = loss1 + loss2

        # Training-specific losses
        if self.training:
            # Zero-inflation prediction loss
            if self.add_zero:
                nonz_pos = zero_idx
                loss3 = self.bce_loss1(z_prob1, nonz_pos, mask_gene)
                loss4 = self.bce_loss2(z_prob2, nonz_pos, mask_gene)
                loss = loss + loss3 + loss4

            # Elastic Cell Similarity (ECS) Loss
            if self.ecs:
                # Normalize cell-wise predictions
                cell_emb_normed = F.normalize(cw_pred, p=2, dim=1)

                # Compute cosine similarity matrix [B, B]
                # Note: This computes similarity between cells in the batch
                cos_sim = torch.matmul(cell_emb_normed, cell_emb_normed.transpose(0, 1))

                # Mask out diagonal (self-similarity)
                eye_mask = torch.eye(cos_sim.size(0), device=cos_sim.device, dtype=cos_sim.dtype)
                cos_sim = cos_sim * (1 - eye_mask)

                # ReLU to only penalize positive similarity? (Check original logic)
                # Original: F.relu(cos_sim) implies we only care about positive correlations
                cos_sim = F.relu(cos_sim)

                # Loss forces similarity towards threshold (or minimizes variance from it)
                loss += torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

            return loss

        return loss1, loss2
