"""scGPT: single-cell Generative Pretrained Transformer - Unified Model Architecture

This module contains all scGPT model variants unified under an inheritance-based architecture.
All models preserve exact attribute names from the original implementation to ensure compatibility
with pre-trained weights.

Model Hierarchy:
    scGPTBaseModel (abstract)
      ├── scGPTModel              (standard single-modality)
      ├── scGPTMultiOmicModel     (multi-omics extension)
      └── scGPTPerturbationModel  (specialized for generation/perturbation)

Method Overview:
================

scGPT is a foundation model for single-cell multi-omics that uses transformer-based
masked gene expression modeling (analogous to BERT's masked language modeling) to learn
universal representations of cells and genes. The model can be pre-trained on large
unlabeled datasets and fine-tuned for various downstream tasks.

Core Capabilities:
    1. Cell Embedding Generation
       - Universal cell representations for clustering, visualization, integration
       - Multiple embedding extraction strategies (CLS token, average pooling, weighted pooling)

    2. Batch Integration
       - Domain-Specific Batch Normalization (DSBN) for multi-dataset training
       - Domain Adversarial Training (DAB) for learning batch-invariant features
       - Explicit batch label encoding

    3. Masked Value Prediction
       - Pretraining objective: predict masked gene expression values
       - MVC (Masked Value prediction for Cell embeddings) decoder
       - Optional explicit zero-inflation modeling

    4. Cell Type Annotation
       - Classification decoder for supervised cell type prediction
       - Transfer learning from pre-trained representations

    5. Multi-omics Integration (scGPTMultiOmicModel)
       - Joint modeling of RNA, ATAC, protein, etc.
       - Modality-specific encoders and embeddings

    6. Gene Expression Generation (scGPTPerturbationModel)
       - Conditional generation of gene expression profiles
       - Perturbation-aware generation

Architecture Components:
========================

1. Input Encoding
-----------------
Gene expressions are tokenized and encoded through three parallel pathways:

a. Gene Encoder (GeneEncoder)
   - Vocabulary-based embedding layer for gene IDs
   - Layer normalization
   - Maps gene indices → d_model dimensions

b. Value Encoder (Multiple styles)
   Continuous Style (ContinuousValueEncoder):
     • Two-layer MLP: [1 → d_model → d_model]
     • ReLU activation + LayerNorm
     • Dropout regularization
     • Max value clipping (default: 512)

   Category Style (CategoryValueEncoder):
     • Binned expression levels
     • Embedding lookup + LayerNorm
     • Requires pre-binning of expression values

   Scaling Style:
     • Identity function (raw scaled values)
     • Expression values directly multiply gene embeddings

c. Batch/Modality Encoder (BatchLabelEncoder)
   - Embedding layer for batch or modality labels
   - Layer normalization
   - Used for batch correction or multi-omics integration

2. Positional Encoding
----------------------
Standard sinusoidal positional encoding (PositionalEncoding):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Added to combined gene + value embeddings before transformer.

[... Detailed component documentation preserved from original header ...]

Copyright:
==========
Copyright (c) 2023 Bo Wang Lab
Licensed under the MIT License

Adapted for PerturbLab with:
  - Config-based initialization (replaces args dict)
  - Inheritance-based architecture (reduces code duplication)
  - Unified Input/Output Schema (scGPTInput/scGPTOutput)
  - Comprehensive documentation (Google-style docstrings)
  - Weight compatibility preserved (exact attribute names)

Last Updated: 2024-12-24
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import trange

from perturblab.models.scgpt.config import scGPTConfig
from perturblab.models.scgpt.io import scGPTInput, scGPTOutput
from .components import (
    GeneEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    BatchLabelEncoder,
    Similarity,
    ExprDecoder,
    MVCDecoder,
    ClsDecoder,
    AdversarialDiscriminator,
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
    DomainSpecificBatchNorm1d,
)


__all__ = [
    "scGPTBaseModel",
    "scGPTModel",
    "scGPTMultiOmicModel",
    "scGPTPerturbationModel",
]


# ============================================================================
# scGPTBaseModel - Abstract Base Class
# ============================================================================


class scGPTBaseModel(nn.Module):
    """Abstract base class for all scGPT model variants.

    This class contains shared initialization and encoding logic that is
    common across all scGPT models. Subclasses should override specific
    methods to implement variant-specific behavior.

    CRITICAL: All attribute names (self.encoder, self.decoder, etc.) are
    preserved exactly as in the original implementation to ensure weight
    compatibility. Do NOT rename these attributes.

    Args:
        config (scGPTConfig): Model configuration object.
        vocab (dict): Gene vocabulary mapping.
        pad_token (str, optional): Padding token string. Defaults to "<pad>".
        **kwargs: Additional arguments for subclass-specific initialization.
    """

    def __init__(self, config: scGPTConfig, vocab: Any, pad_token: str = "<pad>", **kwargs):
        super().__init__()

        # Store config
        self.config = config

        # Extract common parameters
        self.model_type = "Transformer"
        self.d_model = config.d_model
        self.do_dab = config.do_dab
        self.ecs_threshold = config.ecs_threshold
        self.use_batch_labels = config.use_batch_labels
        self.domain_spec_batchnorm = config.domain_spec_batchnorm
        self.input_emb_style = config.input_emb_style
        self.cell_emb_style = config.cell_emb_style
        self.explicit_zero_prob = config.explicit_zero_prob
        self.norm_scheme = "pre" if config.pre_norm else "post"

        # Validate parameters
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {config.input_emb_style}"
            )
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {config.cell_emb_style}")

        # Build components (can be customized by subclasses)
        self._build_encoders(config, vocab, pad_token, **kwargs)
        self._build_transformer(config, **kwargs)
        self._build_decoders(config, **kwargs)
        self._build_additional_components(config, **kwargs)

        # Initialize weights
        self.init_weights()

    def _build_encoders(self, config: scGPTConfig, vocab: Any, pad_token: str, **kwargs):
        """Build encoder components.

        This method can be overridden by subclasses to customize encoder building.
        """
        # Gene encoder (CRITICAL: attribute name must be 'encoder')
        self.encoder = GeneEncoder(config.ntoken, config.d_model, padding_idx=vocab[pad_token])

        # Value encoder (CRITICAL: attribute name must be 'value_encoder')
        if config.input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(config.d_model, config.dropout)
        elif config.input_emb_style == "category":
            assert config.n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                config.n_input_bins, config.d_model, padding_idx=config.pad_value
            )
        else:
            self.value_encoder = nn.Identity()

        # Batch encoder (CRITICAL: attribute name must be 'batch_encoder')
        if config.use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(config.num_batch_labels, config.d_model)

    def _build_transformer(self, config: scGPTConfig, **kwargs):
        """Build transformer encoder.

        This method can be overridden by subclasses to customize transformer building.
        """
        # Batch normalization
        if config.domain_spec_batchnorm is True or config.domain_spec_batchnorm == "dsbn":
            use_affine = True if config.domain_spec_batchnorm == "do_affine" else False
            self.dsbn = DomainSpecificBatchNorm1d(
                config.d_model, config.num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        elif config.domain_spec_batchnorm == "batchnorm":
            self.bn = nn.BatchNorm1d(config.d_model, eps=6.1e-5)

        # Transformer encoder (CRITICAL: attribute name must be 'transformer_encoder')
        if config.use_fast_transformer:
            if config.fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    config.d_model, config.nhead, config.d_hid, config.nlayers, config.dropout
                )
            elif config.fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    config.d_model,
                    config.nhead,
                    config.d_hid,
                    config.dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                config.d_model, config.nhead, config.d_hid, config.dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)

    def _build_decoders(self, config: scGPTConfig, **kwargs):
        """Build decoder components.

        This method can be overridden by subclasses to customize decoder building.
        """
        # Expression decoder (CRITICAL: attribute name must be 'decoder')
        self.decoder = ExprDecoder(
            config.d_model,
            explicit_zero_prob=config.explicit_zero_prob,
            use_batch_labels=config.use_batch_labels,
        )

        # Classification decoder (CRITICAL: attribute name must be 'cls_decoder')
        self.cls_decoder = ClsDecoder(config.d_model, config.n_cls, nlayers=config.nlayers_cls)

        # MVC decoder (CRITICAL: attribute name must be 'mvc_decoder')
        if config.do_mvc:
            self.mvc_decoder = MVCDecoder(
                config.d_model,
                arch_style=config.mvc_decoder_style,
                explicit_zero_prob=config.explicit_zero_prob,
                use_batch_labels=config.use_batch_labels,
            )

        # Adversarial discriminator (CRITICAL: attribute name must be 'grad_reverse_discriminator')
        if config.do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                config.d_model,
                n_cls=config.num_batch_labels,
                reverse_grad=True,
            )

    def _build_additional_components(self, config: scGPTConfig, **kwargs):
        """Hook for subclass-specific components.

        Override this method to add model-specific components.
        """
        # Similarity module
        self.sim = Similarity(temp=0.5)
        self.creterion_cce = nn.CrossEntropyLoss()

    def init_weights(self) -> None:
        """Initialize model weights."""
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Shared encoding logic for all scGPT variants.

        Args:
            src (Tensor): Gene indices, shape (batch, seq_len).
            values (Tensor): Expression values, shape (batch, seq_len).
            src_key_padding_mask (Tensor): Padding mask, shape (batch, seq_len).
            batch_labels (Tensor, optional): Batch labels, shape (batch,).

        Returns:
            Tensor: Transformer output, shape (batch, seq_len, d_model).
        """
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(0, 2, 1)
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        return output

    def _get_cell_emb_from_layer(self, layer_output: Tensor, weights: Tensor = None) -> Tensor:
        """Extract cell embedding from transformer output.

        Args:
            layer_output (Tensor): Transformer output, shape (batch, seq_len, embsize).
            weights (Tensor, optional): Weights for weighted pooling, shape (batch, seq_len).

        Returns:
            Tensor: Cell embedding, shape (batch, embsize).
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)
        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        """Validate batch labels."""
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )

    def forward(self, inputs: scGPTInput) -> scGPTOutput:
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward() method")


# ============================================================================
# scGPTModel - Standard scGPT
# ============================================================================


class scGPTModel(scGPTBaseModel):
    """Standard scGPT Transformer model for masked gene expression modeling.

    This is the main scGPT model that supports:
    - Masked Value Prediction (MVC)
    - Elastic Cell Similarity (ECS)
    - Contrastive Cell Embedding (CCE)
    - Domain Adversarial Training (DAB)

    Args:
        config (scGPTConfig): Model configuration object.
        vocab (dict): Gene vocabulary mapping gene names to indices.
        pad_token (str, optional): Padding token string in vocabulary. Defaults to "<pad>".
    """

    def __init__(
        self,
        config: scGPTConfig,
        vocab: Any,
        pad_token: str = "<pad>",
    ):
        super().__init__(config, vocab, pad_token=pad_token)

    def forward(self, inputs: scGPTInput) -> scGPTOutput:
        """Unified forward pass.

        Args:
            inputs (scGPTInput): Container with data tensors and task control flags.

        Returns:
            scGPTOutput: Container with all computed outputs.
        """
        # 1. Encode
        transformer_output = self._encode(
            inputs.src, inputs.values, inputs.src_key_padding_mask, inputs.batch_labels
        )

        # 2. Prepare Decoder Input (Optional Batch Concat)
        decoder_input = transformer_output
        batch_emb = None
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(inputs.batch_labels)
            decoder_input = torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            )

        # 3. Masked Language Modeling (MLM)
        mlm_raw = self.decoder(decoder_input)

        # Handle zero-inflated logic
        mlm_pred = mlm_raw["pred"]
        mlm_zeros = mlm_raw["zero_probs"] if self.explicit_zero_prob else None

        if self.explicit_zero_prob and inputs.do_sample:
            bernoulli = Bernoulli(probs=mlm_zeros)
            mlm_pred = bernoulli.sample() * mlm_pred

        # 4. Cell Embedding Extraction
        cell_emb = self._get_cell_emb_from_layer(transformer_output, inputs.values)

        # 5. Initialize Output Container
        output = scGPTOutput(mlm_output=mlm_pred, mlm_zero_probs=mlm_zeros, cell_emb=cell_emb)

        # 6. Task Specific Branches (Controlled by Input Flags)

        # --- Classification (CLS) ---
        if inputs.CLS:
            output.cls_output = self.cls_decoder(cell_emb)

        # --- Contrastive Cell Embedding (CCE) ---
        if inputs.CCE:
            # Need a second view for contrastive loss
            transformer_output2 = self._encode(
                inputs.src, inputs.values, inputs.src_key_padding_mask, inputs.batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather from other GPUs if distributed
            cell1 = cell_emb
            if dist.is_initialized() and self.training:
                cls1_list = [torch.zeros_like(cell1) for _ in range(dist.get_world_size())]
                cls2_list = [torch.zeros_like(cell2) for _ in range(dist.get_world_size())]
                dist.all_gather(cls1_list, cell1.contiguous())
                dist.all_gather(cls2_list, cell2.contiguous())
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2
                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)

            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output.loss_cce = self.creterion_cce(cos_sim, labels)

        # --- Masked Value Prediction (MVC) ---
        if inputs.MVC:
            mvc_input = cell_emb
            if self.use_batch_labels:
                mvc_input = torch.cat([cell_emb, batch_emb], dim=1)

            mvc_raw = self.mvc_decoder(mvc_input, self.cur_gene_token_embs)
            mvc_pred = mvc_raw["pred"]
            mvc_zeros = mvc_raw["zero_probs"] if self.explicit_zero_prob else None

            if self.explicit_zero_prob and inputs.do_sample:
                bernoulli = Bernoulli(probs=mvc_zeros)
                mvc_pred = bernoulli.sample() * mvc_pred

            output.mvc_output = mvc_pred
            output.mvc_zero_probs = mvc_zeros

        # --- Elastic Cell Similarity (ECS) ---
        if inputs.ECS:
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            cos_sim = F.relu(cos_sim)
            output.loss_ecs = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        # --- Domain Adversarial Batch (DAB) ---
        if self.do_dab:
            output.dab_output = self.grad_reverse_discriminator(cell_emb)

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """Encode data in batches (for large datasets).

        Args:
            src (Tensor): Gene indices, shape (N, seq_len).
            values (Tensor): Expression values, shape (N, seq_len).
            src_key_padding_mask (Tensor): Padding mask, shape (N, seq_len).
            batch_size (int): Batch size for encoding.
            batch_labels (Tensor, optional): Batch labels, shape (N,).
            output_to_cpu (bool, optional): Move output to CPU. Defaults to True.
            time_step (int, optional): Specific time step to return. If None, returns all.
            return_np (bool, optional): Return numpy array. Defaults to False.

        Returns:
            Tensor or ndarray: Encoded output, shape (N, seq_len, d_model) or (N, d_model).
        """
        N = src.size(0)
        device = next(self.parameters()).device

        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (N, self.d_model) if time_step is not None else (N, src.size(1), self.d_model)
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device) if batch_labels is not None else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs


# ============================================================================
# scGPTMultiOmicModel - Multi-omics Extension
# ============================================================================


class scGPTMultiOmicModel(scGPTBaseModel):
    """Multi-omics extension of scGPT Transformer model.

    Extends the base scGPT model to support multiple data modalities
    (e.g., RNA, ATAC, protein) simultaneously.

    Args:
        config (scGPTConfig): Model configuration object. Must have use_mod=True.
        vocab (dict): Gene vocabulary mapping.
        vocab_mod (dict, optional): Modality vocabulary mapping. Required if use_mod=True.
        pad_token (str, optional): Padding token string. Defaults to "<pad>".
    """

    def __init__(
        self,
        config: scGPTConfig,
        vocab: Any,
        vocab_mod: Optional[Any] = None,
        pad_token: str = "<pad>",
    ):
        # Validate multi-omics parameters
        if config.use_mod:
            if vocab_mod is None:
                raise ValueError("vocab_mod is required when use_mod=True")
            if config.ntokens_mod is None:
                raise ValueError("ntokens_mod is required when use_mod=True")

        # Store for _build_additional_components
        self._vocab_mod = vocab_mod
        self._pad_token = pad_token

        super().__init__(config, vocab, pad_token=pad_token)

    def _build_additional_components(self, config: scGPTConfig, **kwargs):
        """Add modality encoder for multi-omics."""
        super()._build_additional_components(config, **kwargs)

        # Modality encoder (CRITICAL: attribute name must be 'mod_encoder')
        if config.use_mod:
            self.mod_encoder = BatchLabelEncoder(
                config.ntokens_mod, config.d_model, padding_idx=self._vocab_mod[self._pad_token]
            )
        self.use_mod = config.use_mod

    def _build_decoders(self, config: scGPTConfig, **kwargs):
        """Build decoders with multi-omics support."""
        # Use multi-omics aware decoders (adjust d_in dimension)
        use_mod = config.use_mod

        # Expression decoder (CRITICAL: attribute name must be 'decoder')
        # Multi-omics version adjusts input dimension
        use_batch_or_mod = config.use_batch_labels or use_mod

        # We need special decoder that handles use_mod
        # For now, create standard one and rely on concatenation in forward
        self.decoder = ExprDecoder(
            config.d_model,
            explicit_zero_prob=config.explicit_zero_prob,
            use_batch_labels=use_batch_or_mod,  # This adjusts d_in
        )

        self.cls_decoder = ClsDecoder(config.d_model, config.n_cls, nlayers=config.nlayers_cls)

        if config.do_mvc:
            self.mvc_decoder = MVCDecoder(
                config.d_model,
                arch_style=config.mvc_decoder_style,
                explicit_zero_prob=config.explicit_zero_prob,
                use_batch_labels=use_batch_or_mod,
            )

        if config.do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                config.d_model,
                n_cls=config.num_batch_labels,
                reverse_grad=True,
            )

    def forward(self, inputs: scGPTInput) -> scGPTOutput:
        """Forward pass with multi-omics support.

        Args:
            inputs (scGPTInput): Container with data tensors and task control flags.

        Returns:
            scGPTOutput: Output dictionary with predictions and losses.
        """
        # 1. Encode
        transformer_output = self._encode(
            inputs.src, inputs.values, inputs.src_key_padding_mask, inputs.batch_labels
        )

        batch_emb = None
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(inputs.batch_labels)

        mod_emb = None
        if self.use_mod:
            if inputs.mod_types is None:
                raise ValueError("mod_types required for MultiOmic model")
            mod_emb = self.mod_encoder(inputs.mod_types)

        output = scGPTOutput()

        # 2. Prepare concatenation for Decoder
        cat_suffix = None
        if self.use_batch_labels and self.use_mod:
            cat_suffix = batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1) + mod_emb
        elif self.use_batch_labels and not self.use_mod:
            cat_suffix = batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)
        elif self.use_mod and not self.use_batch_labels:
            cat_suffix = mod_emb

        decoder_input = transformer_output
        if cat_suffix is not None:
            decoder_input = torch.cat([transformer_output, cat_suffix], dim=2)

        # 3. MLM
        mlm_raw = self.decoder(decoder_input)
        mlm_pred = mlm_raw["pred"]
        mlm_zeros = mlm_raw["zero_probs"] if self.explicit_zero_prob else None

        if self.explicit_zero_prob and inputs.do_sample:
            bernoulli = Bernoulli(probs=mlm_zeros)
            mlm_pred = bernoulli.sample() * mlm_pred

        output.mlm_output = mlm_pred
        output.mlm_zero_probs = mlm_zeros

        cell_emb = self._get_cell_emb_from_layer(transformer_output, inputs.values)
        output.cell_emb = cell_emb

        # 4. Task Branches
        if inputs.CLS:
            output.cls_output = self.cls_decoder(cell_emb)

        if inputs.CCE:
            # Reusing standard CCE logic from base model
            cell1 = cell_emb
            transformer_output2 = self._encode(
                inputs.src, inputs.values, inputs.src_key_padding_mask, inputs.batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            if dist.is_initialized() and self.training:
                cls1_list = [torch.zeros_like(cell1) for _ in range(dist.get_world_size())]
                cls2_list = [torch.zeros_like(cell2) for _ in range(dist.get_world_size())]
                dist.all_gather(cls1_list, cell1.contiguous())
                dist.all_gather(cls2_list, cell2.contiguous())
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2
                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)

            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output.loss_cce = self.creterion_cce(cos_sim, labels)

        if inputs.MVC:
            # Multi-omic concatenation logic for MVC
            cat_1, cat_2 = None, None
            if self.use_batch_labels and self.use_mod:
                cat_1 = batch_emb + self._get_cell_emb_from_layer(mod_emb)
                cat_2 = batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1) + mod_emb
            elif self.use_batch_labels:
                cat_1 = batch_emb
                cat_2 = batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)
            elif self.use_mod:
                cat_1 = self._get_cell_emb_from_layer(mod_emb)
                cat_2 = mod_emb

            mvc_cell_input = cell_emb if cat_1 is None else torch.cat([cell_emb, cat_1], dim=1)
            mvc_gene_input = (
                self.cur_gene_token_embs
                if cat_2 is None
                else torch.cat([self.cur_gene_token_embs, cat_2], dim=2)
            )

            mvc_raw = self.mvc_decoder(mvc_cell_input, mvc_gene_input)
            output.mvc_output = mvc_raw["pred"]
            if self.explicit_zero_prob:
                output.mvc_zero_probs = mvc_raw["zero_probs"]

        if inputs.ECS:
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            cos_sim = F.relu(cos_sim)
            output.loss_ecs = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output.dab_output = self.grad_reverse_discriminator(cell_emb)

        return output


# ============================================================================
# scGPTPerturbationModel - For Generation Tasks
# ============================================================================


class scGPTPerturbationModel(scGPTBaseModel):
    """scGPT Transformer model specialized for generation tasks.

    This variant is optimized for generating gene expression profiles
    and includes perturbation-specific encoders.

    Args:
        config (scGPTConfig): Model configuration object.
        vocab (dict): Gene vocabulary mapping.
        pad_token (str, optional): Padding token string. Defaults to "<pad>".
        pert_pad_id (int, optional): Padding ID for perturbations. Defaults to 2.
    """

    def __init__(
        self,
        config: scGPTConfig,
        vocab: Any,
        pad_token: str = "<pad>",
        pert_pad_id: int = 2,
    ):
        self._pert_pad_id = pert_pad_id
        super().__init__(config, vocab, pad_token=pad_token)

    def _build_additional_components(self, config: scGPTConfig, **kwargs):
        """Add perturbation encoder for generation."""
        super()._build_additional_components(config, **kwargs)

        # Perturbation encoder (CRITICAL: attribute name must be 'pert_encoder')
        self.pert_encoder = nn.Embedding(3, config.d_model, padding_idx=self._pert_pad_id)
        self.pert_pad_id = self._pert_pad_id

    def forward(self, inputs: scGPTInput) -> scGPTOutput:
        """Forward pass for generation.

        Args:
            inputs (scGPTInput): Container with data tensors.

        Returns:
            scGPTOutput: Container containing mlm_output.
        """
        self._check_batch_labels(inputs.batch_labels)

        src_emb = self.encoder(inputs.src)
        self.cur_gene_token_embs = src_emb

        values_emb = self.value_encoder(inputs.values)
        if self.input_emb_style == "scaling":
            values_emb = values_emb.unsqueeze(2)
            total_embs = src_emb * values_emb
        else:
            total_embs = src_emb + values_emb

        # Add perturbation embeddings if provided
        if inputs.pert_flags is not None:
            pert_emb = self.pert_encoder(inputs.pert_flags)
            total_embs = total_embs + pert_emb

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(inputs.batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(0, 2, 1)
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=inputs.src_key_padding_mask
        )

        decoder_input = transformer_output
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(inputs.batch_labels)
            decoder_input = torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            )

        mlm_raw = self.decoder(decoder_input)

        # Generator typically returns just prediction in mlm_output
        return scGPTOutput(mlm_output=mlm_raw["pred"])
