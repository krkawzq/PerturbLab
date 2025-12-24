"""scFoundation: Large-Scale Foundation Model for Single-Cell Transcriptomics

This module implements the scFoundation model, a Masked Autoencoder (MAE) with
automatic binning (AutoBin) for single-cell RNA-sequencing data.

Overview
--------
scFoundation is designed to learn universal representations of cellular states
by pretraining on large-scale single-cell transcriptomics datasets. The model
uses a novel auto-discretization embedding strategy that learns to bin continuous
gene expression values into discrete categories, enabling more effective
representation learning.

Model Architecture
------------------
The scFoundation model follows a Masked Autoencoder architecture:

1. **Encoder Branch**:
   - Auto-Discretization Embedding: Learns soft binning of expression values
   - Positional Embedding: Encodes gene position information
   - Transformer Encoder: Processes masked gene expression patterns
   
2. **Decoder Branch**:
   - Linear Projection: Projects encoder outputs to decoder dimension
   - Transformer Decoder: Reconstructs masked gene expressions
   - Prediction Head: Outputs reconstructed expression values

3. **Training Objective**:
   - Mask a random subset of genes in each cell
   - Reconstruct the masked gene expression values
   - Minimize reconstruction loss (typically MSE or similar)

Key Innovations
---------------
1. **Auto-Discretization Embedding**:
   - Unlike traditional discretization (fixed bins), scFoundation learns to
     adaptively bin expression values using a 2-layer MLP
   - Soft binning via softmax enables gradient flow
   - Handles special tokens (mask, padding) explicitly
   
2. **Flexible Transformer Backends**:
   - Supports multiple transformer architectures:
     * Standard PyTorch Transformer (memory efficient, slower)
     * Performer (linear attention, faster for long sequences)
     * Reversible Transformer (gradient checkpointing for memory)
   
3. **Gene-Level Masking**:
   - Masks individual genes rather than tokens
   - Enables learning of gene-gene interactions
   - Supports both random and structured masking strategies

Training Strategy
-----------------
scFoundation is typically pretrained using:
1. Large-scale human scRNA-seq datasets (millions of cells)
2. Random gene masking (e.g., 15-30% masking rate)
3. Reconstruction loss on masked genes
4. Optional: Batch effect correction via adversarial training

After pretraining, the model can be finetuned for downstream tasks:
- Cell type annotation
- Batch integration
- Gene expression imputation
- Perturbation response prediction

Model Inputs
------------
The model expects the following inputs (see scFoundationInput dataclass):

- x: Gene expression values [batch_size, seq_len]
- padding_label: Padding mask [batch_size, seq_len]
- encoder_position_gene_ids: Gene position IDs for encoder [batch_size, seq_len]
- encoder_labels: Masking labels [batch_size, seq_len]
- decoder_data: Decoder input expressions [batch_size, seq_len]
- decoder_position_gene_ids: Gene position IDs for decoder [batch_size, seq_len]
- decoder_data_padding_labels: Decoder padding mask [batch_size, seq_len]
- mask_labels: Target labels for masked positions [batch_size, num_masked]

Model Outputs
-------------
The model returns scFoundationOutput containing:
- predictions: Reconstructed gene expression [batch_size, seq_len]
- attention_weights (optional): Attention maps if requested

Weight Compatibility
--------------------
This implementation preserves the original scFoundation model's state_dict
structure to ensure compatibility with pretrained checkpoints. Key attribute
names are maintained:
- self.token_emb: Auto-discretization embedding
- self.pos_emb: Positional embedding
- self.encoder: Transformer encoder
- self.decoder: Transformer decoder
- self.decoder_embed: Encoder-to-decoder projection
- self.norm: Final layer normalization
- self.to_final: Prediction head

Usage Example
-------------
```python
from perturblab.models.scfoundation import scFoundationConfig, scFoundationInput
from perturblab.models.scfoundation._modeling import scFoundationModel

# Create model
config = scFoundationConfig(
    num_tokens=19264,
    max_seq_len=3001,
    embed_dim=512,
    decoder_embed_dim=512,
    transformer_type="pytorch",
    depth=12,
    heads=8,
)
model = scFoundationModel(config)

# Prepare inputs
inputs = scFoundationInput(
    x=gene_expr,
    padding_label=padding_mask,
    encoder_position_gene_ids=enc_pos_ids,
    encoder_labels=mask_labels,
    decoder_data=decoder_expr,
    decoder_position_gene_ids=dec_pos_ids,
    decoder_data_padding_labels=dec_padding,
    mask_labels=target_labels,
)

# Forward pass
outputs = model(inputs)
predictions = outputs.predictions

```

## Integration with PerturbLab

This implementation integrates scFoundation into the PerturbLab framework:

1. Config-based initialization (scFoundationConfig)
2. Typed input/output schemas (scFoundationInput, scFoundationOutput)
3. Registry-based model management
4. Lazy loading for optional dependencies (Performer, flash-attn)
5. Google-style documentation throughout

## References

[1] Wang et al. (2023). "scFoundation: Large-scale Foundation Model for
Single-cell Transcriptomics." bioRxiv.
https://www.biorxiv.org/content/10.1101/2023.05.29.542705v3

[2] Original implementation:
https://github.com/biomap-research/scFoundation

## Performance Notes

* For sequences < 1000: PyTorch Transformer is recommended (stable, fast)
* For sequences > 1000: Performer is recommended (linear complexity)
* For memory-constrained settings: Reversible Transformer with checkpointing
* Flash Attention can be used with Performer for further speedups (requires flash-attn)

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License (see forks/scFoundation/LICENSE for details)
Migrated to PerturbLab by the PerturbLab team.

==============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from perturblab.models.scfoundation.config import scFoundationConfig
    from perturblab.models.scfoundation.io import scFoundationInput, scFoundationOutput

__all__ = ["scFoundationModel"]

def exists(val):
    """Checks if a value is not None.

    Args:
        val: Value to check.

    Returns:
        bool: True if value is not None, otherwise False.
    """
    return val is not None

class scFoundationModel(nn.Module):
    """scFoundation Masked Autoencoder with Auto-Binning.

    This is the main scFoundation model implementing a Masked Autoencoder (MAE)
    architecture with learnable auto-discretization of gene expression values.

    The model consists of:
      1. Auto-discretization embedding layer (soft binning)
      2. Positional embedding (learnable gene positions)
      3. Transformer encoder (configurable backend)
      4. Decoder projection and transformer decoder
      5. Prediction head (outputs reconstructed expressions)

    Attributes:
        max_seq_len (int): Maximum sequence length (number of genes).
        num_tokens (int): Size of gene vocabulary.
        pad_token_id (Optional[float]): Padding token value.
        mask_token_id (Optional[float]): Mask token value.
        token_emb (AutoDiscretizationEmbedding): Expression embedding layer.
        pos_emb (nn.Embedding): Positional embedding layer.
        encoder (nn.Module): Transformer encoder module.
        decoder (nn.Module): Transformer decoder module.
        decoder_embed (nn.Linear): Encoder-to-decoder projection.
        norm (nn.LayerNorm): Final layer normalization.
        to_final (nn.Linear): Prediction head.

    Note:
        Attribute names (e.g., `token_emb`, `encoder`, `decoder`) are preserved
        from the original implementation to ensure pretrained weight compatibility.
    """

    def __init__(
        self,
        config: "scFoundationConfig",
    ):
        """Initializes the scFoundation model from configuration.

        Args:
            config (scFoundationConfig): Model configuration object containing
                all hyperparameters (embedding dims, transformer settings, etc.).

        Raises:
            ValueError: If transformer_type is not supported.
        """
        super().__init__()

        self.max_seq_len = config.max_seq_len
        self.num_tokens = config.num_tokens
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = config.mask_token_id

        # Build embedding layers
        self._build_embeddings(config)

        # Build transformer encoder and decoder
        self.encoder = None
        self.decoder = None
        self._build_backbone(config)

        # Build decoder components
        self.decoder_embed = nn.Linear(
            config.embed_dim,
            config.decoder_embed_dim,
            bias=True
        )
        self.norm = nn.LayerNorm(config.decoder_embed_dim)
        self.to_final = nn.Linear(config.decoder_embed_dim, 1)

    def _build_embeddings(self, config: "scFoundationConfig"):
        """Builds the auto-discretization and positional embedding layers.

        Args:
            config (scFoundationConfig): Model configuration.
        """
        from .components import AutoDiscretizationEmbedding, RandomPositionalEmbedding

        # Auto-discretization embedding (learns soft binning)
        self.token_emb = AutoDiscretizationEmbedding(
            dim=config.embed_dim,
            bin_num=config.bin_num,
            bin_alpha=config.bin_alpha,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )

        # Positional embedding (max_seq_len + 1 for flexibility)
        self.pos_emb = nn.Embedding(config.max_seq_len + 1, config.embed_dim)

    def _build_backbone(self, config: "scFoundationConfig"):
        """Builds the encoder and decoder transformer modules.

        Args:
            config (scFoundationConfig): Model configuration.

        Raises:
            ValueError: If transformer_type is not supported.
            ImportError: If required dependencies are not installed.
        """
        if config.transformer_type == "pytorch":
            self._build_transformer(config)
        elif config.transformer_type == "performer":
            self._build_performer(config)
        elif config.transformer_type == "reversible":
            self._build_reversible_transformer(config)
        else:
            raise ValueError(
                f"Unsupported transformer_type: {config.transformer_type}. "
                f"Supported types: ['pytorch', 'performer', 'reversible']"
            )

    def _build_transformer(self, config: "scFoundationConfig"):
        """Builds standard PyTorch Transformer encoder and decoder.

        Args:
            config (scFoundationConfig): Model configuration.
        """
        from .components import Transformer

        self.encoder = Transformer(
            max_seq_len=config.max_seq_len,
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.heads,
            ff_mult=config.ff_mult,
            norm_first=config.norm_first,
        )

        self.decoder = Transformer(
            max_seq_len=config.max_seq_len,
            dim=config.decoder_embed_dim,
            depth=config.depth,
            heads=config.heads,
            ff_mult=config.ff_mult,
            norm_first=config.norm_first,
        )

    def _build_performer(self, config: "scFoundationConfig"):
        """Builds Performer (FAVOR+) encoder and decoder.

        Args:
            config (scFoundationConfig): Model configuration.

        Raises:
            ImportError: If performer dependencies are not available.
        """
        try:
            from .components.performer import PerformerModule
        except ImportError as e:
            raise ImportError(
                "Performer backend requires additional dependencies. "
                "Please install them with: pip install local-attention einops"
            ) from e

        self.encoder = PerformerModule(
            max_seq_len=config.max_seq_len,
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.heads,
            dim_head=config.dim_head,
            local_attn_heads=config.local_attn_heads,
            local_window_size=config.local_window_size,
            causal=config.causal,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
            reversible=config.reversible,
            ff_dropout=config.dropout,
            attn_dropout=config.dropout,
            generalized_attention=config.generalized_attention,
            kernel_fn=nn.ReLU() if config.kernel_fn == "relu" else nn.GELU(),
            no_projection=config.no_projection,
        )

        self.decoder = PerformerModule(
            max_seq_len=config.max_seq_len,
            dim=config.decoder_embed_dim,
            depth=config.depth,
            heads=config.heads,
            dim_head=config.dim_head,
            local_attn_heads=config.local_attn_heads,
            local_window_size=config.local_window_size,
            causal=config.causal,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
            reversible=config.reversible,
            ff_dropout=config.dropout,
            attn_dropout=config.dropout,
            generalized_attention=config.generalized_attention,
            kernel_fn=nn.ReLU() if config.kernel_fn == "relu" else nn.GELU(),
            no_projection=config.no_projection,
        )

    def _build_reversible_transformer(self, config: "scFoundationConfig"):
        """Builds Reversible Transformer encoder and decoder.

        Reversible transformers use gradient checkpointing to reduce memory
        consumption at the cost of slightly increased computation time.

        Args:
            config (scFoundationConfig): Model configuration.

        Raises:
            NotImplementedError: Reversible transformer not yet implemented.
        """
        raise NotImplementedError(
            "Reversible transformer backend is not yet implemented. "
            "Please use 'pytorch' or 'performer' instead."
        )

    def forward(
        self,
        inputs: Union["scFoundationInput", dict],
        **kwargs,
    ) -> "scFoundationOutput":
        """Forward pass of scFoundation model.

        Args:
            inputs (Union[scFoundationInput, dict]): Model inputs. Can be either
                a scFoundationInput dataclass or a dictionary with the same fields.
            **kwargs: Additional keyword arguments (for backward compatibility). Supported keys match scFoundationInput attributes.

        Returns:
            scFoundationOutput: Model outputs containing predictions and optional attention weights.

        Raises:
            AssertionError: If sequence length exceeds max_seq_len.
        """
        # Handle both dataclass and dict inputs for backward compatibility
        if isinstance(inputs, dict):
            from ..io import scFoundationInput
            inputs = scFoundationInput(**inputs, **kwargs)
        elif kwargs:
            # Update inputs with kwargs if provided
            from dataclasses import replace
            inputs = replace(inputs, **kwargs)

        # Extract inputs
        x = inputs.x
        padding_label = inputs.padding_label
        encoder_position_gene_ids = inputs.encoder_position_gene_ids
        encoder_labels = inputs.encoder_labels
        decoder_data = inputs.decoder_data
        decoder_position_gene_ids = inputs.decoder_position_gene_ids
        decoder_data_padding_labels = inputs.decoder_data_padding_labels
        mask_gene_name = inputs.mask_gene_name
        output_attentions = inputs.output_attentions

        # Validate input shapes
        b, n = x.shape
        device = x.device
        assert n <= self.max_seq_len, (
            f"Sequence length {n} exceeds maximum allowed {self.max_seq_len}"
        )

        # === Encoder Forward Pass ===

        # 1. Token embedding (auto-discretization)
        # Add unsqueeze to match expected input shape [B, N, 1]
        x = self.token_emb(torch.unsqueeze(x, 2), return_weights=False)

        # Enable gradients for attention visualization if requested
        if output_attentions:
            x.requires_grad_()

        # 2. Add positional embedding
        position_emb = self.pos_emb(encoder_position_gene_ids)
        x = x + position_emb

        # 3. Encode with transformer
        x = self.encoder(x, padding_mask=padding_label)

        # === Decoder Forward Pass ===

        # 1. Embed decoder inputs
        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))

        # 2. Add positional embedding
        position_emb = self.pos_emb(decoder_position_gene_ids)

        # 3. Handle gene name masking (optional, currently not implemented)
        if mask_gene_name:
            raise NotImplementedError(
                "Gene name masking is not yet implemented. "
                "Please set mask_gene_name=False."
            )

        # 4. Inject encoder outputs into decoder inputs at masked positions
        # encoder_labels: True at valid (non-masked) encoder positions
        # We copy these valid encoder outputs to corresponding decoder positions
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        # 5. Add positional embedding to decoder data
        decoder_data = decoder_data + position_emb

        # 6. Project to decoder dimension
        decoder_data = self.decoder_embed(decoder_data)

        # 7. Decode with transformer
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        # === Prediction Head ===

        # 1. Layer normalization
        x = self.norm(x)

        # 2. Final linear layer to predict expression values
        if exists(self.to_final):
            x = self.to_final(x)
            predictions = x.squeeze(2)  # [B, N, 1] -> [B, N]
        else:
            predictions = x  # [B, N, decoder_embed_dim]

        # === Create Output ===

        from ..io import scFoundationOutput

        return scFoundationOutput(
            predictions=predictions,
            attention_weights=None,  # TODO: Extract attention if requested
        )
