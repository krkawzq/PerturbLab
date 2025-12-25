r"""UCE: Universal Cell Embeddings - Unified Model Architecture

This module implements the UCE foundation model for generating universal single-cell
representations. It leverages a Transformer-based architecture to process gene
expression profiles as token sequences, producing robust embeddings applicable
across diverse biological contexts.

Method Overview:
================

UCE is a foundation model designed to learn a universal latent space for single-cell
biology. Unlike gene-centric models (e.g., scGPT) that focus on reconstructing gene
networks, UCE focuses on cell-centric representation learning. It treats a cell's
gene expression profile as a document of gene tokens and uses a Transformer encoder
to compress this information into a single, high-quality vector (via the CLS token).

Core Capabilities:
    1. Universal Cell Embedding
       - Generates 1280-d fixed-dimensional vectors for any cell.
       - Zero-shot integration: Embeddings from different datasets naturally align
         without batch correction if the biological signals are strong.
       - L2-normalized output suitable for cosine similarity searches.

    2. Gene Expression Prediction
       - Can predict the expression value of any gene given a cell embedding.
       - Supports zero-shot prediction for genes not seen in the input.

Architecture Components:
========================

1. Input Encoding Strategy
--------------------------
   - **Tokenization**: Genes are mapped to learnable embeddings (token_dim).
   - **Scaling**: Inputs are scaled by sqrt(d_model) before adding position info.
   - **Positional Encoding**: Uses fixed sinusoidal encodings. Since gene expression
     is technically a set, not a sequence, UCE relies on a canonical ordering or
     random permutation during training, but positional encoding helps the
     Transformer distinguish separate input slots.

2. Transformer Backbone
-----------------------
   - Standard PyTorch ``TransformerEncoder``.
   - Processes the sequence of gene tokens.
   - Uses self-attention to capture gene-gene co-occurrence patterns globally.

3. Decoders
-----------
   a. **Cell Embedding Decoder** (Primary Head)
      - Architecture: MLP (d_model -> 1024 -> output_dim -> output_dim)
      - Processing: Applied to the output of the CLS token (index 0).
      - Normalization: Final output is strictly L2-normalized.

   b. **Binary Decoder** (Prediction Head)
      - Architecture: MLP taking concatenated [Cell_Emb || Gene_Emb].
      - Purpose: Reconstructs the expression value for a specific gene-cell pair.
      - Used for fine-tuning or imputation tasks.

Input/Output Specifications:
============================

Input Format (UCEInput):
------------------------
src (Tensor):                  Gene expression tokens.
                               Shape: (seq_len, batch_size, token_dim)
                               Note: UCE uses sequence-first convention.

mask (Tensor):                 Padding mask.
                               Shape: (batch_size, seq_len)
                               Values: 1 for valid genes, 0 for padding.

Output Format (UCEOutput):
--------------------------
cell_embedding (Tensor):       The universal cell representation.
                               Shape: (batch_size, output_dim)
                               Property: L2-normalized (|x|=1).

gene_embeddings (Tensor):      Contextualized gene representations.
                               Shape: (seq_len, batch_size, output_dim)

Model Workflow:
===============

1. **Encode**:
   - Project gene tokens to model dimension.
   - Scale by $\sqrt{d_{model}}$.
   - Add sinusoidal positional encodings.

2. **Transform**:
   - Pass through N layers of Transformer Encoder.
   - Apply padding mask to ignore empty slots.

3. **Decode & Pool**:
   - Pass encoder output through the MLP decoder.
   - Extract the token at index 0 (CLS token) as the cell representation.

4. **Normalize**:
   - Apply L2 normalization to the CLS vector.
   - Return the final Universal Cell Embedding.

Hyperparameters:
================

Model Architecture:
  token_dim (int):             Input token dimension (default: 5120 from ESM2/vocab).
  d_model (int):               Transformer hidden dimension (default: 1280).
  nhead (int):                 Number of attention heads (default: 20).
  d_hid (int):                 FFN hidden dimension (default: 5120).
  nlayers (int):               Number of transformer layers (default: 33).
  output_dim (int):            Size of final cell embedding (default: 1280).
  max_len (int):               Maximum sequence length (default: 1536).

Training parameters:
  dropout (float):             Dropout probability (default: 0.1).

Comparison with Other Models:
=============================

1. **vs. scGPT**:
   - **Focus**: UCE optimizes for cell-level similarity (integration, clustering).
     scGPT optimizes for gene-level generative tasks (perturbation, GRN inference).
   - **Embedding**: UCE uses a CLS token. scGPT often uses average pooling or
     generative readout.
   - **Normalization**: UCE mandates L2 normalization; scGPT uses LayerNorm.

2. **vs. Geneformer**:
   - **Input**: UCE uses continuous expression values (via protein embeddings or
     learned tokens). Geneformer uses rank-based discrete tokens.
   - **Backbone**: UCE is standard Transformer. Geneformer is BERT-based.

References:
===========

[1] Rosen, Y., Roohani, Y., Agrawal, A., et al. (2023).
    "Universal Cell Embeddings: A Foundation Model for Cell Biology."
    bioRxiv. https://doi.org/10.1101/2023.11.28.568918

[2] Official Implementation:
    https://github.com/snap-stanford/UCE

Copyright:
==========
Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License

Adapted for PerturbLab with:
  - Unified Input/Output Schema (UCEInput/UCEOutput)
  - Config-based initialization
  - Refactored modular components

Last Updated: 2024-12-25
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from perturblab.models.uce.config import UCEConfig
from perturblab.models.uce.io import UCEInput, UCEOutput, UCEPredictInput, UCEPredictOutput

__all__ = [
    "PositionalEncoding",
    "UCEModel",
]


def _make_mlp_block(in_features: int, out_features: int, p_drop: float = 0.1) -> nn.Sequential:
    """Helper to construct a standard Linear-LayerNorm-GELU-Dropout block."""
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs.

    Injects information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        """Initializes the PositionalEncoding module.

        Args:
            d_model: The embedding dimension.
            dropout: The dropout probability.
            max_len: The maximum length of the incoming sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            Tensor of the same shape with positional encoding added.
        """
        # Slice pe to match the sequence length of x
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class UCEModel(nn.Module):
    """The Universal Cell Embeddings (UCE) Transformer model.

    A foundation model that learns universal representations of cells from
    gene expression profiles. It consists of a Transformer encoder to process
    gene tokens and multiple decoder heads for embedding extraction and
    expression prediction.

    Attributes:
        config (UCEConfig): The configuration object defining model hyperparameters.
    """

    def __init__(self, config: UCEConfig):
        """Initializes the UCE model.

        Args:
            config: Model configuration containing dimensions and architectural params.
        """
        super().__init__()
        self.config = config
        self.model_type = "Transformer"

        # ---------------------------------------------------------------------
        # 1. Input Projection
        # Projects raw gene tokens to the transformer model dimension.
        # ---------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(config.token_dim, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model)
        )

        # ---------------------------------------------------------------------
        # 2. Positional Encoding
        # ---------------------------------------------------------------------
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        self.d_model = config.d_model

        # ---------------------------------------------------------------------
        # 3. Transformer Backbone
        # Standard PyTorch TransformerEncoder.
        # ---------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_hid,
            dropout=config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.nlayers)

        # ---------------------------------------------------------------------
        # 4. Cell Embedding Decoder
        # Post-processes the transformer output to generate the final cell embedding.
        # ---------------------------------------------------------------------
        self.decoder = nn.Sequential(
            _make_mlp_block(config.d_model, 1024, config.dropout),
            _make_mlp_block(1024, config.output_dim, config.dropout),
            _make_mlp_block(config.output_dim, config.output_dim, config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # ---------------------------------------------------------------------
        # 5. Binary Decoder (Prediction Head)
        # Used for predicting gene expression values from cell + gene embeddings.
        # Input dim is output_dim (cell) + 1280 (gene, assumed fixed in orig impl).
        # We use d_model here assuming gene_embedding_layer projects to d_model.
        # ---------------------------------------------------------------------
        self.binary_decoder = nn.Sequential(
            _make_mlp_block(config.output_dim + config.d_model, 2048, config.dropout),
            _make_mlp_block(2048, 512, config.dropout),
            _make_mlp_block(512, 128, config.dropout),
            nn.Linear(128, 1),
        )

        # ---------------------------------------------------------------------
        # 6. Auxiliary Gene Projection
        # Used during prediction to project query genes into the shared space.
        # ---------------------------------------------------------------------
        self.gene_embedding_layer = nn.Sequential(
            nn.Linear(config.token_dim, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model)
        )

    def forward(self, inputs: UCEInput) -> UCEOutput:
        """Standard forward pass to generate cell embeddings.

        Args:
            inputs: A UCEInput object containing:
                - src: Gene expression tokens (seq_len, batch, dim).
                - mask: Padding mask (batch, seq_len), 1 for valid, 0 for pad.

        Returns:
            A UCEOutput object containing:
                - cell_embedding: The L2 normalized CLS token.
                - gene_embeddings: Contextualized gene representations.
        """
        # Unpack from Schema
        src = inputs.src
        mask = inputs.mask

        # 1. Input Embedding & Scaling
        # Scale inputs by sqrt(d_model) as per 'Attention is All You Need'
        src = self.encoder(src) * math.sqrt(self.d_model)

        # 2. Add Positional Encoding
        src = self.pos_encoder(src)

        # 3. Transformer Encoding
        # Convert mask: PyTorch expects True for Padding, False for Valid.
        # Input mask is 1 (Valid), 0 (Padding) -> (1-mask) -> 1 (Padding), 0 (Valid)
        padding_mask = (1 - mask).bool()

        transformer_output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)

        # 4. Decoding to Feature Space
        # (seq_len, batch, output_dim)
        gene_embeddings = self.decoder(transformer_output)

        # 5. Extract CLS Token
        # The CLS token is assumed to be at index 0
        cls_token = gene_embeddings[0, :, :]  # (batch, output_dim)

        # 6. Normalization
        # L2 normalize the cell embedding for cosine similarity usage
        cell_embedding = F.normalize(cls_token, p=2, dim=1)

        return UCEOutput(cell_embedding=cell_embedding, gene_embeddings=gene_embeddings)

    def predict(self, inputs: UCEPredictInput) -> UCEPredictOutput:
        """Predicts gene expression values using the binary decoder head.

        This method projects query genes into the model space, concatenates them
        with the cell embedding, and passes them through the binary decoder.

        Args:
            inputs: A UCEPredictInput object containing:
                - cell_embedding: (batch, output_dim)
                - gene_embeddings: Query gene tokens (batch, seq_len, token_dim)
                                   or (batch, token_dim)

        Returns:
            A UCEPredictOutput object containing the predicted expression values.
        """
        cell_embedding = inputs.cell_embedding
        query_gene_tokens = inputs.gene_embeddings

        # 1. Project gene tokens to model space (d_model)
        gene_repr = self.gene_embedding_layer(query_gene_tokens)

        # 2. Handle Dimension Mismatch for Concatenation
        # If gene_repr is a sequence (batch, seq, dim) and cell_emb is (batch, dim),
        # we need to broadcast cell_emb to match the sequence length.
        if gene_repr.ndim == 3 and cell_embedding.ndim == 2:
            # Expand cell_embedding: (B, D) -> (B, 1, D) -> (B, Seq, D)
            cell_repr_expanded = cell_embedding.unsqueeze(1).expand(-1, gene_repr.size(1), -1)
            # Concatenate along the feature dimension
            combined_repr = torch.cat((cell_repr_expanded, gene_repr), dim=-1)
        else:
            # Standard concatenation (e.g. single gene prediction)
            combined_repr = torch.cat((cell_embedding, gene_repr), dim=-1)

        # 3. Decode to Expression Value
        prediction = self.binary_decoder(combined_repr)

        return UCEPredictOutput(gene_expression_preds=prediction)
