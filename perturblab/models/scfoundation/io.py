"""scFoundation Input/Output Schemas.

This module defines the input and output data structures for scFoundation models,
using dataclasses for type-safe and structured data passing.

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License (see forks/scFoundation/LICENSE for details)
"""

from typing import Optional

import torch

from perturblab.core.model_io import ModelIO

class scFoundationInput(ModelIO):
    """Input schema for scFoundation model forward pass.

    This dataclass encapsulates all required inputs for a scFoundation MAE-Autobin
    model's forward method, including encoder and decoder inputs as well as masking information.

    Attributes:
        x: Input gene expression values of shape [batch_size, seq_len].
        padding_label: Padding mask for encoder (True = padding), shape [batch_size, seq_len].
        encoder_position_gene_ids: Gene position IDs for encoder, shape [batch_size, seq_len].
        encoder_labels: Masking labels for encoder (True = masked), shape [batch_size, seq_len].
        decoder_data: Input gene expression for decoder, shape [batch_size, seq_len].
        decoder_position_gene_ids: Gene position IDs for decoder, shape [batch_size, seq_len].
        decoder_data_padding_labels: Padding mask for decoder, shape [batch_size, seq_len].
        mask_labels: Target labels for masked positions, shape [batch_size, num_masked].
        mask_gene_name: Whether to mask gene names (not implemented). Defaults to False.
        output_attentions: Whether to output attention maps. Defaults to False.

    Example:
        inputs = scFoundationInput(
            x=gene_expr,
            padding_label=padding_mask,
            encoder_position_gene_ids=enc_pos_ids,
            encoder_labels=encoder_mask_labels,
            decoder_data=decoder_expr,
            decoder_position_gene_ids=dec_pos_ids,
            decoder_data_padding_labels=dec_padding,
            mask_labels=target_labels,
        )
        outputs = model(inputs)
    """

    x: torch.Tensor
    padding_label: torch.Tensor
    encoder_position_gene_ids: torch.Tensor
    encoder_labels: torch.Tensor
    decoder_data: torch.Tensor
    decoder_position_gene_ids: torch.Tensor
    decoder_data_padding_labels: torch.Tensor
    mask_labels: torch.Tensor
    mask_gene_name: bool = False
    output_attentions: bool = False


class scFoundationOutput(ModelIO):
    """Output schema for scFoundation model forward pass.

    This dataclass contains the outputs from a scFoundation model,
    including predicted gene expression (reconstructions) and optional attention weights.

    Attributes:
        predictions: Reconstructed gene expression values, 
            of shape [batch_size, seq_len] or [batch_size, seq_len, decoder_dim]
            depending on model configuration.
        attention_weights: Optional attention weights if requested by user.
            The shape depends on the underlying transformer architecture.

    Example:
        outputs = model(inputs)
        reconstructed = outputs.predictions
        loss = criterion(reconstructed[mask], targets[mask])
    """

    predictions: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None

    def to_dict(self):
        """Converts the output to a dictionary for backward compatibility.

        Returns:
            dict: A dictionary containing 'predictions' as well as 'attention_weights' if present.
        """
        result = {"predictions": self.predictions}
        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights
        return result
