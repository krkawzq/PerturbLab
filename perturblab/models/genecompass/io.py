"""GeneCompass Input/Output Schemas.

This module defines dataclasses describing the input and output schemas
for the GeneCompass model. These schemas enable type safety and standardize
interfaces for model pipelines.
"""

from typing import Optional, Tuple

from torch import Tensor

from perturblab.core.model_io import ModelIO

__all__ = ["GeneCompassInput", "GeneCompassOutput", "MaskedLMOutputBoth"]


class GeneCompassInput(ModelIO):
    """Input schema for the GeneCompass model.

    Attributes:
        input_ids (Tensor):
            Tensor of gene token IDs, with shape [batch_size, sequence_length].
            Use pad_token_id for padding.
        values (Optional[Tensor]):
            Tensor of gene expression values (typically normalized). Shape must match input_ids.
            Used for value regression; optional.
        attention_mask (Optional[Tensor]):
            Tensor with values {0, 1} indicating which tokens are real (1) or padding (0),
            shape [batch_size, sequence_length]. Optional—default attends to all tokens.
        species (Optional[Tensor]):
            Species indicator tensor, shape [batch_size, sequence_length].
            0 = human, 1 = mouse. Optional; default is all human (0).
        token_type_ids (Optional[Tensor]):
            Segment/token type IDs for each token (similar to BERT), [batch_size, sequence_length].
            Optional; default is all zeros.
        position_ids (Optional[Tensor]):
            Position indices for each token. If None, positions are assigned as [0, 1, ...].
        labels (Optional[Tensor]):
            Gene ID labels for computing masked language modeling loss.
            Shape [batch_size, sequence_length]. Use -100 for positions to ignore.
            Optional; only needed during training.
        labels_values (Optional[Tensor]):
            Expression value labels for computing value prediction loss.
            Shape [batch_size, sequence_length]. Only used if use_values=True.
            Optional; only needed during training.
        output_attentions (bool):
            Whether to return attention weights from all encoder layers. Default is False.
        output_hidden_states (bool):
            Whether to return hidden states from all layers. Default is False.

    Example:
        ```python
        import torch
        from perturblab.models.genecompass import GeneCompassInput

        # Inference usage
        inputs = GeneCompassInput(
            input_ids=torch.tensor([[1, 234, 567, 890, 2]]),
            values=torch.tensor([[0.0, 2.3, 1.5, 3.2, 0.0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1]])
        )

        # Training usage with labels
        inputs = GeneCompassInput(
            input_ids=gene_tokens,
            values=expression_values,
            labels=target_gene_ids,
            labels_values=target_expression,
            attention_mask=mask
        )

        # Cross-species batch
        inputs = GeneCompassInput(
            input_ids=gene_tokens,
            values=expression_values,
            species=torch.zeros_like(gene_tokens),  # Human
            output_attentions=True
        )
        ```

    Notes:
        - All input tensors should be on the same device.
        - input_ids values should be in [0, vocab_size).
        - For masked language modeling, set masked positions in input_ids to mask_token_id.
        - labels with value -100 are ignored in loss computation.
    """

    input_ids: Tensor
    values: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    species: Optional[Tensor] = None
    token_type_ids: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    labels_values: Optional[Tensor] = None
    output_attentions: bool = False
    output_hidden_states: bool = False


class GeneCompassOutput(ModelIO):
    """Output schema for the GeneCompass model.

    Attributes:
        logits (Tensor):
            Gene ID logits prior to softmax, with shape [batch_size, sequence_length, vocab_size].
        loss (Optional[Tensor]):
            Combined training loss (MLM and/or value loss), if computed. Scalar.
        value_logits (Optional[Tensor]):
            Predicted expression values for each token, shape [batch_size, sequence_length].
            Present if use_values=True in config.
        hidden_states (Optional[Tensor]):
            Output sequence representations from final encoder layer,
            [batch_size, sequence_length, hidden_size]. Useful for downstream tasks.
        attentions (Optional[Tuple[Tensor, ...]]):
            Tuple of attention maps for each layer (if requested).
            Each: [batch_size, num_heads, sequence_length, sequence_length].

    Example:
        ```python
        outputs = model(inputs)

        # Gene prediction
        gene_probs = torch.softmax(outputs.logits, dim=-1)
        predicted_genes = torch.argmax(gene_probs, dim=-1)

        # Expression value prediction
        if outputs.value_logits is not None:
            predicted_values = outputs.value_logits

        # Extract embeddings
        embeddings = outputs.hidden_states
        ```

    Notes:
        - loss is only computed during training (when labels present).
        - value_logits are raw predictions (no activation, unless handled downstream).
        - hidden_states can be useful for clustering or additional tasks.
    """

    logits: Tensor
    loss: Optional[Tensor] = None
    value_logits: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


class MaskedLMOutputBoth(ModelIO):
    """Internal output for GeneCompass masked language modeling (MLM + value).

    This structure is mainly for compatibility with HuggingFace Transformers' model outputs.
    End users should use `GeneCompassOutput` in most cases.

    Attributes:
        loss (Optional[Tensor]):
            The total training loss (weighted sum), if computed.
        value_loss (Optional[Tensor]):
            Expression value prediction loss (e.g., MSE), if computed.
        id_loss (Optional[Tensor]):
            Gene ID (MLM token) loss (e.g., CrossEntropy), if computed.
        logits (Optional[Tensor]):
            Predicted gene ID logits, [batch, seq_len, vocab_size].
        hidden_states (Optional[Tuple[Tensor, ...]]):
            Tuple of all hidden states, including embeddings, length = num_hidden_layers + 1.
        attentions (Optional[Tuple[Tensor, ...]]):
            Tuple of all attention maps, length = num_hidden_layers.

    Notes:
        This dataclass is for internal/intermediate use (checkpoint loading etc).
        Most application logic returns or expects GeneCompassOutput.
    """

    loss: Optional[Tensor] = None
    value_loss: Optional[Tensor] = None
    id_loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
