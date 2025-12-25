"""scBERT: Single-Cell BERT Model Implementation.

This module implements the scBERT architecture, which adapts the standard BERT
model for single-cell RNA-sequencing data by incorporating biological prior
knowledge embeddings and continuous value prediction.

Original source: https://github.com/TencentAILabHealthcare/scBERT
Adapted for PerturbLab.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

# Internal imports
from perturblab.models.genecompass.io import MaskedLMOutputBoth
from perturblab.utils.logging import get_logger

from .embeddings import KnowledgeBertEmbeddings  # From your previous refactor

logger = get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"


# =============================================================================
# Prediction Heads
# =============================================================================


class BertLMPredictionHead(nn.Module):
    """Standard BERT LM Head for token classification."""

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertLMPredictionHead_value(nn.Module):
    """Regression head for predicting continuous expression values."""

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # Output dim is 1 for regression value
        self.decoder = nn.Linear(config.hidden_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyMLMHead_value(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead_value(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class EmbeddingWarmup(nn.Module):
    """Warmup scheduler for prior knowledge embedding scaling factor."""

    def __init__(self, warmup_steps, emb_warmup_steps):
        super().__init__()
        alpha = torch.tensor(1.0, dtype=torch.float)
        steps = torch.tensor(0, dtype=torch.long)
        self.alpha = torch.nn.Parameter(alpha, requires_grad=False)
        self.steps = torch.nn.Parameter(steps, requires_grad=False)
        # Note: Original code set self.warmup_steps = 0 explicitly, overriding arg
        self.warmup_steps = 0
        self.per_step = 1.0 / emb_warmup_steps

    def forward(self):
        self.steps += 1
        if self.steps > self.warmup_steps and self.alpha < 1.0:
            self.alpha += self.per_step


# =============================================================================
# Main Models
# =============================================================================

BERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices.
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs.
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """
    scBERT Base Model.

    Modified to use `KnowledgeBertEmbeddings` and accept `knowledges` during init.
    """

    def __init__(self, config, knowledges: dict[str, Any], add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Use specialized embeddings
        self.embeddings = KnowledgeBertEmbeddings(config, knowledges)

        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__
        self.use_cls_token = cfg_dict.get("use_cls_token", False)

        if self.use_cls_token:
            self.cls_embedding = nn.Embedding(2, config.hidden_size)

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        emb_warmup_alpha: float | None = 1.0,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if self.use_cls_token:
            attention_mask = torch.cat(
                [torch.ones(batch_size, 1).to(device), attention_mask], dim=1
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            values=values,
            species=species,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            emb_warmup_alpha=emb_warmup_alpha,
        )

        if self.use_cls_token:
            if species is None:
                species = torch.zeros(batch_size, 1).to(device).long()
            cls_embedding = self.cls_embedding(species)
            embedding_output = torch.cat((cls_embedding, embedding_output), dim=1)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING
)
class BertForMaskedLM(BertPreTrainedModel):
    """
    scBERT For Masked Language Modeling.

    Features:
    - Dual heads: One for token ID prediction (standard MLM), one for continuous value regression.
    - Embedding Warmup: Progressively scales prior knowledge influence.
    """

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"predictions.decoder.bias",
        r"cls.predictions.decoder.weight",
    ]

    def __init__(self, config, knowledges):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, knowledges, add_pooling_layer=False)

        self.cls = BertOnlyMLMHead(config)
        self.cls4value = BertOnlyMLMHead_value(config)

        self.use_values = config.use_values
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__
        self.use_cls_token = cfg_dict.get("use_cls_token", False)

        # Initialize warmup scheduler
        # Note: warmup_steps and emb_warmup_steps must be in config
        self.emb_warmup = EmbeddingWarmup(config.warmup_steps, config.emb_warmup_steps)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        labels_values: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | MaskedLMOutputBoth:

        # Step warmup
        self.emb_warmup()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            values=values,
            attention_mask=attention_mask,
            species=species,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            emb_warmup_alpha=self.emb_warmup.alpha,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Handle CLS token stripping if used
        if self.use_cls_token:
            # cls_output = sequence_output[:, 0]
            sequence_output = sequence_output[:, 1:]

        # Head 1: ID Prediction
        prediction_scores = self.cls(sequence_output)

        # Head 2: Value Prediction (Optional)
        prediction_values = None
        if self.use_values:
            prediction_values = self.cls4value(sequence_output).squeeze(-1)

        masked_lm_loss = None
        loss_values = None
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

            if self.use_values and labels_values is not None:
                loss_mse = MSELoss()
                # Filter valid positions based on labels mask
                valid_mask = labels.view(-1) != -100
                pred_val_flat = prediction_values.view(-1)[valid_mask]
                true_val_flat = labels_values.view(-1)[valid_mask]

                loss_values = loss_mse(pred_val_flat, true_val_flat)
                # Weighted multi-task loss
                loss = masked_lm_loss * 0.2 + loss_values * 0.8
            else:
                loss_values = torch.tensor(0.0, device=prediction_scores.device)
                loss = masked_lm_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutputBoth(
            loss=loss,
            value_loss=loss_values,
            id_loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Note: BERT for Sequence Classification, Token Classification etc.
# follow standard patterns and reuse the modified BertModel base.
