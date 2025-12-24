"""GeneCompass Model Implementation.

This module provides a PerturbLab-compatible wrapper for the GeneCompass model,
a knowledge-informed cross-species foundation model based on BERT.

GeneCompass Architecture and Methodology
=========================================

Algorithm Overview
-----------------
GeneCompass is a transformer-based foundation model designed for single-cell RNA-seq
analysis with cross-species knowledge transfer capabilities. It builds upon the BERT
architecture with several key enhancements:

1. **Knowledge-Informed Embeddings**: Integrates five types of biological prior knowledge
   into the embedding layer:
   - Gene2Vec: Pre-trained gene embeddings from literature
   - Protein-Protein Interactions (PPI): Gene interaction networks
   - Gene Ontology (GO): Functional annotations
   - ChIP-seq: Transcription factor binding sites
   - Co-expression: Gene co-expression patterns

2. **Cross-Species Transfer**: Supports human and mouse data with species-specific embeddings,
   enabling knowledge transfer across model organisms.

3. **Dual Prediction Heads**: Simultaneously predicts:
   - Gene identities (classification over vocabulary)
   - Expression values (regression)

4. **Masked Language Modeling**: Pre-trained using MLM objective where:
   - Gene tokens are randomly masked
   - Model predicts both masked gene IDs and their expression values

Architecture Components
----------------------

**Embeddings Layer** (KnowledgeBertEmbeddings):
    - Token embeddings: Gene vocabulary embeddings
    - Position embeddings: Sequence position encoding
    - Token type embeddings: Segment embeddings
    - Species embeddings: Human/mouse indicator
    - Prior knowledge embeddings: 5 types of biological knowledge
    - All embeddings are summed with learned weights (emb_warmup_alpha)

**Encoder** (BertEncoder):
    - Standard BERT encoder with 12 transformer layers
    - Multi-head self-attention (12 heads)
    - Feed-forward networks with GELU activation
    - Layer normalization and residual connections

**Prediction Heads**:
    - MLM Head (cls): Predicts gene IDs
        * Transform layer with GELU activation
        * Linear projection to vocabulary size
        * Tied weights with input embeddings
    - Value Head (cls4value): Predicts expression values
        * Transform layer with GELU activation
        * Linear projection to scalar output
        * MSE loss for regression

**Embedding Warmup**:
    - Gradually increases the weight of prior knowledge embeddings
    - Alpha parameter increases linearly during training
    - Helps stabilize early training

Input/Output Format
------------------

**Input** (GeneCompassInput):
    - input_ids: Gene token IDs [batch, seq_len]
    - values: Gene expression values [batch, seq_len] (optional)
    - attention_mask: Attention mask [batch, seq_len] (optional)
    - species: Species indicator (0=human, 1=mouse) [batch, seq_len] (optional)
    - token_type_ids: Token type IDs [batch, seq_len] (optional)
    - position_ids: Position IDs [batch, seq_len] (optional)

**Output** (GeneCompassOutput):
    - logits: Gene ID predictions [batch, seq_len, vocab_size]
    - value_logits: Expression value predictions [batch, seq_len, 1] (if use_values=True)
    - hidden_states: Encoder hidden states [batch, seq_len, hidden_size] (optional)
    - attentions: Attention weights (optional)
    - loss: Combined loss (if labels provided)

Training Workflow
-----------------

1. **Data Preparation**:
    - Tokenize gene sequences using gene vocabulary
    - Normalize expression values (log1p, z-score, etc.)
    - Apply random masking (15% of tokens)
    - Add species indicators

2. **Forward Pass**:
    - Embed genes with knowledge-informed embeddings
    - Apply transformer encoder
    - Generate predictions for gene IDs and values

3. **Loss Computation**:
    - MLM loss: CrossEntropyLoss for gene ID prediction
    - Value loss: MSELoss for expression value prediction
    - Combined loss: 0.2 * mlm_loss + 0.8 * value_loss (configurable)

4. **Embedding Warmup**:
    - Alpha parameter increases each step
    - Prior knowledge influence grows gradually
    - Stabilizes early training dynamics

Model Design Principles
-----------------------

1. **Knowledge Integration**: Prior biological knowledge is explicitly encoded into
   the model architecture, not just the training data. This improves interpretability
   and leverages domain expertise.

2. **Multi-Task Learning**: Jointly predicting gene identities and expression values
   forces the model to learn richer representations that capture both discrete
   (gene identity) and continuous (expression) aspects.

3. **Cross-Species Generalization**: Species embeddings enable the model to learn
   shared and species-specific patterns, facilitating transfer learning.

4. **Flexible Inference**: The model can be used for:
   - Zero-shot prediction on new cell types
   - Gene expression imputation
   - Cross-species expression prediction
   - Cell type annotation via hidden states

Key Hyperparameters
------------------

Training:
    - vocab_size: Gene vocabulary size (~20k-30k)
    - hidden_size: 768 (BERT-base) or 1024 (BERT-large)
    - num_hidden_layers: 12 (base) or 24 (large)
    - num_attention_heads: 12 (base) or 16 (large)
    - intermediate_size: 3072 (base) or 4096 (large)
    - max_position_embeddings: 2048 (longer sequences than standard BERT)
    - warmup_steps: Learning rate warmup steps
    - emb_warmup_steps: Embedding alpha warmup steps

Loss Weights:
    - mlm_weight: 0.2 (gene ID prediction weight)
    - value_weight: 0.8 (expression value prediction weight)

References
----------
.. [1] Wang et al. (2024) "GeneCompass: A Knowledge-Informed Cross-Species Foundation
       Model for Single-Cell RNA-seq Analysis"
       https://github.com/xyz/GeneCompass

Copyright and License
--------------------
Original implementation: Copyright (c) 2024 GeneCompass Authors
PerturbLab wrapper: Copyright (c) 2024 PerturbLab Authors
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from perturblab.models.genecompass.config import GeneCompassConfig
from perturblab.models.genecompass.io import GeneCompassInput, GeneCompassOutput
from perturblab.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["GeneCompassModel"]


class GeneCompassModel(nn.Module):
    """GeneCompass model wrapper for PerturbLab.

    This class wraps the original GeneCompass implementation, providing a unified interface
    consistent with other PerturbLab models and preserving compatibility with pre-trained checkpoints.
    It leverages Hugging Face Transformers under the hood.

    Attributes:
        config (GeneCompassConfig): Model configuration.
        bert (BertModel): Core BERT encoder with knowledge-based embeddings.
        cls (BertOnlyMLMHead): Prediction head for gene identifiers (MLM).
        cls4value (BertOnlyMLMHead_value): Prediction head for gene expression values.
        emb_warmup (EmbeddingWarmup): Scheduler for embedding warmup.
        use_values (bool): Whether to predict expression values.
        use_cls_token (bool): Whether to use CLS token for pooled outputs.

    Example:
        ```python
        from perturblab.models import GeneCompassModel, GeneCompassConfig
        from perturblab.models.genecompass import GeneCompassInput

        config = GeneCompassConfig(
            vocab_size=25000,
            hidden_size=768,
            num_hidden_layers=12,
            use_values=True
        )
        knowledges = load_knowledge_bases()
        model = GeneCompassModel(config, knowledges)

        inputs = GeneCompassInput(
            input_ids=gene_ids,
            values=expression_values,
            attention_mask=mask,
            species=species_ids
        )
        outputs = model(inputs)

        gene_logits = outputs.logits
        value_predictions = outputs.value_logits
        embeddings = outputs.hidden_states
        ```

    Notes:
        - Attribute names (`bert`, `cls`, `cls4value`, `emb_warmup`) are maintained for checkpoint compatibility.
        - Embedding warmup is handled automatically during training.
        - Set `model.eval()` during inference to disable dropout.
    """

    def __init__(
        self,
        config: GeneCompassConfig,
        knowledges: Dict[str, torch.Tensor],
        **kwargs,
    ):
        """Initializes GeneCompassModel.

        Args:
            config (GeneCompassConfig): Model configuration.
            knowledges (Dict[str, torch.Tensor]): Dictionary of prior knowledge embedding matrices:
                - "gene2vec": Gene2Vec embeddings [vocab_size, emb_dim]
                - "ppi": Protein-protein interaction network [vocab_size, emb_dim]
                - "go": Gene ontology annotations [vocab_size, emb_dim]
                - "chipseq": ChIP-seq binding sites [vocab_size, emb_dim]
                - "coexp": Co-expression patterns [vocab_size, emb_dim]
                All matrices must share the same embedding dimension.
            **kwargs: Additional keyword arguments (ignored, for compatibility).
        """
        super().__init__()
        self.config = config
        self._build_model(config, knowledges)
        self.use_values = config.use_values
        self.use_cls_token = config.use_cls_token

        logger.info(
            f"Initialized GeneCompassModel: vocab_size={config.vocab_size}, "
            f"hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
            f"use_values={config.use_values}"
        )

    def _build_model(self, config: GeneCompassConfig, knowledges: Dict[str, torch.Tensor]) -> None:
        """Builds model components based on configuration.

        This method sets up the BERT encoder and prediction heads, ensuring attribute names
        support checkpoint compatibility.

        Args:
            config (GeneCompassConfig): Model configuration.
            knowledges (Dict[str, torch.Tensor]): Dictionary of prior knowledge embeddings.

        Raises:
            ImportError: If transformers is not installed.
        """
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "GeneCompass requires the transformers library. "
                "Install via: pip install perturblab[genecompass]"
            )

        from perturblab.models.genecompass._modeling.components import BertForMaskedLM

        bert_config = self._create_bert_config(config)
        core_model = BertForMaskedLM(bert_config, knowledges)

        # Always keep these attribute names for checkpoint loading
        self.bert = core_model.bert
        self.cls = core_model.cls
        self.cls4value = core_model.cls4value
        self.emb_warmup = core_model.emb_warmup

    def _create_bert_config(self, config: GeneCompassConfig):
        """Converts PerturbLab config to Hugging Face BertConfig.

        Args:
            config (GeneCompassConfig): PerturbLab model configuration.

        Returns:
            BertConfig: Hugging Face BERT configuration.
        """
        from transformers import BertConfig

        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=config.pad_token_id,
            position_embedding_type=config.position_embedding_type,
        )
        # Add GeneCompass-specific configurations
        bert_config.use_values = config.use_values
        bert_config.use_cls_token = config.use_cls_token
        bert_config.warmup_steps = config.warmup_steps
        bert_config.emb_warmup_steps = config.emb_warmup_steps
        return bert_config

    def forward(
        self,
        inputs: GeneCompassInput,
    ) -> GeneCompassOutput:
        """Runs the forward pass of GeneCompassModel.

        Args:
            inputs (GeneCompassInput): Model input, including gene tokens, expression values,
                masks, labels, etc. All arguments (including labels) are provided in this schema.

        Returns:
            GeneCompassOutput: Outputs including logits, losses, and optionally hidden states
                and attentions.

        Notes:
            - Embedding warmup is applied automatically during training.
            - Loss is computed only if `inputs.labels` is given.
            - Value loss requires both `inputs.labels` and `inputs.labels_values`.
        """
        # 1. Embedding warmup (update alpha)
        self.emb_warmup()

        # 2. BERT encoding
        encoder_outputs = self._encode(inputs)
        sequence_output = encoder_outputs["sequence_output"]

        # 3. Handle CLS token if used
        if self.use_cls_token:
            cls_output = sequence_output[:, 0]
            sequence_output = sequence_output[:, 1:]

        # 4. Generate predictions
        predictions = self._predict(sequence_output)

        # 5. Compute loss if labels are available
        loss = None
        if inputs.labels is not None:
            loss = self._compute_loss(
                predictions["logits"],
                predictions["value_logits"],
                inputs.labels,
                inputs.labels_values,
            )

        return GeneCompassOutput(
            logits=predictions["logits"],
            loss=loss,
            value_logits=predictions["value_logits"],
            hidden_states=encoder_outputs["hidden_states"],
            attentions=encoder_outputs.get("attentions"),
        )

    def _encode(self, inputs: GeneCompassInput) -> Dict[str, Any]:
        """Encodes inputs using the BERT encoder.

        Args:
            inputs (GeneCompassInput): Model input.

        Returns:
            Dict[str, Any]: Dictionary with:
                - `sequence_output`: Hidden states from final encoder layer.
                - `hidden_states`: All layer hidden states (if requested).
                - `attentions`: Attention weights (if requested).
        """
        bert_outputs = self.bert(
            input_ids=inputs.input_ids,
            values=inputs.values,
            attention_mask=inputs.attention_mask,
            species=inputs.species,
            token_type_ids=inputs.token_type_ids,
            position_ids=inputs.position_ids,
            output_attentions=inputs.output_attentions,
            output_hidden_states=inputs.output_hidden_states,
            emb_warmup_alpha=self.emb_warmup.alpha,
            return_dict=True,
        )

        return {
            "sequence_output": bert_outputs[0],
            "hidden_states": bert_outputs.last_hidden_state,
            "attentions": bert_outputs.attentions if hasattr(bert_outputs, "attentions") else None,
        }

    def _predict(self, sequence_output: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Generates predictions from encoder outputs.

        Args:
            sequence_output (torch.Tensor): Output hidden states from encoder.
                Shape: [batch_size, sequence_length, hidden_size].

        Returns:
            Dict[str, Optional[torch.Tensor]]: Dictionary with:
                - `logits`: Gene identifier prediction logits.
                - `value_logits`: Expression value predictions (if applicable).
        """
        # Gene ID predictions
        prediction_scores = self.cls(sequence_output)
        prediction_values = None
        if self.use_values:
            prediction_values = self.cls4value(sequence_output).squeeze(-1)

        return {
            "logits": prediction_scores,
            "value_logits": prediction_values,
        }

    def _compute_loss(
        self,
        logits: torch.Tensor,
        value_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        labels_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes training loss for the model.

        Args:
            logits (torch.Tensor): Gene identifier prediction logits.
            value_logits (Optional[torch.Tensor]): Predicted expression values.
            labels (torch.Tensor): Gene ID labels. Ignore positions where label is -100.
            labels_values (Optional[torch.Tensor]): True expression value labels.

        Returns:
            torch.Tensor: Weighted combined loss (scalar).
        """
        from torch.nn import CrossEntropyLoss, MSELoss

        # Masked language modeling loss (gene IDs)
        loss_fct = CrossEntropyLoss()  # -100 is the ignore index
        masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Value prediction loss (if enabled)
        if self.use_values and value_logits is not None and labels_values is not None:
            loss_mse = MSELoss()
            valid_mask = labels.view(-1) != -100
            pred_vals = value_logits.view(-1)[valid_mask]
            true_vals = labels_values.view(-1)[valid_mask]
            loss_values = loss_mse(pred_vals, true_vals)

            total_loss = (
                self.config.mlm_weight * masked_lm_loss + self.config.value_weight * loss_values
            )
        else:
            total_loss = masked_lm_loss

        return total_loss

    def encode(self, inputs: GeneCompassInput) -> torch.Tensor:
        """Extracts encoder hidden states for downstream analysis.

        This method returns hidden states without predictions, enabling applications like
        clustering or representation learning.

        Args:
            inputs (GeneCompassInput): Model inputs.

        Returns:
            torch.Tensor: Hidden states from the final encoder layer,
                shape [batch_size, sequence_length, hidden_size].

        Example:
            ```
            embeddings = model.encode(inputs)
            from sklearn.cluster import KMeans
            cell_embeddings = embeddings.mean(dim=1).cpu().numpy()
            clusters = KMeans(n_clusters=10).fit_predict(cell_embeddings)
            ```
        """
        outputs = self.bert(
            input_ids=inputs.input_ids,
            values=inputs.values,
            attention_mask=inputs.attention_mask,
            species=inputs.species,
            token_type_ids=inputs.token_type_ids,
            position_ids=inputs.position_ids,
            output_hidden_states=False,
            emb_warmup_alpha=self.emb_warmup.alpha,
            return_dict=True,
        )
        return outputs.last_hidden_state
