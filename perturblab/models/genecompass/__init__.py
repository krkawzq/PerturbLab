"""GeneCompass: Knowledge-Informed Cross-Species Foundation Model.

GeneCompass is a BERT-based foundation model for single-cell RNA-seq analysis.
It integrates five types of biological prior knowledge to enable enhanced gene expression prediction,
cross-species transfer learning (human ↔ mouse), zero-shot cell type prediction, and gene expression imputation.

Key Features:
    * Knowledge-Informed Embeddings: Integrates Gene2Vec, PPI, GO, ChIP-seq, and co-expression knowledge.
    * Dual Prediction: Predicts both gene identities and expression values simultaneously.
    * Cross-Species: Supports species embeddings (human and mouse) for domain adaptation.
    * Transformer Architecture: Utilizes BERT with configurable number of layers (6-24).
    * Embedding Warmup: Gradually incorporates prior knowledge during training.

Model Architecture:
    Input: genes, expression values, and species indicators
      ↓
    Knowledge-Informed Embeddings
      ↓
    BERT Encoder (6-24 layers)
      ↓
      ├─→ MLM Head (gene ID prediction)
      └─→ Value Head (expression value regression)

Example usage:
    ```python
    from perturblab.models import GeneCompassModel, GeneCompassConfig
    from perturblab.models.genecompass import GeneCompassInput

    # Initialize model configuration
    config = GeneCompassConfig(
        vocab_size=25000,
        hidden_size=768,
        num_hidden_layers=12,
        use_values=True,
    )
    knowledges = {
        "gene2vec": gene2vec_embeddings,
        "ppi": ppi_network,
        "go": go_annotations,
        "chipseq": chipseq_data,
        "coexp": coexp_network,
    }
    model = GeneCompassModel(config, knowledges)

    # Construct model inputs
    inputs = GeneCompassInput(
        input_ids=gene_tokens,
        values=expression_values,
        species=species_indicators,
    )
    outputs = model(inputs)

    # Retrieve predictions
    gene_predictions = outputs.logits
    value_predictions = outputs.value_logits
    embeddings = outputs.hidden_states
    ```

References:
    [1] GeneCompass: Decoding Universal Gene Expression Signatures Across Species and Sequencing Platforms.
        bioRxiv 2023. https://github.com/xyz/GeneCompass

Dependencies:
    Required: transformers
    Install: pip install perturblab[genecompass]
"""

from perturblab.models import MODELS
from perturblab.utils import create_lazy_loader

# Import configuration and IO schemas
from .config import GeneCompassConfig, requirements, dependencies
from .io import GeneCompassInput, GeneCompassOutput, MaskedLMOutputBoth

# Mapping for lazy loading core model module.
_LAZY_MODULES = {
    "GeneCompassModel": "_modeling",
}

# Setup lazy loading for the core GeneCompassModel
__getattr__, __dir__ = create_lazy_loader(
    requirements=requirements,
    dependencies=dependencies,
    lazy_modules=_LAZY_MODULES,
    package_name=__package__,
    install_hint="pip install perturblab[genecompass]",
)

# ============================================================================
# Model Registry
# ============================================================================

GENECOMPASS_REGISTRY = {
    "genecompass": "GeneCompassModel",
    "genecompass_base": "GeneCompassModel",
    "genecompass_small": "GeneCompassModel",
}

# Register model classes into the global MODELS registry only if not present.
for model_name, model_class in GENECOMPASS_REGISTRY.items():
    if model_name not in MODELS:
        MODELS[model_name] = {
            "class": model_class,
            "module": "perturblab.models.genecompass",
            "config": GeneCompassConfig,
            "input": GeneCompassInput,
            "output": GeneCompassOutput,
            "description": "Knowledge-informed cross-species foundation model",
        }

# ============================================================================
# Component Registry
# ============================================================================

GENECOMPASS_COMPONENTS = {
    # Embedding Components
    "knowledge_embeddings": {
        "class": "KnowledgeBertEmbeddings",
        "module": "perturblab.models.genecompass._modeling.components.embeddings",
        "description": "BERT embeddings incorporating biological prior knowledge.",
    },
    "continuous_value_encoder": {
        "class": "ContinuousValueEncoder",
        "module": "perturblab.models.genecompass._modeling.components.embeddings",
        "description": "Encodes continuous gene expression values.",
    },
    "prior_embedding": {
        "class": "PriorEmbedding",
        "module": "perturblab.models.genecompass._modeling.components.embeddings",
        "description": "Embedding layer for prior knowledge integration.",
    },
    # BERT Core Components
    "bert_model": {
        "class": "BertModel",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "BERT encoder with knowledge-informed embeddings.",
    },
    "bert_for_masked_lm": {
        "class": "BertForMaskedLM",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "BERT for masked language modeling with dual prediction heads.",
    },
    # Prediction Heads
    "bert_lm_prediction_head": {
        "class": "BertLMPredictionHead",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "Predicts gene (token) identities.",
    },
    "bert_lm_prediction_head_value": {
        "class": "BertLMPredictionHead_value",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "Predicts expression values (regression).",
    },
    "bert_mlm_head": {
        "class": "BertOnlyMLMHead",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "MLM head for gene ID prediction.",
    },
    "bert_mlm_head_value": {
        "class": "BertOnlyMLMHead_value",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "MLM head for value prediction.",
    },
    # Training Components
    "embedding_warmup": {
        "class": "EmbeddingWarmup",
        "module": "perturblab.models.genecompass._modeling.components.bert",
        "description": "Warmup scheduler for prior knowledge embeddings.",
    },
}

__all__ = [
    # Public API: config and I/O
    "GeneCompassConfig",
    "GeneCompassInput",
    "GeneCompassOutput",
    "MaskedLMOutputBoth",
    # Main model (lazy-loaded)
    "GeneCompassModel",
    # Registries
    "GENECOMPASS_REGISTRY",
    "GENECOMPASS_COMPONENTS",
]
