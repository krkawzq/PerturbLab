"""GeneCompass model implementations."""

from .model import GeneCompassModel
from .components.embeddings import (
    KnowledgeBertEmbeddings,
    ContinuousValueEncoder,
    PriorEmbedding,
    QuickGELU,
)
from .components.bert import (
    BertModel,
    BertForMaskedLM,
    BertLMPredictionHead,
    BertLMPredictionHead_value,
    BertOnlyMLMHead,
    BertOnlyMLMHead_value,
    EmbeddingWarmup,
)

__all__ = [
    # Main model
    "GeneCompassModel",
    # Embeddings
    "KnowledgeBertEmbeddings",
    "ContinuousValueEncoder",
    "PriorEmbedding",
    "QuickGELU",
    # BERT core
    "BertModel",
    "BertForMaskedLM",
    # Prediction heads
    "BertLMPredictionHead",
    "BertLMPredictionHead_value",
    "BertOnlyMLMHead",
    "BertOnlyMLMHead_value",
    # Training utilities
    "EmbeddingWarmup",
]
