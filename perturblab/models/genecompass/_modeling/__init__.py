"""GeneCompass model implementations."""

from .components.bert import (
    BertForMaskedLM,
    BertLMPredictionHead,
    BertLMPredictionHead_value,
    BertModel,
    BertOnlyMLMHead,
    BertOnlyMLMHead_value,
    EmbeddingWarmup,
)
from .components.embeddings import (
    ContinuousValueEncoder,
    KnowledgeBertEmbeddings,
    PriorEmbedding,
    QuickGELU,
)
from .model import GeneCompassModel

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
