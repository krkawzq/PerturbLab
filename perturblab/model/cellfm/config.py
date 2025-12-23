import logging
from typing import Optional

from ..configuration import ModelConfig

logger = logging.getLogger(__name__)


class CellFMConfig(ModelConfig):
    """Configuration for CellFM (Cell Foundation Model).
    
    CellFM is a large-scale foundation model pre-trained on transcriptomics
    of 100 million human cells using retention-based architecture.
    """
    
    def __init__(
        self,
        model_series: str = 'cellfm',
        model_name: str = '80M',
        n_genes: int = 27855,
        enc_dims: int = 1536,
        enc_nlayers: int = 2,
        enc_num_heads: int = 48,
        enc_dropout: float = 0.1,
        value_enc_hidden: int = 256,
        dropout: float = 0.1,
        nonz_len: int = 2048,
        mask_len: int = 2048,
        filt_len: int = 200,
        mask_ratio: float = 0.5,
        add_zero: bool = True,
        pad_zero: bool = True,
        cellwise_use_bias: bool = True,
        lora: int = 0,
        alpha: float = 0.0,
        temp: float = 0.2,
        eps: float = 0.01,
        label: bool = False,
        num_cls: Optional[int] = None,
        recompute: bool = True,
        sim: float = 0.8,
        ecs: bool = False,
        ecs_threshold: float = 0.8,
        **kwargs
    ):
        """Initialize CellFMConfig with model architecture hyperparameters.

        Args:
            model_series: Series identifier for the model.
            model_name: Specific model name (e.g., '80M', '400M').
            n_genes: Number of genes in the vocabulary.
            enc_dims: Dimension of encoder embeddings.
            enc_nlayers: Number of encoder retention layers.
            enc_num_heads: Number of attention heads in encoder.
            enc_dropout: Dropout rate in encoder.
            value_enc_hidden: Hidden dimension in value encoder FFN.
            dropout: General dropout rate.
            nonz_len: Maximum length of non-zero genes.
            mask_len: Maximum length of masked genes.
            filt_len: Filter length for preprocessing.
            mask_ratio: Ratio of genes to mask during training.
            add_zero: Whether to add zero prediction head.
            pad_zero: Whether to pad with zero tokens.
            cellwise_use_bias: Whether to use bias in cellwise decoder map layer.
            lora: LoRA rank (0 means no LoRA).
            alpha: LoRA alpha parameter.
            temp: Temperature for contrastive learning.
            eps: Epsilon for numerical stability.
            label: Whether to include classification head.
            num_cls: Number of classes for classification.
            recompute: Whether to use gradient checkpointing.
            sim: Similarity threshold.
            ecs: Whether to use ECS (Expression Consistency Score).
            ecs_threshold: Threshold for ECS filtering.
            **kwargs: Additional configuration arguments.
        """
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type='foundation_model',
            **kwargs
        )
        
        if enc_dims % enc_num_heads != 0:
            logger.warning(
                "enc_dims (%d) is not divisible by enc_num_heads (%d).",
                enc_dims, enc_num_heads
            )
        
        if label and num_cls is None:
            logger.warning("label=True but num_cls is None. Defaulting to 1.")
            num_cls = 1
        
        self._set_all(locals())

    @classmethod
    def _get_key_mapping(cls) -> dict[str, str]:
        """Get mapping for legacy key names."""
        return {}
