import logging
from typing import Optional

from ..configuration import ModelConfig

logger = logging.getLogger(__name__)

class scELMoConfig(ModelConfig):
    """Configuration for scELMo model.
    
    scELMo uses language model embeddings to extract features from single-cell data.
    It supports loading pre-computed gene embeddings and various aggregation strategies.
    """
    
    def __init__(
        self,
        model_series: str = 'scelmo',
        model_name: str = 'gpt-3.5-turbo',
        embedding_dim: int = 1536,
        aggregation_mode: str = 'wa',
        api_model: str = 'text-embedding-ada-002',
        **kwargs
    ):
        """Initialize scELMoConfig with embedding and aggregation parameters.

        Args:
            model_series: Series identifier for the model.
            model_name: Specific model name or engine.
            embedding_dim: Dimension of the language model embeddings.
            aggregation_mode: Strategy for aggregating embeddings ('wa' for weighted 
                average or 'aa' for average).
            api_model: The underlying API model used for generating embeddings.
            **kwargs: Additional configuration arguments.
        """
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type='embedding_extractor',
            **kwargs
        )
        
        if aggregation_mode not in ['wa', 'aa']:
            logger.error("Invalid aggregation_mode: %s. Expected 'wa' or 'aa'.", aggregation_mode)
            raise ValueError(f"aggregation_mode must be 'wa' or 'aa', got {aggregation_mode}")

        self._set_all(locals())
