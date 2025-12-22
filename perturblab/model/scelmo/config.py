from typing import Optional

from ..configuration import ModelConfig


class scELMoConfig(ModelConfig):
    """
    Configuration for scELMo model.
    
    scELMo uses language model embeddings to extract features from single-cell data.
    It supports loading pre-computed gene embeddings from pickle files.
    """
    
    def __init__(
        self,
        model_series: str = 'scelmo',
        model_name: str = 'gpt-3.5-turbo',
        # Embedding configuration
        embedding_dim: int = 1536,  # Default GPT-3.5 embedding dimension
        aggregation_mode: str = 'wa',  # 'wa' (weighted average) or 'aa' (average)
        # API model configuration (for generating embeddings on-the-fly)
        api_model: str = 'text-embedding-ada-002',  # OpenAI embedding model
        **kwargs
    ):
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type='embedding_extractor',
            **kwargs
        )
        self._set_all(locals())
        
        if aggregation_mode not in ['wa', 'aa']:
            raise ValueError(f"aggregation_mode must be 'wa' or 'aa', got {aggregation_mode}")
