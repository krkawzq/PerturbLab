import logging
from typing import Optional

from ..configuration import ModelConfig

logger = logging.getLogger(__name__)

class UCEModelConfig(ModelConfig):
    """Configuration for UCE (Universal Cell Embeddings) model.
    
    UCE is a transformer-based foundation model for generating universal cell embeddings
    from single-cell RNA-seq data. It uses protein embeddings (ESM2) as gene tokens.
    
    Source: https://github.com/snap-stanford/UCE
    """
    
    def __init__(
        self,
        model_series: str = 'uce',
        model_name: str = '4layer',
        nlayers: int = 4,
        d_model: int = 1280,
        nhead: int = 20,
        d_hid: int = 5120,
        output_dim: int = 1280,
        token_dim: int = 5120,
        dropout: float = 0.05,
        batch_first: bool = True,
        pad_length: int = 1536,
        sample_size: int = 1024,
        pad_token_idx: int = 0,
        chrom_token_left_idx: int = 1,
        chrom_token_right_idx: int = 2,
        cls_token_idx: int = 3,
        CHROM_TOKEN_OFFSET: int = 143574,
        species: str = 'human',
        embedding_model: str = 'ESM2',
        spec_chrom_csv_path: Optional[str] = None,
        token_file: Optional[str] = None,
        protein_embeddings_dir: Optional[str] = None,
        offset_pkl_path: Optional[str] = None,
        device: str = 'cuda',
        **kwargs
    ):
        """Initialize UCEModelConfig with model hyperparameters.

        Args:
            model_series: Series identifier for the model.
            model_name: Specific model name (e.g., '4layer').
            nlayers: Number of transformer layers (typically 4 or 33).
            d_model: Model dimension (fixed at 1280 for UCE).
            nhead: Number of attention heads (fixed at 20 for UCE).
            d_hid: Hidden dimension in feedforward network.
            output_dim: Output embedding dimension.
            token_dim: Token/protein embedding dimension.
            dropout: Dropout rate.
            batch_first: Whether input is in batch_first format.
            pad_length: Maximum sequence length (padded).
            sample_size: Number of genes sampled per cell.
            pad_token_idx: Index for padding token.
            chrom_token_left_idx: Index for left chromosome token.
            chrom_token_right_idx: Index for right chromosome token.
            cls_token_idx: Index for CLS token.
            CHROM_TOKEN_OFFSET: Offset for chromosome tokens.
            species: Default species name.
            embedding_model: Protein embedding model used (e.g., 'ESM2').
            spec_chrom_csv_path: Path to species chromosome CSV.
            token_file: Path to token file.
            protein_embeddings_dir: Directory containing protein embeddings.
            offset_pkl_path: Path to offset pickle file.
            device: Computing device ('cuda' or 'cpu').
            **kwargs: Additional configuration arguments.
        """
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type='embedding_extractor',
            **kwargs
        )
        
        if nlayers not in [4, 33]:
            logger.error("Invalid nlayers: %d. UCE expects 4 or 33.", nlayers)
            raise ValueError(f"nlayers must be 4 or 33, got {nlayers}")

        self._set_all(locals())
