from typing import Optional

from ..configuration import ModelConfig


class UCEModelConfig(ModelConfig):
    """
    Configuration for UCE (Universal Cell Embeddings) model.
    
    UCE is a transformer-based foundation model for generating universal cell embeddings
    from single-cell RNA-seq data. It uses protein embeddings (ESM2) as gene tokens.
    
    Original source: https://github.com/snap-stanford/UCE
    """
    
    def __init__(
        self,
        model_series: str = 'uce',
        model_name: str = '4layer',
        # Model architecture
        nlayers: int = 4,  # Number of transformer layers (4 or 33)
        d_model: int = 1280,  # Model dimension (fixed at 1280 for UCE)
        nhead: int = 20,  # Number of attention heads (fixed at 20 for UCE)
        d_hid: int = 5120,  # Hidden dimension in feedforward network
        output_dim: int = 1280,  # Output embedding dimension
        token_dim: int = 5120,  # Token/protein embedding dimension
        dropout: float = 0.05,  # Dropout rate
        batch_first: bool = True,  # Whether input is batch_first format
        # Data processing
        pad_length: int = 1536,  # Maximum sequence length (padded)
        sample_size: int = 1024,  # Number of genes sampled per cell
        # Token indices
        pad_token_idx: int = 0,
        chrom_token_left_idx: int = 1,
        chrom_token_right_idx: int = 2,
        cls_token_idx: int = 3,
        CHROM_TOKEN_OFFSET: int = 143574,  # Offset for chromosome tokens
        # Species and embeddings
        species: str = 'human',  # Default species
        embedding_model: str = 'ESM2',  # Protein embedding model
        # File paths (will be auto-downloaded if not provided)
        spec_chrom_csv_path: Optional[str] = None,
        token_file: Optional[str] = None,
        protein_embeddings_dir: Optional[str] = None,
        offset_pkl_path: Optional[str] = None,
        # Other
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type='embedding_extractor',
            **kwargs
        )
        self._set_all(locals())
        
        # Validate nlayers
        if nlayers not in [4, 33]:
            raise ValueError(f"nlayers must be 4 or 33, got {nlayers}")

