import json
import logging
import os
from typing import Any, Dict, List, Optional

from ..configuration import ModelConfig

logger = logging.getLogger(__name__)

class scFoundationConfig(ModelConfig):
    """Configuration class for scFoundation model.
    
    Provides parameters for both the core scFoundation architecture (encoder/decoder)
    and the GEARS perturbation head.
    """

    def __init__(
        self,
        model_series: str = 'scfoundation',
        model_name: str = 'default',
        num_tokens: int = 19264,
        encoder_hidden_dim: int = 768,
        decoder_hidden_dim: int = 512,
        encoder_depth: int = 12,
        decoder_depth: int = 6,
        encoder_heads: int = 12,
        decoder_heads: int = 8,
        encoder_dim_head: int = 64,
        decoder_dim_head: int = 64,
        bin_num: int = 100,
        bin_alpha: float = 1.0,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        encoder_module_type: str = 'performer',
        decoder_module_type: str = 'performer',
        model_type: str = 'mae_autobin',
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        gears_hidden_size: int = 64,
        gears_num_go_gnn_layers: int = 1,
        gears_num_gene_gnn_layers: int = 1,
        gears_decoder_hidden_size: int = 16,
        gears_num_similar_genes_go_graph: int = 20,
        gears_num_similar_genes_co_express_graph: int = 20,
        gears_coexpress_threshold: float = 0.4,
        gears_uncertainty: bool = False,
        gears_uncertainty_reg: float = 1,
        gears_direction_lambda: float = 1e-1,
        gears_no_perturb: bool = False,
        gears_go_graph_threshold: float = 0.1,
        gears_go_graph_top_k: int = 20,
        gears_coexpress_top_k: int = 20,
        **kwargs
    ):
        """Initialize scFoundationConfig with architecture and perturbation parameters."""
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type=model_type,
            **kwargs
        )
        self._set_all(locals())
        
        # Initialize default token IDs if not provided
        if self.pad_token_id is None:
            self.pad_token_id = self.num_tokens
        if self.mask_token_id is None:
            self.mask_token_id = self.num_tokens + 1
            
        self.max_seq_len = self.num_tokens + 2

    def get_gears_config(self) -> 'GearsConfig':
        """Convert scFoundation parameters to a GEARS specific configuration.

        Returns:
            GearsConfig: Configuration object for the GEARS perturbation head.
        """
        from ..gears.config import GearsConfig
        
        return GearsConfig(
            hidden_size=self.gears_hidden_size,
            num_go_gnn_layers=self.gears_num_go_gnn_layers,
            num_gene_gnn_layers=self.gears_num_gene_gnn_layers,
            decoder_hidden_size=self.gears_decoder_hidden_size,
            num_similar_genes_go_graph=self.gears_num_similar_genes_go_graph,
            num_similar_genes_co_express_graph=self.gears_num_similar_genes_co_express_graph,
            coexpress_threshold=self.gears_coexpress_threshold,
            uncertainty=self.gears_uncertainty,
            uncertainty_reg=self.gears_uncertainty_reg,
            direction_lambda=self.gears_direction_lambda,
            no_perturb=self.gears_no_perturb,
            go_graph_threshold=self.gears_go_graph_threshold,
            go_graph_top_k=self.gears_go_graph_top_k,
            coexpress_top_k=self.gears_coexpress_top_k,
        )

    def to_model_config_dict(self) -> Dict[str, Any]:
        """Generate a nested dictionary for model internal initialization.

        Returns:
            Dict[str, Any]: Dictionary containing model, encoder, and decoder configs.
        """
        return {
            'model': self.model_type,
            'n_class': self.num_tokens,
            'seq_len': self.max_seq_len,
            'pad_token_id': self.pad_token_id,
            'mask_token_id': self.mask_token_id,
            'bin_alpha': self.bin_alpha,
            'bin_num': self.bin_num,
            'encoder': {
                'module_type': self.encoder_module_type,
                'hidden_dim': self.encoder_hidden_dim,
                'depth': self.encoder_depth,
                'heads': self.encoder_heads,
                'dim_head': self.encoder_dim_head,
                'ff_dropout': self.ff_dropout,
                'attn_dropout': self.attn_dropout,
            },
            'decoder': {
                'module_type': self.decoder_module_type,
                'hidden_dim': self.decoder_hidden_dim,
                'depth': self.decoder_depth,
                'heads': self.decoder_heads,
                'dim_head': self.decoder_dim_head,
                'ff_dropout': self.ff_dropout,
                'attn_dropout': self.attn_dropout,
            }
        }


def load_gene_list(path: Optional[str] = None) -> List[str]:
    """Load the gene list from a JSON file.

    Args:
        path: Custom path to gene list file. Defaults to source/gene_index.json.

    Returns:
        List[str]: A list of gene names.
    """
    gene_list_path = path or os.path.join(
        os.path.dirname(__file__), 'source', 'gene_index.json'
    )
    
    if not os.path.exists(gene_list_path):
        logger.error("Gene list not found at: %s", gene_list_path)
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")
    
    with open(gene_list_path, 'r') as f:
        return json.load(f)

def load_default_gene_list() -> List[str]:
    """Load the default gene list.

    Returns:
        List[str]: Default gene list from source directory.
    """
    return load_gene_list()

def load_config(path: str) -> Dict[str, Any]:
    """Utility to load JSON configuration files.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not os.path.exists(path):
        logger.error("Config file not found: %s", path)
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def load_default_model_config() -> Dict[str, Any]:
    """Load default model configuration."""
    return load_config(os.path.join(os.path.dirname(__file__), 'source', 'configs', 'config.json'))

def load_default_training_config() -> Dict[str, Any]:
    """Load default training configuration."""
    return load_config(os.path.join(os.path.dirname(__file__), 'source', 'configs', 'training_config.json'))
