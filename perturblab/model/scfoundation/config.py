import json
import os
from typing import Any, Dict, Optional

from ..configuration import ModelConfig


class scFoundationConfig(ModelConfig):
    """
    Configuration class for scFoundation model.
    
    This config can be loaded/saved using the base class methods:
    - config.save(path): Save to JSON file
    - scFoundationConfig.load(path): Load from JSON file
    
    Default config can be found at: source/configs/config.json
    """
    def __init__(
        self,
        model_series: str = 'scfoundation',
        model_name: str = 'default',
        # Model architecture
        num_tokens: int = 19264,
        encoder_hidden_dim: int = 768,
        decoder_hidden_dim: int = 512,
        encoder_depth: int = 12,
        decoder_depth: int = 6,
        encoder_heads: int = 12,
        decoder_heads: int = 8,
        encoder_dim_head: int = 64,
        decoder_dim_head: int = 64,
        # Embedding parameters
        bin_num: int = 100,
        bin_alpha: float = 1.0,
        # Special tokens
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        # Module types
        encoder_module_type: str = 'performer',
        decoder_module_type: str = 'performer',
        model_type: str = 'mae_autobin',
        # Dropout
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(
            model_series=model_series,
            model_name=model_name,
            model_type=model_type,
            **kwargs
        )
        self._set_all(locals())
        
        if self.pad_token_id is None:
            self.pad_token_id = self.num_tokens
        if self.mask_token_id is None:
            self.mask_token_id = self.num_tokens + 1
            
        self.max_seq_len = self.num_tokens + 2

    def to_model_config_dict(self) -> Dict[str, Any]:
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


def load_gene_list(path: str = None) -> list:
    """
    Load the gene list from source/gene_index.json.
    
    Args:
        path (str, optional): Custom path to gene list file. If None, uses default source/gene_index.json.
    
    Returns:
        list: List of 19264 gene names
        
    Example:
        ```python
        from perturblab.model.scfoundation.config import load_gene_list
        
        genes = load_gene_list()
        print(len(genes))  # 19264
        ```
    """
    if path is None:
        gene_list_path = os.path.join(
            os.path.dirname(__file__),
            'source',
            'gene_index.json'
        )
    else:
        gene_list_path = path
    
    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")
    
    with open(gene_list_path, 'r') as f:
        return json.load(f)

def load_default_gene_list() -> list:
    return load_gene_list()

def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")

def load_default_model_config() -> dict:
    return load_config(os.path.join(os.path.dirname(__file__), 'source', 'configs', 'config.json'))

def load_default_training_config() -> dict:
    return load_config(os.path.join(os.path.dirname(__file__), 'source', 'configs', 'training_config.json'))
