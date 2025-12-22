from typing import Literal, Optional

from ..configuration import ModelConfig


class GearsConfig(ModelConfig):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        
        num_similar_genes_go_graph: int = 20,
        num_similar_genes_co_express_graph: int = 20,
        coexpress_threshold: float = 0.4,
        go_graph_threshold: float = 0.1,
        go_graph_top_k: int = 20,
        coexpress_top_k: int = 20,
        
        uncertainty: bool = False,
        no_perturb: bool = False,
        
        uncertainty_reg: float = 1.0,
        direction_lambda: float = 1e-1,
        
        cell_fitness_pred: bool = False,
        weight_bias_track: bool = False,
        
        num_genes: Optional[int] = None,
        num_perts: Optional[int] = None,
        
        go_graph_path: Optional[str] = None,
        
        **kwargs,
    ):
        super().__init__(
            model_series='gears',
            model_name='',
            model_type='gnn',
            **kwargs
        )
        self._set_all(locals())
