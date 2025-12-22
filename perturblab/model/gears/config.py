import logging
from typing import Literal, Optional

from ..configuration import ModelConfig

logger = logging.getLogger(__name__)

class GearsConfig(ModelConfig):
    """Configuration class for GEARS (Graph-based Estimation of Agent Response Surfaces) model.
    
    GEARS combines a Gene Ontology (GO) Graph and a Co-expression Graph with GNNs
    to predict gene expression responses under perturbations.
    """

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
        direction_lambda: float = 0.1,
        cell_fitness_pred: bool = False,
        weight_bias_track: bool = False,
        num_genes: Optional[int] = None,
        num_perts: Optional[int] = None,
        go_graph_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize GearsConfig with GNN architecture and graph processing parameters.

        Args:
            hidden_size: Dimension of the hidden layers in GNN.
            num_go_gnn_layers: Number of GNN layers for the GO graph.
            num_gene_gnn_layers: Number of GNN layers for the gene graph.
            decoder_hidden_size: Hidden dimension of the MLP decoder.
            num_similar_genes_go_graph: Number of top similar genes to consider in GO graph.
            num_similar_genes_co_express_graph: Number of similar genes in co-expression graph.
            coexpress_threshold: Pearson correlation threshold for co-expression edges.
            go_graph_threshold: Semantic similarity threshold for GO graph edges.
            go_graph_top_k: Keep top K edges per node in GO graph.
            coexpress_top_k: Keep top K edges per node in co-expression graph.
            uncertainty: Whether to predict uncertainty using a heteroscedastic loss.
            no_perturb: If True, uses identity mapping (debugging mode).
            uncertainty_reg: Regularization weight for the uncertainty loss.
            direction_lambda: Weight for the directionality loss component.
            cell_fitness_pred: Whether to include a cell fitness prediction head.
            weight_bias_track: Internal flag for tracking bias weights during training.
            num_genes: Number of genes in the input expression data.
            num_perts: Number of unique perturbations in the vocabulary.
            go_graph_path: Path to pre-built GO graph file.
            **kwargs: Additional configuration arguments.
        """
        super().__init__(
            model_series='gears',
            model_name='',
            model_type='gnn',
            **kwargs
        )
        
        # Batch set attributes from locals
        self._set_all(locals())
