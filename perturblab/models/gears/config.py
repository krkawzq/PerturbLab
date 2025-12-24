"""Configuration definition for GEARS model."""

from perturblab.core.config import Config

__all__ = ["GEARSConfig"]


class GEARSConfig(Config):
    """Configuration for GEARS (Graph-Enhanced gene Activation and Repression Simulator) model.

    GEARS predicts transcriptional outcomes of genetic perturbations using graph neural networks.

    Args:
        num_genes (int):
            Number of genes in the dataset.
        num_perts (int):
            Number of unique perturbations.
        hidden_size (int, optional):
            Hidden dimension for embeddings and GNN layers. Defaults to 64.
        num_go_gnn_layers (int, optional):
            Number of GNN layers for Gene Ontology (GO) graph. Defaults to 1.
        num_gene_gnn_layers (int, optional):
            Number of GNN layers for gene co-expression graph. Defaults to 1.
        decoder_hidden_size (int, optional):
            Hidden dimension for gene-specific decoder. Defaults to 16.
        num_similar_genes_go_graph (int, optional):
            Number of maximum similar K genes in the GO graph. Defaults to 20.
        num_similar_genes_co_express_graph (int, optional):
            Number of maximum similar K genes in the co-expression graph. Defaults to 20.
        coexpress_threshold (float, optional):
            Pearson correlation threshold when constructing co-expression graph. Defaults to 0.4.
        uncertainty (bool, optional):
            Whether to enable uncertainty quantification mode. Defaults to False.
        uncertainty_reg (float, optional):
            Regularization term to balance uncertainty loss and prediction loss. Defaults to 1.0.
        direction_lambda (float, optional):
            Regularization term to balance direction loss and prediction loss. Defaults to 0.1.
        no_perturb (bool, optional):
            If True, predict no perturbation condition (baseline mode). Defaults to False.
    """

    num_genes: int
    num_perts: int
    hidden_size: int = 64
    num_go_gnn_layers: int = 1
    num_gene_gnn_layers: int = 1
    decoder_hidden_size: int = 16
    num_similar_genes_go_graph: int = 20
    num_similar_genes_co_express_graph: int = 20
    coexpress_threshold: float = 0.4
    uncertainty: bool = False
    uncertainty_reg: float = 1.0
    direction_lambda: float = 0.1
    no_perturb: bool = False
