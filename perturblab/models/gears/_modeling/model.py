"""GEARS: Graph-Enhanced gene Activation and Repression Simulator

This module implements the GEARS model for predicting transcriptional outcomes
of genetic perturbations using graph neural networks.

Model Overview:
    GEARS is a deep learning model that predicts cellular responses to single and
    multi-gene perturbations by leveraging:
      1. Gene co-expression networks
      2. Gene Ontology (GO) similarity graphs
      3. Gene-specific and cross-gene decoders

Architecture Components:

    1. Gene Embeddings
       - Base gene embeddings: Learnable embeddings for each gene (num_genes × hidden_size)
       - Positional embeddings: Enhanced by co-expression GNN

    2. Graph Neural Networks (GNNs)
       a. Co-expression GNN (Gene-Gene)
          - Input: Gene co-expression adjacency matrix
          - Purpose: Capture gene-gene relationships from expression patterns
          - Layers: Simplified Graph Convolution (SGConv)
          - Output: Positional embeddings (added to base gene embeddings with λ=0.2)
       b. GO Similarity GNN (Perturbation-Perturbation)
          - Input: GO term-based perturbation similarity graph
          - Purpose: Augment perturbation embeddings with functional relationships
          - Layers: SGConv layers
          - Output: Enhanced perturbation embeddings

    3. Perturbation Integration
       - Global perturbation embeddings: Learned for each perturbation
       - Enhanced via GO similarity GNN
       - Fused and added to corresponding genes in perturbed samples

    4. Decoder Architecture
       a. Shared MLP Decoder
          - Input: Perturbed gene embeddings
          - Architecture: [hidden_size, hidden_size×2, hidden_size]
          - Purpose: Extract shared features across all genes
       b. Gene-Specific Decoder
          - Individual parameters per gene: indv_w1, indv_b1 (first layer)
          - Purpose: Model gene-specific response patterns
       c. Cross-Gene Decoder
          - Input: All gene states in a sample
          - Architecture: MLP([num_genes, hidden_size, hidden_size])
          - Purpose: Capture gene-gene interactions in response
          - Final layer: indv_w2, indv_b2 (gene-specific parameters)

    5. Uncertainty Quantification (Optional)
       - Additional MLP head predicting log-variance
       - Enables uncertainty-aware predictions

Input data format:
    The model expects PyTorch Geometric Data objects with:
    x : torch.Tensor, shape (num_graphs × num_genes,)
        Baseline gene expression values (e.g., control condition).
        Flattened across all samples in the batch.
    pert_idx : list of list of int, length num_graphs
        Perturbation indices for each sample. Each element is a list of perturbed gene indices.
        Use [-1] for control samples (no perturbation).
        Example: [[0, 1], [2], [-1]]
          # Sample 1: genes 0,1 perturbed
          # Sample 2: gene 2 perturbed
          # Sample 3: control
    batch : torch.Tensor, shape (num_graphs × num_genes,)
        Batch assignment vector (0, 0, ..., 0, 1, 1, ..., 1, ...).
        Indicates which sample each gene expression belongs to.

Output format:
    predictions : torch.Tensor, shape (num_graphs, num_genes)
        Predicted gene expression values after perturbation.
    log_variance : torch.Tensor, shape (num_graphs, num_genes), optional
        Predicted log-variance (uncertainty) for each prediction (if uncertainty=True).

Model Workflow:
    1. Encode genes:
       - Lookup base gene embeddings
       - Apply co-expression GNN → positional embeddings
       - Combine: base_emb + 0.2 × pos_emb
    2. Encode perturbations:
       - Lookup perturbation embeddings
       - Apply GO similarity GNN
       - Aggregate perturbations per sample
    3. Integrate perturbation signals:
       - Add perturbation embeddings to corresponding gene embeddings
       - Apply batch normalization
    4. Decode to predictions:
       - Shared MLP decoder
       - Gene-specific first layer
       - Cross-gene MLP (captures interactions)
       - Gene-specific second layer
       - Add residual connection with baseline expression
    5. (Optional) Predict uncertainty:
       - Separate MLP head for log-variance

Key Design Principles:
    - Graph-based prior knowledge: Leverages biological networks (co-expression, GO)
    - Gene-specific parameters: Allows modeling unique gene responses
    - Cross-gene interactions: MLP aggregates information across all genes
    - Residual learning: Predicts changes relative to baseline
    - Scalability: Handles multi-gene perturbations and large gene sets

Hyperparameters:
    hidden_size (default: 64):           Embedding dimension
    num_go_gnn_layers (default: 1):      GO GNN depth
    num_gene_gnn_layers (default: 1):    Co-expression GNN depth
    decoder_hidden_size (default: 16):   Gene-specific decoder dimension
    uncertainty (default: False):        Enable uncertainty quantification

Training Considerations:
    1. Loss function: MSE on predicted vs. actual gene expression
    2. Regularization: Uncertainty regularization (if enabled)
    3. Direction loss: Optional regularization on prediction direction
    4. Graph construction:
        - Co-expression: Pearson correlation > threshold
        - GO similarity: Jaccard/overlap coefficient on GO terms

References:
    Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional
        outcomes of novel multigene perturbations with GEARS.
        Nat. Biotechnol. 42, 927–935 (2024).
        https://doi.org/10.1038/s41587-023-01905-6

    Original implementation:
        https://github.com/snap-stanford/GEARS

Copyright:
    Copyright (c) 2023 SNAP Lab, Stanford University
    Licensed under the MIT License

Adapted for PerturbLab with config-based API and modular design.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SGConv

from perturblab.models.gears.config import GEARSConfig
from perturblab.models.gears.io import GEARSInput, GEARSOutput

warnings.filterwarnings("ignore")

__all__ = ["MLP", "GEARS_Model"]


class MLP(nn.Module):
    """Multi-layer perceptron with optional batch normalization.

    This helper block is used throughout GEARS.

    Args:
        sizes (List[int]): List containing layer dimensions ([input_dim, hidden_dim, ...]).
        batch_norm (bool, optional): If True, applies BatchNorm1d after each layer except last. Default: True.
        last_layer_act (str, optional): Type for last activation ("linear" or activation name). Default: 'linear'.
    """

    def __init__(self, sizes: List[int], batch_norm: bool = True, last_layer_act: str = "linear"):
        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[s], sizes[s + 1]))
            if batch_norm and s < len(sizes) - 2:
                layers.append(nn.BatchNorm1d(sizes[s + 1]))
            if s < len(sizes) - 2 or last_layer_act.lower() != "linear":
                layers.append(nn.ReLU())
        # Remove extraneous ReLU if linear is requested last
        if last_layer_act.lower() == "linear" and isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class GEARSModel(nn.Module):
    r"""GEARS model for predicting transcriptional outcomes after genetic perturbation.

    This model uses graph neural networks over gene co-expression and Gene Ontology graphs to
    augment gene and perturbation embeddings, supporting multi-gene perturbations
    and gene/cross-gene decoders.

    Args:
        config (GEARSConfig):           Configuration with all model hyperparameters.
        G_coexpress (Tensor):           Gene co-expression adjacency matrix (edge indices).
        G_coexpress_weight (Tensor):    Weights for G_coexpress edges.
        G_go (Tensor):                  Gene Ontology similarity graph (edge indices).
        G_go_weight (Tensor):           Weights for G_go edges.
        device (str or torch.device):   Device to place all tensors. Default: 'cuda'.

    """

    def __init__(
        self,
        config: GEARSConfig,
        G_coexpress: Tensor,
        G_coexpress_weight: Tensor,
        G_go: Tensor,
        G_go_weight: Tensor,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device

        self.num_genes = config.num_genes
        self.num_perts = config.num_perts
        hidden_size = config.hidden_size
        self.uncertainty = config.uncertainty
        self.num_layers = config.num_go_gnn_layers
        self.indv_out_hidden_size = config.decoder_hidden_size
        self.num_layers_gene_pos = config.num_gene_gnn_layers
        self.no_perturb = config.no_perturb
        self.pert_emb_lambda = 0.2

        # Graph buffers (could be registered as buffer, legacy for compatibility)
        self.G_coexpress = G_coexpress.to(self.device)
        self.G_coexpress_weight = G_coexpress_weight.to(self.device)
        self.G_sim = G_go.to(self.device)
        self.G_sim_weight = G_go_weight.to(self.device)

        # Embeddings
        self.pert_w = nn.Linear(1, hidden_size)
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # Feature transformation
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act="ReLU")
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act="ReLU")

        # Co-expression GNN
        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = nn.ModuleList(
            [SGConv(hidden_size, hidden_size, 1) for _ in range(self.num_layers_gene_pos)]
        )

        # GO similarity GNN
        self.sim_layers = nn.ModuleList(
            [SGConv(hidden_size, hidden_size, 1) for _ in range(self.num_layers)]
        )

        # Decoders
        self.recovery_w = MLP([hidden_size, hidden_size * 2, hidden_size], last_layer_act="linear")

        self.indv_w1 = nn.Parameter(torch.empty(self.num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.empty(self.num_genes, 1))
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        self.cross_gene_state = MLP([self.num_genes, hidden_size, hidden_size])

        self.indv_w2 = nn.Parameter(torch.empty(1, self.num_genes, hidden_size + 1))
        self.indv_b2 = nn.Parameter(torch.empty(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        # Normalization layers
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        # Optional uncertainty head
        if self.uncertainty:
            self.uncertainty_w = MLP(
                [hidden_size, hidden_size * 2, hidden_size, 1], last_layer_act="linear"
            )

    def forward(self, inputs: Union[GEARSInput, Any]) -> GEARSOutput:
        """Forward pass for GEARS_Model.

        Supports both structured GEARSInput and legacy PyG Data objects.

        Args:
            inputs (GEARSInput or torch_geometric.data.Data):
                Input data containing fields:
                  - gene_expression (Tensor): Flattened input, shape (num_graphs * num_genes,)
                  - pert_idx (list[list[int]]): List of perturbed gene indices for each sample
                  - graph_batch_indices (Tensor): Batch assignment for each gene (optional)

        Returns:
            GEARSOutput: Output structure containing:
                predictions (Tensor): Predicted gene expression (num_graphs, num_genes)
                log_variance (Tensor, optional): Predicted log-variance (num_graphs, num_genes)
        """
        # 1. Unpack Inputs
        if isinstance(inputs, GEARSInput):
            x = inputs.gene_expression
            pert_idx = inputs.pert_idx
            batch_idx = inputs.graph_batch_indices
        else:
            x = inputs.x
            pert_idx = inputs.pert_idx
            batch_idx = inputs.batch

        # 2. Early exit if no perturbation mode
        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return GEARSOutput(predictions=torch.stack(out))

        # 3. Determine number of graphs (i.e., batch size)
        num_graphs = len(torch.unique(batch_idx)) if batch_idx is not None else 1

        # 4. Gene Embeddings & Co-expression GNN
        gene_indices = torch.arange(self.num_genes, device=self.device).long().repeat(num_graphs)

        emb = self.gene_emb(gene_indices)
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)

        pos_emb = self.emb_pos(gene_indices)
        for idx, layer in enumerate(self.layers_emb_pos):
            pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                pos_emb = torch.relu(pos_emb)
        base_emb = base_emb + self.pert_emb_lambda * pos_emb
        base_emb = self.emb_trans_v2(base_emb)

        # 5. Perturbation embedding lookup & graph propagation
        pert_index_list = []
        for batch_i, perts in enumerate(pert_idx):
            for p in perts:
                if p != -1:
                    pert_index_list.append([batch_i, p])

        if pert_index_list:
            pert_index_tensor = torch.tensor(pert_index_list, device=self.device).T
        else:
            pert_index_tensor = torch.empty((2, 0), device=self.device, dtype=torch.long)

        pert_global_indices = torch.arange(self.num_perts, device=self.device).long()
        pert_global_emb = self.pert_emb(pert_global_indices)
        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = torch.relu(pert_global_emb)

        # 6. Fuse perturbations into the relevant gene embeddings
        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)
        if pert_index_tensor.shape[1] > 0:
            batch_indices = pert_index_tensor[0]
            pert_genes = pert_index_tensor[1]
            active_pert_embs = pert_global_emb[pert_genes]  # (N_active_perts, hidden)

            if active_pert_embs.shape[0] == 1:
                fused_pert = self.pert_fuse(
                    torch.stack([active_pert_embs[0], active_pert_embs[0]])
                )[:1]
            else:
                fused_pert = self.pert_fuse(active_pert_embs)
            for k in range(pert_index_tensor.shape[1]):
                b_idx = batch_indices[k]
                p_gene_idx = pert_genes[k]
                base_emb[b_idx, p_gene_idx] = base_emb[b_idx, p_gene_idx] + fused_pert[k]

        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        base_emb = self.bn_pert_base(base_emb)

        # 7. Decoding sequence (shared, gene-specific, cross-gene, second gene-specific)
        base_emb = self.transform(base_emb)
        out = self.recovery_w(base_emb)
        out = out.reshape(num_graphs, self.num_genes, -1)

        # First gene-specific decoder layer
        out = out.unsqueeze(-1) * self.indv_w1
        w = torch.sum(out, dim=2)  # (Batch, Genes, 1)
        out = w + self.indv_b1  # (Batch, Genes, 1)

        # Cross-gene decoder
        cross_gene_embed = self.cross_gene_state(
            out.reshape(num_graphs, self.num_genes, -1).squeeze(2)
        )
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)
        cross_gene_embed = cross_gene_embed.reshape(num_graphs, self.num_genes, -1)

        cross_gene_out = torch.cat([out, cross_gene_embed], dim=2)  # (Batch, Genes, Hidden+1)

        # Second gene-specific decoder layer
        cross_gene_out = cross_gene_out * self.indv_w2  # broadcasting
        cross_gene_out = torch.sum(cross_gene_out, dim=2)  # (Batch, Genes)
        out = cross_gene_out + self.indv_b2

        # Residual with input expression
        out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)

        predictions = out.reshape(num_graphs, self.num_genes)

        # 8. Uncertainty head (if enabled)
        log_var = None
        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            log_var = out_logvar.reshape(num_graphs, self.num_genes)

        return GEARSOutput(predictions=predictions, log_variance=log_var)
