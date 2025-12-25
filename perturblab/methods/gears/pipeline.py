"""GEARS Method Pipeline Interface.

This module provides high-level API for GEARS training, handling graph construction,
model initialization, trainer setup, and data loading with proper gene alignment.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from perturblab.models import MODELS
from perturblab.models.gears import GEARSConfig, GEARSModel
from perturblab.types import GeneGraph, GeneVocab, PerturbationData
from perturblab.utils import get_logger

from .graph import build_coexpression_graph, build_go_similarity_graph
from .loss import build_loss
from .processing import build_collate_fn

logger = get_logger()

__all__ = [
    "build_graphs",
    "build_model",
    "build_trainer",
    "create_loader",
    "build_training_components",
    "build_training",
]


def build_graphs(
    data: PerturbationData,
    gene_vocab: GeneVocab | None = None,
    go_threshold: float = 0.1,
    go_k: int | None = None,
    coexp_threshold: float = 0.4,
    coexp_k: int = 20,
    use_control_only: bool = True,
) -> tuple[GeneGraph, GeneGraph, PerturbationData]:
    """Constructs knowledge graphs with optional gene alignment.

    If gene_vocab is provided, aligns the data to match the vocabulary and builds
    graphs on the aligned gene set. This ensures consistency between data and graphs.

    Args:
        data: Input PerturbationData.
        gene_vocab: Optional GeneVocab for gene alignment. If provided, data will
            be aligned to this vocabulary. If None, uses data's gene list.
        go_threshold: Similarity threshold for GO graph edges. Defaults to 0.1.
        go_k: Top-k filtering for GO graph nodes. Defaults to None.
        coexp_threshold: Pearson correlation threshold for co-expression. Defaults to 0.4.
        coexp_k: Top-k neighbors per gene for co-expression. Defaults to 20.
        use_control_only: Use only control cells for co-expression. Defaults to True.

    Returns:
        Tuple of (go_graph, coexp_graph, aligned_data).
            - If gene_vocab provided: data is aligned, graphs use vocab genes
            - If gene_vocab is None: data unchanged, graphs use data's genes

    Examples:
        >>> from perturblab.methods.gears import build_graphs
        >>> from perturblab.types import GeneVocab
        >>>
        >>> # Without alignment (uses data's genes)
        >>> go_graph, coexp_graph, data = build_graphs(data)
        >>>
        >>> # With alignment (forces consistency)
        >>> target_vocab = GeneVocab(['TP53', 'KRAS', 'MYC', ...])
        >>> go_graph, coexp_graph, aligned_data = build_graphs(
        ...     data,
        ...     gene_vocab=target_vocab
        ... )
        >>> # aligned_data.gene_names matches target_vocab.itos
        >>> # Graphs are built on target_vocab genes
    """
    logger.info("🏗️  Building GEARS knowledge graphs...")

    # Align data if vocab provided
    if gene_vocab is not None:
        logger.info(f"Aligning data to provided vocabulary ({len(gene_vocab)} genes)...")
        aligned_data = data.align_genes(gene_vocab.itos, fill_value=0.0)
        genes_for_graph = gene_vocab
    else:
        aligned_data = data
        genes_for_graph = list(data.gene_names)

    # Build GO similarity graph
    go_graph = build_go_similarity_graph(
        gene_vocab=genes_for_graph,
        similarity_metric="jaccard",
        threshold=go_threshold,
        k=go_k,
        show_progress=False,
    )

    # Build co-expression graph
    coexp_graph = build_coexpression_graph(
        aligned_data,
        threshold=coexp_threshold,
        k=coexp_k,
        vocab=gene_vocab,  # Pass vocab for consistency
        use_control_only=use_control_only,
    )

    logger.info(
        f"✅ Graphs ready: "
        f"GO({go_graph.n_nodes} nodes, {go_graph.graph.n_unique_edges} edges), "
        f"CoExp({coexp_graph.n_nodes} nodes, {coexp_graph.graph.n_unique_edges} edges)"
    )

    return go_graph, coexp_graph, aligned_data


def build_model(
    data: PerturbationData,
    go_graph: GeneGraph,
    coexp_graph: GeneGraph,
    hidden_size: int = 64,
    num_go_gnn_layers: int = 1,
    num_gene_gnn_layers: int = 1,
    decoder_hidden_size: int = 16,
    uncertainty: bool = False,
    device: str | torch.device = "cuda",
    **kwargs,
) -> GEARSModel:
    """Instantiates GEARS model with graph tensors.

    Args:
        data: PerturbationData (should be aligned with graphs).
        go_graph: Pre-built GO GeneGraph.
        coexp_graph: Pre-built Co-expression GeneGraph.
        hidden_size: Embedding dimension. Defaults to 64.
        num_go_gnn_layers: GO GNN depth. Defaults to 1.
        num_gene_gnn_layers: Gene GNN depth. Defaults to 1.
        decoder_hidden_size: Gene-specific decoder dim. Defaults to 16.
        uncertainty: Enable uncertainty quantification. Defaults to False.
        device: Computational device. Defaults to 'cuda'.
        **kwargs: Extra GEARSConfig arguments.

    Returns:
        Initialized GEARS model on specified device.

    Examples:
        >>> model = build_model(data, go_graph, coexp_graph, hidden_size=64)
    """
    logger.info(f"🏗️  Initializing GEARS model (dim={hidden_size}, device={device})...")

    # Validate alignment
    if len(data.gene_names) != go_graph.n_nodes:
        logger.warning(
            f"Data genes ({len(data.gene_names)}) != GO graph nodes ({go_graph.n_nodes}). "
            f"This may cause dimension mismatch."
        )

    # Create config
    config = GEARSConfig(
        num_genes=len(data.gene_names),
        num_perts=data.n_perturbations,
        hidden_size=hidden_size,
        num_go_gnn_layers=num_go_gnn_layers,
        num_gene_gnn_layers=num_gene_gnn_layers,
        decoder_hidden_size=decoder_hidden_size,
        uncertainty=uncertainty,
        **kwargs,
    )

    # Convert GeneGraph to PyG edge format
    def _to_pyg_format(gene_graph: GeneGraph):
        edges = gene_graph.graph.edges
        edge_index = torch.tensor(edges[:, :2].T, dtype=torch.long)
        edge_weight = torch.tensor(edges[:, 2], dtype=torch.float32)
        return edge_index, edge_weight

    go_edge_index, go_edge_weight = _to_pyg_format(go_graph)
    coexp_edge_index, coexp_edge_weight = _to_pyg_format(coexp_graph)

    # Build model
    model = MODELS.build(
        "GEARS.default",
        config=config,
        G_go=go_edge_index,
        G_go_weight=go_edge_weight,
        G_coexpress=coexp_edge_index,
        G_coexpress_weight=coexp_edge_weight,
        device=device,
    )

    logger.info(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def build_trainer(
    model: GEARSModel,
    data: PerturbationData,
    optimizer: optim.Optimizer | None = None,
    learning_rate: float = 1e-3,
    loss_type: str = "standard",
    direction_lambda: float = 1e-3,
    uncertainty_reg: float = 1.0,
    device: str | torch.device = "cuda",
    distributed: bool = False,
    **trainer_kwargs,
) -> Any:
    """Configures trainer with loss function and optimizer.

    Args:
        model: GEARS model to train.
        data: PerturbationData (for extracting control expression).
        optimizer: Optional pre-configured optimizer. If None, uses Adam.
        learning_rate: Learning rate for new optimizer. Defaults to 1e-3.
        loss_type: 'standard' or 'uncertainty'. Defaults to 'standard'.
        direction_lambda: Direction loss weight. Defaults to 1e-3.
        uncertainty_reg: Uncertainty regularization weight. Defaults to 1.0.
        device: Device. Defaults to 'cuda'.
        distributed: Use DistributedTrainer. Defaults to False.
        **trainer_kwargs: Additional trainer arguments.

    Returns:
        Trainer or DistributedTrainer instance.

    Examples:
        >>> trainer = build_trainer(model, data, learning_rate=1e-3)
    """
    logger.info("⚙️  Configuring trainer...")

    # Compute control expression for direction loss
    if data.is_control.any():
        ctrl_expr = torch.tensor(
            data.adata.X[data.is_control].mean(axis=0).A1, dtype=torch.float32, device=device
        )
    else:
        logger.warning("⚠️  No control cells found, direction loss disabled.")
        ctrl_expr = None

    # Build loss function
    loss_fn = build_loss(
        loss_type=loss_type,
        direction_lambda=direction_lambda,
        uncertainty_reg=uncertainty_reg if loss_type == "uncertainty" else None,
        ctrl_expression=ctrl_expr,
    )

    # Build optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate trainer
    if distributed:
        from perturblab.engine import DistributedTrainer

        trainer = DistributedTrainer(
            model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, **trainer_kwargs
        )
    else:
        from perturblab.engine import Trainer

        trainer = Trainer(
            model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, **trainer_kwargs
        )

    logger.info("✅ Trainer configured")
    return trainer


def create_loader(
    data: PerturbationData,
    split: str | None = None,
    indices: list[int] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """Creates DataLoader with automatic caching for performance.

    CellData/PerturbationData can be used directly as PyTorch Dataset.
    This function handles split filtering, cache enabling, and collate function creation.

    Args:
        data: Source PerturbationData.
        split: Split name ('train', 'val', 'test'). Filters by obs['split'].
        indices: Explicit indices. Overrides split if provided.
        batch_size: Batch size. Defaults to 32.
        shuffle: Whether to shuffle. Defaults to True.
        num_workers: Data loading workers. Defaults to 0.
        distributed: Use DistributedSampler. Defaults to False.

    Returns:
        PyTorch DataLoader with batch_dict format.

    Examples:
        >>> train_loader = create_loader(data, split='train', batch_size=32)
        >>> val_loader = create_loader(data, split='val', shuffle=False)
    """
    # Filter to split
    if indices is not None:
        subset_data = data[indices]
    elif split is not None:
        if "split" not in data.adata.obs:
            raise ValueError("Split column not found. Run data.split() first.")

        mask = data.adata.obs["split"] == split
        if not mask.any():
            logger.warning(f"⚠️  Split '{split}' is empty")
            return DataLoader([], batch_size=batch_size)

        subset_data = data[mask.values]
    else:
        subset_data = data

    # Enable cache for performance
    if not subset_data.is_cache_enabled:
        subset_data.enable_cache(verbose=False)

    # Create collate function
    vocab = GeneVocab(list(subset_data.gene_names))
    collate_fn = build_collate_fn(vocab)

    # Create DataLoader
    if distributed:
        from perturblab.engine import DistributedTrainer

        return DistributedTrainer.create_distributed_loader(
            subset_data,
            batch_size=batch_size,
            is_train=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    else:
        return DataLoader(
            subset_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def build_training_components(
    data: PerturbationData,
    gene_vocab: GeneVocab | None = None,
    hidden_size: int = 64,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    distributed: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Builds all components for GEARS training with optional gene alignment.

    One-stop function that creates graphs, model, trainer, and dataloaders.
    If gene_vocab is provided, aligns data to ensure consistency.

    Args:
        data: PerturbationData (must have 'split' column for train/val).
        gene_vocab: Optional GeneVocab for forced alignment. If None, uses data's genes.
        hidden_size: Model embedding size. Defaults to 64.
        batch_size: Training batch size. Defaults to 32.
        learning_rate: Optimizer learning rate. Defaults to 1e-3.
        device: Device. Defaults to 'cuda'.
        distributed: Enable DDP. Defaults to False.
        **kwargs: Additional arguments:
            - go_threshold, coexp_threshold, coexp_k
            - num_go_gnn_layers, num_gene_gnn_layers
            - direction_lambda, uncertainty_reg
            - num_workers

    Returns:
        Dictionary with:
            - 'model': GEARS model
            - 'trainer': Trainer
            - 'train_loader': Training DataLoader
            - 'val_loader': Validation DataLoader
            - 'go_graph': GO similarity graph
            - 'coexp_graph': Co-expression graph
            - 'data': Aligned data (if gene_vocab provided) or original data

    Examples:
        >>> from perturblab.methods.gears import build_training_components
        >>>
        >>> # Simple usage
        >>> components = build_training_components(data, hidden_size=64)
        >>> trainer = components['trainer']
        >>> trainer.fit(components['train_loader'], components['val_loader'], epochs=20)
        >>>
        >>> # With gene alignment
        >>> from perturblab.types import GeneVocab
        >>> target_vocab = GeneVocab(reference_genes)
        >>> components = build_training_components(
        ...     data,
        ...     gene_vocab=target_vocab,  # Forces alignment
        ...     hidden_size=64
        ... )
    """
    logger.info("🏗️  Building GEARS training pipeline...")

    # 1. Build graphs (with alignment if vocab provided)
    go_graph, coexp_graph, aligned_data = build_graphs(
        data,
        gene_vocab=gene_vocab,
        go_threshold=kwargs.get("go_threshold", 0.1),
        go_k=kwargs.get("go_k", None),
        coexp_threshold=kwargs.get("coexp_threshold", 0.4),
        coexp_k=kwargs.get("coexp_k", 20),
        use_control_only=kwargs.get("use_control_only", True),
    )

    # 2. Build model
    model = build_model(
        aligned_data,
        go_graph,
        coexp_graph,
        hidden_size=hidden_size,
        num_go_gnn_layers=kwargs.get("num_go_gnn_layers", 1),
        num_gene_gnn_layers=kwargs.get("num_gene_gnn_layers", 1),
        decoder_hidden_size=kwargs.get("decoder_hidden_size", 16),
        uncertainty=kwargs.get("uncertainty", False),
        device=device,
    )

    # 3. Build trainer
    trainer = build_trainer(
        model,
        aligned_data,
        learning_rate=learning_rate,
        loss_type=kwargs.get("loss_type", "standard"),
        direction_lambda=kwargs.get("direction_lambda", 1e-3),
        uncertainty_reg=kwargs.get("uncertainty_reg", 1.0),
        device=device,
        distributed=distributed,
    )

    # 4. Create dataloaders
    num_workers = kwargs.get("num_workers", 0)

    train_loader = create_loader(
        aligned_data,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        distributed=distributed,
        num_workers=num_workers,
    )

    val_loader = create_loader(
        aligned_data,
        split="val",
        batch_size=batch_size * 2,
        shuffle=False,
        distributed=distributed,
        num_workers=num_workers,
    )

    logger.info("✅ All training components built!")

    return {
        "model": model,
        "trainer": trainer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "go_graph": go_graph,
        "coexp_graph": coexp_graph,
        "data": aligned_data,  # Return aligned data
    }


# Alias for convenience
build_training = build_training_components
