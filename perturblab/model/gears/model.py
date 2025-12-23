import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from ...data import GeneGraph, PerturbationData
from ..base import PerturbationModel
from .config import GearsConfig
from .source.model import GEARS_Model
from .source.utils import loss_fct, uncertainty_loss_fct

logger = logging.getLogger(__name__)


class GearsModel(PerturbationModel):
    """GEARS Model wrapper for single-cell perturbation prediction.

    Integrates Gene Ontology (GO) and co-expression graphs with the core
    GEARS architecture for graph-based perturbation response modeling.
    """

    def __init__(
        self,
        config: GearsConfig,
        gene_list: List[str],
        pert_list: List[str],
        go_graph: GeneGraph,
        co_graph: GeneGraph,
        pert_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings: Optional[torch.Tensor] = None,
        device: str = 'cpu',
    ):
        """Initializes the GearsModel.

        Args:
            config: GEARS configuration object.
            gene_list: List of gene names.
            pert_list: List of perturbation names.
            go_graph: Gene Ontology graph object.
            co_graph: Co-expression graph object.
            pert_embeddings: Optional pre-trained perturbation embeddings.
            gene_embeddings: Optional pre-trained gene embeddings.
            device: Device to run the model on.
        """
        super().__init__(config)

        self.gene_list = gene_list
        self.pert_list = pert_list

        # Align graphs with gene/perturbation lists
        if co_graph.gene_list != self.gene_list:
            co_graph = co_graph.subset(self.gene_list)
        self.co_graph = co_graph
        
        if go_graph.gene_list != self.pert_list:
            go_graph = go_graph.subset(self.pert_list)
        self.go_graph = go_graph

        # Use as_tensor to support both Tensor and Numpy inputs
        G_coexpress = torch.as_tensor(co_graph.edge_index, dtype=torch.long)
        if co_graph.edge_weight is not None:
            G_coexpress_weight = torch.as_tensor(co_graph.edge_weight, dtype=torch.float)
        else:
            num_edges = G_coexpress.shape[1] if G_coexpress.shape[1] > 0 else 0
            G_coexpress_weight = torch.ones(num_edges, dtype=torch.float32)

        G_go = torch.as_tensor(go_graph.edge_index, dtype=torch.long)
        if go_graph.edge_weight is not None:
            G_go_weight = torch.as_tensor(go_graph.edge_weight, dtype=torch.float)
        else:
            num_edges = G_go.shape[1] if G_go.shape[1] > 0 else 0
            G_go_weight = torch.ones(num_edges, dtype=torch.float32)
            
        # Initialize core GEARS model
        self.gears_model = GEARS_Model(dict(
            num_genes=len(self.gene_list),
            num_perts=len(self.pert_list),
            hidden_size=config.hidden_size,
            uncertainty=config.uncertainty,
            num_go_gnn_layers=config.num_go_gnn_layers,
            decoder_hidden_size=config.decoder_hidden_size,
            num_gene_gnn_layers=config.num_gene_gnn_layers,
            no_perturb=config.no_perturb,
            G_coexpress=G_coexpress,
            G_coexpress_weight=G_coexpress_weight,
            G_go=G_go,
            G_go_weight=G_go_weight,
            device=device,
        ))

        # Track embedding layer types
        self.pert_embedding_layer_type = "default"
        self.gene_embedding_layer_type = "default"

        # Set pre-trained embeddings if provided
        if pert_embeddings is not None:
            self.set_pert_embeddings(pert_embeddings, trainable=True)
        if gene_embeddings is not None:
            self.set_gene_embeddings(gene_embeddings, trainable=True)
            
    def to(self, device: str):
        """Moves the model to a device."""
        self.gears_model.to(device)
        return self

    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        self.gears_model.train(mode)
        return self
    
    def eval(self):
        """Sets the model to evaluation mode."""
        self.gears_model.eval()
        return self

    @classmethod
    def init_from_dataset(
        cls,
        dataset: PerturbationData,
        config: GearsConfig,
        gene_list: Optional[List[str]] = None,
        pert_list: Optional[List[str]] = None,
    ):
        """
        Initializes the model directly from a PerturbationData dataset.

        Args:
            dataset: The dataset containing AnnData.
            config: GEARS configuration.
            device: Computation device.
            gene_list: Optional subset of genes to use.
            pert_list: Optional subset of perturbations to use.

        Returns:
            An instance of GearsModel.
        """
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")

        dataset_gene_list = dataset.adata.var["gene_name"].tolist()
        
        # Extract perturbation list
        if "pert_list" in dataset.adata.uns:
            dataset_pert_list = dataset.adata.uns["pert_list"]
        else:
            all_conditions = dataset.adata.obs["condition"].unique()
            pert_set = set()
            for cond in all_conditions:
                if cond != "ctrl":
                    perts = [p for p in cond.split("+") if p != "ctrl"]
                    pert_set.update(perts)
            dataset_pert_list = sorted(list(pert_set))

        # Filter gene list
        if gene_list is None:
            gene_list = dataset_gene_list
        else:
            gene_list = [g for g in gene_list if g in dataset_gene_list]
            if not gene_list:
                raise ValueError("No genes from provided gene_list found in dataset.")
            logger.info(f"Using {len(gene_list)} genes from intersection.")

        # Filter perturbation list
        if pert_list is None:
            pert_list = dataset_pert_list
        else:
            pert_list = [p for p in pert_list if p in dataset_pert_list]
            if not pert_list:
                raise ValueError("No perturbations from provided pert_list found in dataset.")
            logger.info(f"Using {len(pert_list)} perturbations from intersection.")

        # Construct graphs
        go_graph = GeneGraph.from_go(
            pert_list,
            path=config.go_graph_path,
            threshold=config.go_graph_threshold,
            top_k=config.go_graph_top_k,
        )

        co_graph = GeneGraph.from_coexpression(
            dataset.adata,
            gene_list,
            threshold=config.coexpress_threshold,
            top_k=config.coexpress_top_k,
        )

        return cls(
            config=config,
            gene_list=gene_list,
            pert_list=pert_list,
            go_graph=go_graph,
            co_graph=co_graph,
        )
        
    @staticmethod
    def get_dataloader(
        dataset: PerturbationData,
        batch_size: int,
        split: Optional[str] = 'train', # None or 'all' to return all splits
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> Dict[str, DataLoader]:
        """
        Creates a Dictionary of optimized PyG DataLoaders.
        
        Args:
            dataset: Input perturbation dataset.
            batch_size: Batch size.
            split: 
                - None: Returns loaders for ALL splits found in dataset.obs['split'].
                - 'all': Returns a single loader containing the ENTIRE dataset.
                - 'train'/'val'/'test': Returns a loader for the specific split.
            shuffle: Whether to shuffle the data.
            drop_last: Whether to drop the last incomplete batch.
            num_workers: Number of worker processes.
            
        Returns:
            Dict[str, DataLoader]: A dictionary mapping split names to DataLoaders.
                                   e.g., {'train': loader, 'val': loader}
                                   Keys for empty/invalid splits are omitted.
        """
        from torch.utils.data import Dataset as TorchDataset
        # Ensure we use the PyG DataLoader for proper collation of Data objects
        from torch_geometric.loader import DataLoader
        
        # --- 1. Data Validation and Preprocessing ---
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")
        
        if "rank_genes_groups_cov_all" not in dataset.adata.uns:
            logger.warning("DE genes not found. Computing DE genes...")
            dataset.compute_de_genes()
        
        if "ctrl_indices" not in dataset.adata.obsm:
            raise ValueError("Control cell pairing not found. Please call dataset.pair_cells() first.")
        
        # --- 2. Prepare Shared Metadata (Once for all splits) ---
        gene_list = dataset.adata.var["gene_name"].tolist()
        gene_name_to_idx = {name: i for i, name in enumerate(gene_list)}
        
        if "pert_list" not in dataset.adata.uns:
            dataset.adata.uns["pert_list"] = sorted(list({
                p for c in dataset.adata.obs["condition"].unique() if c != "ctrl"
                for p in c.split("+") if p != "ctrl"
            }))
        
        # Build DE Gene Map
        de_gene_map = {}
        if "rank_genes_groups_cov_all" in dataset.adata.uns:
            rank_data = dataset.adata.uns["rank_genes_groups_cov_all"]
            top_n = dataset.adata.uns.get("top_de_n", 20)
            
            # Handle dict or structured array
            if isinstance(rank_data, dict):
                iterator = rank_data.items()
            else:
                iterator = ((n, rank_data["names"][n]) for n in rank_data["names"].dtype.names)
            
            for cond, genes in iterator:
                if cond in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
                    continue
                genes_flat = genes if isinstance(genes, (list, np.ndarray)) else [genes]
                de_gene_map[cond] = [
                    gene_name_to_idx.get(g, -1) 
                    for g in genes_flat[:top_n] 
                    if g in gene_name_to_idx
                ]

        # --- 3. Define Internal Dataset Class ---
        # Modified to accept specific indices for flexibility
        class _LazyGearsDataset(TorchDataset):
            def __init__(self, cell_indices, conditions):
                # References to large shared data (Closure)
                self.X = dataset.adata.X
                self.ctrl_indices = dataset.adata.obsm["ctrl_indices"]
                self.num_samples = self.ctrl_indices.shape[1]
                
                # Split-specific data
                self.cell_indices = cell_indices
                self.conditions = conditions
                
                # Metadata
                self.gene_name_to_idx = gene_name_to_idx
                self.de_gene_map = de_gene_map
                self.is_sparse = hasattr(self.X, "toarray") or hasattr(self.X, "A")
                self.n_cells = len(self.cell_indices)
            
            def __len__(self):
                return self.n_cells * self.num_samples
            
            def __getitem__(self, idx):
                # Math Indexing: idx = i * num_samples + j
                cell_ptr = idx // self.num_samples
                sample_ptr = idx % self.num_samples
                
                abs_cell_idx = self.cell_indices[cell_ptr]
                condition = self.conditions[cell_ptr]
                
                # 1. Y (Perturbed)
                if self.is_sparse:
                    y = self.X[abs_cell_idx].toarray().flatten()
                else:
                    y = self.X[abs_cell_idx]
                
                # 2. X (Control)
                abs_ctrl_idx = self.ctrl_indices[abs_cell_idx, sample_ptr]
                if self.is_sparse:
                    x = self.X[abs_ctrl_idx].toarray().flatten()
                else:
                    x = self.X[abs_ctrl_idx]
                
                # 3. Pert Index
                if condition == "ctrl":
                    pert_idx = [-1]
                else:
                    pert_idx = [
                        self.gene_name_to_idx.get(g, -1) 
                        for g in condition.split("+") 
                        if g != "ctrl"
                    ]
                    if not pert_idx or all(i == -1 for i in pert_idx):
                        pert_idx = [-1]
                
                # 4. DE Index
                de_idx = self.de_gene_map.get(condition, [-1] * 20)
                
                return Data(
                    x=torch.tensor(x, dtype=torch.float).unsqueeze(1),
                    y=torch.tensor(y, dtype=torch.float).unsqueeze(1),
                    pert_idx=torch.tensor(pert_idx, dtype=torch.long),
                    pert=condition,
                    de_idx=torch.tensor(de_idx, dtype=torch.long)
                )

        # --- 4. Determine Target Splits ---
        loaders = {}
        target_splits = []

        if split is None:
            # Auto-detect all splits
            if "split" in dataset.adata.obs:
                target_splits = dataset.adata.obs["split"].unique().tolist()
            else:
                logger.warning("No 'split' column found in adata.obs. Returning empty dict.")
                return {}
        else:
            # User requested specific split (or 'all')
            target_splits = [split]

        # --- 5. Iterate and Create Loaders ---
        for target_split in target_splits:
            # A. Filter Indices
            if target_split == "all":
                # Special case: use everything
                indices = dataset.adata.obs["index_col"].values
                conds = dataset.adata.obs["condition"].values
                split_name = "all"
            else:
                # Standard split filtering
                if "split" not in dataset.adata.obs:
                    logger.warning(f"Skipping '{target_split}': 'split' column missing.")
                    continue
                
                mask = dataset.adata.obs["split"] == target_split
                
                # Check for empty split
                if not np.any(mask):
                    logger.warning(f"Split '{target_split}' contains no data. Skipping.")
                    continue
                
                indices = dataset.adata.obs.loc[mask, "index_col"].values
                conds = dataset.adata.obs.loc[mask, "condition"].values
                split_name = str(target_split)

            # Double check length
            if len(indices) == 0:
                continue

            # B. Create Dataset & Loader
            logger.info(f"Creating loader for '{split_name}' with {len(indices)} cells...")
            ds = _LazyGearsDataset(indices, conds)
            
            loaders[split_name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
            )

        if not loaders:
            logger.warning("No valid data loaders were created (check split names or data availability).")

        return loaders

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            batch: Input batch data.
            
        Returns:
            Dictionary with 'pred' key and optionally 'logvar' key.
        """
        if self.config.uncertainty:
            pred, logvar = self.gears_model(batch)
            return {
                "pred": pred,
                "logvar": logvar,
            }
        else:
            pred = self.gears_model(batch)
            return {
                "pred": pred,
            }

    def compute_loss(
        self,
        batch,
        ctrl_expression: Optional[torch.Tensor] = None,
        dict_filter: Optional[Dict] = None,
        output_dict: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Computes the loss for the given batch.

        Args:
            batch: Input batch of data.
            ctrl_expression: Baseline control expression vector.
            dict_filter: Dictionary for filtering (e.g., DE genes).
            output_dict: Pre-computed model outputs.
            
        Returns:
            Dictionary with 'loss' key containing the computed loss.
        """
        if output_dict is None:
            output_dict = self.forward(batch)

        pred = output_dict["pred"]

        # Get ground truth
        if not hasattr(batch, "y") or batch.y is None:
            raise ValueError("Batch must contain 'y' attribute (ground truth values)")

        y = batch.y
        # Reshape if necessary [n_nodes, 1] -> [n_samples, n_genes]
        if y.dim() == 2 and y.shape[1] == 1:
            num_samples = len(batch.batch.unique())
            y = y.reshape(num_samples, -1)

        # Get perturbation categories
        perts = batch.pert if hasattr(batch, "pert") else ["ctrl"] * len(batch.batch.unique())
        if isinstance(perts, torch.Tensor):
            perts = perts.tolist()
        elif not isinstance(perts, (list, tuple)):
            perts = [perts]

        # Calculate or use provided ctrl_expression
        if ctrl_expression is None:
            if hasattr(batch, "x") and batch.x is not None:
                batch_x = batch.x
                if batch_x.dim() == 2 and batch_x.shape[1] == 1:
                    batch_x = batch_x.squeeze(1)
                num_samples = len(batch.batch.unique())
                batch_x_reshaped = batch_x.reshape(num_samples, -1)
                ctrl_expression = batch_x_reshaped.mean(dim=0)
            else:
                logger.warning("No batch.x found, using zeros as ctrl_expression")
                ctrl_expression = torch.zeros(pred.shape[1], device=pred.device, dtype=pred.dtype)

        # Build dict_filter from batch if not provided
        if dict_filter is None:
            dict_filter = {}
            if hasattr(batch, "de_idx") and batch.de_idx is not None:
                unique_perts = set(perts)
                for pert in unique_perts:
                    if pert != "ctrl":
                        try:
                            idx = perts.index(pert)
                            de_idx = batch.de_idx[idx]

                            if isinstance(de_idx, torch.Tensor):
                                de_idx = de_idx.tolist()

                            # Filter invalid indices (-1)
                            dict_filter[pert] = [idx for idx in de_idx if idx >= 0]
                        except (ValueError, IndexError):
                            continue

        # Compute loss
        if self.config.uncertainty:
            logvar = output_dict.get("logvar")
            if logvar is None:
                raise ValueError("Uncertainty mode is enabled but logvar not found in output_dict")

            loss = uncertainty_loss_fct(
                pred=pred,
                logvar=logvar,
                y=y,
                perts=perts,
                reg=self.config.uncertainty_reg,
                ctrl=ctrl_expression,
                direction_lambda=self.config.direction_lambda,
                dict_filter=dict_filter,
            )
        else:
            loss = loss_fct(
                pred=pred,
                y=y,
                perts=perts,
                ctrl=ctrl_expression,
                direction_lambda=self.config.direction_lambda,
                dict_filter=dict_filter,
            )

        result = {
            "pred": pred,
            "loss": loss,
        }
        if self.config.uncertainty:
            result["logvar"] = output_dict.get("logvar")

        return result

    def predict_perturbation(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: Optional[str] = None,
        return_numpy: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Predicts perturbation effects on the dataset.
        
        Args:
            dataset: Input perturbation dataset.
            batch_size: Batch size for inference.
            split: Specific split to predict on ('train', 'val', 'test'), or None for all splits.
            return_numpy: Whether to return numpy arrays.
            
        Returns:
            Dictionary with 'pred' key containing predictions.
            If split is None and dataset has splits, returns nested dict.
        """
        # Determine split behavior
        has_split = "split" in dataset.adata.obs
        
        if split is not None:
            # Process specific split
            return self._predict_perturbation_single(
                dataset, batch_size, split, return_numpy
            )
        elif has_split:
            # Process all splits separately
            split_names = dataset.adata.obs["split"].unique()
            result = {}
            for split_name in split_names:
                pred_result = self._predict_perturbation_single(
                    dataset, batch_size, str(split_name), return_numpy
                )
                result[str(split_name)] = pred_result
            return result
        else:
            # No split info, treat as train
            return {'train': self._predict_perturbation_single(
                dataset, batch_size, 'train', return_numpy
            )}
    
    def _predict_perturbation_single(
        self,
        dataset: PerturbationData,
        batch_size: int,
        split: str,
        return_numpy: bool,
        device: str = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu',
    ) -> Dict[str, np.ndarray]:
        """Predicts perturbation for a single split."""
        loader = self.get_dataloader(dataset, batch_size, split=split, shuffle=False)

        self.gears_model = self.gears_model.to(device)
        self.gears_model.eval()

        pert_cat = []
        pred = []
        truth = []
        logvar = []

        uncertainty_mode = self.config.uncertainty

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pert_cat.extend(batch.pert)

                if uncertainty_mode:
                    p, unc = self.gears_model(batch)
                    logvar.append(unc.cpu())
                else:
                    p = self.gears_model(batch)

                pred.append(p.cpu())
                if hasattr(batch, "y") and batch.y is not None:
                    truth.append(batch.y.cpu())

        if len(pred) == 0:
            return {}

        pred = torch.cat(pred, dim=0)

        results = {
            "pred": pred.detach().cpu().numpy() if return_numpy else pred.detach().cpu(),
            "pert_cat": np.array(pert_cat),
        }

        if truth:
            truth = torch.cat(truth, dim=0)
            results["truth"] = truth.detach().cpu().numpy() if return_numpy else truth.detach().cpu()

        if uncertainty_mode and logvar:
            logvar = torch.cat(logvar, dim=0)
            results["logvar"] = logvar.detach().cpu().numpy() if return_numpy else logvar.detach().cpu()

        return results

    def predict_embeddings(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: Optional[str] = None,
        return_numpy: bool = True,
    ) -> Dict[str, Any]:
        """Extracts latent embeddings from the model.
        
        Args:
            dataset: Input perturbation dataset.
            batch_size: Batch size for inference.
            split: Specific split to predict on ('train', 'val', 'test'), or None for all splits.
            return_numpy: Whether to return numpy arrays.
            
        Returns:
            Dictionary with 'cell' key containing cell embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        has_split = "split" in dataset.adata.obs
        
        if split is not None:
            return self._predict_embeddings_single(
                dataset, batch_size, split, return_numpy
            )
        elif has_split:
            split_names = dataset.adata.obs["split"].unique()
            result = {}
            for split_name in split_names:
                emb_result = self._predict_embeddings_single(
                    dataset, batch_size, str(split_name), return_numpy
                )
                result[str(split_name)] = emb_result
            return result
        else:
            return {'train': self._predict_embeddings_single(
                dataset, batch_size, 'train', return_numpy
            )}
    
    def _predict_embeddings_single(
        self,
        dataset: PerturbationData,
        batch_size: int,
        split: str,
        return_numpy: bool,
        device: str = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu',
    ) -> Dict[str, Any]:
        """Extracts embeddings for a single split."""
        loader = self.get_dataloader(dataset, batch_size, split=split, shuffle=False)

        self.gears_model = self.gears_model.to(device)
        self.gears_model.eval()

        embeddings_list = []
        pert_cat = []

        def get_embedding_hook(module, input, output):
            embeddings_list.append(output.detach().cpu())

        hook_handle = self.gears_model.transform.register_forward_hook(get_embedding_hook)

        try:
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    pert_cat.extend(batch.pert)
                    _ = self.gears_model(batch)
        finally:
            hook_handle.remove()

        if not embeddings_list:
            raise RuntimeError("No embeddings were captured.")

        embeddings = torch.cat(embeddings_list, dim=0)
        num_samples = len(pert_cat)
        num_genes = len(self.gene_list)
        hidden_size = embeddings.shape[-1]
        embeddings = embeddings.reshape(num_samples, num_genes, hidden_size)

        return {
            "cell": embeddings.numpy() if return_numpy else embeddings,
            "pert_cat": np.array(pert_cat),
        }

    def get_gene_embedding_layer(self):
        if not self.gene_embedding_layer_type == "custom":
            return self.gears_model.gene_emb
        else:
            raise ValueError("Gene embedding layer is not custom.")

    def get_pert_embedding_layer(self):
        if not self.pert_embedding_layer_type == "custom":
            return self.gears_model.pert_emb
        else:
            raise ValueError("Perturbation embedding layer is not custom.")

    def gene_embedding(self, gene: str):
        gene_idx = self.gene_list.get(gene, None)
        if gene_idx is None:
            raise ValueError(f"Gene {gene} not found in gene list.")
        return self.gears_model.gene_emb.weight[gene_idx]

    def pert_embedding(self, pert: str):
        pert_idx = self.pert_list.get(pert, None)
        if pert_idx is None:
            raise ValueError(f"Perturbation {pert} not found in perturbation list.")
        return self.gears_model.pert_emb.weight[pert_idx]

    def set_pert_embeddings(self, pert_embeddings: torch.Tensor, trainable: bool = True):
        """Sets weights for perturbation embeddings."""
        pert_embeddings = pert_embeddings.to(self.gears_model.device)

        if pert_embeddings.shape[0] != self.gears_model.num_perts:
            raise ValueError(f"pert_embeddings must have {self.gears_model.num_perts} rows")

        self.gears_model.pert_emb.weight.data = pert_embeddings.float()
        self.gears_model.pert_emb.weight.requires_grad = trainable
        self.pert_embedding_layer_type = "custom"

    def set_gene_embeddings(self, gene_embeddings: torch.Tensor, trainable: bool = True):
        """Sets weights for gene embeddings."""
        gene_embeddings = gene_embeddings.to(self.gears_model.device)

        if gene_embeddings.shape[0] != self.gears_model.num_genes:
            raise ValueError(f"gene_embeddings must have {self.gears_model.num_genes} rows")

        self.gears_model.gene_emb.weight.data = gene_embeddings.float()
        self.gears_model.gene_emb.weight.requires_grad = trainable
        self.gene_embedding_layer_type = "custom"

    def set_gene_embedding_layer(self, embedding_layer: Optional[nn.Module] = None):
        self.gears_model.gene_emb = (
            embedding_layer
            if embedding_layer is not None
            else nn.Embedding(self.gears_model.num_genes, self.gears_model.hidden_size)
        ).to(self.gears_model.device)

    def set_pert_embedding_layer(self, embedding_layer: Optional[nn.Module] = None):
        self.gears_model.pert_emb = (
            embedding_layer
            if embedding_layer is not None
            else nn.Embedding(self.gears_model.num_perts, self.gears_model.hidden_size)
        ).to(self.gears_model.device)

    def train_model(
        self,
        dataset: PerturbationData,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        train_split: str = "train",
        val_split: str = "val",
        log_interval: int = 50,
        save_best: bool = True,
        save_path: Optional[str] = None,
        scheduler_step_size: int = 5,
        scheduler_gamma: float = 0.5,
    ) -> Dict[str, List[float]]:
        """Trains the GEARS model.
        
        Args:
            dataset: Input perturbation dataset.
            epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            batch_size: Training batch size.
            train_split: Name of training split.
            val_split: Name of validation split.
            log_interval: Logging interval in steps.
            save_best: Whether to save the best model.
            save_path: Path to save the best model.
            scheduler_step_size: Step size for learning rate scheduler.
            scheduler_gamma: Gamma for learning rate scheduler.
            
        Returns:
            Dictionary with training history.
        """
        # Data Preparation
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")

        if "rank_genes_groups_cov_all" not in dataset.adata.uns:
            logger.warning("DE genes not found. Computing DE genes...")
            dataset.compute_de_genes()

        if "ctrl_indices" not in dataset.adata.obsm:
            logger.warning("Control cell pairing not found. Pairing cells...")
            dataset.pair_cells()

        # Compute control expression
        ctrl_cells = dataset.adata[dataset.adata.obs["condition"] == "ctrl"]
        ctrl_expression = torch.tensor(
            np.array(ctrl_cells.X.mean(axis=0)).flatten(),
            dtype=torch.float32,
            device=self.gears_model.device,
        )

        # Build DE gene filter dictionary
        dict_filter = {}
        if "rank_genes_groups_cov_all" in dataset.adata.uns:
            rank_data = dataset.adata.uns["rank_genes_groups_cov_all"]
            top_n = dataset.adata.uns.get("top_de_n", 20)
            gene_list = dataset.adata.var["gene_name"].tolist()
            gene_name_to_idx = {name: i for i, name in enumerate(gene_list)}

            if isinstance(rank_data, dict):
                for cond, top_genes in rank_data.items():
                    if cond not in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
                        if isinstance(top_genes, list):
                            genes_iter = top_genes
                        elif hasattr(top_genes, "__iter__"):
                            genes_iter = list(top_genes)
                        else:
                            genes_iter = [top_genes]
                        dict_filter[cond] = [
                            gene_name_to_idx.get(g, -1)
                            for g in genes_iter[:top_n]
                            if g in gene_name_to_idx
                        ]
            else:
                if hasattr(rank_data, "dtype") and hasattr(rank_data.dtype, "names"):
                    for cond in rank_data["names"].dtype.names:
                        top_genes = rank_data["names"][cond][:top_n]
                        dict_filter[cond] = [
                            gene_name_to_idx.get(g, -1)
                            for g in top_genes
                            if g in gene_name_to_idx
                        ]

        # Initialize DataLoaders
        train_loader = self.get_dataloader(dataset, batch_size, split=train_split, shuffle=True)
        val_loader = self.get_dataloader(dataset, batch_size, split=val_split, shuffle=False)

        # Use state_dict instead of deepcopy to save GPU memory
        best_model_state = {k: v.cpu().clone() for k, v in self.gears_model.state_dict().items()}
        optimizer = optim.Adam(
            self.gears_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        min_val = np.inf
        history = {
            "train_loss": [],
            "train_mse": [],
            "train_mse_de": [],
            "val_mse": [],
            "val_mse_de": [],
        }

        logger.info("Start Training...")

        for epoch in range(epochs):
            self.gears_model.train()
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                batch = batch.to(self.gears_model.device)
                optimizer.zero_grad()

                loss_dict = self.compute_loss(
                    batch,
                    ctrl_expression=ctrl_expression,
                    dict_filter=dict_filter,
                )
                loss = loss_dict["loss"]

                loss.backward()
                nn.utils.clip_grad_value_(self.gears_model.parameters(), clip_value=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if step % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch + 1} Step {step + 1} Train Loss: {loss.item():.4f}"
                    )

            scheduler.step()

            # Evaluation
            train_res = self.evaluate(dataset, batch_size=batch_size, split=train_split)
            val_res = self.evaluate(dataset, batch_size=batch_size, split=val_split)

            train_metrics = self.compute_metrics(train_res)["mse"]  # simplified extraction
            train_metrics_full = self.compute_metrics(train_res)
            val_metrics_full = self.compute_metrics(val_res)

            # Record history
            history["train_loss"].append(epoch_loss / num_batches)
            history["train_mse"].append(train_metrics_full["mse"])
            history["train_mse_de"].append(train_metrics_full["mse_de"])
            history["val_mse"].append(val_metrics_full["mse"])
            history["val_mse_de"].append(val_metrics_full["mse_de"])

            logger.info(
                f"Epoch {epoch + 1}: "
                f"Train MSE: {train_metrics_full['mse']:.4f}, "
                f"Val MSE: {val_metrics_full['mse']:.4f}, "
                f"Val DE MSE: {val_metrics_full['mse_de']:.4f}"
            )

            # Save best model
            if val_metrics_full["mse_de"] < min_val:
                min_val = val_metrics_full["mse_de"]
                # Save state dict to CPU to save GPU memory
                best_model_state = {k: v.cpu().clone() for k, v in self.gears_model.state_dict().items()}
                logger.info(f"New best model found! Val DE MSE: {min_val:.4f}")

        logger.info("Training Done!")
        # Restore best model weights
        self.gears_model.load_state_dict(best_model_state)

        if save_best and save_path is not None:
            self.save(save_path)

        return history

    def save(self, model_path: str):
        """
        Saves the GEARS model, including weights, config, graphs, and metadata.

        Args:
            path: Target directory path.
        """
        # 1. Ensure directory exists
        os.makedirs(model_path, exist_ok=True)

        # 2. Save Configuration
        config_path = os.path.join(model_path, "config.json")
        self.config.save(config_path)

        # 3. Save Vocabularies (Gene & Perturbation Lists)
        gene_list_path = os.path.join(model_path, "gene_list.json")
        with open(gene_list_path, "w") as f:
            json.dump(self.gene_list, f, indent=2)

        pert_list_path = os.path.join(model_path, "pert_list.json")
        with open(pert_list_path, "w") as f:
            json.dump(self.pert_list, f, indent=2)

        # 4. Save Graphs
        go_graph_path = os.path.join(model_path, "go_graph.pkl")
        self.go_graph.save(go_graph_path)

        co_graph_path = os.path.join(model_path, "co_graph.pkl")
        self.co_graph.save(co_graph_path)

        # 5. Save Metadata (e.g. embedding types)
        metadata = {
            "gene_embedding_layer_type": getattr(self, "gene_embedding_layer_type", "default"),
            "pert_embedding_layer_type": getattr(self, "pert_embedding_layer_type", "default"),
        }
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 6. Save Model Weights
        # Save weights last to ensure structure files are written successfully first
        model_path = os.path.join(model_path, "model.pt")
        torch.save(self.gears_model.state_dict(), model_path)

        logger.info(f"Model content successfully saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu') -> "GearsModel":
        """
        Loads the GEARS model from a saved directory.

        Args:
            model_path: Path to the saved model directory.
            device: Target device ('cpu', 'cuda', etc.).
            **kwargs: Additional arguments for model initialization.

        Returns:
            Loaded GearsModel instance.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # 1. Load Configuration
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        config = GearsConfig.load(config_path)

        # 2. Load Vocabularies
        gene_list_path = os.path.join(model_path, "gene_list.json")
        with open(gene_list_path, "r") as f:
            gene_list = json.load(f)

        pert_list_path = os.path.join(model_path, "pert_list.json")
        with open(pert_list_path, "r") as f:
            pert_list = json.load(f)

        # 3. Load Graphs
        go_graph_path = os.path.join(model_path, "go_graph.pkl")
        co_graph_path = os.path.join(model_path, "co_graph.pkl")
        
        if not os.path.exists(go_graph_path) or not os.path.exists(co_graph_path):
             raise FileNotFoundError(f"Graph files missing in {model_path}")

        go_graph = GeneGraph.load(go_graph_path)
        co_graph = GeneGraph.load(co_graph_path)

        # 4. Initialize Model Structure
        model = cls(
            config=config,
            gene_list=gene_list,
            pert_list=pert_list,
            go_graph=go_graph,
            co_graph=co_graph,
            # Note: Embeddings are initialized to default/random here.
            # They will be overwritten by state_dict later.
        )

        # 5. Load Metadata and Apply to Model State
        metadata_path = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            model.gene_embedding_layer_type = metadata.get("gene_embedding_layer_type", "default")
            model.pert_embedding_layer_type = metadata.get("pert_embedding_layer_type", "default")

        # 6. Load Weights
        weights_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        # Load to CPU first to avoid CUDA OOM or device mismatch errors
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Load state dict
        # strict=False is used to allow flexibility, but we warn on missing keys
        keys_missing, keys_unexpected = model.gears_model.load_state_dict(state_dict, strict=False)
        
        if keys_missing:
            logger.warning(f"Missing keys in state_dict: {keys_missing}")
        if keys_unexpected:
            logger.warning(f"Unexpected keys in state_dict: {keys_unexpected}")

        # 7. Move to Target Device
        model.to(device)
        
        logger.info(f"✓ Model loaded successfully from {model_path} (Device: {device})")
        return model
