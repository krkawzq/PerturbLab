import json
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, Any

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
    """
    GEARS Model wrapper for single-cell perturbation prediction.

    Integrates Gene Ontology (GO) and co-expression graphs with the core
    GEARS architecture.
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
        device: str = "cuda",
    ):
        """
        Initializes the GearsModel.

        Args:
            config: Configuration object for GEARS.
            gene_list: List of gene names.
            pert_list: List of perturbation names.
            go_graph: Gene Ontology graph object.
            co_graph: Co-expression graph object.
            pert_embeddings: Pre-trained perturbation embeddings.
            gene_embeddings: Pre-trained gene embeddings.
            device: Computation device ('cuda' or 'cpu').
        """
        super().__init__(config)
        
        if device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        else:
            self.device = "cpu"

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
        self.gears_model = GEARS_Model(
            dict(
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
                device=self.device,
            )
        ).to(self.device)

        # Track embedding layer types
        self.pert_embedding_layer_type = "default"
        self.gene_embedding_layer_type = "default"

        # Set pre-trained embeddings if provided
        if pert_embeddings is not None:
            self.set_pert_embeddings(pert_embeddings, trainable=True)
        if gene_embeddings is not None:
            self.set_gene_embeddings(gene_embeddings, trainable=True)

    @classmethod
    def init_from_dataset(
        cls,
        dataset: PerturbationData,
        config: GearsConfig,
        device: str = "cuda",
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
            device=device,
        )

    @staticmethod
    def get_dataloader(
        dataset: PerturbationData,
        batch_size: int,
        split: str = "train",
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Creates a PyG DataLoader for specific data split.
        """
        # 1. Basic Validation
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")

        if "rank_genes_groups_cov_all" not in dataset.adata.uns:
            logger.warning("DE genes not found. Computing DE genes...")
            dataset.compute_de_genes()

        if "ctrl_indices" not in dataset.adata.obsm:
            raise ValueError("Control cell pairing not found. Please call dataset.pair_cells() first.")

        # 2. Metadata Preparation
        gene_list = dataset.adata.var["gene_name"].tolist()
        gene_name_to_idx = {name: i for i, name in enumerate(gene_list)}

        if "pert_list" not in dataset.adata.uns:
            all_conditions = dataset.adata.obs["condition"].unique()
            pert_set = set()
            for cond in all_conditions:
                if cond != "ctrl":
                    pert_set.update([p for p in cond.split("+") if p != "ctrl"])
            dataset.adata.uns["pert_list"] = sorted(list(pert_set))

        # 3. Prepare DE Genes Map
        de_gene_map = {}
        if "rank_genes_groups_cov_all" in dataset.adata.uns:
            rank_data = dataset.adata.uns["rank_genes_groups_cov_all"]
            top_n = dataset.adata.uns.get("top_de_n", 20)

            # Handle different rank_data formats (dict or structured array)
            if isinstance(rank_data, dict):
                for cond, top_genes in rank_data.items():
                    if cond not in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
                        # Ensure compatibility with lists or iterables
                        if isinstance(top_genes, list):
                            genes_iter = top_genes
                        elif hasattr(top_genes, "__iter__"):
                            genes_iter = list(top_genes)
                        else:
                            genes_iter = [top_genes]
                        de_gene_map[cond] = [
                            gene_name_to_idx.get(g, -1)
                            for g in genes_iter[:top_n]
                            if g in gene_name_to_idx
                        ]
            else:
                if hasattr(rank_data, "dtype") and hasattr(rank_data.dtype, "names"):
                    for cond in rank_data["names"].dtype.names:
                        top_genes = rank_data["names"][cond][:top_n]
                        de_gene_map[cond] = [
                            gene_name_to_idx.get(g, -1)
                            for g in top_genes
                            if g in gene_name_to_idx
                        ]

        # 4. Optimization: Filter by split early to reduce memory usage and loop time
        obs_df = dataset.adata.obs[["condition", "index_col", "split"]].copy()
        if split != "all":
            obs_df = obs_df[obs_df["split"] == split]
            if obs_df.empty:
                available_splits = dataset.adata.obs["split"].unique()
                raise ValueError(f"Split '{split}' not found. Available: {available_splits}")

        logger.info(f"Generating PyG Data objects for split '{split}' ({len(obs_df)} samples)...")

        data_list = []
        X = dataset.adata.X
        ctrl_indices = dataset.adata.obsm["ctrl_indices"]
        num_samples = ctrl_indices.shape[1]

        # Optimization: Check sparsity once before the loop
        is_sparse = hasattr(X, "toarray") or hasattr(X, "A")

        for condition, group_df in tqdm(obs_df.groupby("condition"), desc="Processing conditions"):
            # Prepare perturbation indices
            if condition == "ctrl":
                pert_idx = [-1]
            else:
                genes = [p for p in condition.split("+") if p != "ctrl"]
                pert_idx = [gene_name_to_idx.get(g, -1) for g in genes]
                if not pert_idx or all(i == -1 for i in pert_idx):
                    pert_idx = [-1]

            de_idx = de_gene_map.get(condition, [-1] * 20)
            current_idx = group_df["index_col"].values

            # Optimization: Batch extract X to reduce indexing overhead
            if is_sparse:
                y_batch_all = X[current_idx].toarray() if hasattr(X, "toarray") else X[current_idx].A
            else:
                y_batch_all = X[current_idx]

            # Construct Data objects
            for i, cell_idx in enumerate(current_idx):
                paired_idx = ctrl_indices[cell_idx, :]

                # Get paired control expression
                if is_sparse:
                    x_batch = X[paired_idx].toarray() if hasattr(X, "toarray") else X[paired_idx].A
                else:
                    x_batch = X[paired_idx]

                y_tensor = torch.tensor(y_batch_all[i], dtype=torch.float).unsqueeze(1)

                for j in range(num_samples):
                    data_obj = Data(
                        x=torch.tensor(x_batch[j], dtype=torch.float).unsqueeze(1),
                        y=y_tensor,  # Reuse y_tensor
                        pert_idx=pert_idx,
                        pert=condition,
                        de_idx=de_idx,
                    )
                    data_obj.split = split
                    data_list.append(data_obj)

        logger.info(f"Created {len(data_list)} Data objects.")

        return DataLoader(
            data_list,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
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
        """
        Computes the loss for the given batch.

        Args:
            batch: The input batch of data.
            ctrl_expression: Baseline control expression vector.
            dict_filter: Dictionary for filtering (e.g., DE genes).
            output_dict: Optional pre-computed model outputs.
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
        split: str = "test",
        return_numpy: bool = True,
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Runs inference on the dataset.
        """
        loader = self.get_dataloader(dataset, batch_size, split=split, shuffle=False)

        self.gears_model = self.gears_model.to(self.device)
        self.gears_model.eval()

        pert_cat = []
        pred = []
        truth = []
        logvar = []

        uncertainty_mode = self.config.uncertainty

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pert_cat.extend(batch.pert)

                if uncertainty_mode:
                    p, unc = self.gears_model(batch)
                    logvar.append(unc.cpu())
                else:
                    p = self.gears_model(batch)

                pred.append(p.cpu())
                if hasattr(batch, "y") and batch.y is not None:
                    truth.append(batch.y.cpu())

        # Use torch.cat instead of stack to handle variable batch sizes
        if len(pred) > 0:
            pred = torch.cat(pred, dim=0)  # [total_samples, n_genes]
        else:
            return {}

        results = {
            "pred": pred.detach().cpu().numpy() if return_numpy else pred.detach().cpu(),
            "pert_cat": np.array(pert_cat),
        }

        if truth:
            truth = torch.cat(truth, dim=0)
            results["truth"] = truth.detach().cpu().numpy() if return_numpy else truth.detach().cpu()

        if uncertainty_mode and logvar:
            logvar = torch.cat(logvar, dim=0)
            uncertainty_dict = {
                "logvar": logvar.detach().cpu().numpy() if return_numpy else logvar.detach().cpu(),
            }
            return results, uncertainty_dict

        return results

    def predict_embeddings(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: str = "test",
        return_numpy: bool = True,
    ) -> Dict[str, Any]:
        """
        Extracts latent embeddings from the model.
        """
        loader = self.get_dataloader(dataset, batch_size, split=split, shuffle=False)

        self.gears_model = self.gears_model.to(self.device)
        self.gears_model.eval()

        embeddings_list = []
        pert_cat = []
        batch_sizes = []

        def get_embedding_hook(module, input, output):
            embeddings_list.append(output.detach().cpu())

        hook_handle = self.gears_model.transform.register_forward_hook(get_embedding_hook)

        try:
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    pert_cat.extend(batch.pert)

                    # Track batch size
                    num_graphs_in_batch = (
                        batch.ptr.shape[0] - 1
                        if hasattr(batch, "ptr")
                        else len(batch.batch.unique())
                    )
                    batch_sizes.append(num_graphs_in_batch)

                    _ = self.gears_model(batch)
        finally:
            hook_handle.remove()

        if embeddings_list:
            # Use torch.cat for proper concatenation
            embeddings = torch.cat(embeddings_list, dim=0)

            num_samples = len(pert_cat)
            num_genes = len(self.gene_list)
            hidden_size = embeddings.shape[-1]

            embeddings = embeddings.reshape(num_samples, num_genes, hidden_size)

            results = {
                "embeddings": embeddings.numpy() if return_numpy else embeddings,
                "pert_cat": np.array(pert_cat),
            }

            return results
        else:
            raise RuntimeError("No embeddings were captured.")

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
        pert_embeddings = pert_embeddings.to(self.device)

        if pert_embeddings.shape[0] != self.gears_model.num_perts:
            raise ValueError(f"pert_embeddings must have {self.gears_model.num_perts} rows")

        self.gears_model.pert_emb.weight.data = pert_embeddings.float()
        self.gears_model.pert_emb.weight.requires_grad = trainable
        self.pert_embedding_layer_type = "custom"

    def set_gene_embeddings(self, gene_embeddings: torch.Tensor, trainable: bool = True):
        """Sets weights for gene embeddings."""
        gene_embeddings = gene_embeddings.to(self.device)

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
        )

    def set_pert_embedding_layer(self, embedding_layer: Optional[nn.Module] = None):
        self.gears_model.pert_emb = (
            embedding_layer
            if embedding_layer is not None
            else nn.Embedding(self.gears_model.num_perts, self.gears_model.hidden_size)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Loads a GEARS model from a saved directory.

        Args:
            model_name_or_path: Path to the saved model directory.
            device: Computation device.

        Returns:
            Loaded GearsModel instance.
        """
        if not os.path.exists(model_name_or_path):
            raise ValueError(f"Path '{model_name_or_path}' does not exist.")

        # Load config
        config_path = os.path.join(model_name_or_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        config = GearsConfig.load(config_path)

        # Load gene and perturbation lists
        gene_list_path = os.path.join(model_name_or_path, "gene_list.json")
        if not os.path.exists(gene_list_path):
            raise FileNotFoundError(f"Gene list file not found at {gene_list_path}")
        with open(gene_list_path, "r") as f:
            gene_list = json.load(f)

        pert_list_path = os.path.join(model_name_or_path, "pert_list.json")
        if not os.path.exists(pert_list_path):
            raise FileNotFoundError(f"Perturbation list file not found at {pert_list_path}")
        with open(pert_list_path, "r") as f:
            pert_list = json.load(f)

        # Load metadata
        metadata_path = os.path.join(model_name_or_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            gene_embedding_layer_type = metadata.get("gene_embedding_layer_type", "default")
            pert_embedding_layer_type = metadata.get("pert_embedding_layer_type", "default")
        else:
            gene_embedding_layer_type = "default"
            pert_embedding_layer_type = "default"

        # Load graphs
        go_graph_path = os.path.join(model_name_or_path, "go_graph.pkl")
        if not os.path.exists(go_graph_path):
            raise FileNotFoundError(f"GO graph file not found at {go_graph_path}")
        go_graph = GeneGraph.load(go_graph_path)

        co_graph_path = os.path.join(model_name_or_path, "co_graph.pkl")
        if not os.path.exists(co_graph_path):
            raise FileNotFoundError(f"Co-expression graph file not found at {co_graph_path}")
        co_graph = GeneGraph.load(co_graph_path)

        # Create model instance
        model = cls(
            config=config,
            gene_list=gene_list,
            pert_list=pert_list,
            go_graph=go_graph,
            co_graph=co_graph,
            device=device,
        )

        # Load weights
        model_path = os.path.join(model_name_or_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        model.gears_model.load_state_dict(state_dict)

        # Restore embedding metadata
        model.gene_embedding_layer_type = gene_embedding_layer_type
        model.pert_embedding_layer_type = pert_embedding_layer_type

        return model

    def save(self, path: str):
        """
        Saves the GEARS model, including weights, config, and graphs.

        Args:
            path: Target directory path.
        """
        os.makedirs(path, exist_ok=True)

        # Save configuration
        config_path = os.path.join(path, "config.json")
        self.config.save(config_path)

        # Save model weights
        model_path = os.path.join(path, "model.pt")
        torch.save(self.gears_model.state_dict(), model_path)

        # Save graphs
        go_graph_path = os.path.join(path, "go_graph.pkl")
        self.go_graph.save(go_graph_path)

        co_graph_path = os.path.join(path, "co_graph.pkl")
        self.co_graph.save(co_graph_path)

        # Save lists
        gene_list_path = os.path.join(path, "gene_list.json")
        with open(gene_list_path, "w") as f:
            json.dump(self.gene_list, f, indent=2)

        pert_list_path = os.path.join(path, "pert_list.json")
        with open(pert_list_path, "w") as f:
            json.dump(self.pert_list, f, indent=2)

        # Save metadata
        metadata = {
            "gene_embedding_layer_type": getattr(self, "gene_embedding_layer_type", "default"),
            "pert_embedding_layer_type": getattr(self, "pert_embedding_layer_type", "default"),
        }
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    def train(self):
        """Sets the model to training mode."""
        self.gears_model.train()
        return self

    def eval(self):
        """Sets the model to evaluation mode."""
        self.gears_model.eval()
        return self

    def evaluate(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: str = "val",
    ) -> Dict[str, np.ndarray]:
        """
        Evaluates the model on a specific data split.
        """
        loader = self.get_dataloader(dataset, batch_size, split=split, shuffle=False)

        self.gears_model = self.gears_model.to(self.device)
        self.gears_model.eval()

        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        logvar = []

        uncertainty_mode = self.config.uncertainty

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pert_cat.extend(batch.pert)

                if uncertainty_mode:
                    p, unc = self.gears_model(batch)
                    logvar.append(unc.cpu())
                else:
                    p = self.gears_model(batch)

                t = batch.y
                # Reshape ground truth if necessary
                if t.dim() == 2 and t.shape[1] == 1:
                    num_samples = len(batch.batch.unique())
                    t = t.reshape(num_samples, -1)

                pred.append(p.cpu())
                truth.append(t.cpu())

                # Collect differentially expressed genes
                if hasattr(batch, "de_idx") and batch.de_idx is not None:
                    for idx, de_idx in enumerate(batch.de_idx):
                        if isinstance(de_idx, torch.Tensor):
                            de_idx = de_idx.tolist()
                        valid_de_idx = [i for i in de_idx if i >= 0]
                        if valid_de_idx:
                            pred_de.append(p[idx, valid_de_idx].cpu())
                            truth_de.append(t[idx, valid_de_idx].cpu())

        # Consolidate results
        results = {
            "pert_cat": np.array(pert_cat),
            "pred": torch.cat(pred, dim=0).detach().cpu().numpy(),
            "truth": torch.cat(truth, dim=0).detach().cpu().numpy(),
        }

        if pred_de:
            results["pred_de"] = torch.stack(pred_de).detach().cpu().numpy()
            results["truth_de"] = torch.stack(truth_de).detach().cpu().numpy()

        if uncertainty_mode and logvar:
            results["logvar"] = torch.cat(logvar, dim=0).detach().cpu().numpy()

        return results

    def compute_metrics(self, results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Computes MSE and Pearson correlation metrics.

        Args:
            results: Dictionary containing prediction results.

        Returns:
            Dictionary with aggregated metrics.
        """
        def mse(pred, truth):
            return np.mean((pred - truth) ** 2)

        metrics = {}
        metrics_pert = {}

        metric2fct = {"mse": mse, "pearson": pearsonr}

        for m in metric2fct.keys():
            metrics[m] = []
            metrics[m + "_de"] = []

        for pert in np.unique(results["pert_cat"]):
            metrics_pert[pert] = {}
            p_idx = np.where(results["pert_cat"] == pert)[0]

            for m, fct in metric2fct.items():
                if m == "pearson":
                    val = fct(
                        results["pred"][p_idx].mean(0), results["truth"][p_idx].mean(0)
                    )[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(
                        results["pred"][p_idx].mean(0), results["truth"][p_idx].mean(0)
                    )

                metrics_pert[pert][m] = val
                metrics[m].append(metrics_pert[pert][m])

            if pert != "ctrl" and "pred_de" in results:
                for m, fct in metric2fct.items():
                    if m == "pearson":
                        val = fct(
                            results["pred_de"][p_idx].mean(0),
                            results["truth_de"][p_idx].mean(0),
                        )[0]
                        if np.isnan(val):
                            val = 0
                    else:
                        val = fct(
                            results["pred_de"][p_idx].mean(0),
                            results["truth_de"][p_idx].mean(0),
                        )

                    metrics_pert[pert][m + "_de"] = val
                    metrics[m + "_de"].append(metrics_pert[pert][m + "_de"])
            else:
                for m in metric2fct.keys():
                    metrics_pert[pert][m + "_de"] = 0

        for m in metric2fct.keys():
            metrics[m] = np.mean(metrics[m])
            metrics[m + "_de"] = np.mean(metrics[m + "_de"])

        return {
            "mse": metrics["mse"],
            "mse_de": metrics["mse_de"],
            "pearson": metrics["pearson"],
            "pearson_de": metrics["pearson_de"],
            "per_perturbation": metrics_pert,
        }

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
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
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
            device=self.device,
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

        # Optimizer & Scheduler
        self.gears_model = self.gears_model.to(self.device)
        best_model = deepcopy(self.gears_model)
        optimizer = optim.Adam(
            self.gears_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

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
                batch = batch.to(self.device)
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
                best_model = deepcopy(self.gears_model)
                logger.info(f"New best model found! Val DE MSE: {min_val:.4f}")

        logger.info("Training Done!")
        self.gears_model = best_model

        if save_best and save_path is not None:
            self.save(save_path)

        return history
