import json
import logging
import os
import sys
import warnings
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from ...data import GeneGraph, PerturbationData
from ...utils import download_from_huggingface
from ..base import PerturbationModel
from ..gears.source import GEARS, PertData
from ..gears.source.inference import (compute_metrics, deeper_analysis,
                                      evaluate, non_dropout_analysis)
from ..gears.source.utils import print_sys
from .config import scFoundationConfig
from .source.load import gatherData, getEncoerDecoderData
from .source.pretrainmodels import select_model

logger = logging.getLogger(__name__)


def adata_row_to_pyg_data(
    adata: AnnData,
    idx: int,
    gene_names: List[str],
    pert_name_to_id: Dict[str, int],
    de_gene_map: Dict[str, List[int]],
    ctrl_indices: np.ndarray,
    sample_idx: int = 0,
    original_idx: Optional[int] = None,
    original_adata: Optional[AnnData] = None
) -> Data:
    """
    Convert a single AnnData row to PyTorch Geometric Data object.
    
    This is a pure function that extracts all necessary information from adata
    and converts it to the format expected by GEARS model.
    
    Args:
        adata: AnnData object containing expression data (may be sliced)
        idx: Row index in adata
        gene_names: List of gene names aligned with model vocabulary
        pert_name_to_id: Mapping from perturbation name to index ID
        de_gene_map: Mapping from condition to list of DE gene indices
        ctrl_indices: Control pairing indices from original adata [n_cells_full, n_samples]
        sample_idx: Which control sample to use (default: 0)
        original_idx: Original adata index (if adata is sliced)
        original_adata: Original (full) AnnData for accessing control cells
    
    Returns:
        PyG Data object with x, y, pert_idx, de_idx, pert attributes
    """
    # 1. Get target expression (Y) from current adata
    y_vec = adata.X[idx]
    if sparse.issparse(y_vec):
        y_vec = y_vec.toarray().flatten()
    else:
        y_vec = np.asarray(y_vec).flatten()
    
    # 2. Get condition and DE genes
    condition = adata.obs['condition'].iloc[idx]
    de_idx = de_gene_map.get(condition, [-1] * 20)
    
    # 3. Get perturbation indices
    if condition == 'ctrl':
        pert_idx = [-1]
    else:
        # Handle multi-gene perturbations (e.g., "GeneA+GeneB")
        perts = [p for p in condition.split('+') if p != 'ctrl']
        pert_idx = [pert_name_to_id.get(p, -1) for p in perts]
        if not pert_idx or all(i == -1 for i in pert_idx):
            pert_idx = [-1]
    
    # 4. Get control/basal expression (X)
    # GEARS strategy: Pair current cell with a sampled control cell
    # Use original_idx for ctrl_indices lookup if provided (for sliced adata)
    lookup_idx = original_idx if original_idx is not None else idx
    paired_ctrl_idx = ctrl_indices[lookup_idx, sample_idx]
    
    # Get control expression from original_adata if provided, otherwise from adata
    ctrl_adata = original_adata if original_adata is not None else adata
    x_vec = ctrl_adata.X[paired_ctrl_idx]
    if sparse.issparse(x_vec):
        x_vec = x_vec.toarray().flatten()
    else:
        x_vec = np.asarray(x_vec).flatten()
    
    # 5. Construct PyG Data
    return Data(
        x=torch.tensor(x_vec, dtype=torch.float).unsqueeze(1),  # [Genes, 1]
        y=torch.tensor(y_vec, dtype=torch.float).unsqueeze(1),  # [Genes, 1]
        pert_idx=torch.tensor(pert_idx, dtype=torch.long),
        de_idx=torch.tensor(de_idx, dtype=torch.long),
        pert=condition
    )


class PerturbationPyGDataset(Dataset):
    """
    Standard PyTorch Dataset that converts AnnData rows into PyG Data objects.
    
    Decoupled from GEARS internal logic. Uses pure function adata_row_to_pyg_data
    for conversion.
    
    Note: This dataset works with sliced AnnData (e.g., train/test splits).
    It maintains a mapping from dataset indices to original adata indices.
    """
    
    def __init__(
        self,
        adata: AnnData,
        gene_names: List[str],
        pert_name_to_id: Dict[str, int],
        de_gene_map: Dict[str, List[int]],
        ctrl_indices: np.ndarray,
        num_samples_per_cell: int = 1,
        original_adata: Optional[AnnData] = None
    ):
        """
        Args:
            adata: AnnData object (may be sliced, e.g., train split)
            gene_names: List of gene names aligned with the model
            pert_name_to_id: Mapping from perturbation name to index ID
            de_gene_map: Mapping from condition to list of DE gene indices
            ctrl_indices: Control pairing indices from original (full) adata [n_cells_full, n_samples]
            num_samples_per_cell: Number of control samples to pair with each treated cell
            original_adata: Original (full) AnnData if adata is sliced. If None, assumes adata is full.
        """
        self.adata = adata
        self.original_adata = original_adata if original_adata is not None else adata
        self.gene_names = gene_names
        self.pert_name_to_id = pert_name_to_id
        self.de_gene_map = de_gene_map
        self.ctrl_indices = ctrl_indices  # Always from original adata
        self.num_samples = num_samples_per_cell
        
        # Map from dataset index to original adata index
        # If adata is sliced, we need to map dataset indices to original indices
        if original_adata is not None and len(adata) < len(original_adata):
            # adata is sliced, create mapping
            self._idx_map = {}
            original_indices = original_adata.obs.index
            for i, obs_name in enumerate(adata.obs_names):
                self._idx_map[i] = original_indices.get_loc(obs_name)
        else:
            # adata is full, direct mapping
            self._idx_map = {i: i for i in range(len(adata))}
        
        # Pre-compute indices for efficient access
        # For num_samples > 1, we create multiple entries per cell
        self._indices = []
        for i in range(len(adata)):
            for j in range(num_samples_per_cell):
                self._indices.append((i, j))
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx):
        cell_idx, sample_idx = self._indices[idx]
        # Map dataset index to original adata index
        original_idx = self._idx_map[cell_idx]
        
        return adata_row_to_pyg_data(
            self.adata,  # Use sliced adata for target expression (Y)
            cell_idx,    # Index in sliced adata
            self.gene_names,
            self.pert_name_to_id,
            self.de_gene_map,
            self.ctrl_indices,  # Uses original indices
            sample_idx,
            original_idx,  # Pass original index for ctrl_indices lookup
            self.original_adata  # Use original adata for control expression (X)
        )


class scFoundationModel(PerturbationModel):
    """
    scFoundation model for single-cell foundation embeddings.
    
    scFoundation is a large-scale foundation model trained on diverse single-cell datasets.
    It supports:
    - Cell embedding extraction (with different pooling strategies)
    - Gene embedding extraction
    - Both single-cell and bulk RNA-seq data
    
    Args:
        config (scFoundationConfig): Model configuration.
        device (str): Device to run the model on ('cuda' or 'cpu'). Default: 'cuda'
        gene_list (Optional[list]): List of gene names. If None, will load from default gene index file.
    
    Example:
        ```python
        # Load pretrained model
        model = scFoundationModel.from_pretrained('scfoundation-cell', device='cuda')
        
        # Get cell embeddings
        embeddings = model.predict_embedding(adata, output_type='cell', pool_type='all')
        
        # Get gene embeddings
        gene_embs = model.predict_embedding(adata, output_type='gene')
        ```
    """
    
    def __init__(
        self,
        config: scFoundationConfig,
        device: str = 'cuda',
        gene_list: Optional[list] = None,
    ):
        """
        Initialize scFoundation model.
        
        Args:
            config (scFoundationConfig): Model configuration
            device (str): Device to run model on
            gene_list (Optional[list]): List of gene names for vocabulary
        """
        self.config = config
        self.device = device
        
        # Load gene list
        if gene_list is None:
            gene_list_path = os.path.join(
                os.path.dirname(__file__), 
                'source', 
                'gene_index.json'
            )
            if os.path.exists(gene_list_path):
                import json
                with open(gene_list_path, 'r') as f:
                    self.gene_list = json.load(f)
            else:
                warnings.warn(
                    f"Gene index file not found at {gene_list_path}. "
                    "Gene alignment will not be performed automatically."
                )
                self.gene_list = None
        else:
            self.gene_list = gene_list
        
        # Build model
        model_config = config.to_model_config_dict()
        self.model = select_model(model_config)
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"Initialized scFoundation model with {config.num_tokens} tokens")
    
    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        return self
    
    def forward(
        self,
        x: torch.Tensor,
        padding_label: torch.Tensor,
        encoder_position_gene_ids: torch.Tensor,
        encoder_labels: torch.Tensor,
        decoder_data: torch.Tensor,
        decoder_position_gene_ids: torch.Tensor,
        decoder_data_padding_labels: torch.Tensor,
        mask_gene_name: bool = False,
        mask_labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the scFoundation model.
        
        Args:
            x (torch.Tensor): Input gene expression data [B, N]
            padding_label (torch.Tensor): Padding mask for encoder [B, N]
            encoder_position_gene_ids (torch.Tensor): Position IDs for encoder [B, N]
            encoder_labels (torch.Tensor): Labels for encoder [B, N]
            decoder_data (torch.Tensor): Input data for decoder [B, N]
            decoder_position_gene_ids (torch.Tensor): Position IDs for decoder [B, N]
            decoder_data_padding_labels (torch.Tensor): Padding mask for decoder [B, N]
            mask_gene_name (bool): Whether to mask gene names
            mask_labels (Optional[torch.Tensor]): Mask labels if applicable
            output_attentions (bool): Whether to output attention maps
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Model output
        """
        return self.model(
            x=x,
            padding_label=padding_label,
            encoder_position_gene_ids=encoder_position_gene_ids,
            encoder_labels=encoder_labels,
            decoder_data=decoder_data,
            mask_gene_name=mask_gene_name,
            mask_labels=mask_labels,
            decoder_position_gene_ids=decoder_position_gene_ids,
            decoder_data_padding_labels=decoder_data_padding_labels,
            output_attentions=output_attentions,
            **kwargs
        )
    
    def _align_genes(self, adata: AnnData) -> pd.DataFrame:
        """
        Align genes in adata to the model's gene list.
        
        Args:
            adata (AnnData): Input data
            
        Returns:
            pd.DataFrame: Gene-aligned expression matrix
        """
        if self.gene_list is None:
            raise ValueError(
                "Gene list is not available. Cannot perform gene alignment. "
                "Please provide gene_list during model initialization."
            )
        
        # Get gene expression as DataFrame
        if issparse(adata.X):
            expr = pd.DataFrame(
                adata.X.toarray(),
                index=adata.obs_names,
                columns=adata.var_names
            )
        else:
            expr = pd.DataFrame(
                adata.X,
                index=adata.obs_names,
                columns=adata.var_names
            )
        
        # Align genes
        to_fill_columns = list(set(self.gene_list) - set(expr.columns))
        if to_fill_columns:
            padding_df = pd.DataFrame(
                np.zeros((expr.shape[0], len(to_fill_columns))),
                columns=to_fill_columns,
                index=expr.index
            )
            expr = pd.concat([expr, padding_df], axis=1)
        
        # Reorder to match gene_list
        expr = expr[self.gene_list]
        
        return expr
    
    def predict_embedding(
        self,
        dataset: PerturbationData,
        output_type: Literal['cell', 'gene', 'gene_batch'] = 'cell',
        input_type: Literal['singlecell', 'bulk'] = 'singlecell',
        pool_type: Literal['all', 'max'] = 'all',
        tgthighres: str = 't4',
        pre_normalized: Literal['F', 'T', 'A'] = 'F',
        batch_size: int = 32,
        return_adata: bool = False,
        use_batch: bool = True,
    ) -> Union[np.ndarray, AnnData]:
        """
        Generate embeddings from input data.
        
        Args:
            dataset (PerturbationData): Input data using the library's unified interface.
                The PerturbationData object manages the underlying AnnData losslessly.
            output_type (str): Type of output embedding:
                - 'cell': Cell embeddings (default)
                - 'gene': Gene embeddings (processed individually)
                - 'gene_batch': Gene embeddings (batch processing)
            input_type (str): Type of input data ('singlecell' or 'bulk')
            pool_type (str): Pooling strategy for cell embeddings ('all' or 'max').
                Only valid when output_type='cell'
            tgthighres (str): Target high resolution parameter (e.g., 't4', 'a5', 'f2').
                - 't{N}': Target resolution T = N
                - 'a{N}': Additive resolution T = S + N
                - 'f{N}': Fold change resolution T = S * N
                Only valid for singlecell input
            pre_normalized (str): Normalization status:
                - 'F': Not normalized (will apply normalization)
                - 'T': Already normalized+log1p
                - 'A': Normalized+log1p with total count appended
            batch_size (int): Batch size for processing. Default: 32
            return_adata (bool): If True, return AnnData with embeddings in obsm/varm
            use_batch (bool): Whether to use batch processing. Default: True
            
        Returns:
            Union[np.ndarray, AnnData]: Embeddings array or AnnData object
            
        Examples:
            ```python
            from perturblab.data import PerturbationData
            
            # Create PerturbationData with unified interface
            pert_data = PerturbationData(adata, perturb_col='condition')
            
            # Cell embeddings with 'all' pooling
            cell_emb = model.predict_embedding(pert_data, output_type='cell', pool_type='all')
            # Shape: (n_cells, hidden_dim * 4)
            
            # Cell embeddings with 'max' pooling (with batch processing)
            cell_emb = model.predict_embedding(pert_data, output_type='cell', pool_type='max', 
                                             batch_size=64, use_batch=True)
            # Shape: (n_cells, hidden_dim)
            
            # Gene embeddings
            gene_emb = model.predict_embedding(pert_data, output_type='gene')
            # Shape: (n_cells, n_genes, hidden_dim)
            ```
        """
        # Extract AnnData from PerturbationData (lossless access)
        adata = dataset.adata
        
        # Align genes
        gexpr_feature = self._align_genes(adata)
        
        # Normalize if needed (bulk data)
        if pre_normalized == 'F' and input_type == 'bulk':
            tmp_adata = sc.AnnData(gexpr_feature.values)
            sc.pp.normalize_total(tmp_adata)
            sc.pp.log1p(tmp_adata)
            gexpr_feature = pd.DataFrame(
                tmp_adata.X,
                index=gexpr_feature.index,
                columns=gexpr_feature.columns
            )
        
        embeddings = []
        
        # Determine if we should use batch processing
        if not use_batch or output_type == 'gene':
            batch_size = 1  # Process one by one for gene embeddings or when disabled
        
        n_samples = gexpr_feature.shape[0]
        
        # Process data in batches
        with torch.no_grad():
            for start_idx in tqdm(range(0, n_samples, batch_size), desc="Generating embeddings"):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_data = gexpr_feature.iloc[start_idx:end_idx]
                
                # Prepare batch tensor
                batch_tensors = []
                
                for i in range(len(batch_data)):
                    # Prepare input tensor based on input_type
                    if input_type == 'bulk':
                        if pre_normalized == 'T':
                            totalcount = batch_data.iloc[i, :].sum()
                        elif pre_normalized == 'F':
                            totalcount = np.log10(batch_data.iloc[i, :].sum())
                        else:
                            raise ValueError('For bulk data, pre_normalized must be T or F')
                        
                        tmpdata = batch_data.iloc[i, :].tolist()
                        gene_x = torch.tensor(
                            tmpdata + [totalcount, totalcount],
                            device=self.device
                        )
                        batch_tensors.append(gene_x)
                    
                    elif input_type == 'singlecell':
                        # Pre-normalization
                        if pre_normalized == 'F':
                            tmpdata = np.log1p(
                                batch_data.iloc[i, :] / batch_data.iloc[i, :].sum() * 1e4
                            ).tolist()
                        elif pre_normalized == 'T':
                            tmpdata = batch_data.iloc[i, :].tolist()
                        elif pre_normalized == 'A':
                            tmpdata = batch_data.iloc[i, :-1].tolist()
                        else:
                            raise ValueError('pre_normalized must be T, F, or A')
                        
                        # Calculate total count
                        if pre_normalized == 'A':
                            totalcount = batch_data.iloc[i, -1]
                        else:
                            totalcount = batch_data.iloc[i, :].sum()
                        
                        # Select resolution
                        if tgthighres[0] == 'f':
                            high_res = np.log10(totalcount * float(tgthighres[1:]))
                        elif tgthighres[0] == 'a':
                            high_res = np.log10(totalcount) + float(tgthighres[1:])
                        elif tgthighres[0] == 't':
                            high_res = float(tgthighres[1:])
                        else:
                            raise ValueError('tgthighres must start with f, a, or t')
                        
                        gene_x = torch.tensor(
                            tmpdata + [high_res, np.log10(totalcount)],
                            device=self.device
                        )
                        batch_tensors.append(gene_x)
                    else:
                        raise ValueError("input_type must be 'singlecell' or 'bulk'")
                
                # Stack batch tensors
                if batch_tensors:
                    batch_gene_x = torch.stack(batch_tensors, dim=0)
                    
                    # Generate embeddings based on output_type
                    if output_type == 'cell':
                        cell_emb = self._get_cell_embedding(batch_gene_x, pool_type)
                        embeddings.append(cell_emb.detach().cpu().numpy())
                    
                    elif output_type == 'gene':
                        for j in range(batch_gene_x.shape[0]):
                            gene_emb = self._get_gene_embedding(batch_gene_x[j:j+1], single=True)
                            embeddings.append(gene_emb.detach().cpu().numpy())
                    
                    elif output_type == 'gene_batch':
                        # Process all cells in one batch
                        gene_emb = self._get_gene_embedding(batch_gene_x, single=False)
                        embeddings.append(gene_emb.detach().cpu().numpy())
                    else:
                        raise ValueError("output_type must be 'cell', 'gene', or 'gene_batch'")
        
        # Convert to array
        if embeddings:
            if output_type == 'gene_batch':
                embeddings = np.concatenate(embeddings, axis=0) if len(embeddings) > 1 else embeddings[0]
            else:
                embeddings = np.concatenate(embeddings, axis=0) if len(embeddings) > 1 else embeddings[0]
                if embeddings.ndim == 3 and embeddings.shape[0] == 1:
                    embeddings = embeddings.squeeze(0)
        else:
            raise ValueError("No embeddings were generated")
        
        # Return as AnnData if requested
        if return_adata:
            result_adata = adata.copy()
            if output_type == 'cell':
                result_adata.obsm['X_scfoundation'] = embeddings
            elif output_type in ['gene', 'gene_batch']:
                # Store gene embeddings in varm (per gene, averaged over cells)
                result_adata.varm['scfoundation_gene_emb'] = embeddings.mean(axis=0)
                # Optionally store per-cell gene embeddings in layers
                result_adata.layers['scfoundation_gene_emb'] = embeddings
            return result_adata
        
        return embeddings
    
    def _get_cell_embedding(
        self,
        gene_x: torch.Tensor,
        pool_type: str
    ) -> torch.Tensor:
        """
        Get cell embedding from gene expression.
        
        Args:
            gene_x (torch.Tensor): Gene expression tensor
            pool_type (str): Pooling type ('all' or 'max')
            
        Returns:
            torch.Tensor: Cell embedding
        """
        model_config = self.config.to_model_config_dict()
        
        # Prepare data
        data_gene_ids = torch.arange(
            self.config.max_seq_len,
            device=gene_x.device
        ).repeat(gene_x.shape[0], 1)
        value_labels = gene_x > 0
        x, x_padding = gatherData(gene_x, value_labels, model_config['pad_token_id'])
        position_gene_ids, _ = gatherData(
            data_gene_ids,
            value_labels,
            model_config['pad_token_id']
        )
        
        # Forward pass through model
        x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids)
        x += position_emb
        gene_emb = self.model.encoder(x, x_padding)
        
        # Pool embeddings
        gene_emb1 = gene_emb[:, -1, :]  # Last token
        gene_emb2 = gene_emb[:, -2, :]  # Second to last token
        gene_emb3, _ = torch.max(gene_emb[:, :-2, :], dim=1)  # Max pool
        gene_emb4 = torch.mean(gene_emb[:, :-2, :], dim=1)  # Mean pool
        
        if pool_type == 'all':
            # Concatenate all pooling strategies
            cell_emb = torch.cat([gene_emb1, gene_emb2, gene_emb3, gene_emb4], dim=1)
        elif pool_type == 'max':
            # Use max pooling only
            cell_emb, _ = torch.max(gene_emb, dim=1)
        else:
            raise ValueError("pool_type must be 'all' or 'max'")
        
        return cell_emb
    
    def _get_gene_embedding(
        self,
        gene_x: torch.Tensor,
        single: bool = True
    ) -> torch.Tensor:
        """
        Get gene embedding from gene expression.
        
        Args:
            gene_x (torch.Tensor): Gene expression tensor
            single (bool): Whether processing single cell or batch
            
        Returns:
            torch.Tensor: Gene embedding
        """
        model_config = self.config.to_model_config_dict()
        
        # Prepare encoder-decoder data
        (encoder_data, encoder_position_gene_ids, encoder_data_padding,
         encoder_labels, decoder_data, decoder_data_padding,
         new_data_raw, data_mask_labels, decoder_position_gene_ids) = \
            getEncoerDecoderData(gene_x, gene_x, model_config)
        
        # Temporarily remove final layer for embedding extraction
        original_to_final = self.model.to_final
        self.model.to_final = None
        
        # Forward pass
        out = self.model.forward(
            x=encoder_data,
            padding_label=encoder_data_padding,
            encoder_position_gene_ids=encoder_position_gene_ids,
            encoder_labels=encoder_labels,
            decoder_data=decoder_data,
            mask_gene_name=False,
            mask_labels=None,
            decoder_position_gene_ids=decoder_position_gene_ids,
            decoder_data_padding_labels=decoder_data_padding,
        )
        
        # Restore final layer
        self.model.to_final = original_to_final
        
        # Extract gene embeddings (first num_tokens positions)
        gene_emb = out[:, :self.config.num_tokens, :].contiguous()
        
        return gene_emb
    
    def train_model(
        self,
        dataset: PerturbationData,
        output_dir: str,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        mask_ratio: float = 0.3,
        warmup_steps: int = 1000,
        gradient_clip_val: float = 1.0,
        save_interval: int = 1,
        eval_interval: int = 1,
        device: Optional[str] = None,
    ):
        """
        Train the scFoundation model on masked autoencoding task.
        
        Args:
            dataset (PerturbationData): Training data
            output_dir (str): Directory to save checkpoints and logs
            epochs (int): Number of training epochs. Default: 10
            batch_size (int): Batch size for training. Default: 32
            learning_rate (float): Learning rate. Default: 1e-4
            weight_decay (float): Weight decay for optimizer. Default: 0.0
            mask_ratio (float): Ratio of genes to mask. Default: 0.3
            warmup_steps (int): Number of warmup steps. Default: 1000
            gradient_clip_val (float): Gradient clipping value. Default: 1.0
            save_interval (int): Save checkpoint every N epochs. Default: 1
            eval_interval (int): Evaluate every N epochs. Default: 1
            device (str): Device to train on. Default: self.device
        
        Note:
            This is a basic training implementation for MAE pretraining.
            For advanced features like PyTorch Lightning, mixed precision,
            or distributed training, please refer to the original scFoundation repository.
        """
        if device is None:
            device = self.device
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Linear warmup scheduler
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function (MSE for reconstruction)
        criterion = nn.MSELoss()
        
        self.train()
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Get data
            adata = dataset.adata
            
            # Split data if available
            if 'split' in adata.obs:
                train_adata = adata[adata.obs['split'] == 'train']
            else:
                train_adata = adata
            
            # Align genes
            gexpr_feature = self._align_genes(train_adata)
            n_samples = gexpr_feature.shape[0]
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            # Training loop
            progress_bar = tqdm(
                range(0, n_samples, batch_size),
                desc=f"Epoch {epoch+1}/{epochs}"
            )
            
            for start_idx in progress_bar:
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = gexpr_feature.iloc[batch_indices]
                
                # Prepare batch tensors
                batch_gene_x = []
                for i in range(len(batch_data)):
                    # Normalize if needed
                    tmpdata = np.log1p(
                        batch_data.iloc[i, :] / batch_data.iloc[i, :].sum() * 1e4
                    ).tolist()
                    totalcount = batch_data.iloc[i, :].sum()
                    
                    # Add control tokens (high_res and total_count)
                    gene_x = torch.tensor(
                        tmpdata + [4.0, np.log10(totalcount)],  # Default high_res=4
                        device=device
                    )
                    batch_gene_x.append(gene_x)
                
                batch_gene_x = torch.stack(batch_gene_x, dim=0)
                
                # Prepare encoder/decoder data
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, \
                decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, \
                decoder_position_gene_ids = getEncoerDecoderData(
                    batch_gene_x, batch_gene_x, self.config.to_model_config_dict()
                )
                
                # Forward pass
                optimizer.zero_grad()
                
                output = self.forward(
                    x=encoder_data,
                    padding_label=encoder_data_padding,
                    encoder_position_gene_ids=encoder_position_gene_ids,
                    encoder_labels=encoder_labels,
                    decoder_data=decoder_data,
                    decoder_position_gene_ids=decoder_position_gene_ids,
                    decoder_data_padding_labels=decoder_data_padding,
                    mask_gene_name=False,
                    mask_labels=None
                )
                
                # Calculate loss (only on masked positions)
                loss = criterion(output[data_mask_labels], new_data_raw[data_mask_labels])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        gradient_clip_val
                    )
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_dir = os.path.join(output_dir, f'checkpoint-epoch-{epoch+1}')
                self.save(checkpoint_dir)
                logger.info(f"Checkpoint saved to {checkpoint_dir}")
            
            # Evaluation
            if (epoch + 1) % eval_interval == 0 and 'split' in adata.obs:
                if 'val' in adata.obs['split'].values:
                    self.eval()
                    val_loss = self._evaluate(dataset, batch_size, device)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    self.train()
        
        # Save final model
        final_dir = os.path.join(output_dir, 'final_model')
        self.save(final_dir)
        logger.info(f"Training complete! Final model saved to {final_dir}")
    
    def _evaluate(
        self,
        dataset: PerturbationData,
        batch_size: int,
        device: str
    ) -> float:
        """Evaluate model on validation set."""
        adata = dataset.adata
        val_adata = adata[adata.obs['split'] == 'val']
        
        if len(val_adata) == 0:
            return 0.0
        
        gexpr_feature = self._align_genes(val_adata)
        n_samples = gexpr_feature.shape[0]
        
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_data = gexpr_feature.iloc[start_idx:end_idx]
                
                # Prepare batch (similar to training)
                batch_gene_x = []
                for i in range(len(batch_data)):
                    tmpdata = np.log1p(
                        batch_data.iloc[i, :] / batch_data.iloc[i, :].sum() * 1e4
                    ).tolist()
                    totalcount = batch_data.iloc[i, :].sum()
                    gene_x = torch.tensor(
                        tmpdata + [4.0, np.log10(totalcount)],
                        device=device
                    )
                    batch_gene_x.append(gene_x)
                
                batch_gene_x = torch.stack(batch_gene_x, dim=0)
                
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, \
                decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, \
                decoder_position_gene_ids = getEncoerDecoderData(
                    batch_gene_x, batch_gene_x, self.config.to_model_config_dict()
                )
                
                output = self.forward(
                    x=encoder_data,
                    padding_label=encoder_data_padding,
                    encoder_position_gene_ids=encoder_position_gene_ids,
                    encoder_labels=encoder_labels,
                    decoder_data=decoder_data,
                    decoder_position_gene_ids=decoder_position_gene_ids,
                    decoder_data_padding_labels=decoder_data_padding,
                    mask_gene_name=False,
                    mask_labels=None
                )
                
                loss = criterion(output[data_mask_labels], new_data_raw[data_mask_labels])
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save(self, save_directory: str):
        """
        Save model and configuration to directory.
        
        Args:
            save_directory (str): Directory to save model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, 'config.json')
        self.config.save(config_path)
        
        # Save model weights
        model_path = os.path.join(save_directory, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = 'cuda',
        **kwargs
    ) -> 'scFoundationModel':
        """
        Load a pretrained scFoundation model.
        
        Args:
            model_name_or_path (str): Model name or path to model directory.
                Can be:
                - Local directory path: "/path/to/model"
                - Built-in model name: "scfoundation-cell", "scfoundation-gene", "scfoundation-rde"
                - HuggingFace repo ID: "perturblab/scfoundation-cell"
            device (str): Device to load model on. Default: 'cuda'
            **kwargs: Additional arguments
                - Model initialization: gene_list, etc.
                - HuggingFace download: revision, token, cache_dir, force_download, etc.
            
        Returns:
            scFoundationModel: Loaded model
            
        Examples:
            ```python
            # Load from local directory
            model = scFoundationModel.from_pretrained('weights/scfoundation-cell', device='cuda')
            
            # Load built-in model
            model = scFoundationModel.from_pretrained('scfoundation-cell', device='cuda')
            
            # Load from HuggingFace
            model = scFoundationModel.from_pretrained('perturblab/scfoundation-cell', device='cuda')
            
            # Load gene embedding model
            model = scFoundationModel.from_pretrained('scfoundation-gene', device='cuda')
            ```
        """
        # Separate HuggingFace kwargs from model init kwargs
        hf_kwargs = {}
        model_init_kwargs = {}
        hf_keys = {'revision', 'token', 'cache_dir', 'force_download', 'resume_download'}
        
        for key, value in kwargs.items():
            if key in hf_keys:
                hf_kwargs[key] = value
            else:
                model_init_kwargs[key] = value
        
        # Resolve model path
        if os.path.exists(model_name_or_path):
            # It's a local path
            model_path = model_name_or_path
            logger.info(f"Loading model from local path: {model_path}")
        else:
            # Try built-in models in weights directory
            weights_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'weights')
            model_path = os.path.join(weights_dir, model_name_or_path)
            
            if not os.path.isdir(model_path):
                # Try HuggingFace download
                try:
                    logger.info(f"Attempting to download '{model_name_or_path}' from HuggingFace...")
                    model_path = download_from_huggingface(
                        model_name_or_path,
                        organization="perturblab",
                        **hf_kwargs
                    )
                    logger.info(f"✓ Model cached at: {model_path}")
                except Exception as e:
                    raise ValueError(
                        f"Model not found: {model_name_or_path}\n"
                        f"Tried local: {model_path}\n"
                        f"Tried HuggingFace: perturblab/{model_name_or_path}\n"
                        f"Please provide a valid model directory or one of: "
                        f"scfoundation-cell, scfoundation-gene, scfoundation-rde\n"
                        f"Error: {str(e)}"
                    )
        
        # Load config
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = scFoundationConfig.load(config_path)
        logger.info(f"Loaded config from {config_path}")
        
        # Initialize model
        model = cls(config, device=device, **model_init_kwargs)
        
        # Load model weights
        model_file = os.path.join(model_path, 'model.pt')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        state_dict = torch.load(model_file, map_location=device)
        model.model.load_state_dict(state_dict)
        logger.info(f"✓ Model loaded successfully from {model_file}")
        
        model.eval()
        
        return model







class scFoundationPerturbationModel(scFoundationModel):
    def __init__(
        self,
        config,
        device: str = 'cuda',
        gene_list: Optional[list[str]] = None,
        go_graph: Optional[GeneGraph] = None,
        gene_graph: Optional[GeneGraph] = None,
        **kwargs
    ):
        super().__init__(config, device, gene_list, **kwargs)
        
        # 状态容器
        self.gears_model = None  # GEARS 实例 (Head)
        
        # 图对象 (GeneGraph 实例)
        self.go_graph = go_graph
        self.gene_graph = gene_graph
        self.gears_config = None
        
        # Metadata containers (extracted from dataset)
        self.pert_names = []
        self.pert_name_to_id = {}
        self.de_gene_map = {}
        self.ctrl_expression = None
        self.dict_filter = {}



    def prepare_graphs(
        self, 
        dataset: PerturbationData, 
        temp_dir: str = '.gears_temp',
        force_recompute: bool = False,
        coexp_threshold: float = 0.4
    ):
        """
        准备 GEARS 所需的图结构。
        
        Args:
            dataset: 统一格式的 PerturbationData
            temp_dir: GEARS 处理数据的临时目录（用于 GO 图缓存）
            force_recompute: 是否强制重新计算
            coexp_threshold: 共表达图的阈值
        """
        if not dataset.gears_format:
            raise ValueError("Dataset must be in GEARS format. Call dataset.set_gears_format() first.")
        
        # 准备图结构 (GeneGraph)
        # 如果用户初始化时没有提供图，我们在这里构建
        
        # A. 构建共表达图 (Gene Graph)
        if self.gene_graph is None or force_recompute:
            logger.info("Building Co-expression Graph using GeneGraph...")
            self.gene_graph = GeneGraph.from_coexpression(
                dataset.adata, 
                gene_list=self.gene_list,
                threshold=coexp_threshold
            )
        
        # B. 构建 GO 图 (GO Graph)
        if self.go_graph is None or force_recompute:
            logger.info("Building GO Graph using GeneGraph (Default Logic)...")
            self.go_graph = GeneGraph.from_go(
                gene_list=self.gene_list,
                path=None, 
                cache_dir=os.path.join(temp_dir, 'go_graph')
            )
        
        logger.info("✓ Graphs prepared successfully.")

    def _requires_perturbation_head(func):
        """Decorator to ensure perturbation head is initialized before calling method."""
        def wrapper(self, *args, **kwargs):
            if self.gears_model is None:
                raise RuntimeError(
                    f"{func.__name__} requires perturbation head to be initialized. "
                    "Please call `init_perturbation_head()` first."
                )
            return func(self, *args, **kwargs)
        return wrapper

    def init_perturbation_head(
        self,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        num_similar_genes_go_graph: int = 20,
        num_similar_genes_co_express_graph: int = 20,
        coexpress_threshold: float = 0.4,
        uncertainty: bool = False,
        uncertainty_reg: float = 1,
        direction_lambda: float = 1e-1,
        no_perturb: bool = False,
        cell_fitness_pred: bool = False,
        weight_bias_track: bool = False,
        proj_name: str = 'GEARS',
        exp_name: str = 'scFoundation-GEARS',
        **kwargs
    ):
        """
        初始化 GEARS 下游头（Head），并注入 GeneGraph 定义的图结构。
        
        Args:
            dataset: PerturbationData 对象，用于构造 GEARS 所需的数据结构
            go_graph: GO 图对象（可选）
            gene_graph: 共表达图对象（可选）
            其他参数: GEARS 模型配置参数
        """
        # 1. 确定使用的图对象
        # 优先级: 参数传入 > self.成员变量
        active_go_graph = self.go_graph
        active_gene_graph =  self.gene_graph
        
        if active_go_graph is None or active_gene_graph is None:
            raise RuntimeError(
                "Graph objects missing. Please provide `go_graph/gene_graph` or call `prepare_graphs`."
            )
        
        # 2. Extract metadata from dataset (one-time setup)
        logger.info("Extracting metadata from dataset...")
        if 'ctrl_indices' not in adata.obsm:
            raise ValueError("Run dataset.pair_cells() first.")
        
        # 2.1 Perturbation mapping
        pert_conditions = [c for c in adata.obs['condition'].unique() if c != 'ctrl']
        unique_perts = set()
        for p in pert_conditions:
            unique_perts.update([g for g in p.split('+') if g != 'ctrl'])
        self.pert_names = sorted(list(unique_perts))
        self.pert_name_to_id = {p: i for i, p in enumerate(self.pert_names)}
        
        # 2.2 Control expression (basal state)
        ctrl_mask = adata.obs['condition'] == 'ctrl'
        ctrl_X = adata.X[ctrl_mask]
        if sparse.issparse(ctrl_X):
            ctrl_X = ctrl_X.toarray()
        else:
            ctrl_X = np.asarray(ctrl_X)
        self.ctrl_expression = torch.tensor(np.mean(ctrl_X, axis=0), dtype=torch.float).to(self.device)
        
        
        # Convert DE gene names to indices in our gene_list
        gene_name_to_idx = {name: i for i, name in enumerate(adata.var['gene_name'])}
        gene_list_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}
        
        rank_data = adata.uns['rank_genes_groups_cov_all']
        top_n = adata.uns.get('top_de_n', 20)
        
        self.de_gene_map = {}
        if hasattr(rank_data, 'dtype') and hasattr(rank_data.dtype, 'names'):
            for cond in rank_data['names'].dtype.names:
                top_genes = rank_data['names'][cond][:top_n]
                # Map from adata gene names to model gene_list indices
                de_indices = []
                for g in top_genes:
                    if g in gene_name_to_idx:
                        gene_name = adata.var['gene_name'].iloc[gene_name_to_idx[g]]
                        if gene_name in gene_list_to_idx:
                            de_indices.append(gene_list_to_idx[gene_name])
                self.de_gene_map[cond] = de_indices[:top_n] if de_indices else [-1] * top_n
        else:
            for cond in rank_data.keys():
                if cond not in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']:
                    top_genes = rank_data[cond][:top_n]
                    de_indices = []
                    for g in top_genes:
                        if g in gene_name_to_idx:
                            gene_name = adata.var['gene_name'].iloc[gene_name_to_idx[g]]
                            if gene_name in gene_list_to_idx:
                                de_indices.append(gene_list_to_idx[gene_name])
                    self.de_gene_map[cond] = de_indices[:top_n] if de_indices else [-1] * top_n
        
        # 2.4 Dict filter (non_zeros_gene_idx)
        self.dict_filter = {}
        if 'non_zeros_gene_idx' in adata.uns:
            pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)
            self.dict_filter = {
                pert_full_id2pert[i]: j 
                for i, j in adata.uns['non_zeros_gene_idx'].items() 
                if i in pert_full_id2pert
            }

        # 3. 图对齐 (Graph Alignment) 
        # 模型的 gene_list 可能与图的 gene_list 不同。
        # 我们使用 GeneGraph.subset 方法将图"切"成模型需要的大小和顺序。
        logger.info(f"Aligning graphs to model gene list (Size: {len(self.gene_list)})...")
        
        aligned_go = active_go_graph.subset(self.gene_list)
        aligned_gene = active_gene_graph.subset(self.gene_list)
        
        # 提取 PyG 格式的 edge_index
        edge_index_go = aligned_go.edge_index.to(self.device)
        edge_weight_go = aligned_go.edge_weight.to(self.device) if aligned_go.edge_weight is not None else None
        
        edge_index_gene = aligned_gene.edge_index.to(self.device)
        edge_weight_gene = aligned_gene.edge_weight.to(self.device) if aligned_gene.edge_weight is not None else None

        # 4. 保存配置 (不包含 Tensor 数据，只保存超参)
        self.gears_config = {
            'hidden_size': hidden_size,
            'num_go_gnn_layers': num_go_gnn_layers,
            'num_gene_gnn_layers': num_gene_gnn_layers,
            'decoder_hidden_size': decoder_hidden_size,
            'num_similar_genes_go_graph': num_similar_genes_go_graph,
            'num_similar_genes_co_express_graph': num_similar_genes_co_express_graph,
            'coexpress_threshold': coexpress_threshold,
            'uncertainty': uncertainty,
            'uncertainty_reg': uncertainty_reg,
            'direction_lambda': direction_lambda,
            'no_perturb': no_perturb,
            'cell_fitness_pred': cell_fitness_pred,
            'weight_bias_track': weight_bias_track,
            'proj_name': proj_name,
            'exp_name': exp_name,
            **kwargs
        }

        # 5. 创建一个最小化的 PertData 对象用于 GEARS 初始化
        # GEARS 需要从 pert_data 中读取一些属性，我们创建一个简单的包装类
        class MinimalPertData:
            def __init__(self, adata, pert_names, gene_names):
                self.adata = adata
                self.node_map = {g: i for i, g in enumerate(gene_names)}
                self.node_map_pert = {p: i for i, p in enumerate(pert_names)}
                self.gene_names = pd.Series(gene_names)
                self.pert_names = np.array(pert_names)
                self.set2conditions = {}
                if 'split' in adata.obs:
                    self.set2conditions = adata.obs.groupby('split')['condition'].unique().to_dict()
                    self.set2conditions = {k: list(v) for k, v in self.set2conditions.items()}
                self.split = adata.obs.get('split', 'custom').iloc[0] if 'split' in adata.obs else 'custom'
                self.data_path = ''
                self.dataset_name = 'custom'
                self.default_pert_graph = False
                self._dataloader = None
            
            def get_dataloader(self, batch_size, test_batch_size=None):
                # This will be set externally via gears_model.dataloader
                return self._dataloader
        
        # 创建最小化的 PertData 对象
        gene_names = adata.var['gene_name'].tolist()
        minimal_pert_data = MinimalPertData(adata, self.pert_names, gene_names)
        
        # 6. 初始化 GEARS 实例
        logger.info(f"Initializing GEARS Head...")
        
        self.gears_model = GEARS(
            minimal_pert_data,
            device=self.device,
            weight_bias_track=weight_bias_track,
            proj_name=proj_name,
            exp_name=exp_name
        )
        
        # 7. 手动设置 GEARS 需要的额外属性
        self.gears_model.ctrl_expression = self.ctrl_expression
        self.gears_model.dict_filter = self.dict_filter
        self.gears_model.ctrl_adata = adata[adata.obs['condition'] == 'ctrl']
        
        # 构造 pert2gene 映射 (pert index -> gene index in model gene_list)
        gene_list_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}
        self.gears_model.pert2gene = {
            i: gene_list_to_idx[pert] 
            for i, pert in enumerate(self.pert_names) 
            if pert in gene_list_to_idx
        }

        # 8. 模型初始化与注入
        self.gears_model.model_initialize(
            model_type='scfoundation',
            load_path=None, 
            **self.gears_config
        )
        
        # 9. 【关键步骤】注入 GeneGraph 的图结构
        # 覆盖 GEARS 内部自动生成的图，强制使用我们要的图
        # GEARS model 内部通常将 GO 图存储在 model.sim_network (或者 model.edge_index_go，视版本而定)
        # 这里假设是标准的 GEARS SG 模块接口
        
        # 注入 GO 图
        if hasattr(self.gears_model.model, 'sim_network'):
            self.gears_model.model.sim_network = Data(
                edge_index=edge_index_go, 
                edge_weight=edge_weight_go, 
                num_nodes=len(self.gene_list)
            )
        
        # 注入 scFoundation Encoder
        self.gears_model.model.singlecell_model = self.model
        self.gears_model.model.pretrained = True
        
        self.gears_model.model.to(self.device)
        
        logger.info("✓ GEARS Head initialized with aligned GeneGraphs.")

    @_requires_perturbation_head
    def train_model(self, dataset: PerturbationData, epochs=20, lr=1e-3, batch_size=32, result_dir='./results', **kwargs):
        """
        训练模型 (需要先初始化 Head)
        
        使用 PerturbationPyGDataset 构造 DataLoader，完全解耦 GEARS 的数据结构。
        """
        adata = dataset.adata
        
        if 'split' not in adata.obs:
            raise ValueError("Dataset must have 'split' column. Call dataset.split_data() first.")
        
        if 'ctrl_indices' not in adata.obsm:
            raise ValueError("Run dataset.pair_cells() first.")
        
        # 1. Create Datasets using PerturbationPyGDataset
        # Pass original adata for proper index mapping
        train_ds = PerturbationPyGDataset(
            adata[adata.obs['split'] == 'train'],
            self.gene_list,
            self.pert_name_to_id,
            self.de_gene_map,
            adata.obsm['ctrl_indices'],
            num_samples_per_cell=1,
            original_adata=adata
        )
        
        val_ds = PerturbationPyGDataset(
            adata[adata.obs['split'] == 'val'],
            self.gene_list,
            self.pert_name_to_id,
            self.de_gene_map,
            adata.obsm['ctrl_indices'],
            num_samples_per_cell=1,
            original_adata=adata
        )
        
        # 2. Create Loaders
        loaders = {
            'train_loader': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            'val_loader': DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        }
        
        # Add test loader if available
        if 'test' in adata.obs['split'].values:
            test_ds = PerturbationPyGDataset(
                adata[adata.obs['split'] == 'test'],
                self.gene_list,
                self.pert_name_to_id,
                self.de_gene_map,
                adata.obsm['ctrl_indices'],
                num_samples_per_cell=1,
                original_adata=adata
            )
            loaders['test_loader'] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # 3. Inject Loaders into GEARS and Train
        self.gears_model.dataloader = loaders
        
        logger.info(f"Starting training in {result_dir}...")
        self.gears_model.train(epochs=epochs, lr=lr, result_dir=result_dir, **kwargs)
        # Note: GEARS internally maintains best_model in self.gears_model.best_model
        # Weight saving should be handled externally via save_pretrained()
        return {}

    @_requires_perturbation_head
    def predict_perturbation(self, dataset: PerturbationData, batch_size=32, **kwargs):
        """
        预测 (需要先初始化 Head)
        
        使用 PerturbationPyGDataset 构造测试 DataLoader。
        """
        from ..gears.source.inference import evaluate
        
        adata = dataset.adata
        
        if 'split' not in adata.obs or 'test' not in adata.obs['split'].values:
            raise ValueError("Dataset must have test split. Call dataset.split_data() first.")
        
        if 'ctrl_indices' not in adata.obsm:
            raise ValueError("Run dataset.pair_cells() first.")
        
        # Create test dataset
        test_ds = PerturbationPyGDataset(
            adata[adata.obs['split'] == 'test'],
            self.gene_list,
            self.pert_name_to_id,
            self.de_gene_map,
            adata.obsm['ctrl_indices'],
            num_samples_per_cell=1,
            original_adata=adata
        )
        
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # Use best_model from GEARS if available, otherwise use current model
        model_to_use = getattr(self.gears_model, 'best_model', None) or self.gears_model.model
        test_res = evaluate(
            test_loader, 
            model_to_use,
            self.gears_config.get('uncertainty', False),
            self.device
        )
        
        return test_res

    def save_pretrained(self, save_directory: str):
        """
        保存模型：包含基础模型权重、配置以及 GEARS Head 的配置和权重。
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存 scFoundation 基础配置 (config.json) 和权重 (model.pt)
        # 调用父类的 save 方法 (假设父类只保存基础部分)
        super().save(save_directory)
        
        # 2. 如果 GEARS Head 存在，保存其特有信息
        if self.gears_model is not None and self.gears_config is not None:
            # 保存 GEARS 配置
            gears_config_path = os.path.join(save_directory, 'gears_config.json')
            with open(gears_config_path, 'w') as f:
                json.dump(self.gears_config, f, indent=2)
            
            # 保存 GEARS Head 权重 (包含 GNN, Decoder 等)
            # 这里的策略是保存整个 state_dict (包含 encoder)，加载时再处理
            # 或者仅保存 head 部分。为简单起见，GEARS 的 save_model 通常保存整个模型。
            # 我们将保存一个专门的 gears_head.pt
            head_path = os.path.join(save_directory, 'gears_head.pt')
            torch.save(self.gears_model.model.state_dict(), head_path)
            logger.info(f"Saved GEARS head config and weights to {save_directory}")
        else:
            logger.info("Only base model saved (GEARS head not initialized).")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = 'cuda',
        **kwargs
    ) -> 'scFoundationPerturbationModel':
        """
        加载模型。
        
        逻辑：
        1. 加载基础 scFoundation 模型。
        2. 检查 model_name_or_path 是否为路径且包含 'gears_config.json'。
        3. 如果存在，则自动恢复 GEARS Head。
        4. 如果不存在 (或是模型名称)，则仅返回基础模型，GEARS Head 为 None。
        """
        # 1. 实例化基础部分 (利用父类的加载逻辑)
        # 注意：这里我们调用父类的 from_pretrained，但父类返回的是 scFoundationModel 类实例
        # 我们需要一种方式将其转换为 scFoundationPerturbationModel，
        # 或者直接在该类中重新实现加载逻辑。
        
        # 更稳健的做法是：复用父类逻辑获取 config 和 state_dict，然后用 cls 初始化
        
        # 假设父类逻辑能处理路径解析
        base_model = super().from_pretrained(model_name_or_path, device, **kwargs)
        
        # 创建当前类的实例 (复制基础模型的属性)
        model = cls(base_model.config, device=device, gene_list=base_model.gene_list)
        model.model = base_model.model # 共享权重
        
        # 2. 检查是否存在 GEARS Head 组件
        gears_config_path = os.path.join(model_name_or_path, 'gears_config.json')
        gears_weight_path = os.path.join(model_name_or_path, 'gears_head.pt')
        
        if os.path.exists(gears_config_path) and os.path.exists(gears_weight_path):
            logger.info(f"Found GEARS head configuration at {gears_config_path}. Restoring...")
            
            # 加载配置
            with open(gears_config_path, 'r') as f:
                gears_config = json.load(f)
            
            # 这是一个特殊的恢复情况：
            # 我们没有 PertData 数据对象 (因为它通常太大不保存)，但我们需要初始化架构。
            # 这是一个挑战：GEARS 初始化通常强依赖数据来推断 num_genes 等。
            # 解决方案：我们需要在 gears_config 中保存所有必要的维度参数。
            
            # 假设 gears_config 包含了所有初始化参数
            # 并且我们使用一种不依赖数据的 "Dummy" 初始化方式，
            # 或者用户需要在加载后手动 set data。
            
            # 这里我们尝试仅恢复架构 (Assuming GEARS supports instantiation without full PertData if dims are known)
            # 如果 GEARS 库不支持无数据初始化，这里会报错。
            # 为了代码的鲁棒性，我们通常在此处仅加载 Config，提示用户需要调用 prepare_graphs
            
            logger.warning(
                "Detected GEARS checkpoint. "
                "The configuration is loaded to `model.gears_config`. "
                "Please call `model.prepare_graphs(dataset)` followed by "
                "`model.init_perturbation_head(**model.gears_config)` to fully restore the head structure, "
                "then the weights will be loaded automatically if you call load_weights."
            )
            # 实际上，为了方便，我们可以尝试部分初始化，但最安全的是让用户提供数据
            model.gears_config = gears_config
            
            # 我们可以保存权重路径供后续使用
            model._pending_gears_weights = gears_weight_path
            
        else:
            logger.info(f"No GEARS head found in {model_name_or_path}. Initialized as base model only.")
            
        return model

    def load_head_weights(self):
        """
        在 prepare_graphs 和 init_perturbation_head 之后调用，
        用于从 _pending_gears_weights 加载实际权重。
        """
        if hasattr(self, '_pending_gears_weights') and self._pending_gears_weights:
            logger.info(f"Loading GEARS weights from {self._pending_gears_weights}...")
            state_dict = torch.load(self._pending_gears_weights, map_location=self.device)
            self.gears_model.model.load_state_dict(state_dict)
            del self._pending_gears_weights
            logger.info("✓ GEARS weights restored.")
        else:
            logger.warning("No pending weights to load.")

