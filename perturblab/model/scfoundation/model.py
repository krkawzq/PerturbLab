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
from ..gears import GearsModel, GearsConfig
from .config import scFoundationConfig
from .source.load import gatherData, getEncoerDecoderData
from .source.pretrainmodels import select_model

logger = logging.getLogger(__name__)

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
        
        # Get data first to calculate total steps for scheduler
        adata = dataset.adata
        if 'split' in adata.obs:
            train_adata = adata[adata.obs['split'] == 'train']
        else:
            train_adata = adata
        
        # Align genes
        gexpr_feature = self._align_genes(train_adata)
        n_samples = gexpr_feature.shape[0]
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_steps = epochs * steps_per_epoch
        
        # Setup optimizer (对齐源代码：AdamW, lr=1e-4, weight_decay=0)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Warmup scheduler (对齐源代码：warmup_steps=9766, max_lr=1e-4)
        # 使用线性 warmup，之后保持固定学习率（对齐源代码配置）
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup (对齐源代码)
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 保持固定学习率（对齐源代码：没有使用 cosine annealing）
                return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function (MSE for reconstruction，对齐源代码)
        criterion = nn.MSELoss(reduction='mean')
        
        self.model.train()
        
        logger.info("="*60)
        logger.info("Starting scFoundation training")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Gradient clip value: {gradient_clip_val}")
        logger.info(f"Total samples: {n_samples}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*60)
        
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle indices each epoch (对齐源代码)
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
                
                # Prepare batch tensors (对齐源代码的数据准备方式)
                batch_gene_x = []
                batch_gene_x_raw = []
                for i in range(len(batch_data)):
                    # Normalize: log1p(normalize to 10k)
                    tmpdata = batch_data.iloc[i, :].values
                    totalcount = tmpdata.sum()
                    
                    # Normalize and log transform
                    normalized = np.log1p(tmpdata / totalcount * 1e4)
                    totalcount_log = np.log10(totalcount) if totalcount > 0 else 0.0
                    
                    # Add control tokens: [high_res_token, total_count_token]
                    # high_res 默认使用 4.0 (对应 tgthighres=a5 或其他配置)
                    # 根据源代码，high_res 可以是固定值或基于 totalcount 计算
                    high_res_token = 4.0  # 默认值，可根据配置调整
                    
                    gene_x = torch.tensor(
                        normalized.tolist() + [high_res_token, totalcount_log],
                        device=device,
                        dtype=torch.float32
                    )
                    batch_gene_x.append(gene_x)
                    batch_gene_x_raw.append(gene_x.clone())
                
                batch_gene_x = torch.stack(batch_gene_x, dim=0)
                batch_gene_x_raw = torch.stack(batch_gene_x_raw, dim=0)
                
                # Prepare encoder/decoder data (对齐源代码)
                model_config = self.config.to_model_config_dict()
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, \
                decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, \
                decoder_position_gene_ids = getEncoerDecoderData(
                    batch_gene_x, batch_gene_x_raw, model_config
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
                
                # Calculate loss (对齐源代码：MSE loss on all decoder positions)
                # 注意：根据源代码，data_mask_labels 可能为 None，此时计算所有位置的损失
                if data_mask_labels is not None:
                    loss = criterion(output[data_mask_labels], new_data_raw[data_mask_labels])
                else:
                    # 如果没有 mask labels，计算所有非 padding 位置的损失
                    valid_mask = ~decoder_data_padding
                    loss = criterion(output[valid_mask], new_data_raw[valid_mask])
                
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
                
                # Update progress bar (对齐源代码的日志格式)
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}',
                    'lr': f'{current_lr:.6f}',
                    'step': f'{global_step}/{total_steps}'
                })
                
                # Log at intervals (对齐源代码的日志间隔，每100步)
                if global_step % 100 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {loss.item():.4f} | Avg Loss: {epoch_loss/num_batches:.4f} | "
                        f"LR: {current_lr:.6f}"
                    )
            
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
    """
    scFoundation model for perturbation prediction using GEARS framework.
    
    This model extends scFoundationModel to perform gene perturbation prediction.
    It uses the encapsulated GearsModel with scFoundation as the base encoder.
    
    Architecture:
        scFoundation (encoder) → GEARS (GNN + decoder) → Perturbation prediction
    
    Features:
        - Graph Neural Networks for gene relationships
        - Perturbation-specific embeddings
        - Gene-specific decoders
        - Support for single and combinatorial perturbations
    
    Example:
        ```python
        from perturblab.data import PerturbationData
        from perturblab.model.scfoundation import scFoundationPerturbationModel
        
        # Create model
        model = scFoundationPerturbationModel.from_pretrained(
            'scfoundation-cell',
            device='cuda'
        )
        
        # Prepare data (must be in GEARS format)
        pert_data = PerturbationData(adata, perturb_col='condition')
        pert_data.set_gears_format(fallback_cell_type='unknown')
        pert_data.split_data(split_type='simulation', seed=1)
        pert_data.pair_cells()
        pert_data.compute_de_genes(n_top_genes=20)
        
        # Extract perturbation list from dataset
        pert_conditions = [c for c in pert_data.adata.obs['condition'].unique() if c != 'ctrl']
        unique_perts = set()
        for p in pert_conditions:
            unique_perts.update([g for g in p.split('+') if g != 'ctrl'])
        pert_list = sorted(list(unique_perts))
        
        # Build graphs using GeneGraph built-in methods
        from perturblab.data import GeneGraph
        go_graph = GeneGraph.from_go(pert_list, cache_dir='.gears_temp/go_graph')
        co_graph = GeneGraph.from_coexpression(pert_data.adata, gene_list=model.gene_list)
        
        # Initialize perturbation head
        gears_config = model.config.get_gears_config()
        model.init_perturbation_head(
            gears_config=gears_config,
            gene_list=model.gene_list,
            pert_list=pert_list,
            go_graph=go_graph,
            co_graph=co_graph
        )
        
        # Train
        model.train_model(pert_data, epochs=20, lr=1e-3)
        
        # Predict
        predictions = model.predict_perturbation(pert_data, split='test')
        ```
    """
    
    def __init__(
        self,
        config,
        device: str = 'cuda',
        gene_list: Optional[list[str]] = None,
        go_graph: Optional[GeneGraph] = None,
        gene_graph: Optional[GeneGraph] = None,
        # GEARS head initialization parameters
        gears_config: Optional[GearsConfig] = None,
        pert_list: Optional[list[str]] = None,
        pert_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Initialize scFoundation perturbation model.
        
        Args:
            config: Model configuration
            device: Device to use. Default: 'cuda'
            gene_list: List of gene names
            go_graph: Pre-built GO graph (optional)
            gene_graph: Pre-built co-expression graph (optional)
            gears_config: GearsConfig for perturbation head initialization (optional)
            pert_list: List of perturbations (required if initializing head in __init__)
            pert_embeddings: Pre-trained perturbation embeddings (optional)
            gene_embeddings: Pre-trained gene embeddings (optional)
            **kwargs: Additional arguments
        
        Note:
            If `gears_config`, `pert_list`, `go_graph`, and `co_graph` (gene_graph) are all provided,
            the perturbation head will be automatically initialized in __init__.
        """
        super().__init__(config, device, gene_list, **kwargs)
        
        # 使用封装的 GearsModel 而不是原生 GEARS
        self.gears_model: Optional[GearsModel] = None  # GearsModel 实例 (Head)
        
        # 图对象 (GeneGraph 实例)
        self.go_graph = go_graph
        self.gene_graph = gene_graph
        self.gears_config: Optional[GearsConfig] = None
        
        # Metadata containers (extracted from dataset)
        self.pert_names: list[str] = []
        self.pert_name_to_id: dict[str, int] = {}
        self.de_gene_map: dict[str, list[int]] = {}
        
        # 如果提供了所有必要的参数，自动初始化 GEARS head
        if pert_list is not None and go_graph is not None and gene_graph is not None:
            if gene_list is None:
                raise ValueError("gene_list is required when initializing perturbation head in __init__")
            
            # 使用 config 中的 GEARS 参数创建配置（如果未提供）
            if gears_config is None:
                gears_config = self.config.get_gears_config()
            
            # 自动调用 init_perturbation_head
            self.init_perturbation_head(
                gears_config=gears_config,
                gene_list=gene_list,
                pert_list=pert_list,
                go_graph=go_graph,
                co_graph=gene_graph,
                pert_embeddings=pert_embeddings,
                gene_embeddings=gene_embeddings
            )
            logger.info("✓ Perturbation head initialized automatically in __init__")
        else:
            logger.info("Initialized scFoundation perturbation model (head not initialized)")




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
        gears_config: GearsConfig,
        gene_list: list[str],
        pert_list: list[str],
        go_graph: GeneGraph,
        co_graph: GeneGraph,
        pert_embeddings: torch.Tensor = None,
        gene_embeddings: torch.Tensor = None,
    ):
        """
        初始化 GEARS 下游头（Head），直接传入图结构和配置。
        
        类似于 GearsModel.__init__，直接使用提供的图结构和配置初始化。
        
        Args:
            gears_config: GearsConfig 配置对象
            gene_list: 基因列表
            pert_list: 扰动列表
            go_graph: GO 图（GeneGraph 对象）
            co_graph: 共表达图（GeneGraph 对象）
            pert_embeddings: 预训练的扰动嵌入（可选）
            gene_embeddings: 预训练的基因嵌入（可选）
        
        Example:
            ```python
            from perturblab.model.gears import GearsConfig
            
            # 创建 GEARS 配置
            gears_config = GearsConfig(
                hidden_size=64,
                num_go_gnn_layers=1,
                num_gene_gnn_layers=1
            )
            
            # 初始化扰动头
            model.init_perturbation_head(
                gears_config=gears_config,
                gene_list=gene_list,
                pert_list=pert_list,
                go_graph=go_graph,
                co_graph=co_graph
            )
            ```
        """
        # 1. 创建 GearsModel 实例
        logger.info("Initializing GearsModel with provided graphs and config...")
        self.gears_model = GearsModel(
            config=gears_config,
            gene_list=gene_list,
            pert_list=pert_list,
            go_graph=go_graph,
            co_graph=co_graph,
            pert_embeddings=pert_embeddings,
            gene_embeddings=gene_embeddings,
            device=self.device
        )
        
        # 2. 注入 scFoundation 编码器到 GEARS 模型中（参照原始实现）
        if hasattr(self.gears_model.gears_model, 'singlecell_model'):
            self.gears_model.gears_model.singlecell_model = self.model
            self.gears_model.gears_model.pretrained = True
            logger.info("✓ Injected scFoundation encoder into GEARS model")
        else:
            logger.warning(
                "GEARS model does not have 'singlecell_model' attribute. "
                "scFoundation encoder may not be used directly in GEARS forward pass."
            )
        
        # 3. 保存配置和元数据
        self.gears_config = gears_config
        self.pert_names = pert_list
        self.pert_name_to_id = {p: i for i, p in enumerate(pert_list)}
        
        logger.info("✓ GearsModel Head initialized successfully")
        logger.info(f"  - Base encoder: scFoundation ({self.config.model_series}-{self.config.model_name})")
        logger.info(f"  - Hidden size: {gears_config.hidden_size}")
        logger.info(f"  - GO GNN layers: {gears_config.num_go_gnn_layers}")
        logger.info(f"  - Gene GNN layers: {gears_config.num_gene_gnn_layers}")
        logger.info(f"  - Number of genes: {len(gene_list)}")
        logger.info(f"  - Number of perturbations: {len(pert_list)}")
    
    def init_perturbation_head_from_dataset(
        self,
        dataset: PerturbationData,
        gears_config: Optional[GearsConfig] = None,
        **kwargs
    ):
        """
        从数据集初始化 GEARS 下游头（Head）。
        
        参照原始实现逻辑，从数据集中提取信息并构建图结构，然后初始化 GearsModel。
        如果 gears_config 为 None，则使用 self.config 中的 GEARS 参数创建配置。
        
        Args:
            dataset: PerturbationData 对象，用于构造 GEARS 所需的数据结构
            gears_config: GearsConfig 配置对象。如果为 None，则从 self.config 创建
            **kwargs: 其他 GEARS 配置参数（仅在 gears_config 为 None 时使用）
        
        Raises:
            ValueError: 如果数据集格式不正确
            RuntimeError: 如果图对象缺失
        """
        # 1. 验证数据集格式
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type='unknown')
        
        required_fields = ['condition', 'split', 'ctrl_indices']
        missing = []
        if 'condition' not in dataset.adata.obs:
            missing.append('condition (use set_gears_format)')
        if 'split' not in dataset.adata.obs:
            missing.append('split (use split_data)')
        if 'ctrl_indices' not in dataset.adata.obsm:
            missing.append('ctrl_indices (use pair_cells)')
        
        if missing:
            raise ValueError(
                f"Dataset missing required fields: {', '.join(missing)}"
            )
        
        adata = dataset.adata
        
        # 2. 提取扰动列表
        pert_conditions = [c for c in adata.obs['condition'].unique() if c != 'ctrl']
        unique_perts = set()
        for p in pert_conditions:
            unique_perts.update([g for g in p.split('+') if g != 'ctrl'])
        pert_list = sorted(list(unique_perts))
        
        if not pert_list:
            raise ValueError("No perturbations found in dataset.")
        
        # 3. 构建图对象（如果缺失）
        if self.go_graph is None:
            logger.info("Building GO Graph using GeneGraph.from_go()...")
            self.go_graph = GeneGraph.from_go(
                gene_list=pert_list,
                cache_dir='.gears_temp/go_graph'
            )
        
        if self.gene_graph is None:
            logger.info("Building Co-expression Graph using GeneGraph.from_coexpression()...")
            self.gene_graph = GeneGraph.from_coexpression(
                adata,
                gene_list=self.gene_list
            )
        
        # 4. 创建或使用 GearsConfig
        if gears_config is None:
            # 从 self.config 创建 GearsConfig
            gears_config = self.config.get_gears_config()
            # 如果提供了 kwargs，更新配置
            if kwargs:
                gears_config_dict = gears_config.to_dict()
                gears_config_dict.update(kwargs)
                gears_config = GearsConfig(**gears_config_dict)
        
        # 5. 对齐图到模型的 gene_list 和 pert_list
        logger.info(f"Aligning graphs to model gene list (Size: {len(self.gene_list)}) and pert list (Size: {len(pert_list)})...")
        
        # GO 图应该对齐到 pert_list
        aligned_go = self.go_graph.subset(pert_list)
        # 共表达图应该对齐到 gene_list
        aligned_gene = self.gene_graph.subset(self.gene_list)
        
        # 6. 调用 init_perturbation_head 初始化
        self.init_perturbation_head(
            gears_config=gears_config,
            gene_list=self.gene_list,
            pert_list=pert_list,
            go_graph=aligned_go,
            co_graph=aligned_gene
        )
        
        # 7. 提取 DE 基因映射（如果需要）
        if 'rank_genes_groups_cov_all' in adata.uns:
            gene_name_to_idx = {name: i for i, name in enumerate(adata.var['gene_name'])}
            gene_list_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}
            
            rank_data = adata.uns['rank_genes_groups_cov_all']
            top_n = adata.uns.get('top_de_n', 20)
            
            self.de_gene_map = {}
            if hasattr(rank_data, 'dtype') and hasattr(rank_data.dtype, 'names'):
                for cond in rank_data['names'].dtype.names:
                    top_genes = rank_data['names'][cond][:top_n]
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
                        top_genes = rank_data[cond][:top_n] if isinstance(rank_data[cond], (list, np.ndarray)) else []
                        de_indices = []
                        for g in top_genes:
                            if g in gene_name_to_idx:
                                gene_name = adata.var['gene_name'].iloc[gene_name_to_idx[g]]
                                if gene_name in gene_list_to_idx:
                                    de_indices.append(gene_list_to_idx[gene_name])
                        self.de_gene_map[cond] = de_indices[:top_n] if de_indices else [-1] * top_n
        
        logger.info("✓ GearsModel Head initialized from dataset successfully")

    @_requires_perturbation_head
    def train_model(
        self,
        dataset: PerturbationData,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        train_split: str = 'train',
        val_split: str = 'val',
        result_dir: str = './results',
        log_interval: int = 50,
        save_best: bool = True,
        **kwargs
    ):
        """
        训练扰动预测模型。
        
        参照原始 GEARS 实现逻辑，使用封装的 GearsModel 的训练方法。
        
        Args:
            dataset: 训练数据
            epochs: 训练轮数。Default: 20
            lr: 学习率。Default: 1e-3
            weight_decay: 权重衰减。Default: 5e-4
            batch_size: 批次大小。Default: 32
            train_split: 训练集分割。Default: 'train'
            val_split: 验证集分割。Default: 'val'
            result_dir: 结果保存目录。Default: './results'
            log_interval: 日志打印间隔（步数）。Default: 50
            save_best: 是否保存最佳模型。Default: True
            **kwargs: 其他训练参数
        
        Returns:
            Dict: 训练历史，包含每个 epoch 的指标
                - 'train_loss': 训练损失列表
                - 'train_mse': 训练集整体 MSE 列表
                - 'train_mse_de': 训练集 DE 基因 MSE 列表
                - 'val_mse': 验证集整体 MSE 列表
                - 'val_mse_de': 验证集 DE 基因 MSE 列表
        """
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type='unknown')
        
        if 'split' not in dataset.adata.obs.columns:
            raise ValueError("Dataset must have 'split' column. Call dataset.split_data() first.")
        
        if 'ctrl_indices' not in dataset.adata.obsm:
            raise ValueError("Run dataset.pair_cells() first.")
        
        os.makedirs(result_dir, exist_ok=True)
        
        logger.info("="*60)
        logger.info("Starting perturbation prediction training")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Result directory: {result_dir}")
        logger.info("="*60)
        
        # 使用 GearsModel 的训练方法
        save_path = os.path.join(result_dir, 'best_model') if save_best else None
        history = self.gears_model.train_model(
            dataset=dataset,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            log_interval=log_interval,
            save_best=save_best,
            save_path=save_path,
            **kwargs
        )
        
        logger.info("✓ Training completed")
        
        return history

    @_requires_perturbation_head
    def predict_perturbation(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: str = 'test',
        return_numpy: bool = True,
        **kwargs
    ):
        """
        预测基因表达变化。
        
        参照原始实现逻辑，使用 GearsModel 进行预测。
        
        Args:
            dataset: 测试数据
            batch_size: 批次大小
            split: 数据分割（'train', 'val', 'test', 'all'）
            return_numpy: 是否返回 NumPy 数组
            **kwargs: 其他预测参数
        
        Returns:
            Dict: 包含预测结果和指标的字典
                - 'pred': 预测值
                - 'pert_cat': 扰动类别
                - 'truth': 真实值（如果可用）
                - 'logvar': 不确定性方差（如果启用）
        """
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type='unknown')
        
        if 'split' not in dataset.adata.obs.columns:
            raise ValueError("Dataset must have 'split' column. Call dataset.split_data() first.")
        
        if split != 'all' and split not in dataset.adata.obs['split'].values:
            available_splits = dataset.adata.obs['split'].unique()
            raise ValueError(
                f"Split '{split}' not found in dataset. "
                f"Available splits: {available_splits}"
            )
        
        if 'ctrl_indices' not in dataset.adata.obsm:
            raise ValueError("Run dataset.pair_cells() first.")
        
        logger.info(f"Predicting perturbation effects for split: {split}")
        
        # 直接使用 GearsModel 的预测方法
        results = self.gears_model.predict_perturbation(
            dataset=dataset,
            batch_size=batch_size,
            split=split,
            return_numpy=return_numpy,
            **kwargs
        )
        
        logger.info("✓ Prediction completed")
        if isinstance(results, dict) and 'pred' in results:
            logger.info(f"  - Predictions shape: {results['pred'].shape}")
            if 'pert_cat' in results:
                unique_perts = np.unique(results['pert_cat'])
                logger.info(f"  - Number of unique perturbations: {len(unique_perts)}")
        
        return results

    def save_pretrained(self, save_directory: str):
        """
        保存模型：包含基础模型权重、配置以及 GEARS Head 的配置和权重。
        
        参照原始实现逻辑，分别保存 scFoundation 基础模型和 GEARS Head。
        
        Args:
            save_directory: 保存目录路径
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存 scFoundation 基础配置 (config.json) 和权重 (model.pt)
        # 调用父类的 save 方法
        super().save(save_directory)
        
        # 2. 如果 GearsModel Head 存在，保存其特有信息
        if self.gears_model is not None:
            # 使用 GearsModel 的 save 方法保存到子目录
            gears_save_dir = os.path.join(save_directory, 'gears_head')
            self.gears_model.save(gears_save_dir)
            logger.info(f"✓ Saved GearsModel head to {gears_save_dir}")
        else:
            logger.info("Only base model saved (GearsModel head not initialized).")
        
        logger.info(f"✓ Perturbation model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = 'cuda',
        **kwargs
    ) -> 'scFoundationPerturbationModel':
        """
        加载预训练模型。
        
        参照原始实现逻辑：
        1. 加载基础 scFoundation 模型。
        2. 检查 model_name_or_path 是否为路径且包含 'gears_head' 目录。
        3. 如果存在，则自动恢复 GearsModel Head。
        4. 如果不存在，则仅返回基础模型，GearsModel Head 为 None。
        
        Args:
            model_name_or_path: 模型名称或路径
                - 可以是本地路径
                - 可以是内置模型名称（如 'scfoundation-cell'）
                - 可以是 HuggingFace 模型标识符（如 'perturblab/scfoundation-cell'）
            device: 设备 ('cuda' 或 'cpu')
            **kwargs: 其他加载参数
        
        Returns:
            scFoundationPerturbationModel: 加载的模型实例
        
        Example:
            ```python
            # Load from local directory
            model = scFoundationPerturbationModel.from_pretrained('./models/my_model', device='cuda')
            
            # Load built-in model
            model = scFoundationPerturbationModel.from_pretrained('scfoundation-cell', device='cuda')
            
            # Load from HuggingFace
            model = scFoundationPerturbationModel.from_pretrained('perturblab/scfoundation-cell', device='cuda')
            ```
        """
        # 1. 加载基础 scFoundation 模型
        base_model = super().from_pretrained(model_name_or_path, device, **kwargs)
        
        # 创建当前类的实例 (复制基础模型的属性)
        model = cls(base_model.config, device=device, gene_list=base_model.gene_list)
        model.model = base_model.model  # 共享权重
        
        # 2. 检查是否存在 GearsModel Head
        # 如果 model_name_or_path 是本地路径，检查 gears_head 子目录
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            gears_head_dir = os.path.join(model_name_or_path, 'gears_head')
        else:
            # 如果是模型名称，尝试在可能的路径中查找
            gears_head_dir = None
        
        if gears_head_dir and os.path.exists(gears_head_dir) and os.path.isdir(gears_head_dir):
            logger.info(f"Found GearsModel head at {gears_head_dir}. Loading...")
            
            try:
                # 使用 GearsModel 的 from_pretrained 方法加载
                model.gears_model = GearsModel.from_pretrained(
                    gears_head_dir,
                    device=device
                )
                
                # 重新注入 scFoundation 编码器（如果 GEARS 支持）
                if hasattr(model.gears_model.gears_model, 'singlecell_model'):
                    model.gears_model.gears_model.singlecell_model = model.model
                    model.gears_model.gears_model.pretrained = True
                
                logger.info("✓ GearsModel head loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to load GearsModel head: {e}. "
                    "The model will be initialized without the perturbation head. "
                    "You can build graphs using GeneGraph.from_go() and GeneGraph.from_coexpression(), then call `init_perturbation_head()` or `init_perturbation_head_from_dataset()` to initialize it."
                )
                model.gears_model = None
        else:
            logger.info(f"No GearsModel head found in {model_name_or_path}. Initialized as base model only.")
            
        return model

