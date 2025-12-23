import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...data import GeneGraph, PerturbationData
from ...utils import download_from_huggingface
from ..base import PerturbationModel
from ..gears import GearsConfig, GearsModel
from .config import scFoundationConfig
from .source.load import gatherData, getEncoerDecoderData
from .source.pretrainmodels import select_model

logger = logging.getLogger(__name__)


class scFoundationModel(PerturbationModel):
    """scFoundation model wrapper for single-cell foundation embeddings.

    scFoundation is a large-scale foundation model trained on diverse single-cell datasets.
    It supports:
    - Cell embedding extraction (with various pooling strategies)
    - Gene embedding extraction
    - Inference on both single-cell and bulk RNA-seq data
    """

    def __init__(
        self,
        config: scFoundationConfig,
        device: str = 'cpu',
        gene_list: Optional[List[str]] = None,
    ):
        """Initializes the scFoundation model.

        Args:
            config: scFoundation configuration object.
            device: Computation device.
            gene_list: List of gene names for vocabulary alignment.
        """
        self.config = config
        self.device = device

        # Load gene list
        if gene_list is None:
            gene_list_path = os.path.join(
                os.path.dirname(__file__), "source", "gene_index.json"
            )
            if os.path.exists(gene_list_path):
                with open(gene_list_path, "r") as f:
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

        logger.info(f"Initialized scFoundation model with {config.num_tokens} tokens on {device}.")

    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()
        return self
    
    def to(self, device: str):
        self.model.to(device)
        return self
    
    @staticmethod
    def _get_adata_from_data(data: Union[AnnData, PerturbationData]) -> AnnData:
        """Extracts AnnData from input data."""
        if isinstance(data, PerturbationData):
            return data.adata
        elif isinstance(data, AnnData):
            return data
        else:
            raise TypeError(
                f"Expected AnnData or PerturbationData, got {type(data)}"
            )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the scFoundation model.

        Args:
            batch: Dictionary containing model inputs.
            **kwargs: Additional arguments passed to model.

        Returns:
            Dictionary with 'output' key containing model output tensor.
        """
        # Extract batch components
        x = batch['x']
        padding_label = batch['padding_label']
        encoder_position_gene_ids = batch['encoder_position_gene_ids']
        encoder_labels = batch['encoder_labels']
        decoder_data = batch['decoder_data']
        decoder_position_gene_ids = batch['decoder_position_gene_ids']
        decoder_data_padding_labels = batch['decoder_data_padding_labels']
        
        mask_gene_name = batch.get('mask_gene_name', False)
        mask_labels = batch.get('mask_labels', None)
        output_attentions = batch.get('output_attentions', False)
        
        output = self.model(
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
            **kwargs,
        )
        
        return {'output': output}

    def _align_genes(self, adata: AnnData) -> pd.DataFrame:
        """Aligns genes in adata to the model's vocabulary."""
        if self.gene_list is None:
            raise ValueError(
                "Gene list is not available. Cannot perform alignment. "
                "Provide gene_list during initialization."
            )

        # Extract expression matrix
        if issparse(adata.X):
            expr = pd.DataFrame(
                adata.X.toarray(), index=adata.obs_names, columns=adata.var_names
            )
        else:
            expr = pd.DataFrame(
                adata.X, index=adata.obs_names, columns=adata.var_names
            )

        # Pad missing genes with zeros
        to_fill_columns = list(set(self.gene_list) - set(expr.columns))
        if to_fill_columns:
            padding_df = pd.DataFrame(
                np.zeros((expr.shape[0], len(to_fill_columns))),
                columns=to_fill_columns,
                index=expr.index,
            )
            expr = pd.concat([expr, padding_df], axis=1)

        # Reorder to match model's gene list
        expr = expr[self.gene_list]

        return expr
    
    def get_dataloader(
        self,
        dataset: Union[AnnData, PerturbationData],
        batch_size: int = 32,
        shuffle: bool = True,
        split: Optional[str] = None,
        input_type: Literal["singlecell", "bulk"] = "singlecell",
        tgthighres: str = "t4",
        pre_normalized: Literal["F", "T", "A"] = "F",
        mask_ratio: float = 0.3,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        """Creates DataLoader for scFoundation model.
        
        Args:
            dataset: Input dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            split: Specific split to return, or None for all splits.
            input_type: Data source type ('singlecell' or 'bulk').
            tgthighres: Target high-resolution parameter.
            pre_normalized: Normalization status.
            mask_ratio: Masking ratio for training.
            num_workers: Number of worker threads.
            drop_last: Whether to drop incomplete batches.
            
        Returns:
            DataLoader or dictionary of DataLoaders.
        """
        adata = self._get_adata_from_data(dataset)
        has_split = "split" in adata.obs
        
        # Handle splits
        adata_map = {}
        if split is not None:
            if has_split:
                if split not in adata.obs["split"].values:
                    raise ValueError(f"Split '{split}' not found in dataset")
                adata_map[split] = adata[adata.obs["split"] == split]
            else:
                adata_map[split] = adata
        else:
            if has_split:
                split_names = adata.obs["split"].unique()
                for split_name in split_names:
                    adata_map[str(split_name)] = adata[adata.obs["split"] == split_name]
            else:
                adata_map["train"] = adata
        
        # Create dataset class
        class scFoundationDataset(Dataset):
            def __init__(
                self,
                adata_subset: AnnData,
                gene_list: List[str],
                input_type: str,
                tgthighres: str,
                pre_normalized: str,
                mask_ratio: float,
                device: str,
            ):
                self.adata = adata_subset
                self.gene_list = gene_list
                self.input_type = input_type
                self.tgthighres = tgthighres
                self.pre_normalized = pre_normalized
                self.mask_ratio = mask_ratio
                self.device = device
                
                # Align genes
                if issparse(adata_subset.X):
                    expr = pd.DataFrame(
                        adata_subset.X.toarray(),
                        index=adata_subset.obs_names,
                        columns=adata_subset.var_names
                    )
                else:
                    expr = pd.DataFrame(
                        adata_subset.X,
                        index=adata_subset.obs_names,
                        columns=adata_subset.var_names
                    )
                
                # Pad missing genes with zeros
                to_fill_columns = list(set(gene_list) - set(expr.columns))
                if to_fill_columns:
                    padding_df = pd.DataFrame(
                        np.zeros((expr.shape[0], len(to_fill_columns))),
                        columns=to_fill_columns,
                        index=expr.index,
                    )
                    expr = pd.concat([expr, padding_df], axis=1)
                
                # Reorder to match model's gene list
                self.expr = expr[gene_list].values
            
            def __len__(self):
                return len(self.expr)
            
            def __getitem__(self, idx):
                expr_row = self.expr[idx]
                
                # Prepare input tensor based on input_type
                if self.input_type == "bulk":
                    if self.pre_normalized == "T":
                        totalcount = expr_row.sum()
                    elif self.pre_normalized == "F":
                        totalcount = np.log10(expr_row.sum())
                    else:
                        raise ValueError("For bulk data, pre_normalized must be T or F")
                    
                    tmpdata = expr_row.tolist()
                    gene_x = torch.tensor(
                        tmpdata + [totalcount, totalcount], dtype=torch.float32
                    )
                
                elif self.input_type == "singlecell":
                    # Pre-normalization logic
                    if self.pre_normalized == "F":
                        totalcount = expr_row.sum()
                        tmpdata = np.log1p(expr_row / totalcount * 1e4).tolist()
                    elif self.pre_normalized == "T":
                        tmpdata = expr_row.tolist()
                        totalcount = expr_row.sum()
                    elif self.pre_normalized == "A":
                        tmpdata = expr_row[:-1].tolist()
                        totalcount = expr_row[-1]
                    else:
                        raise ValueError("pre_normalized must be T, F, or A")
                    
                    # Parse target resolution parameter
                    if self.tgthighres[0] == "f":
                        high_res = np.log10(totalcount * float(self.tgthighres[1:]))
                    elif self.tgthighres[0] == "a":
                        high_res = np.log10(totalcount) + float(self.tgthighres[1:])
                    elif self.tgthighres[0] == "t":
                        high_res = float(self.tgthighres[1:])
                    else:
                        raise ValueError("tgthighres must start with f, a, or t")
                    
                    gene_x = torch.tensor(
                        tmpdata + [high_res, np.log10(totalcount)], dtype=torch.float32
                    )
                else:
                    raise ValueError("input_type must be 'singlecell' or 'bulk'")
                
                return gene_x
        
        # Create DataLoaders
        dataloaders = {}
        model_config = self.config.to_model_config_dict()
        
        for split_name, adata_subset in adata_map.items():
            dataset_obj = scFoundationDataset(
                adata_subset=adata_subset,
                gene_list=self.gene_list if self.gene_list else adata_subset.var_names.tolist(),
                input_type=input_type,
                tgthighres=tgthighres,
                pre_normalized=pre_normalized,
                mask_ratio=mask_ratio,
                device=self.device,
            )
            
            def collate_fn(batch):
                """Collate function to prepare batch for forward pass."""
                batch_gene_x = torch.stack(batch)
                
                # Prepare encoder-decoder data
                if mask_ratio > 0:
                    # For training: use masked version
                    batch_gene_x_raw = batch_gene_x.clone()
                    (
                        encoder_data,
                        encoder_position_gene_ids,
                        encoder_data_padding,
                        encoder_labels,
                        decoder_data,
                        decoder_data_padding,
                        new_data_raw,
                        data_mask_labels,
                        decoder_position_gene_ids,
                    ) = getEncoerDecoderData(batch_gene_x, batch_gene_x_raw, model_config)
                else:
                    # For inference: no masking
                    (
                        encoder_data,
                        encoder_position_gene_ids,
                        encoder_data_padding,
                        encoder_labels,
                        decoder_data,
                        decoder_data_padding,
                        new_data_raw,
                        data_mask_labels,
                        decoder_position_gene_ids,
                    ) = getEncoerDecoderData(batch_gene_x, batch_gene_x, model_config)
                
                return {
                    'x': encoder_data,
                    'padding_label': encoder_data_padding,
                    'encoder_position_gene_ids': encoder_position_gene_ids,
                    'encoder_labels': encoder_labels,
                    'decoder_data': decoder_data,
                    'decoder_position_gene_ids': decoder_position_gene_ids,
                    'decoder_data_padding_labels': decoder_data_padding,
                    'mask_gene_name': False,
                    'mask_labels': data_mask_labels,
                    'output_attentions': False,
                    # Additional fields for loss computation
                    'new_data_raw': new_data_raw,
                    'data_mask_labels': data_mask_labels,
                    # Store original gene_x for embedding extraction
                    'gene_x': batch_gene_x,
                }
            
            loader = DataLoader(
                dataset_obj,
                batch_size=batch_size,
                shuffle=shuffle and split_name == "train",
                num_workers=num_workers,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
            
            dataloaders[split_name] = loader
        
        # Return single DataLoader if split is specified, otherwise return dict
        if split is not None:
            return dataloaders[split]
        return dataloaders

    def predict_embedding(
        self,
        dataset: Union[AnnData, PerturbationData],
        output_type: Literal["cell", "gene", "gene_batch"] = "cell",
        input_type: Literal["singlecell", "bulk"] = "singlecell",
        pool_type: Literal["all", "max"] = "all",
        tgthighres: str = "t4",
        pre_normalized: Literal["F", "T", "A"] = "F",
        batch_size: int = 32,
        return_adata: bool = False,
        use_batch: bool = True,
        split: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], np.ndarray, AnnData]:
        """Generates embeddings from input data.

        Args:
            dataset: Input dataset.
            output_type: Type of embedding ('cell', 'gene', or 'gene_batch').
            input_type: Data source type ('singlecell' or 'bulk').
            pool_type: Pooling strategy for cell embeddings.
            tgthighres: Target high-resolution parameter.
            pre_normalized: Normalization status.
            batch_size: Processing batch size.
            return_adata: If True, returns AnnData with embeddings.
            use_batch: Whether to process in batches.
            split: Specific split to return, or None for all splits.

        Returns:
            Dictionary with 'cell' or 'gene' key containing embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        # Extract AnnData from input
        adata = self._get_adata_from_data(dataset)
        
        # Handle splits
        has_split = "split" in adata.obs
        if split is not None:
            if has_split:
                if split not in adata.obs["split"].values:
                    raise ValueError(f"Split '{split}' not found in dataset")
                adata = adata[adata.obs["split"] == split]
            # If no split in data, use all data for requested split
        elif has_split:
            # Process all splits separately
            split_names = adata.obs["split"].unique()
            result = {}
            for split_name in split_names:
                subset = adata[adata.obs["split"] == split_name]
                emb_result = self._predict_embedding_single(
                    subset, output_type, input_type, pool_type, tgthighres,
                    pre_normalized, batch_size, return_adata, use_batch
                )
                if return_adata:
                    result[str(split_name)] = emb_result
                else:
                    # Convert to dict format
                    key = 'cell' if output_type == 'cell' else 'gene'
                    result[str(split_name)] = {key: emb_result}
            return result
        else:
            # No split, process all as 'train'
            return self._predict_embedding_single(
                adata, output_type, input_type, pool_type, tgthighres,
                pre_normalized, batch_size, return_adata, use_batch, split_key='train'
            )
        
        # Process single split
        return self._predict_embedding_single(
            adata, output_type, input_type, pool_type, tgthighres,
            pre_normalized, batch_size, return_adata, use_batch, split_key=split
        )
    
    def _predict_embedding_single(
        self,
        adata: AnnData,
        output_type: Literal["cell", "gene", "gene_batch"],
        input_type: Literal["singlecell", "bulk"],
        pool_type: Literal["all", "max"],
        tgthighres: str,
        pre_normalized: Literal["F", "T", "A"],
        batch_size: int,
        return_adata: bool,
        use_batch: bool,
    ) -> Union[Dict[str, np.ndarray], np.ndarray, AnnData]:
        """Internal method to process a single AnnData subset."""
        
        # Use get_dataloader for batch processing
        if not use_batch or output_type == "gene":
            batch_size = 1
        
        # Create temporary AnnData wrapper (no split column to ensure single loader)
        temp_adata = AnnData(adata.X, obs=adata.obs.copy(), var=adata.var)
        if "split" in temp_adata.obs:
            temp_adata.obs = temp_adata.obs.drop(columns=["split"])
        
        loader = self.get_dataloader(
            dataset=temp_adata,
            batch_size=batch_size,
            shuffle=False,
            split=None,  # Will return 'train' key since no split column
            input_type=input_type,
            tgthighres=tgthighres,
            pre_normalized=pre_normalized,
            mask_ratio=0.0,  # No masking for inference
            num_workers=0,
            drop_last=False,
        )
        
        # Handle single DataLoader (should be dict with 'train' key)
        if isinstance(loader, DataLoader):
            dataloader = loader
        else:
            # Get first (and only) loader
            dataloader = list(loader.values())[0]
        
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                output_dict = self.forward(batch)
                output = output_dict['output']
                
                # Extract embeddings based on output_type
                if output_type == "cell":
                    # Use stored gene_x from batch
                    batch_gene_x = batch['gene_x']
                    cell_emb = self._get_cell_embedding(batch_gene_x, pool_type)
                    embeddings.append(cell_emb.detach().cpu().numpy())
                
                elif output_type == "gene":
                    # Extract gene embeddings from output
                    gene_emb = output[:, :self.config.num_tokens, :].contiguous()
                    for j in range(gene_emb.shape[0]):
                        embeddings.append(gene_emb[j:j+1].detach().cpu().numpy())
                
                elif output_type == "gene_batch":
                    gene_emb = output[:, :self.config.num_tokens, :].contiguous()
                    embeddings.append(gene_emb.detach().cpu().numpy())
                else:
                    raise ValueError(
                        "output_type must be 'cell', 'gene', or 'gene_batch'"
                    )

        # Concatenate results
        if embeddings:
            if output_type == "gene_batch":
                embeddings = (
                    np.concatenate(embeddings, axis=0)
                    if len(embeddings) > 1
                    else embeddings[0]
                )
            else:
                embeddings = (
                    np.concatenate(embeddings, axis=0)
                    if len(embeddings) > 1
                    else embeddings[0]
                )
                if embeddings.ndim == 3 and embeddings.shape[0] == 1:
                    embeddings = embeddings.squeeze(0)
        else:
            raise ValueError("No embeddings were generated")

        # Return as AnnData if requested (legacy behavior)
        if return_adata:
            result_adata = adata.copy()
            if output_type == "cell":
                result_adata.obsm["X_scfoundation"] = embeddings
            elif output_type in ["gene", "gene_batch"]:
                # Store average gene embedding in varm
                result_adata.varm["scfoundation_gene_emb"] = embeddings.mean(axis=0)
                # Optionally store per-cell gene embeddings in layers
                result_adata.layers["scfoundation_gene_emb"] = embeddings
            return result_adata

        # Return unified dictionary format
        if output_type == "cell":
            return {'cell': embeddings}
        elif output_type in ["gene", "gene_batch"]:
            return {'gene': embeddings}
        else:
            return {'embeddings': embeddings}
    
    def predict_embeddings(
        self,
        dataset: Union[AnnData, PerturbationData],
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        split: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Unified embedding prediction method (alias for predict_embedding).
        
        Args:
            dataset: Input dataset.
            batch_size: Batch size.
            embedding_type: Type of embedding ('cell' or 'gene').
            split: Specific split to return, or None for all splits.
            **kwargs: Additional arguments passed to predict_embedding.
            
        Returns:
            Dictionary with 'cell' or 'gene' key containing embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        output_type = embedding_type
        return self.predict_embedding(
            dataset=dataset,
            output_type=output_type,
            batch_size=batch_size,
            split=split,
            return_adata=False,
            **kwargs
        )

    def _get_cell_embedding(
        self, gene_x: torch.Tensor, pool_type: str
    ) -> torch.Tensor:
        """Computes cell embeddings from expression tensors."""
        model_config = self.config.to_model_config_dict()

        # Prepare input data
        data_gene_ids = torch.arange(
            self.config.max_seq_len, device=gene_x.device
        ).repeat(gene_x.shape[0], 1)
        value_labels = gene_x > 0
        x, x_padding = gatherData(gene_x, value_labels, model_config["pad_token_id"])
        position_gene_ids, _ = gatherData(
            data_gene_ids, value_labels, model_config["pad_token_id"]
        )

        # Forward pass
        x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids)
        x += position_emb
        gene_emb = self.model.encoder(x, x_padding)

        # Pooling strategies
        gene_emb1 = gene_emb[:, -1, :]  # Last token
        gene_emb2 = gene_emb[:, -2, :]  # Second to last
        gene_emb3, _ = torch.max(gene_emb[:, :-2, :], dim=1)  # Max pool
        gene_emb4 = torch.mean(gene_emb[:, :-2, :], dim=1)  # Mean pool

        if pool_type == "all":
            cell_emb = torch.cat(
                [gene_emb1, gene_emb2, gene_emb3, gene_emb4], dim=1
            )
        elif pool_type == "max":
            cell_emb, _ = torch.max(gene_emb, dim=1)
        else:
            raise ValueError("pool_type must be 'all' or 'max'")

        return cell_emb

    def _get_gene_embedding(
        self, gene_x: torch.Tensor, single: bool = True
    ) -> torch.Tensor:
        """Computes gene embeddings."""
        model_config = self.config.to_model_config_dict()

        # Prepare encoder-decoder data
        (
            encoder_data,
            encoder_position_gene_ids,
            encoder_data_padding,
            encoder_labels,
            decoder_data,
            decoder_data_padding,
            new_data_raw,
            data_mask_labels,
            decoder_position_gene_ids,
        ) = getEncoerDecoderData(gene_x, gene_x, model_config)

        # Temporarily remove final layer to extract raw embeddings
        original_to_final = self.model.to_final
        self.model.to_final = None

        batch = {
            'x': encoder_data,
            'padding_label': encoder_data_padding,
            'encoder_position_gene_ids': encoder_position_gene_ids,
            'encoder_labels': encoder_labels,
            'decoder_data': decoder_data,
            'decoder_position_gene_ids': decoder_position_gene_ids,
            'decoder_data_padding_labels': decoder_data_padding,
            'mask_gene_name': False,
            'mask_labels': None,
            'output_attentions': False,
        }
        
        output_dict = self.forward(batch)
        out = output_dict['output']

        # Restore final layer
        self.model.to_final = original_to_final

        # Extract gene embeddings corresponding to known tokens
        gene_emb = out[:, : self.config.num_tokens, :].contiguous()

        return gene_emb

    def train_model(
        self,
        dataset: Union[AnnData, PerturbationData],
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
        split: Optional[str] = None,
    ):
        """Trains the scFoundation model using masked autoencoding.

        Args:
            dataset: Input dataset.
            output_dir: Directory to save checkpoints.
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            mask_ratio: Masking ratio for MAE.
            warmup_steps: Number of warmup steps.
            gradient_clip_val: Gradient clipping value.
            save_interval: Interval for saving checkpoints.
            eval_interval: Interval for evaluation.
            device: Device to use.
            split: Specific split to use for training.
        """
        if device is None:
            device = self.device

        os.makedirs(output_dir, exist_ok=True)

        # Use get_dataloader for training
        train_loader = self.get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            split=split if split is not None else "train",
            input_type="singlecell",
            tgthighres="t4",
            pre_normalized="F",
            mask_ratio=mask_ratio,
            num_workers=0,
            drop_last=False,
        )
        
        # Get validation loader if available
        val_loader = None
        adata = self._get_adata_from_data(dataset)
        if "split" in adata.obs and "val" in adata.obs["split"].values:
            val_loader = self.get_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                split="val",
                input_type="singlecell",
                tgthighres="t4",
                pre_normalized="F",
                mask_ratio=0.0,  # No masking for validation
                num_workers=0,
                drop_last=False,
            )
        
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler with linear warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.MSELoss(reduction="mean")

        self.model.train()

        logger.info(
            f"Starting training: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, "
            f"Warmup={warmup_steps}, Total Steps={total_steps}"
        )

        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                optimizer.zero_grad()

                # Forward pass
                output_dict = self.forward(batch)
                output = output_dict['output']

                # Compute loss (MSE)
                new_data_raw = batch['new_data_raw']
                data_mask_labels = batch.get('data_mask_labels', None)
                
                if data_mask_labels is not None:
                    loss = criterion(
                        output[data_mask_labels], new_data_raw[data_mask_labels]
                    )
                else:
                    decoder_data_padding = batch['decoder_data_padding_labels']
                    valid_mask = ~decoder_data_padding
                    loss = criterion(
                        output[valid_mask], new_data_raw[valid_mask]
                    )

                loss.backward()

                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip_val
                    )

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{epoch_loss / num_batches:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

                if global_step % 100 == 0:
                    logger.info(
                        f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f}"
                    )

            logger.info(
                f"Epoch {epoch + 1} Avg Loss: {epoch_loss / num_batches:.4f}"
            )

            if (epoch + 1) % save_interval == 0:
                checkpoint_dir = os.path.join(
                    output_dir, f"checkpoint-epoch-{epoch + 1}"
                )
                self.save(checkpoint_dir)

            if (epoch + 1) % eval_interval == 0 and val_loader is not None:
                self.eval()
                val_loss = self._evaluate_from_loader(val_loader, criterion, device)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                self.train()

        final_dir = os.path.join(output_dir, "final_model")
        self.save(final_dir)
        logger.info(f"Training complete. Model saved to {final_dir}")

    def _evaluate_from_loader(
        self, val_loader: DataLoader, criterion: nn.Module, device: str
    ) -> float:
        """Evaluates model on validation set."""
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                # Forward pass
                output_dict = self.forward(batch)
                output = output_dict['output']

                # Compute loss
                new_data_raw = batch['new_data_raw']
                data_mask_labels = batch.get('data_mask_labels', None)
                
                if data_mask_labels is not None:
                    loss = criterion(
                        output[data_mask_labels], new_data_raw[data_mask_labels]
                    )
                else:
                    decoder_data_padding = batch['decoder_data_padding_labels']
                    valid_mask = ~decoder_data_padding
                    loss = criterion(
                        output[valid_mask], new_data_raw[valid_mask]
                    )
                
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _evaluate(
        self, dataset: Union[AnnData, PerturbationData], batch_size: int, device: str
    ) -> float:
        """Evaluates model on validation set."""
        val_loader = self.get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            split="val",
            input_type="singlecell",
            tgthighres="t4",
            pre_normalized="F",
            mask_ratio=0.0,
            num_workers=0,
            drop_last=False,
        )
        
        criterion = nn.MSELoss()
        return self._evaluate_from_loader(val_loader, criterion, device)

    def save(self, model_path: str):
        """Saves model weights and configuration."""
        os.makedirs(model_path, exist_ok=True)

        config_path = os.path.join(model_path, "config.json")
        self.config.save(config_path)

        model_path = os.path.join(model_path, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = 'cpu',
        **kwargs,
    ) -> "scFoundationModel":
        """
        Loads a scFoundation model from a saved directory.

        Args:
            model_path: Path to the saved model directory.
            device: Device to load the model on.
            **kwargs: Additional args for model initialization.

        Returns:
            Loaded scFoundationModel instance.
        """
        # Load config and weights
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = scFoundationConfig.load(config_path)
        
        model = cls(config, device=device, **kwargs)

        model_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        state_dict = torch.load(model_file, map_location=device)
        model.model.load_state_dict(state_dict)

        model.eval()
        return model
\
class scFoundationPerturbationModel(scFoundationModel):
    """scFoundation model for perturbation prediction using GEARS framework.

    Extends scFoundationModel by using it as a backbone encoder for GEARS.
    Architecture: scFoundation (Encoder) -> GEARS (GNN + Decoder).
    """

    def __init__(
        self,
        config: scFoundationConfig,
        device: str = 'cpu',
        gene_list: Optional[List[str]] = None,
        go_graph: Optional[GeneGraph] = None,
        gene_graph: Optional[GeneGraph] = None,
        gears_config: Optional[GearsConfig] = None,
        pert_list: Optional[List[str]] = None,
        pert_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Initializes the scFoundation Perturbation Model.

        Args:
            config: scFoundation configuration.
            device: Computation device.
            gene_list: List of genes.
            go_graph: Pre-built GO graph.
            gene_graph: Pre-built co-expression graph.
            gears_config: Configuration for the GEARS head.
            pert_list: List of perturbations.
            pert_embeddings: Optional pre-trained perturbation embeddings.
            gene_embeddings: Optional pre-trained gene embeddings.
        """
        super().__init__(config, device, gene_list, **kwargs)

        self.gears_model: Optional[GearsModel] = None
        self.go_graph = go_graph
        self.gene_graph = gene_graph
        self.gears_config: Optional[GearsConfig] = None

        # Metadata extraction containers
        self.pert_names: List[str] = []
        self.pert_name_to_id: Dict[str, int] = {}
        self.de_gene_map: Dict[str, List[int]] = {}

        # Auto-initialize if all components are present
        if pert_list is not None and go_graph is not None and gene_graph is not None:
            if gene_list is None:
                raise ValueError(
                    "gene_list is required for immediate head initialization"
                )

            if gears_config is None:
                gears_config = self.config.get_gears_config()

            self.init_perturbation_head(
                gears_config=gears_config,
                gene_list=gene_list,
                pert_list=pert_list,
                go_graph=go_graph,
                co_graph=gene_graph,
                pert_embeddings=pert_embeddings,
                gene_embeddings=gene_embeddings,
            )
            logger.info("Perturbation head initialized automatically.")
        else:
            logger.info(
                "Initialized base model. Call init_perturbation_head to enable predictions."
            )

    def _requires_perturbation_head(func):
        """Enforces initialization of the GEARS head."""

        def wrapper(self, *args, **kwargs):
            if self.gears_model is None:
                raise RuntimeError(
                    f"{func.__name__} requires perturbation head. "
                    "Call `init_perturbation_head()` first."
                )
            return func(self, *args, **kwargs)

        return wrapper

    def init_perturbation_head(
        self,
        gears_config: GearsConfig,
        gene_list: List[str],
        pert_list: List[str],
        go_graph: GeneGraph,
        co_graph: GeneGraph,
        pert_embeddings: torch.Tensor = None,
        gene_embeddings: torch.Tensor = None,
    ):
        """Initializes the GEARS downstream head with provided graphs.

        Args:
            gears_config: GEARS configuration.
            gene_list: List of genes.
            pert_list: List of perturbations.
            go_graph: Gene Ontology graph.
            co_graph: Co-expression graph.
            pert_embeddings: Optional pre-trained embeddings.
            gene_embeddings: Optional pre-trained embeddings.
        """
        logger.info("Initializing GEARS head...")
        self.gears_model = GearsModel(
            config=gears_config,
            gene_list=gene_list,
            pert_list=pert_list,
            go_graph=go_graph,
            co_graph=co_graph,
            pert_embeddings=pert_embeddings,
            gene_embeddings=gene_embeddings,
            device=self.device,
        )

        # Inject scFoundation encoder into GEARS
        if hasattr(self.gears_model.gears_model, "singlecell_model"):
            self.gears_model.gears_model.singlecell_model = self.model
            self.gears_model.gears_model.pretrained = True
            logger.info("Injected scFoundation encoder into GEARS model.")
        else:
            logger.warning(
                "GEARS model lacks 'singlecell_model' attribute. Injection failed."
            )

        self.gears_config = gears_config
        self.pert_names = pert_list
        self.pert_name_to_id = {p: i for i, p in enumerate(pert_list)}

        logger.info(
            f"Head initialized: {len(gene_list)} genes, {len(pert_list)} perturbations."
        )

    def init_perturbation_head_from_dataset(
        self,
        dataset: PerturbationData,
        gears_config: Optional[GearsConfig] = None,
        **kwargs,
    ):
        """Initializes the GEARS head by extracting metadata from a dataset.

        Args:
            dataset: PerturbationData object.
            gears_config: GEARS configuration.
            **kwargs: Overrides for GearsConfig.
        """
        # Validate dataset format
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")

        required = ["condition", "split"]
        if not all(col in dataset.adata.obs for col in required):
            raise ValueError(
                f"Dataset missing fields: {required}. Check gears format and split."
            )
        if "ctrl_indices" not in dataset.adata.obsm:
            raise ValueError("Dataset missing 'ctrl_indices'. Run pair_cells().")

        adata = dataset.adata

        # Extract perturbations
        pert_conditions = [
            c for c in adata.obs["condition"].unique() if c != "ctrl"
        ]
        unique_perts = set()
        for p in pert_conditions:
            unique_perts.update([g for g in p.split("+") if g != "ctrl"])
        pert_list = sorted(list(unique_perts))

        if not pert_list:
            raise ValueError("No perturbations found in dataset.")

        # Build graphs if missing
        if self.go_graph is None:
            logger.info("Building GO Graph...")
            self.go_graph = GeneGraph.from_go(
                gene_list=pert_list, cache_dir=".gears_temp/go_graph"
            )

        if self.gene_graph is None:
            logger.info("Building Co-expression Graph...")
            self.gene_graph = GeneGraph.from_coexpression(
                adata, gene_list=self.gene_list
            )

        # Config setup
        if gears_config is None:
            gears_config = self.config.get_gears_config()
            if kwargs:
                conf_dict = gears_config.to_dict()
                conf_dict.update(kwargs)
                gears_config = GearsConfig(**conf_dict)

        # Align graphs
        logger.info("Aligning graphs to model vocabulary...")
        aligned_go = self.go_graph.subset(pert_list)
        aligned_gene = self.gene_graph.subset(self.gene_list)

        # Initialize head
        self.init_perturbation_head(
            gears_config=gears_config,
            gene_list=self.gene_list,
            pert_list=pert_list,
            go_graph=aligned_go,
            co_graph=aligned_gene,
        )

        # Extract DE genes metadata if available
        if "rank_genes_groups_cov_all" in adata.uns:
            self._extract_de_genes(adata)

    def _extract_de_genes(self, adata: AnnData):
        """Extracts DE gene indices from AnnData."""
        rank_data = adata.uns["rank_genes_groups_cov_all"]
        top_n = adata.uns.get("top_de_n", 20)
        
        gene_name_to_idx = {
            name: i for i, name in enumerate(adata.var["gene_name"])
        }
        gene_list_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}

        self.de_gene_map = {}

        # Handle structured array vs dict formats
        iterator = (
            rank_data["names"].dtype.names
            if hasattr(rank_data, "dtype")
            else rank_data.keys()
        )

        for cond in iterator:
            if isinstance(rank_data, dict) and cond in [
                "names",
                "scores",
                "pvals",
                "pvals_adj",
                "logfoldchanges",
            ]:
                continue

            if hasattr(rank_data, "dtype"):
                top_genes = rank_data["names"][cond][:top_n]
            else:
                top_genes = (
                    rank_data[cond][:top_n]
                    if isinstance(rank_data[cond], (list, np.ndarray))
                    else []
                )

            de_indices = []
            for g in top_genes:
                if g in gene_name_to_idx:
                    gene_name = adata.var["gene_name"].iloc[gene_name_to_idx[g]]
                    if gene_name in gene_list_to_idx:
                        de_indices.append(gene_list_to_idx[gene_name])
            
            self.de_gene_map[cond] = (
                de_indices[:top_n] if de_indices else [-1] * top_n
            )

    @_requires_perturbation_head
    def train_model(
        self,
        dataset: PerturbationData,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        train_split: str = "train",
        val_split: str = "val",
        result_dir: str = "./results",
        log_interval: int = 50,
        save_best: bool = True,
        **kwargs,
    ):
        """Trains the perturbation model.
        
        Args:
            dataset: Training perturbation dataset.
            epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            batch_size: Batch size.
            train_split: Name of training split.
            val_split: Name of validation split.
            result_dir: Directory to save results.
            log_interval: Logging interval.
            save_best: Whether to save the best model.
            
        Returns:
            Dictionary with training history.
        """
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")
        
        if "split" not in dataset.adata.obs.columns:
            raise ValueError("Dataset missing 'split'. Call split_data().")
        if "ctrl_indices" not in dataset.adata.obsm:
            raise ValueError("Dataset missing 'ctrl_indices'. Call pair_cells().")

        os.makedirs(result_dir, exist_ok=True)
        logger.info(
            f"Starting perturbation training: Epochs={epochs}, LR={lr}, Batch={batch_size}"
        )

        save_path = os.path.join(result_dir, "best_model") if save_best else None
        
        return self.gears_model.train_model(
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
            **kwargs,
        )

    @_requires_perturbation_head
    def predict_perturbation(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: str = "test",
        return_numpy: bool = True,
        **kwargs,
    ):
        """Predicts gene expression changes under perturbation.

        Args:
            dataset: Input perturbation dataset.
            batch_size: Batch size.
            split: Data split to predict on.
            return_numpy: Whether to return numpy arrays.

        Returns:
            Dictionary containing predictions and metadata.
        """
        if not dataset.gears_format:
            dataset.set_gears_format(fallback_cell_type="unknown")

        if split != "all" and split not in dataset.adata.obs["split"].values:
            raise ValueError(
                f"Split '{split}' not found. Available: {dataset.adata.obs['split'].unique()}"
            )

        logger.info(f"Predicting perturbation effects for split: {split}")

        results = self.gears_model.predict_perturbation(
            dataset=dataset,
            batch_size=batch_size,
            split=split,
            return_numpy=return_numpy,
            **kwargs,
        )

        if isinstance(results, dict) and "pred" in results:
            logger.info(f"Prediction shape: {results['pred'].shape}")

        return results

    def save(self, model_path: str):
        """Saves the base model and the GEARS head."""
        os.makedirs(model_path, exist_ok=True)

        # Save base model
        super().save(model_path)

        # Save GEARS head
        if self.gears_model is not None:
            gears_save_dir = os.path.join(model_path, "gears_head")
            self.gears_model.save(gears_save_dir)
            logger.info(f"Saved GEARS head to {gears_save_dir}")
        else:
            logger.info("Saved base model only (no head initialized).")

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = 'cpu',
        **kwargs,
    ) -> "scFoundationPerturbationModel":
        """Loads the model with optional GEARS head.
        
        Args:
            model_path: Path to the saved model directory.
            device: Computation device.
            **kwargs: Additional arguments.
            
        Returns:
            Loaded scFoundationPerturbationModel instance.
        """
        # Load base model
        base_model = super().load(model_path, device=device, **kwargs)

        # Create instance sharing weights
        model = cls(base_model.config, device=device, gene_list=base_model.gene_list)
        model.model = base_model.model

        # Check for GEARS head
        gears_head_dir = os.path.join(model_path, "gears_head")

        if os.path.exists(gears_head_dir):
            logger.info(f"Found GEARS head at {gears_head_dir}. Loading...")
            try:
                model.gears_model = GearsModel.load(gears_head_dir)
                
                # Re-inject encoder
                if hasattr(model.gears_model.gears_model, "singlecell_model"):
                    model.gears_model.gears_model.singlecell_model = model.model
                    model.gears_model.gears_model.pretrained = True
                
                logger.info("GEARS head loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to load GEARS head: {e}. "
                    "Model initialized without perturbation head."
                )
                model.gears_model = None
        else:
            logger.info("No GEARS head found. Initialized as base model only.")

        return model
