import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Optional, Any, Dict, Literal
from anndata import AnnData
from tqdm import tqdm
from torch.utils.data import DataLoader
import anndata as ad
import scanpy as sc

from ..base import PerturbationModel, download_from_huggingface
from .config import scGPTConfig
from .source.tokenizer.gene_tokenizer import get_default_gene_vocab, GeneVocab, random_mask_value
from .source.utils.util import map_raw_id_to_vocab_id, load_pretrained
from .source.model import TransformerModel, TransformerGenerator
from .source.loss import masked_mse_loss, criterion_neg_log_bernoulli
from ...data import PerturbationData
import logging

logger = logging.getLogger('[perturblab.model.scgpt]')


class scGPTModel(PerturbationModel):
    """
    scGPT model for single-cell gene expression analysis.
    Supports pretraining, fine-tuning, and various downstream tasks.
    
    Available pretrained models on HuggingFace (perturblab organization):
    - scgpt-human: General human single-cell model
    - scgpt-blood: Blood cell specialized model
    - scgpt-brain: Brain cell specialized model
    - scgpt-heart: Heart cell specialized model
    - scgpt-kidney: Kidney cell specialized model
    - scgpt-lung: Lung cell specialized model
    - scgpt-pan-cancer: Pan-cancer model
    - scgpt-continual-pretrained: Continual pretrained model
    """
    
    # Available pretrained models on HuggingFace
    PRETRAINED_MODELS = {
        "scgpt-human": "perturblab/scgpt-human",
        "scgpt-blood": "perturblab/scgpt-blood",
        "scgpt-brain": "perturblab/scgpt-brain",
        "scgpt-heart": "perturblab/scgpt-heart",
        "scgpt-kidney": "perturblab/scgpt-kidney",
        "scgpt-lung": "perturblab/scgpt-lung",
        "scgpt-pan-cancer": "perturblab/scgpt-pan-cancer",
        "scgpt-continual-pretrained": "perturblab/scgpt-continual-pretrained",
    }
    
    def __init__(self, config: scGPTConfig, gene_list: list[str] = None, device: str = 'cuda', **kwargs):
        super().__init__(config)
        
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
        else:
            self.device = 'cpu'
        
        if config.use_default_gene_vocab:
            self.vocab = get_default_gene_vocab()
        else:
            if gene_list is None:
                raise ValueError("gene_list is required when config.use_default_gene_vocab is False")
            self.vocab = GeneVocab(gene_list, config.specials, config.special_first, config.default_token)
        
        # Ensure pad_token exists in vocab and set as default index
        if config.pad_token not in self.vocab:
            # Add pad_token to vocab if it doesn't exist
            self.vocab.append_token(config.pad_token)
        # Set default index to pad_token
        self.vocab.set_default_index(self.vocab[config.pad_token])
        
        self.model = TransformerModel(
            ntoken=config.ntoken,
            d_model=config.d_model,
            nhead=config.nhead,
            d_hid=config.d_hid,
            nlayers=config.nlayers,
            nlayers_cls=config.nlayers_cls,
            n_cls=config.n_cls,
            vocab=self.vocab,
            dropout=config.dropout,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            do_mvc=config.do_mvc,
            do_dab=config.do_dab,
            use_batch_labels=config.use_batch_labels,
            num_batch_labels=config.num_batch_labels,
            domain_spec_batchnorm=config.domain_spec_batchnorm,
            input_emb_style=config.input_emb_style,
            n_input_bins=config.n_input_bins,
            cell_emb_style=config.cell_emb_style,
            mvc_decoder_style=config.mvc_decoder_style,
            ecs_threshold=config.ecs_threshold,
            explicit_zero_prob=config.explicit_zero_prob,
            use_fast_transformer=config.use_fast_transformer,
            fast_transformer_backend=config.fast_transformer_backend,
            pre_norm=config.pre_norm,
        ).to(self.device)

        # Loss criteria
        self.criterion_dab = nn.CrossEntropyLoss()

    @classmethod
    def mask_values(
        cls, 
        values: Union[torch.Tensor, np.ndarray], 
        mask_ratio: float = 0.4, 
        mask_value: int = -1, 
        pad_value: int = 0
    ) -> torch.Tensor:
        """
        Class method to perform random masking on values.
        Wraps the tokenizer's random_mask_value function.
        """
        return random_mask_value(
            values,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value
        )

    def prepare_dataloader(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        mask_ratio: float = 0.0,  # Default to 0.0 (no masking) for inference/general usage
        mask_value: int = -1,
        return_split: Optional[str] = None,  # 'all', 'train', 'test', etc. or None for all splits
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Prepare dataloaders for the dataset.
        
        Args:
            dataset: PerturbationData object
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            num_workers: Number of workers for data loading
            mask_ratio: Ratio of values to mask (0.0 = no masking)
            mask_value: Value to use for masking
            return_split: If specified, return only this split as a single DataLoader.
                         If 'all', return all data as a single DataLoader.
                         If None, return dict of all splits.
        
        Returns:
            DataLoader or Dict[str, DataLoader] depending on return_split
        """
        import scipy.sparse
        from torch.utils.data import TensorDataset, DataLoader

        gene_names = dataset.adata.var_names.tolist()
        gene_ids = [
            self.vocab[g] if g in self.vocab else self.vocab[self.config.pad_token]
            for g in gene_names
        ]

        max_len = self.config.max_seq_len if hasattr(self.config, 'max_seq_len') else len(gene_ids)
        if len(gene_ids) > max_len:
            logger.info(f"Truncating genes from {len(gene_ids)} to {max_len}")
            gene_ids = gene_ids[:max_len]
            slice_cols = slice(0, max_len)
        else:
            slice_cols = slice(None)

        # Static src tensor (gene IDs)
        src_tensor = torch.tensor(gene_ids, dtype=torch.long)

        # Collate function with masking logic
        def scgpt_collate(batch):
            # batch structure: [(values_0, label_0?), (values_1, label_1?), ...]
            
            # Stack values: (batch_size, seq_len)
            batch_values = torch.stack([item[0] for item in batch])
            curr_batch_size = batch_values.shape[0]
            
            # Replicate src for the batch
            batch_src = src_tensor.unsqueeze(0).repeat(curr_batch_size, 1)
            
            # Generate padding mask (assuming all selected genes are valid for now)
            batch_padding_mask = torch.zeros_like(batch_src, dtype=torch.bool)

            output = {
                "src": batch_src,
                "values": batch_values, # This will be the TARGET if we mask
                "src_key_padding_mask": batch_padding_mask
            }

            # Apply masking if mask_ratio > 0
            if mask_ratio > 0:
                masked_values = self.mask_values(
                    batch_values,
                    mask_ratio=mask_ratio,
                    mask_value=mask_value,
                    pad_value=self.config.pad_value
                )
                output["masked_values"] = masked_values
                output["target_values"] = batch_values # Original values are targets
            else:
                 output["masked_values"] = batch_values # No masking

            # Handle batch labels
            if len(batch[0]) > 1:
                batch_labels = torch.stack([item[1] for item in batch])
                output["batch_labels"] = batch_labels

            return output

        # Split handling
        adata_map = {}
        if return_split == 'all' or ('split' not in dataset.adata.obs):
            # Return all data as a single loader
            adata_map['all'] = dataset.adata
        elif return_split is not None:
            # Return only specified split
            if 'split' in dataset.adata.obs and return_split in dataset.adata.obs['split'].values:
                subset = dataset.adata[dataset.adata.obs['split'] == return_split]
                adata_map[return_split] = subset
            else:
                raise ValueError(f"Split '{return_split}' not found in dataset")
        else:
            # Return all splits as separate loaders
            if 'split' in dataset.adata.obs:
                split_names = dataset.adata.obs['split'].unique()
                for split_name in split_names:
                    subset = dataset.adata[dataset.adata.obs['split'] == split_name]
                    adata_map[str(split_name)] = subset
            else:
                adata_map['all'] = dataset.adata

        dataloaders = {}

        for split_name, adata_subset in adata_map.items():
            if adata_subset.n_obs == 0:
                continue

            X = adata_subset.X
            if slice_cols != slice(None):
                X = X[:, slice_cols]
            
            if scipy.sparse.issparse(X):
                X = X.toarray()
            
            if self.config.input_emb_style == "category":
                values_tensor = torch.tensor(X, dtype=torch.long)
            else:
                values_tensor = torch.tensor(X, dtype=torch.float32)

            tensors_to_pack = [values_tensor]
            
            if self.config.use_batch_labels:
                batch_label_key = getattr(self.config, 'batch_label_key', 'batch')
                if batch_label_key in adata_subset.obs:
                    batch_labels = adata_subset.obs[batch_label_key].astype('category').cat.codes.values
                    tensors_to_pack.append(torch.tensor(batch_labels, dtype=torch.long))
                else:
                    tensors_to_pack.append(torch.zeros(len(adata_subset), dtype=torch.long))

            dataset_tensor = TensorDataset(*tensors_to_pack)
            
            loader = DataLoader(
                dataset_tensor,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=scgpt_collate
            )
            
            dataloaders[split_name] = loader

        # Return single loader or dict based on return_split
        if return_split is not None:
            return dataloaders[list(dataloaders.keys())[0]]
        return dataloaders
    
    
    def forward(
        self, 
        batch_data: Dict[str, torch.Tensor], 
        CLS: bool = False, 
        CCE: bool = False, 
        MVC: bool = False, 
        ECS: bool = False, 
        do_sample: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass. Only retrieves embeddings and model outputs. 
        Does NOT calculate loss.
        """
        # Move to device
        batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

        src = batch_data["src"]
        # Use masked values if available, otherwise raw values
        values = batch_data.get("masked_values", batch_data["values"])
        src_key_padding_mask = batch_data["src_key_padding_mask"]
        batch_labels = batch_data.get("batch_labels", None)

        # Forward pass through TransformerModel
        output_dict = self.model(
            src=src,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
            CLS=CLS,
            CCE=CCE,
            MVC=MVC,
            ECS=ECS,
            do_sample=do_sample
        )
        
        return output_dict

    def compute_loss(
        self,
        batch_data: Dict[str, torch.Tensor],
        output_dict: Optional[Dict[str, torch.Tensor]] = None,
        CLS: bool = False, 
        CCE: bool = False, 
        MVC: bool = False, 
        ECS: bool = False, 
        do_sample: bool = False,
        mask_value: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the loss for the model.
        If output_dict is provided, uses it. Otherwise calls forward first.
        """
        if output_dict is None:
            output_dict = self.forward(
                batch_data, 
                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS, do_sample=do_sample
            )
        
        losses = {}
        total_loss = 0.0

        if "target_values" in batch_data:
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            target_values = batch_data["target_values"]
            values = batch_data.get("masked_values", batch_data["values"])
            
            masked_positions = values.eq(mask_value)
            
            # 1. Masked MSE Loss (MLM)
            loss_mse = masked_mse_loss(
                output_dict["mlm_output"], target_values, masked_positions
            )
            total_loss += loss_mse
            losses["loss_mse"] = loss_mse

            # 2. Explicit Zero Prob Loss
            if self.config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                total_loss += loss_zero_log_prob
                losses["loss_zero_prob"] = loss_zero_log_prob

            # 3. GEPC Loss (MVC)
            if MVC:
                loss_gepc = masked_mse_loss(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                total_loss += loss_gepc
                losses["loss_gepc"] = loss_gepc

            # 4. ECS Loss
            if ECS and "loss_ecs" in output_dict:
                # Scaling factor 10 is commonly used in scGPT tutorials
                loss_ecs = output_dict["loss_ecs"] * 10 
                total_loss += loss_ecs
                losses["loss_ecs"] = loss_ecs

            # 5. DAB Loss
            if self.config.do_dab and "dab_output" in output_dict and "batch_labels" in batch_data:
                loss_dab = self.criterion_dab(output_dict["dab_output"], batch_data["batch_labels"])
                dab_weight = getattr(self.config, 'dab_weight', 1.0)
                total_loss += loss_dab * dab_weight
                losses["loss_dab"] = loss_dab
            
        losses["loss"] = total_loss
        return losses

    def predict_embeddings(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        **kwargs
    ) -> np.ndarray:
        """
        Unified embedding prediction method.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size for inference
            embedding_type: Type of embedding to predict ("cell" or "gene")
            **kwargs: Additional arguments (unused for scGPTModel)
            
        Returns:
            np.ndarray: Embeddings of shape (n_samples, embedding_dim) for cells
                       or (n_genes, embedding_dim) for genes
        """
        self.eval()
        
        if embedding_type == "gene":
            # For gene embeddings, directly return encoder embeddings
            with torch.no_grad():
                gene_embs = self.model.encoder.embedding.weight.cpu().numpy()
            return gene_embs
        
        elif embedding_type == "cell":
            # For cell embeddings, process through the model
            loader = self.prepare_dataloader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                mask_ratio=0.0,
                return_split='all'
            )
            
            embeddings = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Encoding cells"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    out = self.forward(batch)
                    cell_emb = out["cell_emb"]  # (batch, embsize)
                    embeddings.append(cell_emb.cpu().numpy())
                    
            return np.concatenate(embeddings, axis=0)
        else:
            raise ValueError(f"embedding_type must be 'cell' or 'gene', got {embedding_type}")

    def save(self, save_directory: str):
        """
        Save model weights and config to a directory.
        
        Args:
            save_directory: Directory path to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict as model.pt (scGPT convention)
        model_file = os.path.join(save_directory, "model.pt")
        torch.save(self.model.state_dict(), model_file)
        
        # Save config as args.json (hardcoded, scGPT convention)
        config_file = os.path.join(save_directory, "args.json")
        self.config.save(config_file)
        
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        device: str = 'cuda',
        **kwargs
    ) -> 'scGPTModel':
        """
        Load a pretrained scGPT model.
        
        Supports multiple loading methods:
        1. HuggingFace models (perturblab organization)
        2. Local directory paths
        3. Custom URLs (future support)
        
        Args:
            model_name_or_path: Can be:
                - Model name: "scgpt-human" → downloads from perturblab/scgpt-human
                - Full HuggingFace repo ID: "perturblab/scgpt-human"
                - Local directory path: "/path/to/model"
            device: Device to load the model on ('cuda' or 'cpu')
            **kwargs: Additional arguments
                - Model initialization: gene_list, etc.
                - HuggingFace download: revision, token, cache_dir, force_download, resume_download
            
        Returns:
            Loaded scGPTModel instance
            
        Example:
            >>> # Load from HuggingFace (auto-cached)
            >>> model = scGPTModel.from_pretrained("scgpt-human")
            >>> 
            >>> # Load tissue-specific model
            >>> model = scGPTModel.from_pretrained("scgpt-brain", device='cpu')
            >>> 
            >>> # Load from local path
            >>> model = scGPTModel.from_pretrained("/path/to/model")
            >>> 
            >>> # Load with custom parameters
            >>> model = scGPTModel.from_pretrained(
            ...     "scgpt-human",
            ...     gene_list=my_genes,
            ...     revision="v1.0",
            ...     token="hf_xxx"
            ... )
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
            # Check if it's a registered model name or HuggingFace repo
            # For scGPT models, we use HuggingFace
            try:
                logger.info(f"Downloading model '{model_name_or_path}' from HuggingFace...")
                model_path = download_from_huggingface(
                    model_name_or_path,
                    organization="perturblab",
                    **hf_kwargs
                )
                logger.info(f"✓ Model cached at: {model_path}")
            except Exception as e:
                raise ValueError(
                    f"Failed to load model '{model_name_or_path}'. "
                    f"Make sure it's either:\n"
                    f"  - A valid local directory path\n"
                    f"  - A HuggingFace model name (e.g., 'scgpt-human')\n"
                    f"  - A full repo ID (e.g., 'perturblab/scgpt-human')\n"
                    f"Error: {str(e)}"
                )
        
        # Load config from args.json (hardcoded, scGPT convention)
        config_file = os.path.join(model_path, "args.json")
        config = scGPTConfig.load(config_file)
        
        # If ntoken is None, load from vocab.json
        if config.ntoken is None:
            vocab_file = os.path.join(model_path, "vocab.json")
            if os.path.exists(vocab_file):
                try:
                    with open(vocab_file, 'r') as f:
                        vocab_data = json.load(f)
                        # vocab.json is a dict of {gene: index}, ntoken = max_index + 1
                        if isinstance(vocab_data, dict):
                            config.ntoken = max(vocab_data.values()) + 1
                        else:
                            config.ntoken = len(vocab_data)
                except Exception as e:
                    logger.warning(f"Failed to load ntoken from vocab.json: {e}")
                    config.ntoken = 60697  # Fallback default
            else:
                config.ntoken = 60697  # Fallback default
        
        # Find model file (try both model.pt and best_model.pt)
        model_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_path, "best_model.pt")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Expected 'model.pt' or 'best_model.pt'."
                )
        
        # Create model instance
        model = cls(config, device=device, **model_init_kwargs)
        
        # Load state dict using official load_pretrained (handles flash-attn <-> pytorch conversion)
        state_dict = torch.load(model_file, map_location=device)
        load_pretrained(model.model, state_dict, verbose=False)
        
        logger.info(f"✓ Model loaded successfully")
        
        return model

class scGPTPerturbationModel(PerturbationModel):
    """
    scGPT model specialized for perturbation prediction tasks.
    Uses TransformerGenerator architecture for generating perturbed cell states.
    
    Note: Perturbation models will be added to HuggingFace in the future.
    Currently, use the general scGPT models and fine-tune for perturbation tasks.
    """
    
    # Registry of pretrained models for perturbation (to be added)
    PRETRAINED_MODELS = {
        # Future perturbation-specific models
        # "scgpt-perturb-human": "perturblab/scgpt-perturb-human",
    }
    
    def __init__(self, config: scGPTConfig, gene_list: list[str] = None, device: str = 'cuda', **kwargs):
        super().__init__(config)
        
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
        else:
            self.device = 'cpu'
        
        if config.use_default_gene_vocab:
            self.vocab = get_default_gene_vocab()
        else:
            if gene_list is None:
                raise ValueError("gene_list is required when config.use_default_gene_vocab is False")
            self.vocab = GeneVocab(gene_list, config.specials, config.special_first, config.default_token)
        
        # Ensure pad_token exists in vocab and set as default index
        if config.pad_token not in self.vocab:
            # Add pad_token to vocab if it doesn't exist
            self.vocab.append_token(config.pad_token)
        # Set default index to pad_token
        self.vocab.set_default_index(self.vocab[config.pad_token])
        
        self.model = TransformerGenerator(
            ntoken=config.ntoken,
            d_model=config.d_model,
            nhead=config.nhead,
            d_hid=config.d_hid,
            nlayers=config.nlayers,
            nlayers_cls=config.nlayers_cls,
            n_cls=config.n_cls,
            vocab=self.vocab,
            dropout=config.dropout,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            pert_pad_id=config.pert_pad_id,
            do_mvc=config.do_mvc,
            domain_spec_batchnorm=config.domain_spec_batchnorm,
            n_input_bins=config.n_input_bins,
            cell_emb_style=config.cell_emb_style,
            mvc_decoder_style=config.mvc_decoder_style,
            decoder_activation=config.decoder_activation,
            decoder_adaptive_bias=config.decoder_adaptive_bias,
            ecs_threshold=config.ecs_threshold,
            explicit_zero_prob=config.explicit_zero_prob,
            use_fast_transformer=config.use_fast_transformer,
            fast_transformer_backend=config.fast_transformer_backend,
            pre_norm=config.pre_norm,
        ).to(self.device)

    def prepare_dataloader(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        mask_ratio: float = 0.0,
        return_split: Optional[str] = None,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Prepare dataloaders for perturbation prediction task.
        
        Args:
            dataset: PerturbationData object
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            num_workers: Number of workers for data loading
            mask_ratio: Not used for perturbation task
            return_split: If specified, return only this split as a single DataLoader.
                         If 'all', return all data as a single DataLoader.
                         If None, return dict of all splits.
        
        Returns:
            DataLoader or Dict[str, DataLoader] depending on return_split
        """
        from torch.utils.data import TensorDataset, DataLoader
        import scipy.sparse

        # Ensure we have paired control cells
        if 'ctrl_indices' not in dataset.adata.obsm:
            logger.info("Pairing cells for perturbation task...")
            dataset.pair_cells()

        gene_names = dataset.adata.var_names.tolist()
        gene_ids = [
            self.vocab[g] if g in self.vocab else self.vocab[self.config.pad_token]
            for g in gene_names
        ]

        # Truncate if needed
        max_len = self.config.max_seq_len if hasattr(self.config, 'max_seq_len') else len(gene_ids)
        if len(gene_ids) > max_len:
            gene_ids = gene_ids[:max_len]
            slice_cols = slice(0, max_len)
            gene_names = gene_names[:max_len]
        else:
            slice_cols = slice(None)

        src_tensor = torch.tensor(gene_ids, dtype=torch.long)
        gene_name_to_idx = {name: i for i, name in enumerate(gene_names)}

        # Data Split
        adata_map = {}
        if return_split == 'all' or ('split' not in dataset.adata.obs):
            # Return all data as a single loader
            adata_map['all'] = np.arange(dataset.adata.n_obs)
        elif return_split is not None:
            # Return only specified split
            if 'split' in dataset.adata.obs and return_split in dataset.adata.obs['split'].values:
                indices = np.where(dataset.adata.obs['split'] == return_split)[0]
                adata_map[return_split] = indices
            else:
                raise ValueError(f"Split '{return_split}' not found in dataset")
        else:
            # Return all splits as separate loaders
            if 'split' in dataset.adata.obs:
                for split_name in dataset.adata.obs['split'].unique():
                    indices = np.where(dataset.adata.obs['split'] == split_name)[0]
                    adata_map[str(split_name)] = indices
            else:
                adata_map['all'] = np.arange(dataset.adata.n_obs)

        dataloaders = {}
        
        for split_name, indices in adata_map.items():
            if len(indices) == 0: 
                continue
            
            # Create TensorDataset wrapping just the indices
            index_dataset = TensorDataset(torch.tensor(indices, dtype=torch.long))
            
            # Create collate function with proper closure
            def make_collate_fn(gene_name_to_idx_local, src_tensor_local, slice_cols_local):
                def pert_collate(batch):
                    # batch is list of tuples [(index,), (index,), ...]
                    batch_indices = torch.stack([item[0] for item in batch]).numpy()
                    
                    # Fetch control indices: (batch_size,) - taking first sample [:, 0]
                    ctrl_indices = dataset.adata.obsm['ctrl_indices'][batch_indices, 0]
                    
                    # Fetch expressions
                    X = dataset.adata.X
                    if scipy.sparse.issparse(X):
                        x_val = X[ctrl_indices, slice_cols_local].toarray()
                        y_val = X[batch_indices, slice_cols_local].toarray()
                    else:
                        x_val = X[ctrl_indices, slice_cols_local]
                        y_val = X[batch_indices, slice_cols_local]
                    
                    # Identify perturbed genes
                    conditions = dataset.adata.obs.iloc[batch_indices]['condition'].values
                    pert_flags = np.zeros_like(x_val, dtype=np.int64)
                    
                    for i, cond in enumerate(conditions):
                        if cond != 'ctrl':
                            perts = cond.split('+')
                            for p in perts:
                                if p in gene_name_to_idx_local:
                                    pert_flags[i, gene_name_to_idx_local[p]] = 1
                    
                    # To Tensors
                    input_values = torch.tensor(x_val, dtype=torch.float32)
                    target_values = torch.tensor(y_val, dtype=torch.float32)
                    pert_flags_tensor = torch.tensor(pert_flags, dtype=torch.long)
                    
                    curr_bs = input_values.shape[0]
                    batch_src = src_tensor_local.unsqueeze(0).repeat(curr_bs, 1)
                    batch_padding_mask = torch.zeros_like(batch_src, dtype=torch.bool)
                    
                    return {
                        "src": batch_src,
                        "values": input_values,
                        "target_values": target_values,
                        "input_pert_flags": pert_flags_tensor,
                        "src_key_padding_mask": batch_padding_mask
                    }
                return pert_collate
            
            collate_fn = make_collate_fn(gene_name_to_idx, src_tensor, slice_cols)

            dataloaders[split_name] = DataLoader(
                index_dataset,
                batch_size=batch_size,
                shuffle=(shuffle and split_name == 'train'),
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=collate_fn
            )

        # Return single loader or dict based on return_split
        if return_split is not None:
            return dataloaders[list(dataloaders.keys())[0]]
        return dataloaders

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
        
        output_dict = self.model(
            src=batch_data["src"],
            values=batch_data["values"],
            input_pert_flags=batch_data["input_pert_flags"],
            src_key_padding_mask=batch_data["src_key_padding_mask"],
            CLS=CLS,
            CCE=CCE,
            MVC=MVC,
            ECS=ECS,
            do_sample=do_sample
        )
        return output_dict

    def compute_loss(
        self,
        batch_data: Dict[str, torch.Tensor],
        output_dict: Optional[Dict[str, torch.Tensor]] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        if output_dict is None:
            output_dict = self.forward(
                batch_data,
                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS, do_sample=do_sample
            )
            
        losses = {}
        total_loss = 0.0
        
        if "target_values" in batch_data:
            target_values = batch_data["target_values"].to(self.device)
            # For perturbation prediction, we typically predict all values (or masking some?)
            # Tutorial uses all ones for masked_positions
            masked_positions = torch.ones_like(target_values, dtype=torch.bool)
            
            loss_mse = masked_mse_loss(
                output_dict["mlm_output"], target_values, masked_positions
            )
            total_loss += loss_mse
            losses["loss_mse"] = loss_mse
            
            # Add other losses if needed (ECS, etc.)
            
        losses["loss"] = total_loss
        return losses

    def predict_embeddings(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        **kwargs
    ) -> np.ndarray:
        """
        Unified embedding prediction method.
        
        Args:
            dataset: Input dataset (required for cell embeddings, optional for gene)
            batch_size: Batch size for inference
            embedding_type: Type of embedding to predict ("cell" or "gene")
            **kwargs: Additional arguments (unused for scGPTPerturbationModel)
            
        Returns:
            np.ndarray: Embeddings of shape (n_samples, embedding_dim) for cells
                       or (n_genes, embedding_dim) for genes
        """
        self.eval()
        
        if embedding_type == "gene":
            # For gene embeddings, directly return encoder embeddings
            with torch.no_grad():
                gene_embs = self.model.encoder.embedding.weight.cpu().numpy()
            return gene_embs
        
        elif embedding_type == "cell":
            # For cell embeddings, use internal methods to get cell embeddings
            loader = self.prepare_dataloader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                return_split='all'
            )
            
            embeddings = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Encoding cells"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Use internal encode to get cell embeddings
                    # TransformerGenerator computes cell_emb but doesn't return it in output dict
                    # We need to call internal methods directly
                    src = batch["src"]
                    values = batch["values"]
                    input_pert_flags = batch["input_pert_flags"]
                    src_key_padding_mask = batch["src_key_padding_mask"]
                    
                    transformer_output = self.model._encode(
                        src, values, input_pert_flags, src_key_padding_mask
                    )
                    cell_emb = self.model._get_cell_emb_from_layer(transformer_output, values)
                    
                    embeddings.append(cell_emb.cpu().numpy())
                    
            return np.concatenate(embeddings, axis=0)
        else:
            raise ValueError(f"embedding_type must be 'cell' or 'gene', got {embedding_type}")
    
    def predict_perturbation(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict perturbation effects.
        
        Args:
            dataset: PerturbationData with perturbation conditions
            batch_size: Batch size for inference
            return_numpy: If True, return numpy array; otherwise return torch tensor
            **kwargs: Additional arguments
            
        Returns:
            Predicted gene expressions after perturbation
        """
        self.eval()
        loader = self.prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            return_split='all'
        )
        
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting perturbations"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.forward(batch)
                pred = output["mlm_output"]  # (batch, seq_len)
                
                if return_numpy:
                    predictions.append(pred.cpu().numpy())
                else:
                    predictions.append(pred.cpu())
        
        if return_numpy:
            return np.concatenate(predictions, axis=0)
        else:
            return torch.cat(predictions, dim=0)
    
    def save(self, save_directory: str):
        """
        Save model weights and config to a directory.
        
        Args:
            save_directory: Directory path to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict as model.pt (scGPT convention)
        model_file = os.path.join(save_directory, "model.pt")
        torch.save(self.model.state_dict(), model_file)
        
        # Save config as args.json (hardcoded, scGPT convention)
        config_file = os.path.join(save_directory, "args.json")
        self.config.save(config_file)
        
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        device: str = 'cuda',
        **kwargs
    ) -> 'scGPTPerturbationModel':
        """
        Load a pretrained scGPT perturbation model.
        
        Currently, perturbation models are typically fine-tuned locally.
        Future support for HuggingFace perturbation models will be added.
        
        Args:
            model_name_or_path: Can be:
                - Local directory path: "/path/to/model" (most common)
                - HuggingFace repo ID (future): "perturblab/scgpt-perturb-human"
            device: Device to load the model on ('cuda' or 'cpu')
            **kwargs: Additional arguments
                - Model initialization: gene_list, etc.
                - HuggingFace download: revision, token, etc. (if applicable)
            
        Returns:
            Loaded scGPTPerturbationModel instance
            
        Example:
            >>> # Load from local fine-tuned model
            >>> model = scGPTPerturbationModel.from_pretrained("/path/to/finetuned")
            >>> 
            >>> # Load with custom gene list
            >>> model = scGPTPerturbationModel.from_pretrained(
            ...     "/path/to/model", 
            ...     gene_list=my_genes,
            ...     device='cpu'
            ... )
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
            # Try HuggingFace (for future perturbation models)
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
                    f"Failed to load model '{model_name_or_path}'. "
                    f"Perturbation models are typically fine-tuned locally.\n"
                    f"Make sure the path exists or specify a valid local directory.\n"
                    f"Error: {str(e)}"
                )
        
        # Load config from args.json (hardcoded, scGPT convention)
        config_file = os.path.join(model_path, "args.json")
        config = scGPTConfig.load(config_file)
        
        # If ntoken is None, load from vocab.json
        if config.ntoken is None:
            vocab_file = os.path.join(model_path, "vocab.json")
            if os.path.exists(vocab_file):
                try:
                    with open(vocab_file, 'r') as f:
                        vocab_data = json.load(f)
                        # vocab.json is a dict of {gene: index}, ntoken = max_index + 1
                        if isinstance(vocab_data, dict):
                            config.ntoken = max(vocab_data.values()) + 1
                        else:
                            config.ntoken = len(vocab_data)
                except Exception as e:
                    logger.warning(f"Failed to load ntoken from vocab.json: {e}")
                    config.ntoken = 60697  # Fallback default
            else:
                config.ntoken = 60697  # Fallback default
        
        # Find model file (try both model.pt and best_model.pt)
        model_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_path, "best_model.pt")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Expected 'model.pt' or 'best_model.pt'."
                )
        
        # Create model instance
        model = cls(config, device=device, **model_init_kwargs)
        
        # Load state dict using official load_pretrained (handles flash-attn <-> pytorch conversion)
        state_dict = torch.load(model_file, map_location=device)
        load_pretrained(model.model, state_dict, verbose=False)
        
        logger.info(f"✓ Model loaded successfully")
        
        return model
