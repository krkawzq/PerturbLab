import json
import logging
import os
import time
import warnings
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from anndata import AnnData
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ...data import PerturbationData
from ...utils import download_from_huggingface
from ..base import PerturbationModel
from .config import scGPTConfig
from .source.loss import criterion_neg_log_bernoulli, masked_mse_loss
from .source.model import TransformerGenerator, TransformerModel
from .source.tokenizer.gene_tokenizer import (GeneVocab,
                                              get_default_gene_vocab,
                                              random_mask_value)
from .source.utils.util import load_pretrained

logger = logging.getLogger(__name__)


class scGPTModel(PerturbationModel):
    """scGPT model for single-cell gene expression analysis.

    Supports pretraining, fine-tuning, and various downstream tasks using a
    Transformer-based architecture specialized for single-cell data.

    Available pretrained models on HuggingFace:
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
    _pretrained_models = [
        "scgpt-human",
        "scgpt-blood",
        "scgpt-brain",
        "scgpt-heart",
        "scgpt-kidney",
        "scgpt-lung",
        "scgpt-pan-cancer",
        "scgpt-continual-pretrained",
    ]

    def __init__(
        self,
        config: scGPTConfig,
        gene_list: Optional[List[str]] = None,
        device: str = 'cpu',
        **kwargs,
    ):
        """Initializes the scGPT model.

        Args:
            config: scGPT configuration object.
            gene_list: List of gene names.
            device: Computation device.
        """
        super().__init__(config)

        # Initialize Vocabulary
        if config.use_default_gene_vocab:
            self.vocab = get_default_gene_vocab()
        else:
            if gene_list is None:
                raise ValueError(
                    "gene_list is required when config.use_default_gene_vocab is False"
                )
            self.vocab = GeneVocab(
                gene_list,
                config.specials,
                config.special_first,
                config.default_token,
            )

        # Ensure pad_token exists in vocab and set as default index
        if config.pad_token not in self.vocab:
            self.vocab.append_token(config.pad_token)
        self.vocab.set_default_index(self.vocab[config.pad_token])

        # Initialize Transformer
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
        ).to(device)
    
    def to(self, device: str):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        return self.train(False)
    
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

    @classmethod
    def mask_values(
        cls,
        values: Union[torch.Tensor, np.ndarray],
        mask_ratio: float = 0.4,
        mask_value: int = -1,
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Performs random masking on values for Masked Language Modeling."""
        return random_mask_value(
            values,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

    def get_dataloader(
        self,
        dataset: Union[AnnData, PerturbationData],
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        mask_ratio: float = 0.0,
        mask_value: int = -1,
        split: Optional[str] = None,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Prepares DataLoaders for the dataset.

        Args:
            dataset: Input dataset (AnnData or PerturbationData).
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop incomplete batches.
            num_workers: Number of worker threads.
            mask_ratio: Ratio of values to mask (0.0 = no masking).
            mask_value: Value to use for masking.
            split: Specific split to return ('train', 'val', 'test'), or None for all splits.
                   If dataset has no splits, returns 'train' key.

        Returns:
            DataLoader (if split is specified) or Dictionary of DataLoaders.
        """
        adata = self._get_adata_from_data(dataset)
        gene_names = adata.var_names.tolist()
        gene_ids = [
            self.vocab[g] if g in self.vocab else self.vocab[self.config.pad_token]
            for g in gene_names
        ]

        # Determine max sequence length
        max_len = (
            self.config.max_seq_len
            if self.config.max_seq_len is not None
            else len(gene_ids)
        )
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

            # Generate padding mask (assuming all selected genes are valid)
            batch_padding_mask = torch.zeros_like(batch_src, dtype=torch.bool)

            output = {
                "src": batch_src,
                "values": batch_values,  # Target if masking is applied
                "src_key_padding_mask": batch_padding_mask,
            }

            # Apply masking
            if mask_ratio > 0:
                masked_values = self.mask_values(
                    batch_values,
                    mask_ratio=mask_ratio,
                    mask_value=mask_value,
                    pad_value=self.config.pad_value,
                )
                output["masked_values"] = masked_values
                output["target_values"] = batch_values
            else:
                output["masked_values"] = batch_values

            # Handle batch labels
            if len(batch[0]) > 1:
                batch_labels = torch.stack([item[1] for item in batch])
                output["batch_labels"] = batch_labels

            return output

        # Handle Splits
        adata_map = {}
        has_split = "split" in adata.obs
        
        if split is not None:
            # Return specific split
            if has_split:
                if split in adata.obs["split"].values:
                    subset = adata[adata.obs["split"] == split]
                    adata_map[split] = subset
                else:
                    raise ValueError(f"Split '{split}' not found in dataset")
            else:
                # No split in data, return all as requested split
                adata_map[split] = adata
        else:
            # Return all splits
            if has_split:
                split_names = adata.obs["split"].unique()
                for split_name in split_names:
                    subset = adata[adata.obs["split"] == split_name]
                    adata_map[str(split_name)] = subset
            else:
                # No split in data, return as 'train'
                adata_map["train"] = adata

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
                batch_label_key = self.config.batch_label_key
                if batch_label_key in adata_subset.obs:
                    batch_labels = (
                        adata_subset.obs[batch_label_key]
                        .astype("category")
                        .cat.codes.values
                    )
                    tensors_to_pack.append(torch.tensor(batch_labels, dtype=torch.long))
                else:
                    tensors_to_pack.append(
                        torch.zeros(len(adata_subset), dtype=torch.long)
                    )

            dataset_tensor = TensorDataset(*tensors_to_pack)

            loader = DataLoader(
                dataset_tensor,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=scgpt_collate,
            )

            dataloaders[split_name] = loader

        # Return single DataLoader if split is specified, otherwise return dict
        if split is not None:
            return dataloaders[split]
        return dataloaders

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass retrieves embeddings and model outputs.
        
        Args:
            batch_data: Input batch dictionary.
            CLS: Whether to compute CLS loss.
            CCE: Whether to compute CCE loss.
            MVC: Whether to compute MVC loss.
            ECS: Whether to compute ECS loss.
            do_sample: Whether to sample during generation.
            
        Returns:
            Dictionary containing model outputs.
        """
        batch_data = {k: v.to(self.model.device) for k, v in batch_data.items()}

        src = batch_data["src"]
        # Use masked values if available, otherwise raw values
        values = batch_data.get("masked_values", batch_data["values"])
        src_key_padding_mask = batch_data["src_key_padding_mask"]
        batch_labels = batch_data.get("batch_labels", None)

        output_dict = self.model(
            src=src,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
            CLS=CLS,
            CCE=CCE,
            MVC=MVC,
            ECS=ECS,
            do_sample=do_sample,
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
        mask_value: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Computes model losses (MSE, GEPC, ECS, DAB, Zero-prob).
        
        Args:
            batch_data: Input batch dictionary.
            output_dict: Optional pre-computed outputs.
            CLS: Whether to compute CLS loss.
            CCE: Whether to compute CCE loss.
            MVC: Whether to compute MVC loss.
            ECS: Whether to compute ECS loss.
            do_sample: Whether to sample during generation.
            mask_value: Value used for masking.
            
        Returns:
            Dictionary with 'loss' key containing the total loss.
        """
        if output_dict is None:
            output_dict = self.forward(
                batch_data,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample,
            )

        losses = {}
        total_loss = 0.0

        if "target_values" in batch_data:
            batch_data = {k: v.to(self.model.device) for k, v in batch_data.items()}
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
                # Use configurable ECS loss weight
                loss_ecs = output_dict["loss_ecs"] * self.config.ecs_loss_weight
                total_loss += loss_ecs
                losses["loss_ecs"] = loss_ecs

            # 5. DAB Loss
            if (
                self.config.do_dab
                and "dab_output" in output_dict
                and "batch_labels" in batch_data
            ):
                loss_dab = self.criterion_dab(
                    output_dict["dab_output"], batch_data["batch_labels"]
                )
                total_loss += loss_dab * self.config.dab_weight
                losses["loss_dab"] = loss_dab

        losses["loss"] = total_loss
        return losses

    def predict_embeddings(
        self,
        dataset: Union[AnnData, PerturbationData],
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        split: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Unified embedding prediction method.

        Args:
            dataset: Input dataset.
            batch_size: Batch size.
            embedding_type: Type of embedding ('cell' or 'gene').
            split: Specific split to return, or None for all splits.

        Returns:
            Dictionary with 'cell' or 'gene' key containing embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        self.eval()
        
        if embedding_type == "gene":
            with torch.no_grad():
                gene_embs = self.model.encoder.embedding.weight.cpu().numpy()
            return {'gene': gene_embs}

        elif embedding_type == "cell":
            loaders = self.get_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                mask_ratio=0.0,
                split=split,
            )
            
            # If split is specified, loaders is a single DataLoader
            if split is not None:
                embeddings = []
                with torch.no_grad():
                    for batch in tqdm(loaders, desc=f"Encoding cells ({split})"):
                        batch = {k: v.to(self.model.device) for k, v in batch.items()}
                        out = self.forward(batch)
                        cell_emb = out["cell_emb"]  # (batch, embsize)
                        embeddings.append(cell_emb.cpu().numpy())
                
                return {'cell': np.concatenate(embeddings, axis=0)}
            else:
                # Return dict with split names as keys
                result = {}
                for split_name, loader in loaders.items():
                    embeddings = []
                    with torch.no_grad():
                        for batch in tqdm(loader, desc=f"Encoding cells ({split_name})"):
                            batch = {k: v.to(self.model.device) for k, v in batch.items()}
                            out = self.forward(batch)
                            cell_emb = out["cell_emb"]  # (batch, embsize)
                            embeddings.append(cell_emb.cpu().numpy())
                    
                    result[split_name] = {'cell': np.concatenate(embeddings, axis=0)}
                
                return result
        else:
            raise ValueError(
                f"embedding_type must be 'cell' or 'gene', got {embedding_type}"
            )

    def train_model(
        self,
        dataset: Union[AnnData, PerturbationData],
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        mask_ratio: float = 0.4,
        mask_value: int = -1,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 100,
        save_dir: Optional[str] = None,
        save_interval: Optional[int] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        scheduler_step: int = 1,
        scheduler_gamma: float = 0.99,
        **kwargs,
    ):
        """Trains the scGPT model.

        Args:
            dataset: Training data.
            epochs: Number of epochs.
            batch_size: Batch size.
            lr: Learning rate.
            mask_ratio: Masking ratio for MLM.
            mask_value: Value used for masking.
            use_amp: Whether to use automatic mixed precision.
            grad_clip: Gradient clipping value.
            log_interval: Logging frequency.
            save_dir: Directory to save checkpoints.
            save_interval: Checkpoint saving frequency.
            CLS: Whether to use CLS loss.
            CCE: Whether to use CCE loss.
            MVC: Whether to use MVC loss.
            ECS: Whether to use ECS loss.
            scheduler_step: Scheduler step size.
            scheduler_gamma: Scheduler gamma.

        Returns:
            Dictionary containing training history.
        """
        dataloaders = self.get_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            split=None,  # Get all splits
        )

        train_loader = dataloaders.get("train")
        if train_loader is None:
            raise ValueError("No 'train' split found in dataloaders")
        
        # Try multiple common validation split names
        valid_loader = dataloaders.get("valid") or dataloaders.get("val") or dataloaders.get("test")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        history = {"train_loss": [], "valid_loss": [], "epoch_times": []}

        logger.info(
            f"Starting training: {epochs} epochs | Train batches: {len(train_loader)}"
        )

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Training Phase
            self.train()
            total_loss = 0.0
            num_batches = len(train_loader)

            with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for batch_idx, batch_data in enumerate(pbar, 1):
                    # Forward pass with AMP
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_dict = self.forward(
                            batch_data,
                            CLS=CLS,
                            CCE=CCE,
                            MVC=MVC,
                            ECS=ECS,
                        )

                        losses = self.compute_loss(
                            batch_data,
                            output_dict=output_dict,
                            CLS=CLS,
                            CCE=CCE,
                            MVC=MVC,
                            ECS=ECS,
                            mask_value=mask_value,
                        )
                        loss = losses["loss"]

                    # Backward pass
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    # Gradient clipping
                    with warnings.catch_warnings(record=True) as w:
                        warnings.filterwarnings("always")
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            grad_clip,
                            error_if_nonfinite=False if scaler.is_enabled() else True,
                        )
                        if len(w) > 0:
                            logger.warning(
                                f"Found infinite gradient at batch {batch_idx}. Scale: {scaler.get_scale()}"
                            )

                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()

                    if batch_idx % log_interval == 0:
                        avg_loss = total_loss / batch_idx
                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            }
                        )

                    # Interval Checkpoint
                    if (
                        save_dir
                        and save_interval
                        and batch_idx % save_interval == 0
                    ):
                        ckpt_path = os.path.join(
                            save_dir, f"checkpoint_epoch{epoch}_batch{batch_idx}"
                        )
                        self.save(ckpt_path)

            avg_train_loss = total_loss / num_batches
            history["train_loss"].append(avg_train_loss)

            # Validation Phase
            if valid_loader:
                self.eval()
                valid_loss = 0.0
                num_valid_batches = 0

                with torch.no_grad():
                    for batch_data in valid_loader:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            output_dict = self.forward(
                                batch_data,
                                CLS=CLS,
                                CCE=False,
                                MVC=False,
                                ECS=False,
                            )

                            losses = self.compute_loss(
                                batch_data,
                                output_dict=output_dict,
                                CLS=CLS,
                                mask_value=mask_value,
                            )
                            valid_loss += losses["loss"].item()
                            num_valid_batches += 1

                avg_valid_loss = valid_loss / num_valid_batches
                history["valid_loss"].append(avg_valid_loss)
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                    f"Valid Loss: {avg_valid_loss:.4f} | "
                    f"Time: {time.time() - epoch_start_time:.2f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                    f"Time: {time.time() - epoch_start_time:.2f}s"
                )

            history["epoch_times"].append(time.time() - epoch_start_time)
            scheduler.step()

            # Epoch Checkpoint
            if save_dir:
                epoch_ckpt_path = os.path.join(save_dir, f"epoch_{epoch}")
                self.save(epoch_ckpt_path)

        logger.info("Training completed!")
        return history

    def save(self, model_path: str):
        """
        Saves model weights, configuration, and vocabulary.
        
        Args:
            save_directory: Directory to save the model.
            is_best: If True, saves weights as 'best_model.pt', otherwise 'model.pt'.
        """
        os.makedirs(model_path, exist_ok=True)

        # 1. Save Config
        config_file = os.path.join(model_path, "args.json")
        self.config.save(config_file)
        
        # 2. Save Vocabulary (Crucial for symmetry)
        if hasattr(self, 'vocab') and self.vocab is not None:
            vocab_file = os.path.join(model_path, "vocab.json")
            # Handle both GeneVocab objects and dicts
            if hasattr(self.vocab, 'save_json'):
                self.vocab.save_json(vocab_file)
            elif isinstance(self.vocab, dict):
                with open(vocab_file, 'w') as f:
                    json.dump(self.vocab, f, indent=2)
            logger.info(f"Saved vocab to {vocab_file}")

        # 3. Save Model Weights
        filename = 'model.pt'
        model_file = os.path.join(model_path, filename)
        torch.save(self.model.state_dict(), model_file)

        logger.info(f"Model saved to {model_path} (weights: {filename})")

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu') -> 'scGPTModel':
        """
        Loads a scGPT model from a saved directory.
        Compatible with structure: [args.json, vocab.json, best_model.pt]

        Args:
            model_path: Path to the saved model directory.
            device: Computation device.
            **kwargs: Extra arguments for model initialization.

        Returns:
            Loaded scGPTModel instance.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # 1. Load Config
        config_file = os.path.join(model_path, "args.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")
        
        config = scGPTConfig.load(config_file)
        logger.info(f"Loaded config from {config_file}")

        # 2. Load Vocab & Infer ntoken (Symmetry Step)
        vocab = None
        vocab_file = os.path.join(model_path, "vocab.json")
        
        if os.path.exists(vocab_file):
            try:
                # Try loading as GeneVocab (if available in context) or dict
                try:
                    from .source.tokenizer import GeneVocab 
                    vocab = GeneVocab.from_file(vocab_file)
                    logger.info(f"Loaded GeneVocab object from {vocab_file}")
                except (ImportError, AttributeError):
                    # Fallback to dict
                    with open(vocab_file, "r") as f:
                        vocab = json.load(f)
                    logger.info(f"Loaded vocab dict from {vocab_file}")

                # Infer ntoken if missing in config
                if config.ntoken is None:
                    if hasattr(vocab, '__len__'):
                        config.ntoken = len(vocab)
                    logger.info(f"Inferred ntoken={config.ntoken} from vocab")
                    
            except Exception as e:
                logger.warning(f"Failed to load vocab from {vocab_file}: {e}")
        
        # Fallback for ntoken: Infer from weights later if still None
        if config.ntoken is None:
             logger.warning("ntoken not in config and could not be inferred from vocab. Will attempt to infer from weights.")

        # 3. Locate Weights File (Priority: best_model.pt -> model.pt)
        weights_name = "best_model.pt"
        if not os.path.exists(os.path.join(model_path, weights_name)):
            weights_name = "model.pt"
        
        model_file = os.path.join(model_path, weights_name)
        if not os.path.exists(model_file):
             raise FileNotFoundError(f"No model weights found in {model_path} (checked best_model.pt and model.pt)")

        # 4. Initialize Model
        # If ntoken is still None, this might fail depending on model init logic.
        # We assume config has reasonable defaults or ntoken was found.
        model_wrapper = cls(config, device=device)
        
        # Restore vocab object to wrapper (Symmetry)
        if vocab is not None:
            model_wrapper.vocab = vocab

        # 5. Load State Dict
        logger.info(f"Loading weights from {model_file}...")
        # map_location ensures safety on CPU-only machines or different GPU setups
        state_dict = torch.load(model_file, map_location=device)
        
        # Late inference of ntoken if absolutely necessary
        if config.ntoken is None and "encoder.embedding.weight" in state_dict:
             inferred_ntoken = state_dict["encoder.embedding.weight"].shape[0]
             logger.info(f"Late inference of ntoken={inferred_ntoken} from weights. Re-initializing model...")
             config.ntoken = inferred_ntoken
             model_wrapper = cls(config, device=device) # Re-init
             if vocab is not None: model_wrapper.vocab = vocab

        load_pretrained(model_wrapper.model, state_dict, verbose=False)

        logger.info("✓ Model loaded successfully")
        return model_wrapper


class scGPTPerturbationModel(PerturbationModel):
    """scGPT model specialized for perturbation prediction tasks.

    Uses the TransformerGenerator architecture to predict the gene expression
    state of a cell after perturbation, given a control state and perturbation tokens.
    """

    # Registry of pretrained models for perturbation
    _pretrained_models = []

    def __init__(
        self,
        config: scGPTConfig,
        gene_list: Optional[List[str]] = None,
        device: str = 'cpu',
        **kwargs,
    ):
        """Initializes the scGPT perturbation model.
        
        Args:
            config: scGPT configuration object.
            gene_list: List of gene names.
            device: Computation device.
        """
        super().__init__(config)

        if config.use_default_gene_vocab:
            self.vocab = get_default_gene_vocab()
        else:
            if gene_list is None:
                raise ValueError(
                    "gene_list is required when config.use_default_gene_vocab is False"
                )
            self.vocab = GeneVocab(
                gene_list,
                config.specials,
                config.special_first,
                config.default_token,
            )

        if config.pad_token not in self.vocab:
            self.vocab.append_token(config.pad_token)
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
        ).to(device)

    def to(self, device: str):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def get_dataloader(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        mask_ratio: float = 0.0,
        split: Optional[str] = None,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        """Prepares DataLoaders for perturbation prediction.
        
        Args:
            dataset: PerturbationData object.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop incomplete batches.
            num_workers: Number of worker threads.
            mask_ratio: Ratio of values to mask.
            split: Specific split to return, or None for all splits.
                   
        Returns:
            DataLoader or dictionary of DataLoaders.
        """
        if "ctrl_indices" not in dataset.adata.obsm:
            logger.info("Pairing cells for perturbation task...")
            dataset.pair_cells()

        adata = dataset.adata
        
        # Optimize sparse matrix format for efficient row slicing
        if scipy.sparse.issparse(adata.X) and not scipy.sparse.isspmatrix_csr(adata.X):
            logger.warning(
                "Converting adata.X to CSR format for efficient row slicing. "
                "This may take some time for large matrices."
            )
            adata.X = adata.X.tocsr()
        
        gene_names = adata.var_names.tolist()
        gene_ids = [
            self.vocab[g] if g in self.vocab else self.vocab[self.config.pad_token]
            for g in gene_names
        ]

        max_len = (
            self.config.max_seq_len
            if self.config.max_seq_len is not None
            else len(gene_ids)
        )
        if len(gene_ids) > max_len:
            gene_ids = gene_ids[:max_len]
            slice_cols = slice(0, max_len)
            gene_names = gene_names[:max_len]
        else:
            slice_cols = slice(None)

        src_tensor = torch.tensor(gene_ids, dtype=torch.long)
        gene_name_to_idx = {name: i for i, name in enumerate(gene_names)}

        # Handle Splits
        adata_map = {}
        has_split = "split" in adata.obs
        
        if split is not None:
            # Return specific split
            if has_split:
                if split in adata.obs["split"].values:
                    indices = np.where(adata.obs["split"] == split)[0]
                    adata_map[split] = indices
                else:
                    raise ValueError(f"Split '{split}' not found in dataset")
            else:
                # No split in data, return all as requested split
                adata_map[split] = np.arange(adata.n_obs)
        else:
            # Return all splits
            if has_split:
                for split_name in adata.obs["split"].unique():
                    indices = np.where(adata.obs["split"] == split_name)[0]
                    adata_map[str(split_name)] = indices
            else:
                # No split in data, return as 'train'
                adata_map["train"] = np.arange(adata.n_obs)

        dataloaders = {}

        for split_name, indices in adata_map.items():
            if len(indices) == 0:
                continue

            index_dataset = TensorDataset(torch.tensor(indices, dtype=torch.long))

            def make_collate_fn(
                gene_name_to_idx_local, src_tensor_local, slice_cols_local
            ):
                def pert_collate(batch):
                    batch_indices = torch.stack([item[0] for item in batch]).numpy()

                    # Fetch control indices (using first sample)
                    ctrl_indices = adata.obsm["ctrl_indices"][
                        batch_indices, 0
                    ]

                    # Fetch expressions
                    X = adata.X
                    if scipy.sparse.issparse(X):
                        x_val = X[ctrl_indices, slice_cols_local].toarray()
                        y_val = X[batch_indices, slice_cols_local].toarray()
                    else:
                        x_val = X[ctrl_indices, slice_cols_local]
                        y_val = X[batch_indices, slice_cols_local]

                    # Identify perturbed genes
                    conditions = adata.obs.iloc[batch_indices][
                        "condition"
                    ].values
                    pert_flags = np.zeros_like(x_val, dtype=np.int64)

                    for i, cond in enumerate(conditions):
                        if cond != "ctrl":
                            perts = cond.split(self.config.perturbation_delimiter)
                            for p in perts:
                                p = p.strip()  # Remove whitespace
                                if p in gene_name_to_idx_local:
                                    pert_flags[i, gene_name_to_idx_local[p]] = 1

                    input_values = torch.tensor(x_val, dtype=torch.float32)
                    target_values = torch.tensor(y_val, dtype=torch.float32)
                    pert_flags_tensor = torch.tensor(pert_flags, dtype=torch.long)

                    curr_bs = input_values.shape[0]
                    batch_src = src_tensor_local.unsqueeze(0).repeat(curr_bs, 1)
                    batch_padding_mask = torch.zeros_like(
                        batch_src, dtype=torch.bool
                    )

                    return {
                        "src": batch_src,
                        "values": input_values,
                        "target_values": target_values,
                        "input_pert_flags": pert_flags_tensor,
                        "src_key_padding_mask": batch_padding_mask,
                    }

                return pert_collate

            collate_fn = make_collate_fn(gene_name_to_idx, src_tensor, slice_cols)

            dataloaders[split_name] = DataLoader(
                index_dataset,
                batch_size=batch_size,
                shuffle=(shuffle and split_name == "train"),
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        # Return single DataLoader if split is specified, otherwise return dict
        if split is not None:
            return dataloaders[split]
        return dataloaders

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_data = {k: v.to(self.model.device) for k, v in batch_data.items()}

        output_dict = self.model(
            src=batch_data["src"],
            values=batch_data["values"],
            input_pert_flags=batch_data["input_pert_flags"],
            src_key_padding_mask=batch_data["src_key_padding_mask"],
            CLS=CLS,
            CCE=CCE,
            MVC=MVC,
            ECS=ECS,
            do_sample=do_sample,
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
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample,
            )

        losses = {}
        total_loss = 0.0

        if "target_values" in batch_data:
            target_values = batch_data["target_values"].to(self.model.device)
            # Typically predict all values in perturbation tasks
            masked_positions = torch.ones_like(target_values, dtype=torch.bool)

            loss_mse = masked_mse_loss(
                output_dict["mlm_output"], target_values, masked_positions
            )
            total_loss += loss_mse
            losses["loss_mse"] = loss_mse

        losses["loss"] = total_loss
        return losses

    def predict_embeddings(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        split: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Unified embedding prediction.

        Args:
            dataset: PerturbationData object.
            batch_size: Batch size.
            embedding_type: Type of embedding ('cell' or 'gene').
            split: Specific split to return, or None for all splits.

        Returns:
            Dictionary with 'cell' or 'gene' key containing embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        self.eval()

        if embedding_type == "gene":
            with torch.no_grad():
                gene_embs = self.model.encoder.embedding.weight.cpu().numpy()
            return {'gene': gene_embs}

        elif embedding_type == "cell":
            loaders = self.get_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                split=split,
            )
            
            # If split is specified, loaders is a single DataLoader
            if split is not None:
                embeddings = []
                with torch.no_grad():
                    for batch in tqdm(loaders, desc=f"Encoding cells ({split})"):
                        batch = {k: v.to(self.model.device) for k, v in batch.items()}

                        # Use internal methods to retrieve cell embeddings from TransformerGenerator
                        src = batch["src"]
                        values = batch["values"]
                        input_pert_flags = batch["input_pert_flags"]
                        src_key_padding_mask = batch["src_key_padding_mask"]

                        transformer_output = self.model._encode(
                            src, values, input_pert_flags, src_key_padding_mask
                        )
                        cell_emb = self.model._get_cell_emb_from_layer(
                            transformer_output, values
                        )

                        embeddings.append(cell_emb.cpu().numpy())

                return {'cell': np.concatenate(embeddings, axis=0)}
            else:
                # Return dict with split names as keys
                result = {}
                for split_name, loader in loaders.items():
                    embeddings = []
                    with torch.no_grad():
                        for batch in tqdm(loader, desc=f"Encoding cells ({split_name})"):
                            batch = {k: v.to(self.model.device) for k, v in batch.items()}

                            src = batch["src"]
                            values = batch["values"]
                            input_pert_flags = batch["input_pert_flags"]
                            src_key_padding_mask = batch["src_key_padding_mask"]

                            transformer_output = self.model._encode(
                                src, values, input_pert_flags, src_key_padding_mask
                            )
                            cell_emb = self.model._get_cell_emb_from_layer(
                                transformer_output, values
                            )

                            embeddings.append(cell_emb.cpu().numpy())

                    result[split_name] = {'cell': np.concatenate(embeddings, axis=0)}
                
                return result
        else:
            raise ValueError(
                f"embedding_type must be 'cell' or 'gene', got {embedding_type}"
            )

    def predict_perturbation(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        split: Optional[str] = None,
        return_numpy: bool = True,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Predicts perturbation effects.

        Args:
            dataset: Input perturbation dataset.
            batch_size: Batch size.
            split: Specific split to return, or None for all splits.
            return_numpy: Whether to return numpy array.

        Returns:
            Dictionary with 'pred' key containing predictions.
            If split is None and dataset has splits, returns nested dict.
        """
        self.eval()
        loaders = self.get_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            split=split,
        )

        # If split is specified, loaders is a single DataLoader
        if split is not None:
            predictions = []
            with torch.no_grad():
                for batch in tqdm(loaders, desc=f"Predicting perturbations ({split})"):
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    output = self.forward(batch)
                    pred = output["mlm_output"]  # (batch, seq_len)

                    if return_numpy:
                        predictions.append(pred.cpu().numpy())
                    else:
                        predictions.append(pred.cpu())

            if return_numpy:
                return {'pred': np.concatenate(predictions, axis=0)}
            else:
                return {'pred': torch.cat(predictions, dim=0)}
        else:
            # Return dict with split names as keys
            result = {}
            for split_name, loader in loaders.items():
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"Predicting perturbations ({split_name})"):
                        batch = {k: v.to(self.model.device) for k, v in batch.items()}
                        output = self.forward(batch)
                        pred = output["mlm_output"]  # (batch, seq_len)

                        if return_numpy:
                            predictions.append(pred.cpu().numpy())
                        else:
                            predictions.append(pred.cpu())

                if return_numpy:
                    result[split_name] = {'pred': np.concatenate(predictions, axis=0)}
                else:
                    result[split_name] = {'pred': torch.cat(predictions, dim=0)}
            
            return result

    def train_model(
        self,
        dataset: PerturbationData,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 100,
        save_dir: Optional[str] = None,
        save_interval: Optional[int] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        scheduler_step: int = 1,
        scheduler_gamma: float = 0.99,
        **kwargs,
    ):
        """Trains the scGPT perturbation model.
        
        Args:
            dataset: Training perturbation dataset.
            epochs: Number of epochs.
            batch_size: Batch size.
            lr: Learning rate.
            use_amp: Whether to use automatic mixed precision.
            grad_clip: Gradient clipping value.
            log_interval: Logging frequency.
            save_dir: Directory to save checkpoints.
            save_interval: Checkpoint saving frequency.
            CLS: Whether to use CLS loss.
            CCE: Whether to use CCE loss.
            MVC: Whether to use MVC loss.
            ECS: Whether to use ECS loss.
            scheduler_step: Scheduler step size.
            scheduler_gamma: Scheduler gamma.
            
        Returns:
            Dictionary containing training history.
        """
        dataloaders = self.get_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            split=None,
        )

        train_loader = dataloaders.get("train")
        if train_loader is None:
            raise ValueError("No 'train' split found in dataloaders")
        
        # Try multiple common validation split names
        valid_loader = dataloaders.get("val")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        history = {"train_loss": [], "valid_loss": [], "epoch_times": []}

        logger.info(
            f"Starting perturbation training: {epochs} epochs | Train batches: {len(train_loader)}"
        )

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            self.train()
            total_loss = 0.0
            num_batches = len(train_loader)

            with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for batch_idx, batch_data in enumerate(pbar, 1):
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_dict = self.forward(
                            batch_data,
                            CLS=CLS,
                            CCE=CCE,
                            MVC=MVC,
                            ECS=ECS,
                        )

                        losses = self.compute_loss(
                            batch_data,
                            output_dict=output_dict,
                            CLS=CLS,
                            CCE=CCE,
                            MVC=MVC,
                            ECS=ECS,
                        )
                        loss = losses["loss"]

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    with warnings.catch_warnings(record=True) as w:
                        warnings.filterwarnings("always")
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            grad_clip,
                            error_if_nonfinite=False if scaler.is_enabled() else True,
                        )
                        if len(w) > 0:
                            logger.warning(
                                f"Found infinite gradient at batch {batch_idx}. Scale: {scaler.get_scale()}"
                            )

                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()

                    if batch_idx % log_interval == 0:
                        avg_loss = total_loss / batch_idx
                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            }
                        )

                    if (
                        save_dir
                        and save_interval
                        and batch_idx % save_interval == 0
                    ):
                        ckpt_path = os.path.join(
                            save_dir, f"checkpoint_epoch{epoch}_batch{batch_idx}"
                        )
                        self.save(ckpt_path)

            avg_train_loss = total_loss / num_batches
            history["train_loss"].append(avg_train_loss)

            if valid_loader:
                self.eval()
                valid_loss = 0.0
                num_valid_batches = 0

                with torch.no_grad():
                    for batch_data in valid_loader:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            output_dict = self.forward(
                                batch_data,
                                CLS=CLS,
                                CCE=False,
                                MVC=False,
                                ECS=False,
                            )

                            losses = self.compute_loss(
                                batch_data,
                                output_dict=output_dict,
                                CLS=CLS,
                            )
                            valid_loss += losses["loss"].item()
                            num_valid_batches += 1

                avg_valid_loss = valid_loss / num_valid_batches
                history["valid_loss"].append(avg_valid_loss)
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                    f"Valid Loss: {avg_valid_loss:.4f} | "
                    f"Time: {time.time() - epoch_start_time:.2f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                    f"Time: {time.time() - epoch_start_time:.2f}s"
                )

            history["epoch_times"].append(time.time() - epoch_start_time)
            scheduler.step()

            if save_dir:
                epoch_ckpt_path = os.path.join(save_dir, f"epoch_{epoch}")
                self.save(epoch_ckpt_path)

        logger.info("Perturbation model training completed!")
        return history

    def save(self, model_path: str):
        """
        Saves model weights, configuration, and vocabulary.

        Args:
            save_directory: Directory to save the model.
            is_best: If True, saves weights as 'best_model.pt', otherwise 'model.pt'.
        """
        os.makedirs(model_path, exist_ok=True)

        # 1. Save Config
        config_file = os.path.join(model_path, "args.json")
        self.config.save(config_file)
        
        # 2. Save Vocabulary (Symmetry with load)
        if hasattr(self, 'vocab') and self.vocab is not None:
            vocab_file = os.path.join(model_path, "vocab.json")
            # Handle both GeneVocab objects (with save_json method) and raw dicts
            if hasattr(self.vocab, 'save_json'):
                self.vocab.save_json(vocab_file)
            elif isinstance(self.vocab, dict):
                with open(vocab_file, 'w') as f:
                    json.dump(self.vocab, f, indent=2)
            logger.info(f"Saved vocab to {vocab_file}")

        # 3. Save Model Weights
        filename = 'model.pt'
        model_file = os.path.join(model_path, filename)
        torch.save(self.model.state_dict(), model_file)

        logger.info(f"Model saved to {model_path} (weights: {filename})")

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu') -> 'scGPTPerturbationModel':
        """
        Loads a scGPT perturbation model from a saved directory.
        Compatible with structure: [args.json, vocab.json, best_model.pt/model.pt]

        Args:
            model_path: Path to the saved model directory.
            device: Computation device.
            **kwargs: Extra arguments for model initialization.

        Returns:
            Loaded scGPTPerturbationModel instance.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # 1. Load Config
        config_file = os.path.join(model_path, "args.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")
        
        config = scGPTConfig.load(config_file)
        logger.info(f"Loaded config from {config_file}")

        # 2. Load Vocab & Infer ntoken (Symmetry Step)
        vocab = None
        vocab_file = os.path.join(model_path, "vocab.json")
        
        if os.path.exists(vocab_file):
            try:
                # Try loading as GeneVocab (if available in context) or dict
                try:
                    from .source.tokenizer import GeneVocab 
                    vocab = GeneVocab.from_file(vocab_file)
                    logger.info(f"Loaded GeneVocab object from {vocab_file}")
                except (ImportError, AttributeError):
                    # Fallback to dict if GeneVocab class isn't available
                    with open(vocab_file, "r") as f:
                        vocab = json.load(f)
                    logger.info(f"Loaded vocab dict from {vocab_file}")

                # Infer ntoken if missing in config
                if config.ntoken is None:
                    if hasattr(vocab, '__len__'):
                        config.ntoken = len(vocab)
                    elif isinstance(vocab, dict):
                         # If vocab is a dict (token->id), length is the count
                        config.ntoken = len(vocab)
                    logger.info(f"Inferred ntoken={config.ntoken} from vocab")
                    
            except Exception as e:
                logger.warning(f"Failed to load vocab from {vocab_file}: {e}")
        
        # Fallback for ntoken: Warn if missing (will try to infer from weights later)
        if config.ntoken is None:
             logger.warning("ntoken not in config and could not be inferred from vocab. Will attempt to infer from weights.")

        # 3. Locate Weights File (Priority: best_model.pt -> model.pt)
        weights_name = "best_model.pt"
        if not os.path.exists(os.path.join(model_path, weights_name)):
            weights_name = "model.pt"
        
        model_file = os.path.join(model_path, weights_name)
        if not os.path.exists(model_file):
             raise FileNotFoundError(f"No model weights found in {model_path} (checked best_model.pt and model.pt)")

        # 4. Initialize Model
        # Note: If ntoken is None here, initialization might fail depending on model logic.
        # We assume config has defaults or ntoken was successfully inferred.
        model_wrapper = cls(config, device=device)
        
        # Restore vocab object to wrapper (Symmetry)
        if vocab is not None:
            model_wrapper.vocab = vocab

        # 5. Load State Dict
        logger.info(f"Loading weights from {model_file}...")
        # map_location ensures safety on CPU-only machines or different GPU setups
        state_dict = torch.load(model_file, map_location=device)
        
        # Late inference of ntoken if absolutely necessary (Last Resort)
        if config.ntoken is None and "encoder.embedding.weight" in state_dict:
             inferred_ntoken = state_dict["encoder.embedding.weight"].shape[0]
             logger.info(f"Late inference of ntoken={inferred_ntoken} from weights. Re-initializing model...")
             config.ntoken = inferred_ntoken
             # Re-init model with correct ntoken
             model_wrapper = cls(config, device=device)
             if vocab is not None: model_wrapper.vocab = vocab

        load_pretrained(model_wrapper.model, state_dict, verbose=False)

        logger.info("✓ Model loaded successfully")
        return model_wrapper
