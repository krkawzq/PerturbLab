"""
CellFM (Cell Foundation Model) wrapper for PerturbLab.

CellFM is a large-scale foundation model pre-trained on transcriptomics
of 100 million human cells using retention-based architecture.

Original source: https://github.com/biomed-AI/CellFM
PyTorch version: https://github.com/biomed-AI/CellFM-torch
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import issparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...data import GeneGraph, PerturbationData
from ...utils import download_from_huggingface
from ..base import PerturbationModel
from ..gears import GearsConfig, GearsModel
from .config import CellFMConfig
from .source.cellfm_model import CellFM
from .source.data_utils import CellFMDataProcessor
from .source.dataset import CellFMDataset, collate_fn

logger = logging.getLogger(__name__)


class CellFMModel(PerturbationModel):
    """CellFM (Cell Foundation Model) wrapper.

    CellFM is a retention-based foundation model for single-cell transcriptomics
    that can generate cell embeddings and perform various downstream tasks.
    Unlike standard Transformers, it uses a retention mechanism for O(N) inference cost.

    Features:
    - Pre-trained on 100M cells
    - Retention-based architecture (efficient scaling)
    - Supports cell annotation, batch correction, and perturbation prediction
    """
    
    # List of available pretrained models on HuggingFace
    _pretrained_models = [
        "cellfm-80m",
        "cellfm-400m",
    ]

    def __init__(
        self,
        config: CellFMConfig,
        n_genes: Optional[int] = None,
        device: str = "cuda",
        for_finetuning: bool = False,
    ):
        """Initializes the CellFM model.

        Args:
            config: CellFM configuration object.
            n_genes: Number of genes.
            device: Device to run the model on.
            for_finetuning: Whether to initialize with a classification head.
        """
        super().__init__(config)

        self.config = config
        self.device = device
        self.n_genes = n_genes or config.n_genes
        self.for_finetuning = for_finetuning

        # Initialize base model
        self.model = CellFM(
            n_gene=self.n_genes,
            config=config,
            device=device,
        )

        # Add classification head if needed
        self.cls_head = None
        if for_finetuning and config.num_cls is not None:
            self.cls_head = nn.Sequential(
                nn.Linear(config.enc_dims, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, config.num_cls),
            ).to(device)

        self.model = self.model.to(device)
        self.model_loaded = False

        logger.info(
            f"Initialized CellFM model (n_genes={self.n_genes}, device={device})"
        )

    @staticmethod
    def _extract_adata(data: Union[AnnData, PerturbationData]) -> AnnData:
        """Extracts AnnData from input data."""
        if isinstance(data, PerturbationData):
            return data.adata
        return data

    def load_weights(self, checkpoint_path: str, strict: bool = True):
        """Loads model weights from a PyTorch checkpoint.

        Args:
            checkpoint_path: Path to PyTorch checkpoint file.
            strict: Whether to strictly enforce key matching. Default True for safety.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading weights from {checkpoint_path}...")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Handle different checkpoint formats
        if "base_model" in checkpoint:
            # Format: {'base_model': ..., 'cls_head': ...}
            base_state_dict = checkpoint["base_model"]
            missing, unexpected = self.model.load_state_dict(
                base_state_dict, strict=strict
            )

            # Load classification head if present
            if "cls_head" in checkpoint and self.cls_head is not None:
                self.cls_head.load_state_dict(checkpoint["cls_head"], strict=strict)
                logger.info("Loaded classification head.")

        elif "model_state_dict" in checkpoint:
            # Standard PyTorch convention
            state_dict = checkpoint["model_state_dict"]
            missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)

        elif "state_dict" in checkpoint:
            # Lightning convention
            state_dict = checkpoint["state_dict"]
            missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)

        else:
            # Direct state dict
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=strict)

        # Enhanced validation: warn if too many keys are missing
        if missing:
            total_keys = len(self.model.state_dict())
            missing_ratio = len(missing) / max(1, total_keys)
            if missing_ratio > 0.1:  # More than 10% missing
                logger.error(
                    f"⚠️ Critical: {len(missing)}/{total_keys} keys missing "
                    f"({missing_ratio:.1%}). This may indicate version mismatch!"
                )
                logger.error(f"Missing keys: {missing}")
                if strict:
                    raise RuntimeError("Too many missing keys in checkpoint!")
            else:
                logger.warning(f"Missing keys ({len(missing)}): {missing}")
        
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected}")

        self.model_loaded = True
        logger.info("✓ Weights loaded successfully")

    @staticmethod
    def get_dataloader(
        data: Union[AnnData, PerturbationData],
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pad_length: int = 2048,
        mask_ratio: float = 0.5,
        random_mask: bool = False,
        add_zero: bool = True,
        pad_zero: bool = True,
        target_count: float = 1e4,
        max_genes: int = 2048,
        sample_weighted: bool = True,
        return_split: Optional[str] = None,
    ) -> Dict[str, DataLoader]:
        """
        Creates DataLoaders for CellFM training or inference.

        

        Args:
            data: Preprocessed AnnData or PerturbationData.
            batch_size: Batch size.
            shuffle: Whether to shuffle training data.
            num_workers: Number of worker threads.
            pad_length: Sequence padding length.
            mask_ratio: Ratio of genes to mask for pretraining task.
            random_mask: Enable random masking.
            add_zero: Include zero-expression gene tokens.
            pad_zero: Pad with zero tokens.
            target_count: Normalization target.
            max_genes: Max non-zero genes per cell.
            sample_weighted: Use weighted sampling.
            return_split: Specific split to return ('train', 'val', 'test').

        Returns:
            Dictionary mapping split names to DataLoaders.
        """
        adata = CellFMModel._extract_adata(data)
        has_split = "split" in adata.obs.columns

        if not has_split:
            logger.info("No split info found. Treating all data as 'train'.")
            dataset = CellFMDataset(
                adata=adata,
                pad_length=pad_length,
                mask_ratio=mask_ratio,
                random_mask=random_mask,
                add_zero=add_zero,
                pad_zero=pad_zero,
                target_count=target_count,
                max_genes=max_genes,
                sample_weighted=sample_weighted,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
            return {"train": dataloader}

        # Create DataLoaders for specific splits
        dataloaders = {}
        splits = adata.obs["split"].unique()

        if return_split is not None:
            if return_split not in splits:
                raise ValueError(f"Split '{return_split}' not found in data.")
            splits_to_create = [return_split]
        else:
            splits_to_create = splits

        for split in splits_to_create:
            split_mask = adata.obs["split"] == split
            split_adata = adata[split_mask].copy()

            logger.info(
                f"Creating DataLoader for split '{split}' ({len(split_adata)} cells)"
            )

            dataset = CellFMDataset(
                adata=split_adata,
                pad_length=pad_length,
                mask_ratio=mask_ratio,
                random_mask=random_mask,
                add_zero=add_zero,
                pad_zero=pad_zero,
                target_count=target_count,
                max_genes=max_genes,
                sample_weighted=sample_weighted,
            )

            should_shuffle = shuffle and (split == "train")

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=should_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
            dataloaders[split] = dataloader

        return dataloaders

    @staticmethod
    def prepare_data(
        data: Union[AnnData, PerturbationData],
        gene_list: Optional[List[str]] = None,
        min_genes: int = 200,
        max_genes: int = 2048,
        min_cells: int = 1,
        target_count: float = 1e4,
        filter_cells: bool = True,
        filter_genes: bool = True,
        inplace: bool = False,
    ) -> Union[AnnData, PerturbationData]:
        """
        Prepares data for CellFM (filtering, normalization).

        Args:
            data: Input data.
            gene_list: Target gene vocabulary.
            min_genes: Min genes per cell.
            max_genes: Max genes per cell.
            min_cells: Min cells per gene.
            target_count: Normalization target.
            filter_cells: Filter cells by gene count.
            filter_genes: Filter genes by cell count.
            inplace: Modify data in-place.

        Returns:
            Processed data object.
        """
        is_perturbation_data = isinstance(data, PerturbationData)

        if is_perturbation_data:
            adata = data.adata if inplace else data.adata.copy()
        else:
            adata = data if inplace else data.copy()

        processed_adata = CellFMDataProcessor.prepare_data(
            adata=adata,
            gene_list=gene_list,
            min_genes=min_genes,
            max_genes=max_genes,
            min_cells=min_cells,
            target_count=target_count,
            filter_cells=filter_cells,
            filter_genes=filter_genes,
            inplace=True,
        )

        if is_perturbation_data:
            if inplace:
                data.adata = processed_adata
                return data
            else:
                new_data = PerturbationData(processed_adata)
                if "split" in data.adata.obs.columns:
                    new_data.adata.obs["split"] = data.adata.obs["split"]
                return new_data
        else:
            return processed_adata

    def predict_embeddings(
        self,
        data: Union[AnnData, PerturbationData],
        batch_size: int = 16,
        return_cls_token: bool = True,
        return_gene_embeddings: bool = False,
        preprocess: bool = True,
        split: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Generates cell embeddings from expression data.

        Args:
            data: Input data.
            batch_size: Batch size.
            return_cls_token: Return CLS token (cell) embeddings.
            return_gene_embeddings: Return gene-level embeddings.
            preprocess: Run preprocessing pipeline.
            split: Specific split to predict on, or None for all splits.
            show_progress: Whether to show progress bar.
            **kwargs: Args for preprocessing.

        Returns:
            Dictionary with 'cell' key containing cell embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        if not self.model_loaded:
            logger.warning("⚠️ Model weights not loaded. Using random initialization.")

        adata = self._extract_adata(data)
        logger.info(f"Generating embeddings for {adata.n_obs} cells...")

        if preprocess:
            logger.info("Preprocessing data...")
            processed_data = self.prepare_data(data, inplace=False, **kwargs)
            adata = self._extract_adata(processed_data)

        dataloaders = self.get_dataloader(
            adata,
            batch_size=batch_size,
            shuffle=False,
            return_split=split,
            **kwargs,
        )

        # Handle split logic
        if split is not None:
            return self._predict_embeddings_single(
                dataloaders[split], return_cls_token, return_gene_embeddings,
                show_progress=show_progress
            )
        else:
            # Process all splits
            result = {}
            for split_name, dataloader in dataloaders.items():
                desc = f"Split: {split_name}" if show_progress else None
                if desc:
                    logger.info(desc)
                emb_result = self._predict_embeddings_single(
                    dataloader, return_cls_token, return_gene_embeddings,
                    show_progress=show_progress, desc=desc
                )
                result[split_name] = emb_result
            return result
    
    def _predict_embeddings_single(
        self,
        dataloader: DataLoader,
        return_cls_token: bool,
        return_gene_embeddings: bool,
        show_progress: bool = True,
        desc: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Generates embeddings for a single dataloader.
        
        Args:
            dataloader: DataLoader to generate embeddings from.
            return_cls_token: Whether to return CLS token embeddings.
            return_gene_embeddings: Whether to return gene embeddings.
            show_progress: Whether to show progress bar.
            desc: Description for progress bar.
        
        Returns:
            Dictionary with embeddings.
        """
        self.eval()
        all_cls_tokens = []
        
        # Memory optimization: pre-allocate if dataset size is known
        try:
            total_samples = len(dataloader.dataset)
            if return_cls_token and total_samples > 0:
                # Get embedding dimension from first batch
                first_batch = next(iter(dataloader))
                with torch.no_grad():
                    output = self.forward(first_batch)
                    emb_dim = output["cls_token"].shape[1]
                    
                # Pre-allocate array
                cell_embeddings = np.zeros((total_samples, emb_dim), dtype=np.float32)
                use_preallocated = True
                current_idx = 0
                
                # Store first batch
                batch_size = output["cls_token"].shape[0]
                cell_embeddings[:batch_size] = output["cls_token"].cpu().numpy()
                current_idx = batch_size
        except (StopIteration, AttributeError):
            # Fallback to list accumulation
            use_preallocated = False
            logger.debug("Using list accumulation for embeddings")

        # Configure progress bar
        pbar_kwargs = {"desc": desc or "Generating embeddings", "disable": not show_progress}
        
        with torch.no_grad():
            # Start from second batch if we used first batch for pre-allocation
            data_iter = iter(dataloader)
            if use_preallocated:
                next(data_iter)  # Skip first batch
            
            for batch in tqdm(data_iter, **pbar_kwargs):
                output = self.forward(batch)

                if return_cls_token:
                    cls_token = output["cls_token"].cpu().numpy()
                    if use_preallocated:
                        batch_size = cls_token.shape[0]
                        cell_embeddings[current_idx:current_idx + batch_size] = cls_token
                        current_idx += batch_size
                    else:
                        all_cls_tokens.append(cls_token)

        result = {}
        if return_cls_token:
            if use_preallocated:
                result["cell"] = cell_embeddings
            elif all_cls_tokens:
                result["cell"] = np.concatenate(all_cls_tokens, axis=0)
            
            if "cell" in result:
                logger.info(f"✓ Generated cell embeddings: {result['cell'].shape}")

        if return_gene_embeddings:
            logger.warning(
                "Gene embedding extraction not implemented. Returning CLS tokens only."
            )

        return result

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the CellFM model.

        Args:
            batch: Dictionary from DataLoader.

        Returns:
            Dictionary with 'mask_loss', 'cls_token', and optional 'logits' keys.
        """
        raw_nzdata = batch["raw_nzdata"].to(self.device)
        dw_nzdata = batch["dw_nzdata"].to(self.device)
        ST_feat = batch["ST_feat"].to(self.device)
        nonz_gene = batch["nonz_gene"].to(self.device)
        mask_gene = batch["mask_gene"].to(self.device)
        zero_idx = batch["zero_idx"].to(self.device)

        mask_loss, cls_token = self.model(
            raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx
        )

        output = {
            "mask_loss": mask_loss,
            "cls_token": cls_token,
        }

        if self.for_finetuning and self.cls_head is not None:
            logits = self.cls_head(cls_token)
            output["logits"] = logits

        return output

    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()
        return self

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes reconstruction and classification losses.

        Args:
            batch: Input batch.
            labels: Optional classification labels.
            output_dict: Optional pre-computed output.

        Returns:
            Dictionary with 'loss' key containing the total loss.
        """
        if output_dict is None:
            output_dict = self.forward(batch)

        loss_dict = {}
        mask_loss = output_dict["mask_loss"]
        loss_dict["mask_loss"] = mask_loss
        loss_dict["loss"] = mask_loss

        if self.for_finetuning and self.cls_head is not None:
            if labels is None:
                raise ValueError("Labels must be provided for fine-tuning.")

            logits = output_dict["logits"]
            labels = labels.to(self.device)

            cls_loss = nn.functional.cross_entropy(logits, labels)
            loss_dict["cls_loss"] = cls_loss
            loss_dict["loss"] = mask_loss + cls_loss

        return loss_dict

    def train_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        save_dir: Optional[str] = None,
        save_every: int = 1,
        log_every: int = 100,
        device: Optional[str] = None,
        use_amp: bool = True,
        **kwargs,
    ):
        """Trains the CellFM model.

        Args:
            train_dataloader: Training DataLoader.
            val_dataloader: Validation DataLoader.
            num_epochs: Number of epochs.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            warmup_steps: Learning rate warmup steps.
            max_grad_norm: Gradient clipping norm.
            save_dir: Directory to save checkpoints.
            save_every: Checkpoint saving frequency.
            log_every: Logging frequency.
            device: Device override.
            use_amp: Whether to use automatic mixed precision.
        """
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
            if self.cls_head is not None:
                self.cls_head = self.cls_head.to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        total_steps = len(train_dataloader) * num_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = (
            torch.cuda.amp.GradScaler()
            if use_amp and self.device == "cuda"
            else None
        )

        logger.info(f"Starting training for {num_epochs} epochs ({total_steps} steps)")

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            epoch_mask_loss = 0.0
            epoch_cls_loss = 0.0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(pbar):
                optimizer.zero_grad()

                # Forward & Loss
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.compute_loss(batch)
                        loss = loss_dict["loss"]

                    scaler.scale(loss).backward()

                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = self.compute_loss(batch)
                    loss = loss_dict["loss"]
                    loss.backward()

                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    optimizer.step()

                scheduler.step()

                # Metrics
                epoch_loss += loss.item()
                epoch_mask_loss += loss_dict["mask_loss"].item()
                if "cls_loss" in loss_dict:
                    epoch_cls_loss += loss_dict["cls_loss"].item()

                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = epoch_loss / (step + 1)
                    avg_mask = epoch_mask_loss / (step + 1)
                    msg = f"Loss: {avg_loss:.4f}, Mask: {avg_mask:.4f}"
                    if self.for_finetuning:
                        msg += f", Cls: {epoch_cls_loss / (step + 1):.4f}"
                    pbar.set_postfix_str(msg)

            # Epoch Summary
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} Train Loss: {avg_epoch_loss:.4f}")

            # Validation
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                logger.info(f"Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir is not None:
                        best_path = os.path.join(save_dir, "best_model")
                        self.save(best_path)
                        logger.info(f"✓ Saved best model to {best_path}")

            # Checkpoint
            if save_dir is not None and (epoch + 1) % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                self.save(ckpt_path)
                logger.info(f"✓ Saved checkpoint to {ckpt_path}")

        logger.info("✓ Training complete!")

    def _validate(self, val_dataloader: DataLoader) -> float:
        """Runs validation loop."""
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict["loss"].item()
        return total_loss / len(val_dataloader)

    def save(self, path: str):
        """Saves model weights and config."""
        os.makedirs(path, exist_ok=True)

        config_path = os.path.join(path, "config.json")
        self.config.save(config_path)
        logger.info(f"✓ Saved config to {config_path}")

        model_path = os.path.join(path, "model.pt")
        state_dict = {"base_model": self.model.state_dict()}
        if self.cls_head is not None:
            state_dict["cls_head"] = self.cls_head.state_dict()

        torch.save(state_dict, model_path)
        logger.info(f"✓ Saved model weights to {model_path}")

    @classmethod
    def load(
        cls,
        model_path: str,
        config: Optional[CellFMConfig] = None,
        device: str = "cuda",
        load_weights: bool = True,
        **kwargs,
    ) -> "CellFMModel":
        """
        Loads CellFM model from a local directory.

        Args:
            model_path: Path to model directory.
            config: Optional config override.
            device: Computation device ('cuda' or 'cpu').
            load_weights: Whether to load pretrained weights.
            **kwargs: Extra arguments passed to model initialization.

        Returns:
            Loaded CellFMModel instance.
        """
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            loaded_config = CellFMConfig.load(config_path)
            if config is not None:
                # Merge user-provided config overrides
                for key, value in config.__dict__.items():
                    if not key.startswith("_") and value is not None:
                        setattr(loaded_config, key, value)
            config = loaded_config
            logger.info(f"✓ Loaded config from {config_path}")
        elif config is None:
            logger.warning("Config file not found, using default configuration")
            config = CellFMConfig(**kwargs)

        # Initialize model
        model = cls(
            config=config,
            device=device,
            for_finetuning=kwargs.get("for_finetuning", False),
        )

        # Load pretrained weights
        if load_weights:
            weights_path = os.path.join(model_path, "model.pt")
            if not os.path.exists(weights_path):
                weights_path = os.path.join(model_path, "model.pth")

            if os.path.exists(weights_path):
                model.load_weights(weights_path, strict=False)
                logger.info(f"✓ Loaded pretrained weights from {weights_path}")
            else:
                logger.warning(
                    f"No weights file found at {model_path}. "
                    "Model initialized with random weights."
                )

        return model


class CellFMPerturbationModel(CellFMModel):
    """
    CellFM with GEARS perturbation head.

    

    Integrates CellFM as the encoder and GEARS as the perturbation prediction head.
    Architecture: Input -> CellFM Encoder -> Gene Embeddings -> GEARS Head -> Prediction.
    """

    def __init__(
        self,
        config: CellFMConfig,
        n_genes: Optional[int] = None,
        device: str = "cuda",
        gears_config: Optional[GearsConfig] = None,
        pert_list: Optional[List[str]] = None,
        go_graph: Optional[GeneGraph] = None,
        gene_graph: Optional[GeneGraph] = None,
        pert_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Initializes CellFM with optional GEARS perturbation head.

        Args:
            config: CellFM configuration.
            n_genes: Number of genes.
            device: Device to run on.
            gears_config: GEARS config for immediate initialization.
            pert_list: List of perturbation names.
            go_graph: GO similarity graph.
            gene_graph: Co-expression graph.
            pert_embeddings: Pre-trained perturbation embeddings.
            gene_embeddings: Pre-trained gene embeddings.
        """
        super().__init__(config=config, n_genes=n_genes, device=device, **kwargs)

        self.gears_model = None
        self.gears_config = None
        self.pert_names = None
        self.go_graph = go_graph
        self.gene_graph = gene_graph

        if all([gears_config, pert_list, go_graph, gene_graph]):
            logger.info("Auto-initializing perturbation head...")
            # Placeholder gene list matching n_genes
            gene_list = [f"gene_{i}" for i in range(self.n_genes)]
            self.init_perturbation_head(
                gears_config=gears_config,
                gene_list=gene_list,
                pert_list=pert_list,
                go_graph=go_graph,
                co_graph=gene_graph,
                pert_embeddings=pert_embeddings,
                gene_embeddings=gene_embeddings,
            )

    def init_perturbation_head(
        self,
        gears_config: GearsConfig,
        gene_list: List[str],
        pert_list: List[str],
        go_graph: GeneGraph,
        co_graph: GeneGraph,
        pert_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Initializes the GEARS perturbation head.

        Args:
            gears_config: GEARS configuration.
            gene_list: List of gene names.
            pert_list: List of perturbation names.
            go_graph: GO similarity graph.
            co_graph: Co-expression graph.
            pert_embeddings: Optional perturbation embeddings.
            gene_embeddings: Optional gene embeddings.
        """
        logger.info("Initializing GEARS perturbation head...")
        
        # Critical: Check dimension compatibility between CellFM and GEARS
        if gears_config.hidden_size != self.config.enc_dims:
            logger.warning(
                f"⚠️ Dimension Mismatch Detected: "
                f"GEARS hidden_size={gears_config.hidden_size}, "
                f"CellFM enc_dims={self.config.enc_dims}. "
                f"Overriding GEARS hidden_size to match CellFM."
            )
            gears_config.hidden_size = self.config.enc_dims

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

        # Inject CellFM encoder into GEARS
        if hasattr(self.gears_model.gears_model, "singlecell_model"):
            self.gears_model.gears_model.singlecell_model = self.model
            self.gears_model.gears_model.pretrained = True
            logger.info("✓ Injected CellFM encoder into GEARS model")
        else:
            logger.warning(
                "⚠️ GEARS model lacks 'singlecell_model' attribute. Injection failed."
            )

        self.gears_config = gears_config
        self.pert_names = pert_list
        self.pert_name_to_id = {p: i for i, p in enumerate(pert_list)}

        logger.info(
            f"✓ Perturbation head initialized: {len(gene_list)} genes, "
            f"{len(pert_list)} perturbations"
        )

    def init_perturbation_head_from_dataset(
        self,
        dataset: PerturbationData,
        gears_config: Optional[GearsConfig] = None,
        **kwargs,
    ):
        """
        Initializes GEARS head using metadata from a dataset.

        Args:
            dataset: PerturbationData object.
            gears_config: Optional GEARS configuration.
            **kwargs: Config overrides.
        """
        if not getattr(dataset, "gears_format", False):
            if hasattr(dataset, "set_gears_format"):
                dataset.set_gears_format(fallback_cell_type="unknown")

        required = ["condition", "split"]
        if not all(col in dataset.adata.obs for col in required):
            raise ValueError(
                f"Dataset missing required fields: {required}. "
                "Ensure dataset is in GEARS format with split info."
            )

        adata = dataset.adata
        pert_conditions = [
            c for c in adata.obs["condition"].unique() if c != "ctrl"
        ]
        unique_perts = set()
        for p in pert_conditions:
            unique_perts.update([g for g in p.split("+") if g != "ctrl"])
        pert_list = sorted(list(unique_perts))

        if not pert_list:
            raise ValueError("No perturbations found in dataset.")

        logger.info(f"Found {len(pert_list)} unique perturbations")

        gene_list = adata.var_names.tolist()

        # Build graphs if missing
        if self.go_graph is None:
            logger.info("Building GO similarity graph...")
            self.go_graph = GeneGraph.from_go(
                gene_list=pert_list, cache_dir=".gears_temp/go_graph"
            )

        if self.gene_graph is None:
            logger.info("Building co-expression graph...")
            self.gene_graph = GeneGraph.from_coexpression(
                adata, gene_list=gene_list
            )

        # Setup GEARS config
        if gears_config is None:
            gears_config = GearsConfig(**kwargs)
        else:
            if kwargs:
                conf_dict = gears_config.to_dict()
                conf_dict.update(kwargs)
                gears_config = GearsConfig(**conf_dict)

        logger.info("Aligning graphs to model vocabulary...")
        aligned_go = self.go_graph.subset(pert_list)
        aligned_gene = self.gene_graph.subset(gene_list)

        self.init_perturbation_head(
            gears_config=gears_config,
            gene_list=gene_list,
            pert_list=pert_list,
            go_graph=aligned_go,
            co_graph=aligned_gene,
        )

        if "rank_genes_groups_cov_all" in adata.uns:
            self._extract_de_genes(adata)

    @staticmethod
    def _requires_perturbation_head(func):
        """Decorator to ensure perturbation head is initialized."""

        def wrapper(self, *args, **kwargs):
            if self.gears_model is None:
                raise RuntimeError(
                    "Perturbation head not initialized. "
                    "Call init_perturbation_head() first."
                )
            return func(self, *args, **kwargs)

        return wrapper

    def _extract_de_genes(self, adata: AnnData):
        """Extracts DE gene indices from AnnData for metric calculation.
        
        Note: If a condition has fewer than top_n DE genes, the list is padded
        with None values. Evaluation code should filter out None values.
        """
        rank_data = adata.uns["rank_genes_groups_cov_all"]
        top_n = adata.uns.get("top_de_n", 20)
        gene_name_to_idx = {name: i for i, name in enumerate(adata.var_names)}

        self.de_gene_map = {}
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

            de_indices = [
                gene_name_to_idx[g] for g in top_genes if g in gene_name_to_idx
            ]
            
            # Pad with None instead of -1 to avoid indexing issues
            # Evaluation code should filter out None values before computation
            if len(de_indices) < top_n:
                de_indices.extend([None] * (top_n - len(de_indices)))
                logger.debug(
                    f"Condition {cond}: Found {len([i for i in de_indices if i is not None])}/{top_n} DE genes"
                )
            
            self.de_gene_map[cond] = de_indices[:top_n]

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
        """
        Trains the perturbation model by delegating to GearsModel.
        """
        logger.info("Training CellFM perturbation model...")
        return self.gears_model.train_model(
            dataset=dataset,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            result_dir=result_dir,
            log_interval=log_interval,
            save_best=save_best,
            **kwargs,
        )

    @_requires_perturbation_head
    def predict_perturbation(
        self,
        dataset: PerturbationData,
        split: str = "test",
        batch_size: int = 32,
        return_attention: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts perturbation effects.
        """
        logger.info(f"Predicting perturbations on '{split}' split...")
        return self.gears_model.predict_perturbation(
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            return_attention=return_attention,
            **kwargs,
        )

    @_requires_perturbation_head
    def evaluate(
        self,
        dataset: PerturbationData,
        split: str = "test",
        batch_size: int = 32,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluates prediction performance.
        """
        logger.info(f"Evaluating on '{split}' split...")
        return self.gears_model.evaluate(
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            **kwargs,
        )

    def save(self, save_directory: str):
        """Saves CellFM base model and GEARS head."""
        os.makedirs(save_directory, exist_ok=True)

        super().save(save_directory)

        if self.gears_model is not None:
            gears_dir = os.path.join(save_directory, "gears_head")
            self.gears_model.save(gears_dir)
            logger.info(f"✓ Saved GEARS head to {gears_dir}")

        if self.go_graph is not None:
            go_path = os.path.join(save_directory, "go_graph.pkl")
            self.go_graph.save(go_path)

        if self.gene_graph is not None:
            gene_path = os.path.join(save_directory, "gene_graph.pkl")
            self.gene_graph.save(gene_path)

        logger.info(f"✓ Model saved to {save_directory}")

    @classmethod
    def load(
        cls,
        model_path: str,
        config: Optional[CellFMConfig] = None,
        device: str = "cuda",
        load_gears_head: bool = True,
        **kwargs,
    ) -> "CellFMPerturbationModel":
        """
        Loads CellFM perturbation model from a saved directory.

        Args:
            model_path: Path to the saved model directory.
            config: Optional config override.
            device: Computation device.
            load_gears_head: Whether to load GEARS perturbation head if available.
            **kwargs: Extra arguments.

        Returns:
            Loaded CellFMPerturbationModel instance.
        """
        # Step 1: Load base CellFM model
        base_model = CellFMModel.load(
            model_path=model_path,
            config=config,
            device=device,
            **kwargs,
        )

        # Step 2: Create perturbation model instance
        model = cls(
            config=base_model.config,
            n_genes=base_model.n_genes,
            device=device,
        )
        
        # Transfer base model weights
        model.model = base_model.model
        model.model_loaded = base_model.model_loaded

        # Step 3: Load GEARS head if requested and available
        if load_gears_head:
            gears_dir = os.path.join(model_path, "gears_head")
            if os.path.exists(gears_dir):
                try:
                    model.gears_model = GearsModel.load(gears_dir)
                    logger.info(f"✓ Loaded GEARS head from {gears_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load GEARS head: {e}")
            else:
                logger.debug("No GEARS head found (this is normal for base models)")

        # Step 4: Load graphs if available
        go_path = os.path.join(model_path, "go_graph.pkl")
        if os.path.exists(go_path):
            try:
                model.go_graph = GeneGraph.load(go_path)
                logger.info("✓ Loaded GO graph")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load GO graph: {e}")

        gene_path = os.path.join(model_path, "gene_graph.pkl")
        if os.path.exists(gene_path):
            try:
                model.gene_graph = GeneGraph.load(gene_path)
                logger.info("✓ Loaded gene co-expression graph")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load gene graph: {e}")

        logger.info("✓ CellFM perturbation model loaded successfully")
        return model
