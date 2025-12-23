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
    """
    scGPT model for single-cell gene expression analysis.

    

    Supports pretraining, fine-tuning, and various downstream tasks using a
    Transformer-based architecture specialized for single-cell data.

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

    def __init__(
        self,
        config: scGPTConfig,
        gene_list: Optional[List[str]] = None,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initializes the scGPT model.

        Args:
            config: Model configuration.
            gene_list: List of gene names (required if not using default vocab).
            device: Computation device ('cuda' or 'cpu').
        """
        super().__init__(config)

        if device == "cuda":
            self.device = (
                "cuda"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
        else:
            self.device = "cpu"

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
        ).to(self.device)

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        return self.train(False)

    @classmethod
    def mask_values(
        cls,
        values: Union[torch.Tensor, np.ndarray],
        mask_ratio: float = 0.4,
        mask_value: int = -1,
        pad_value: int = 0,
    ) -> torch.Tensor:
        """
        Performs random masking on values for Masked Language Modeling (MLM).

        
        """
        return random_mask_value(
            values,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

    def prepare_dataloader(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        mask_ratio: float = 0.0,
        mask_value: int = -1,
        return_split: Optional[str] = None,
    ) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Prepares DataLoaders for the dataset.

        Args:
            dataset: Input dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop incomplete batches.
            num_workers: Number of worker threads.
            mask_ratio: Ratio of values to mask (0.0 = no masking).
            mask_value: Value to use for masking.
            return_split: Specific split to return ('train', 'test', 'all'), or None for all.

        Returns:
            DataLoader or Dictionary of DataLoaders.
        """
        gene_names = dataset.adata.var_names.tolist()
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
        if return_split == "all" or ("split" not in dataset.adata.obs):
            adata_map["all"] = dataset.adata
        elif return_split is not None:
            if (
                "split" in dataset.adata.obs
                and return_split in dataset.adata.obs["split"].values
            ):
                subset = dataset.adata[dataset.adata.obs["split"] == return_split]
                adata_map[return_split] = subset
            else:
                raise ValueError(f"Split '{return_split}' not found in dataset")
        else:
            # Return all splits
            if "split" in dataset.adata.obs:
                split_names = dataset.adata.obs["split"].unique()
                for split_name in split_names:
                    subset = dataset.adata[dataset.adata.obs["split"] == split_name]
                    adata_map[str(split_name)] = subset
            else:
                adata_map["all"] = dataset.adata

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
        do_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass. Retrieves embeddings and model outputs without loss calculation.
        """
        batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

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
        """
        Computes model losses (MSE, GEPC, ECS, DAB, Zero-prob).
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
                # Scaling factor 10 is standard in scGPT
                loss_ecs = output_dict["loss_ecs"] * 10
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
        dataset: PerturbationData,
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        **kwargs,
    ) -> np.ndarray:
        """
        Unified embedding prediction method.

        Args:
            dataset: Input dataset.
            batch_size: Batch size.
            embedding_type: 'cell' or 'gene'.

        Returns:
            Numpy array of embeddings.
        """
        self.eval()

        if embedding_type == "gene":
            with torch.no_grad():
                gene_embs = self.model.encoder.embedding.weight.cpu().numpy()
            return gene_embs

        elif embedding_type == "cell":
            loader = self.prepare_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                mask_ratio=0.0,
                return_split="all",
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
            raise ValueError(
                f"embedding_type must be 'cell' or 'gene', got {embedding_type}"
            )

    def train_model(
        self,
        dataset: PerturbationData,
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
        """
        Trains the scGPT model.

        Args:
            dataset: Training data.
            epochs: Number of epochs.
            batch_size: Batch size.
            lr: Learning rate.
            mask_ratio: Masking ratio for MLM.
            mask_value: Value used for masking.
            use_amp: Whether to use Automatic Mixed Precision.
            grad_clip: Gradient clipping value.
            log_interval: Logging frequency.
            save_dir: Directory to save checkpoints.
            save_interval: Checkpoint saving frequency.
            CLS, CCE, MVC, ECS: Loss flags.
            scheduler_step: Scheduler step size.
            scheduler_gamma: Scheduler gamma.

        Returns:
            Dictionary containing training history.
        """
        dataloaders = self.prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            return_split=None,  # Get all splits
        )

        train_loader = dataloaders.get("train", dataloaders.get("all"))
        valid_loader = dataloaders.get("valid", dataloaders.get("test", None))

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

    def save(self, save_directory: str):
        """Saves model weights and configuration."""
        os.makedirs(save_directory, exist_ok=True)

        model_file = os.path.join(save_directory, "model.pt")
        torch.save(self.model.state_dict(), model_file)

        config_file = os.path.join(save_directory, "args.json")
        self.config.save(config_file)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "scGPTModel":
        """
        Loads a pretrained scGPT model from HuggingFace or a local path.

        Args:
            model_name_or_path: HuggingFace ID or local directory.
            device: Computation device.
            **kwargs: Extra arguments for HF download or model init.

        Returns:
            Loaded scGPTModel instance.
        """
        hf_kwargs = {}
        model_init_kwargs = {}
        hf_keys = {
            "revision",
            "token",
            "cache_dir",
            "force_download",
            "resume_download",
        }

        for key, value in kwargs.items():
            if key in hf_keys:
                hf_kwargs[key] = value
            else:
                model_init_kwargs[key] = value

        # Resolve model path
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
            logger.info(f"Loading model from local path: {model_path}")
        else:
            try:
                logger.info(
                    f"Downloading model '{model_name_or_path}' from HuggingFace..."
                )
                model_path = download_from_huggingface(
                    model_name_or_path, organization="perturblab", **hf_kwargs
                )
                logger.info(f"✓ Model cached at: {model_path}")
            except Exception as e:
                raise ValueError(
                    f"Failed to load model '{model_name_or_path}'. "
                    f"Ensure it is a valid local path or HuggingFace ID.\n"
                    f"Error: {str(e)}"
                )

        # Load config
        config_file = os.path.join(model_path, "args.json")
        config = scGPTConfig.load(config_file)

        # Load ntoken from vocab if needed
        if config.ntoken is None:
            vocab_file = os.path.join(model_path, "vocab.json")
            if os.path.exists(vocab_file):
                try:
                    with open(vocab_file, "r") as f:
                        vocab_data = json.load(f)
                        if isinstance(vocab_data, dict):
                            config.ntoken = max(vocab_data.values()) + 1
                        else:
                            config.ntoken = len(vocab_data)
                except Exception as e:
                    logger.warning(f"Failed to load ntoken from vocab.json: {e}")
                    config.ntoken = 60697
            else:
                config.ntoken = 60697

        # Find weights
        model_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_path, "best_model.pt")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Expected 'model.pt' or 'best_model.pt'."
                )

        # Instantiate
        model = cls(config, device=device, **model_init_kwargs)

        # Load weights
        state_dict = torch.load(model_file, map_location=device)
        load_pretrained(model.model, state_dict, verbose=False)

        logger.info("✓ Model loaded successfully")
        return model


class scGPTPerturbationModel(PerturbationModel):
    """
    scGPT model specialized for perturbation prediction tasks.

    

    Uses the TransformerGenerator architecture to predict the gene expression
    state of a cell after perturbation, given a control state and perturbation tokens.
    """

    # Registry of pretrained models for perturbation (placeholder)
    PRETRAINED_MODELS = {}

    def __init__(
        self,
        config: scGPTConfig,
        gene_list: Optional[List[str]] = None,
        device: str = "cuda",
        **kwargs,
    ):
        """Initializes the scGPT perturbation model."""
        super().__init__(config)

        if device == "cuda":
            self.device = (
                "cuda"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
        else:
            self.device = "cpu"

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
        ).to(self.device)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

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
        Prepares DataLoaders for perturbation prediction.
        Requires paired control-perturbation data.
        """
        if "ctrl_indices" not in dataset.adata.obsm:
            logger.info("Pairing cells for perturbation task...")
            dataset.pair_cells()

        gene_names = dataset.adata.var_names.tolist()
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
        if return_split == "all" or ("split" not in dataset.adata.obs):
            adata_map["all"] = np.arange(dataset.adata.n_obs)
        elif return_split is not None:
            if (
                "split" in dataset.adata.obs
                and return_split in dataset.adata.obs["split"].values
            ):
                indices = np.where(dataset.adata.obs["split"] == return_split)[0]
                adata_map[return_split] = indices
            else:
                raise ValueError(f"Split '{return_split}' not found in dataset")
        else:
            if "split" in dataset.adata.obs:
                for split_name in dataset.adata.obs["split"].unique():
                    indices = np.where(dataset.adata.obs["split"] == split_name)[0]
                    adata_map[str(split_name)] = indices
            else:
                adata_map["all"] = np.arange(dataset.adata.n_obs)

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
                    ctrl_indices = dataset.adata.obsm["ctrl_indices"][
                        batch_indices, 0
                    ]

                    # Fetch expressions
                    X = dataset.adata.X
                    if scipy.sparse.issparse(X):
                        x_val = X[ctrl_indices, slice_cols_local].toarray()
                        y_val = X[batch_indices, slice_cols_local].toarray()
                    else:
                        x_val = X[ctrl_indices, slice_cols_local]
                        y_val = X[batch_indices, slice_cols_local]

                    # Identify perturbed genes
                    conditions = dataset.adata.obs.iloc[batch_indices][
                        "condition"
                    ].values
                    pert_flags = np.zeros_like(x_val, dtype=np.int64)

                    for i, cond in enumerate(conditions):
                        if cond != "ctrl":
                            perts = cond.split("+")
                            for p in perts:
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
        do_sample: bool = False,
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
            target_values = batch_data["target_values"].to(self.device)
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
        **kwargs,
    ) -> np.ndarray:
        """
        Unified embedding prediction.
        """
        self.eval()

        if embedding_type == "gene":
            with torch.no_grad():
                gene_embs = self.model.encoder.embedding.weight.cpu().numpy()
            return gene_embs

        elif embedding_type == "cell":
            loader = self.prepare_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                return_split="all",
            )

            embeddings = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Encoding cells"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

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

            return np.concatenate(embeddings, axis=0)
        else:
            raise ValueError(
                f"embedding_type must be 'cell' or 'gene', got {embedding_type}"
            )

    def predict_perturbation(
        self,
        dataset: PerturbationData,
        batch_size: int = 32,
        return_numpy: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Predicts perturbation effects.

        Args:
            dataset: Input data.
            batch_size: Batch size.
            return_numpy: Whether to return numpy array.

        Returns:
            Predicted gene expressions.
        """
        self.eval()
        loader = self.prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            return_split="all",
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
        """
        Trains the scGPT perturbation model.
        """
        dataloaders = self.prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            return_split=None,
        )

        train_loader = dataloaders.get("train", dataloaders.get("all"))
        valid_loader = dataloaders.get("valid", dataloaders.get("test", None))

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

    def save(self, save_directory: str):
        """Saves model weights and configuration."""
        os.makedirs(save_directory, exist_ok=True)

        model_file = os.path.join(save_directory, "model.pt")
        torch.save(self.model.state_dict(), model_file)

        config_file = os.path.join(save_directory, "args.json")
        self.config.save(config_file)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "scGPTPerturbationModel":
        """
        Loads a pretrained scGPT perturbation model.

        Args:
            model_name_or_path: Local path or HF ID.
            device: Computation device.
            **kwargs: Extra arguments.

        Returns:
            Loaded scGPTPerturbationModel instance.
        """
        hf_kwargs = {}
        model_init_kwargs = {}
        hf_keys = {
            "revision",
            "token",
            "cache_dir",
            "force_download",
            "resume_download",
        }

        for key, value in kwargs.items():
            if key in hf_keys:
                hf_kwargs[key] = value
            else:
                model_init_kwargs[key] = value

        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
            logger.info(f"Loading model from local path: {model_path}")
        else:
            try:
                logger.info(
                    f"Downloading model '{model_name_or_path}' from HuggingFace..."
                )
                model_path = download_from_huggingface(
                    model_name_or_path, organization="perturblab", **hf_kwargs
                )
                logger.info(f"✓ Model cached at: {model_path}")
            except Exception as e:
                raise ValueError(
                    f"Failed to load model '{model_name_or_path}'. "
                    f"Ensure path is valid or specify a valid HuggingFace ID.\n"
                    f"Error: {str(e)}"
                )

        config_file = os.path.join(model_path, "args.json")
        config = scGPTConfig.load(config_file)

        if config.ntoken is None:
            vocab_file = os.path.join(model_path, "vocab.json")
            if os.path.exists(vocab_file):
                try:
                    with open(vocab_file, "r") as f:
                        vocab_data = json.load(f)
                        if isinstance(vocab_data, dict):
                            config.ntoken = max(vocab_data.values()) + 1
                        else:
                            config.ntoken = len(vocab_data)
                except Exception as e:
                    logger.warning(f"Failed to load ntoken from vocab.json: {e}")
                    config.ntoken = 60697
            else:
                config.ntoken = 60697

        model_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_path, "best_model.pt")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Expected 'model.pt' or 'best_model.pt'."
                )

        model = cls(config, device=device, **model_init_kwargs)

        state_dict = torch.load(model_file, map_location=device)
        load_pretrained(model.model, state_dict, verbose=False)

        logger.info("✓ Model loaded successfully")
        return model
