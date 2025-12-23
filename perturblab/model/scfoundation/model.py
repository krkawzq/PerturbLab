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
    """
    scFoundation model wrapper for single-cell foundation embeddings.

    scFoundation is a large-scale foundation model trained on diverse single-cell datasets.
    It supports:
    - Cell embedding extraction (with various pooling strategies).
    - Gene embedding extraction.
    - Inference on both single-cell and bulk RNA-seq data.
    """

    def __init__(
        self,
        config: scFoundationConfig,
        device: str = "cuda",
        gene_list: Optional[List[str]] = None,
    ):
        """
        Initializes the scFoundation model.

        Args:
            config: Model configuration object.
            device: Computation device ('cuda' or 'cpu').
            gene_list: List of gene names for vocabulary alignment. If None, loads from default index.
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
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the scFoundation model.

        Args:
            x: Input gene expression data [B, N].
            padding_label: Padding mask for encoder [B, N].
            encoder_position_gene_ids: Position IDs for encoder [B, N].
            encoder_labels: Labels for encoder [B, N].
            decoder_data: Input data for decoder [B, N].
            decoder_position_gene_ids: Position IDs for decoder [B, N].
            decoder_data_padding_labels: Padding mask for decoder [B, N].
            mask_gene_name: Whether to mask gene names.
            mask_labels: Mask labels if applicable.
            output_attentions: Whether to output attention maps.

        Returns:
            Model output tensor.
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
            **kwargs,
        )

    def _align_genes(self, adata: AnnData) -> pd.DataFrame:
        """
        Aligns genes in adata to the model's vocabulary.

        Args:
            adata: Input AnnData object.

        Returns:
            DataFrame containing gene-aligned expression matrix.
        """
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

    def predict_embedding(
        self,
        dataset: PerturbationData,
        output_type: Literal["cell", "gene", "gene_batch"] = "cell",
        input_type: Literal["singlecell", "bulk"] = "singlecell",
        pool_type: Literal["all", "max"] = "all",
        tgthighres: str = "t4",
        pre_normalized: Literal["F", "T", "A"] = "F",
        batch_size: int = 32,
        return_adata: bool = False,
        use_batch: bool = True,
    ) -> Union[np.ndarray, AnnData]:
        """
        Generates embeddings from input data.

        Args:
            dataset: Input data using the unified PerturbationData interface.
            output_type: Type of embedding ('cell', 'gene', or 'gene_batch').
            input_type: Data source type ('singlecell' or 'bulk').
            pool_type: Pooling strategy for cell embeddings ('all' or 'max').
            tgthighres: Target high-resolution parameter (e.g., 't4', 'a5', 'f2').
            pre_normalized: Normalization status ('F': None, 'T': Norm+Log1p, 'A': Norm+Log1p+Total).
            batch_size: Processing batch size.
            return_adata: If True, returns AnnData with embeddings in obsm/varm.
            use_batch: Whether to process in batches.

        Returns:
            Numpy array of embeddings or AnnData object.
        """
        # Lossless access to underlying AnnData
        adata = dataset.adata

        # Gene alignment
        gexpr_feature = self._align_genes(adata)

        # Normalization for bulk data if needed
        if pre_normalized == "F" and input_type == "bulk":
            tmp_adata = sc.AnnData(gexpr_feature.values)
            sc.pp.normalize_total(tmp_adata)
            sc.pp.log1p(tmp_adata)
            gexpr_feature = pd.DataFrame(
                tmp_adata.X, index=gexpr_feature.index, columns=gexpr_feature.columns
            )

        embeddings = []

        # Disable batching for individual gene embedding processing
        if not use_batch or output_type == "gene":
            batch_size = 1

        n_samples = gexpr_feature.shape[0]

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, n_samples, batch_size), desc="Generating embeddings"
            ):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_data = gexpr_feature.iloc[start_idx:end_idx]

                batch_tensors = []

                for i in range(len(batch_data)):
                    # Prepare input tensor based on input_type
                    if input_type == "bulk":
                        if pre_normalized == "T":
                            totalcount = batch_data.iloc[i, :].sum()
                        elif pre_normalized == "F":
                            totalcount = np.log10(batch_data.iloc[i, :].sum())
                        else:
                            raise ValueError(
                                "For bulk data, pre_normalized must be T or F"
                            )

                        tmpdata = batch_data.iloc[i, :].tolist()
                        gene_x = torch.tensor(
                            tmpdata + [totalcount, totalcount], device=self.device
                        )
                        batch_tensors.append(gene_x)

                    elif input_type == "singlecell":
                        # Pre-normalization logic
                        if pre_normalized == "F":
                            # Normalize to 10k and log1p
                            tmpdata = np.log1p(
                                batch_data.iloc[i, :]
                                / batch_data.iloc[i, :].sum()
                                * 1e4
                            ).tolist()
                        elif pre_normalized == "T":
                            tmpdata = batch_data.iloc[i, :].tolist()
                        elif pre_normalized == "A":
                            # 'A' implies total count is appended at the end
                            tmpdata = batch_data.iloc[i, :-1].tolist()
                        else:
                            raise ValueError("pre_normalized must be T, F, or A")

                        # Calculate total count
                        if pre_normalized == "A":
                            totalcount = batch_data.iloc[i, -1]
                        else:
                            totalcount = batch_data.iloc[i, :].sum()

                        # Parse target resolution parameter
                        if tgthighres[0] == "f":
                            high_res = np.log10(totalcount * float(tgthighres[1:]))
                        elif tgthighres[0] == "a":
                            high_res = np.log10(totalcount) + float(tgthighres[1:])
                        elif tgthighres[0] == "t":
                            high_res = float(tgthighres[1:])
                        else:
                            raise ValueError("tgthighres must start with f, a, or t")

                        gene_x = torch.tensor(
                            tmpdata + [high_res, np.log10(totalcount)],
                            device=self.device,
                        )
                        batch_tensors.append(gene_x)
                    else:
                        raise ValueError("input_type must be 'singlecell' or 'bulk'")

                # Stack tensors
                if batch_tensors:
                    batch_gene_x = torch.stack(batch_tensors, dim=0)

                    # Generate embeddings
                    if output_type == "cell":
                        cell_emb = self._get_cell_embedding(batch_gene_x, pool_type)
                        embeddings.append(cell_emb.detach().cpu().numpy())

                    elif output_type == "gene":
                        for j in range(batch_gene_x.shape[0]):
                            gene_emb = self._get_gene_embedding(
                                batch_gene_x[j : j + 1], single=True
                            )
                            embeddings.append(gene_emb.detach().cpu().numpy())

                    elif output_type == "gene_batch":
                        gene_emb = self._get_gene_embedding(batch_gene_x, single=False)
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

        # Return as AnnData if requested
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

        return embeddings

    def _get_cell_embedding(
        self, gene_x: torch.Tensor, pool_type: str
    ) -> torch.Tensor:
        """Helper to compute cell embeddings from expression tensors."""
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
        """Helper to compute gene embeddings."""
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

        # Extract gene embeddings corresponding to known tokens
        gene_emb = out[:, : self.config.num_tokens, :].contiguous()

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
        Trains the scFoundation model using masked autoencoding (MAE).

        Note: Basic implementation. For advanced distributed training,
        refer to the original scFoundation repository.
        """
        if device is None:
            device = self.device

        os.makedirs(output_dir, exist_ok=True)

        adata = dataset.adata
        if "split" in adata.obs:
            train_adata = adata[adata.obs["split"] == "train"]
        else:
            train_adata = adata

        # Align genes
        gexpr_feature = self._align_genes(train_adata)
        n_samples = gexpr_feature.shape[0]
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size
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
            indices = np.random.permutation(n_samples)

            progress_bar = tqdm(
                range(0, n_samples, batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
            )

            for start_idx in progress_bar:
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = gexpr_feature.iloc[batch_indices]

                # Prepare batch
                batch_gene_x = []
                batch_gene_x_raw = []
                for i in range(len(batch_data)):
                    tmpdata = batch_data.iloc[i, :].values
                    totalcount = tmpdata.sum()

                    normalized = np.log1p(tmpdata / totalcount * 1e4)
                    totalcount_log = (
                        np.log10(totalcount) if totalcount > 0 else 0.0
                    )
                    high_res_token = 4.0  # Default high-res token

                    gene_x = torch.tensor(
                        normalized.tolist() + [high_res_token, totalcount_log],
                        device=device,
                        dtype=torch.float32,
                    )
                    batch_gene_x.append(gene_x)
                    batch_gene_x_raw.append(gene_x.clone())

                batch_gene_x = torch.stack(batch_gene_x, dim=0)
                batch_gene_x_raw = torch.stack(batch_gene_x_raw, dim=0)

                # Prepare encoder/decoder inputs
                model_config = self.config.to_model_config_dict()
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
                ) = getEncoerDecoderData(
                    batch_gene_x, batch_gene_x_raw, model_config
                )

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
                    mask_labels=None,
                )

                # Compute loss (MSE)
                if data_mask_labels is not None:
                    loss = criterion(
                        output[data_mask_labels], new_data_raw[data_mask_labels]
                    )
                else:
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

            if (epoch + 1) % eval_interval == 0 and "split" in adata.obs:
                if "val" in adata.obs["split"].values:
                    self.eval()
                    val_loss = self._evaluate(dataset, batch_size, device)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    self.train()

        final_dir = os.path.join(output_dir, "final_model")
        self.save(final_dir)
        logger.info(f"Training complete. Model saved to {final_dir}")

    def _evaluate(
        self, dataset: PerturbationData, batch_size: int, device: str
    ) -> float:
        """Evaluates model on validation set."""
        adata = dataset.adata
        val_adata = adata[adata.obs["split"] == "val"]

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

                batch_gene_x = []
                for i in range(len(batch_data)):
                    tmpdata = np.log1p(
                        batch_data.iloc[i, :]
                        / batch_data.iloc[i, :].sum()
                        * 1e4
                    ).tolist()
                    totalcount = batch_data.iloc[i, :].sum()
                    gene_x = torch.tensor(
                        tmpdata + [4.0, np.log10(totalcount)], device=device
                    )
                    batch_gene_x.append(gene_x)

                batch_gene_x = torch.stack(batch_gene_x, dim=0)

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
                ) = getEncoerDecoderData(
                    batch_gene_x,
                    batch_gene_x,
                    self.config.to_model_config_dict(),
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
                    mask_labels=None,
                )

                loss = criterion(
                    output[data_mask_labels], new_data_raw[data_mask_labels]
                )
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def save(self, save_directory: str):
        """Saves model weights and configuration."""
        os.makedirs(save_directory, exist_ok=True)

        config_path = os.path.join(save_directory, "config.json")
        self.config.save(config_path)

        model_path = os.path.join(save_directory, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "scFoundationModel":
        """
        Loads a pretrained scFoundation model.

        Args:
            model_name_or_path: Path to local directory, built-in model name, or HuggingFace repo ID.
            device: Device to load the model on.
            **kwargs: Additional args for initialization or HuggingFace download.

        Returns:
            Loaded scFoundationModel instance.
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
            weights_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "weights"
            )
            model_path = os.path.join(weights_dir, model_name_or_path)

            if not os.path.isdir(model_path):
                try:
                    logger.info(
                        f"Downloading '{model_name_or_path}' from HuggingFace..."
                    )
                    model_path = download_from_huggingface(
                        model_name_or_path,
                        organization="perturblab",
                        **hf_kwargs,
                    )
                    logger.info(f"Model cached at: {model_path}")
                except Exception as e:
                    raise ValueError(
                        f"Model not found: {model_name_or_path}. Error: {str(e)}"
                    )

        # Load config and weights
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = scFoundationConfig.load(config_path)
        
        model = cls(config, device=device, **model_init_kwargs)

        model_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        state_dict = torch.load(model_file, map_location=device)
        model.model.load_state_dict(state_dict)

        model.eval()
        return model




class scFoundationPerturbationModel(scFoundationModel):
    """
    scFoundation model for perturbation prediction using the GEARS framework.

    Extends scFoundationModel by using it as a backbone encoder for GEARS.
    Architecture: scFoundation (Encoder) -> GEARS (GNN + Decoder).
    """

    def __init__(
        self,
        config: scFoundationConfig,
        device: str = "cuda",
        gene_list: Optional[List[str]] = None,
        go_graph: Optional[GeneGraph] = None,
        gene_graph: Optional[GeneGraph] = None,
        gears_config: Optional[GearsConfig] = None,
        pert_list: Optional[List[str]] = None,
        pert_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Initializes the scFoundation Perturbation Model.

        If graphs and lists are provided, the GEARS head is initialized immediately.
        Otherwise, `init_perturbation_head` must be called later.

        Args:
            config: scFoundation configuration.
            device: Computation device.
            gene_list: List of genes.
            go_graph: Pre-built GO graph.
            gene_graph: Pre-built co-expression graph.
            gears_config: Configuration for the GEARS head.
            pert_list: List of perturbations.
            pert_embeddings: Pre-trained perturbation embeddings.
            gene_embeddings: Pre-trained gene embeddings.
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
        """Decorator to enforce initialization of the GEARS head."""

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
        """
        Initializes the GEARS downstream head with provided graphs.

        Args:
            gears_config: Configuration for GEARS.
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
        """
        Initializes the GEARS head by extracting metadata from a dataset.

        Args:
            dataset: PerturbationData object.
            gears_config: GEARS configuration. Defaults to model config if None.
            **kwargs: Overrides for GearsConfig if gears_config is None.
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
        """Helper to extract DE gene indices from AnnData."""
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
        """
        Trains the perturbation model.

        Delegates to GearsModel.train_model.
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
        """
        Predicts gene expression changes under perturbation.

        Args:
            dataset: Input dataset.
            batch_size: Batch size.
            split: Data split to predict on ('train', 'val', 'test', 'all').
            return_numpy: Whether to return numpy arrays.

        Returns:
            Dictionary containing predictions, ground truth, and metadata.
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

    def save_pretrained(self, save_directory: str):
        """Saves the base model and the GEARS head (if initialized)."""
        os.makedirs(save_directory, exist_ok=True)

        # Save base model
        super().save(save_directory)

        # Save GEARS head
        if self.gears_model is not None:
            gears_save_dir = os.path.join(save_directory, "gears_head")
            self.gears_model.save(gears_save_dir)
            logger.info(f"Saved GEARS head to {gears_save_dir}")
        else:
            logger.info("Saved base model only (no head initialized).")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "scFoundationPerturbationModel":
        """
        Loads the model. Automatically loads GEARS head if found in subdirectory.
        """
        # Load base model
        base_model = super().from_pretrained(model_name_or_path, device, **kwargs)

        # Create instance sharing weights
        model = cls(base_model.config, device=device, gene_list=base_model.gene_list)
        model.model = base_model.model

        # Check for GEARS head
        gears_head_dir = None
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            gears_head_dir = os.path.join(model_name_or_path, "gears_head")

        if gears_head_dir and os.path.exists(gears_head_dir):
            logger.info(f"Found GEARS head at {gears_head_dir}. Loading...")
            try:
                model.gears_model = GearsModel.from_pretrained(
                    gears_head_dir, device=device
                )
                
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
