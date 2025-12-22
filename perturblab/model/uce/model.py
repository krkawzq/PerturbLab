"""
UCE (Universal Cell Embeddings) Model

UCE is a transformer-based foundation model for generating universal cell embeddings
from single-cell RNA-seq data using protein embeddings as gene tokens.

Original source: https://github.com/snap-stanford/UCE
"""

import json
import logging
import os
import pickle
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import issparse, save_npz
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...data import PerturbationData
from ...utils import download_from_huggingface
from ..base import PerturbationModel
from .config import UCEModelConfig
from .source.data_proc.data_utils import (
    adata_path_to_prot_chrom_starts,
    get_spec_chrom_csv,
    get_species_to_pe,
    process_raw_anndata,
)
from .source.eval_data import MultiDatasetSentenceCollator, MultiDatasetSentences
from .source.model import TransformerModel
from .source.utils import figshare_download

# Constants
ESM2_BASE_TOKEN_COUNT = 143574  # ESM2-650M vocabulary size (without chromosome tokens)
CHROM_TOKEN_COUNT = 1895  # Number of chromosome identifier tokens

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class UCEModel(PerturbationModel):
    """
    UCE (Universal Cell Embeddings) Model.

    

    UCE is a transformer-based foundation model that generates universal cell embeddings
    from single-cell RNA-seq data. It uses protein embeddings (ESM2) as gene tokens
    and incorporates genomic position information.

    Features:
    - Multi-species support (human, mouse, etc.)
    - Uses ESM2 protein embeddings as gene tokens
    - Incorporates chromosome and genomic position information
    - Supports 4-layer and 33-layer model variants
    """

    def __init__(
        self,
        config: UCEModelConfig,
        device: str = "cuda",
    ):
        """
        Initializes the UCE model.

        Args:
            config: UCEModelConfig configuration object.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        super().__init__(config)

        self.config = config
        self.device = device

        # Initialize model architecture
        self.model = TransformerModel(
            token_dim=config.token_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            d_hid=config.d_hid,
            nlayers=config.nlayers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

        # Model will be loaded in from_pretrained or load_weights
        self.model_loaded = False
        self.all_pe = None  # Protein embeddings (tokens)

        logger.info(f"Initialized UCE model (nlayers={config.nlayers})")

    def load_weights(self, model_path: str):
        """
        Loads model weights from a standard model.pt file.

        Args:
            model_path: Path to model.pt file containing state_dict.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        logger.info(f"Loading model weights from {model_path}...")

        state_dict = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.model_loaded = True
        logger.info("✓ Model weights loaded successfully")

    def load_token_embeddings(self, token_file: Optional[str] = None):
        """
        Loads protein/token embeddings from tokens.pt file.

        Args:
            token_file: Path to tokens.pt file. If None, uses config.token_file or downloads.
        """
        if token_file is None:
            token_file = self.config.token_file

        if token_file is None:
            token_file = "./model_files/tokens.pt"

        if not os.path.exists(token_file):
            logger.info(
                f"Token file not found at {token_file}, attempting to download..."
            )
            os.makedirs(
                os.path.dirname(token_file) if os.path.dirname(token_file) else ".",
                exist_ok=True,
            )
            figshare_download(
                "https://figshare.com/ndownloader/files/42706585", token_file
            )

        logger.info(f"Loading token embeddings from {token_file}...")

        all_pe = torch.load(token_file, map_location=self.device)

        # Handle different file formats
        if isinstance(all_pe, dict):
            if "tokens" in all_pe:
                all_pe = all_pe["tokens"]
            elif "embeddings" in all_pe:
                all_pe = all_pe["embeddings"]
            else:
                raise ValueError(
                    "Unknown token file format. Expected tensor or dict with 'tokens'/'embeddings' key."
                )

        # Process tokens (add chromosome tokens if needed)
        if all_pe.shape[0] == ESMA_BASE_TOKEN_COUNT:
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(
                mean=0, std=1, size=(CHROM_TOKEN_COUNT, self.config.token_dim)
            )
            all_pe = torch.vstack((all_pe, CHROM_TENSORS))

        all_pe.requires_grad = False

        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.all_pe = all_pe

        logger.info(f"✓ Loaded token embeddings: shape {self.all_pe.shape}")

    @staticmethod
    def get_dataloader(
        data: Union[AnnData, PerturbationData],
        species: str,
        config: UCEModelConfig,
        batch_size: int = 25,
        working_dir: Optional[str] = None,
        filter_genes: bool = True,
        shuffle: bool = False,
        num_workers: int = 0,
        dataset_name: str = "dataset",
    ) -> DataLoader:
        """
        Creates DataLoader for UCE model from PerturbationData or AnnData.

        This method handles the complete preprocessing pipeline:
        1. Data preprocessing (filtering, normalization).
        2. Gene-to-protein-embedding mapping.
        3. Chromosome and position information extraction.
        4. Sentence construction with sampling.
        5. Token indexing.

        Args:
            data: PerturbationData or AnnData object.
            species: Species name (e.g., 'human', 'mouse').
            config: UCEModelConfig with model parameters.
            batch_size: Batch size for DataLoader.
            working_dir: Working directory for intermediate files.
            filter_genes: Whether to filter genes based on protein embeddings availability.
            shuffle: Whether to shuffle data.
            num_workers: Number of workers for DataLoader.
            dataset_name: Name for this dataset (used as key in dicts).

        Returns:
            DataLoader: PyTorch DataLoader ready for model inference.
        """
        if isinstance(data, PerturbationData):
            adata = data.adata
        else:
            adata = data

        if working_dir is None:
            import tempfile

            working_dir = tempfile.mkdtemp(prefix="uce_dataloader_")
        os.makedirs(working_dir, exist_ok=True)

        logger.info(
            f"Preparing UCE DataLoader for {adata.n_obs} cells (species: {species})..."
        )

        # Step 1: Preprocess AnnData
        logger.info("Step 1/4: Preprocessing AnnData...")
        proc_adata, num_cells, num_genes = process_raw_anndata(
            row=type(
                "Row",
                (),
                {"path": "input.h5ad", "covar_col": np.nan, "species": species},
            )(),
            h5_folder_path=working_dir,
            npz_folder_path=working_dir,
            scp="",
            skip=False,
            filter=filter_genes,
            root="",
            adata_input=adata,
        )

        # Step 2: Generate protein embedding indices and chromosome/position info
        logger.info("Step 2/4: Generating gene mappings...")

        species_to_pe = get_species_to_pe(config.protein_embeddings_dir)

        with open(config.offset_pkl_path, "rb") as f:
            species_to_offsets = pickle.load(f)

        gene_to_chrom_pos = get_spec_chrom_csv(config.spec_chrom_csv_path)

        spec_pe_genes = list(species_to_pe[species].keys())
        offset = species_to_offsets[species]

        pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
            proc_adata, species, spec_pe_genes, gene_to_chrom_pos, offset
        )

        # Step 3: Prepare data structures for DataLoader
        logger.info("Step 3/4: Preparing data structures...")

        pe_idx_dict = {dataset_name: pe_row_idxs}
        chroms_dict = {dataset_name: dataset_chroms}
        starts_dict = {dataset_name: dataset_pos}
        shapes_dict = {dataset_name: (num_cells, num_genes)}

        counts_path = os.path.join(working_dir, f"{dataset_name}_counts.npz")
        if issparse(proc_adata.X):
            save_npz(counts_path, proc_adata.X)
        else:
            np.savez_compressed(counts_path, data=proc_adata.X)

        # Step 4: Create DataLoader
        logger.info("Step 4/4: Creating DataLoader...")

        args = type(
            "Args",
            (),
            {
                "pad_length": config.pad_length,
                "pad_token_idx": config.pad_token_idx,
                "chrom_token_left_idx": config.chrom_token_left_idx,
                "chrom_token_right_idx": config.chrom_token_right_idx,
                "cls_token_idx": config.cls_token_idx,
                "CHROM_TOKEN_OFFSET": config.CHROM_TOKEN_OFFSET,
                "sample_size": config.sample_size,
                "CXG": True,
            },
        )()

        dataset = MultiDatasetSentences(
            sorted_dataset_names=[dataset_name],
            shapes_dict=shapes_dict,
            args=args,
            dataset_to_protein_embeddings_path=pe_idx_dict,
            datasets_to_chroms_path=chroms_dict,
            datasets_to_starts_path=starts_dict,
            npzs_dir=working_dir,
        )

        collator = MultiDatasetSentenceCollator(args)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=num_workers,
        )

        logger.info(f"✓ DataLoader created with {len(dataset)} samples")

        return dataloader

    def predict_embeddings(
        self,
        data: Union[AnnData, PerturbationData],
        species: Optional[str] = None,
        batch_size: int = 25,
        store_adata: bool = False,
        output_key: str = "X_uce",
        working_dir: Optional[str] = None,
        filter_genes: bool = True,
        return_gene_embeddings: bool = True,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generates cell embeddings and gene embeddings.

        This method uses get_dataloader() to handle preprocessing and then runs inference.

        Args:
            data: PerturbationData or AnnData object.
            species: Species name (e.g., 'human', 'mouse').
            batch_size: Batch size for inference.
            store_adata: Whether to store the adata object in the result dictionary.
            output_key: Key to store embeddings in adata.obsm.
            working_dir: Working directory for intermediate files.
            filter_genes: Whether to filter genes based on protein embeddings availability.
            return_gene_embeddings: Whether to return gene embeddings.

        Returns:
            Dictionary containing 'cell_embeddings', 'gene_embeddings', and 'gene_list'.
        """
        if not self.model_loaded:
            raise ValueError(
                "Model weights not loaded. Call load_weights() or from_pretrained() first."
            )

        if self.all_pe is None or self.model.pe_embedding is None:
            self.load_token_embeddings()

        if species is None:
            species = self.config.species

        if isinstance(data, PerturbationData):
            adata = data.adata
        else:
            adata = data

        logger.info(
            f"Generating embeddings for {adata.n_obs} cells (species: {species})..."
        )

        dataloader = self.get_dataloader(
            data=data,
            species=species,
            config=self.config,
            batch_size=batch_size,
            working_dir=working_dir,
            filter_genes=filter_genes,
            shuffle=False,
            num_workers=0,
        )

        logger.info("Running inference...")

        self.model.eval()
        cell_embeddings_list = []
        gene_embeddings_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                batch_sentences, mask, idxs = batch[0], batch[1], batch[2]

                # Permute to (seq_len, batch_size)
                batch_sentences = batch_sentences.permute(1, 0)

                # Embed tokens (gene embeddings before transformer)
                gene_emb = self.model.pe_embedding(
                    batch_sentences.long().to(self.device)
                )

                if return_gene_embeddings:
                    gene_embeddings_list.append(gene_emb.detach().cpu().numpy())

                gene_emb_normalized = nn.functional.normalize(gene_emb, dim=2)

                # Forward pass
                mask = mask.to(self.device)
                _, cell_embedding = self.model.forward(gene_emb_normalized, mask=mask)

                cell_embeddings_list.append(cell_embedding.detach().cpu().numpy())

        cell_embeddings = np.vstack(cell_embeddings_list)

        result = {"cell_embeddings": cell_embeddings}

        if return_gene_embeddings and gene_embeddings_list:
            # Concatenate gene embeddings along batch dimension
            gene_embeddings = np.concatenate(gene_embeddings_list, axis=1)
            # Convert to batch_first
            gene_embeddings = np.transpose(gene_embeddings, (1, 0, 2))
            result["gene_embeddings"] = gene_embeddings
            logger.info(f"✓ Gene embeddings shape: {gene_embeddings.shape}")

        if "gene_name" in adata.var.columns:
            result["gene_list"] = adata.var["gene_name"].tolist()
        else:
            result["gene_list"] = adata.var_names.tolist()

        if store_adata and output_key:
            adata.obsm[output_key] = cell_embeddings
            logger.info(f"Stored cell embeddings in adata.obsm['{output_key}']")

            if return_gene_embeddings and "gene_embeddings" in result:
                logger.warning(
                    "Gene embeddings contain special tokens and cannot be directly aligned "
                    "with adata.var. Skipping adata.varm storage."
                )

        logger.info(f"✓ Generated cell embeddings: shape {cell_embeddings.shape}")

        return result

    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()
        return self

    def forward(
        self, batch: Union[torch.Tensor, tuple], return_gene_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the UCE model.

        Args:
            batch: Tensor of token indices or DataLoader tuple.
            return_gene_embeddings: Whether to return gene embeddings.

        Returns:
            Dictionary containing cell and optionally gene embeddings.
        """
        if isinstance(batch, tuple):
            batch_sentences, mask = batch[0], batch[1]
        else:
            batch_sentences = batch
            mask = None

        batch_sentences = batch_sentences.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # UCE expects (seq_len, batch_size) format
        if self.config.batch_first and batch_sentences.dim() == 2:
            batch_sentences = batch_sentences.permute(1, 0)

        gene_embeddings = self.model.pe_embedding(batch_sentences.long())
        gene_embeddings_normalized = nn.functional.normalize(gene_embeddings, dim=2)

        _, cell_embeddings = self.model.forward(gene_embeddings_normalized, mask=mask)

        output = {"cell_embeddings": cell_embeddings}

        if return_gene_embeddings:
            output["gene_embeddings"] = gene_embeddings

        return output

    def compute_loss(
        self,
        batch: Union[torch.Tensor, tuple],
        labels: Optional[torch.Tensor] = None,
        loss_type: str = "contrastive",
        temperature: float = 0.07,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes loss for UCE model.

        Args:
            batch: Batch data.
            labels: Optional labels for supervised learning.
            loss_type: Loss function type ('contrastive', 'mse', 'cosine').
            temperature: Temperature for contrastive loss.

        Returns:
            Dictionary containing loss and embeddings.
        """
        output = self.forward(batch, return_gene_embeddings=False)
        cell_embeddings = output["cell_embeddings"]

        if loss_type == "contrastive":
            cell_embeddings_norm = nn.functional.normalize(cell_embeddings, dim=1)
            similarity_matrix = (
                torch.matmul(cell_embeddings_norm, cell_embeddings_norm.t())
                / temperature
            )
            batch_size = cell_embeddings.shape[0]
            labels_contrastive = torch.arange(batch_size, device=self.device)
            loss = nn.functional.cross_entropy(similarity_matrix, labels_contrastive)

        elif loss_type == "mse" and labels is not None:
            loss = nn.functional.mse_loss(cell_embeddings, labels)

        elif loss_type == "cosine" and labels is not None:
            loss = (
                1
                - nn.functional.cosine_similarity(cell_embeddings, labels, dim=1).mean()
            )

        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        return {"loss": loss, "cell_embeddings": cell_embeddings}

    def train_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        loss_type: str = "contrastive",
        temperature: float = 0.07,
        save_dir: Optional[str] = None,
        save_every: int = 1,
        log_every: int = 100,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Trains the UCE model.

        Note: Basic training loop. Refer to original UCE code for full pipeline.
        """
        if not self.model_loaded:
            raise ValueError(
                "Model weights not loaded. Call load_weights() or from_pretrained() first."
            )

        if device is not None:
            self.device = device
            self.model = self.model.to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_epochs * len(train_dataloader) - current_step)
                / float(max(1, num_epochs * len(train_dataloader) - warmup_steps)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            logger.info(f"Validation samples: {len(val_dataloader.dataset)}")

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()

                loss_dict = self.compute_loss(
                    batch=batch, loss_type=loss_type, temperature=temperature
                )
                loss = loss_dict["loss"]

                loss.backward()

                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_steps += 1
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = train_loss / train_steps
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        {"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"}
                    )

            avg_train_loss = train_loss / train_steps
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}"
            )

            if val_dataloader:
                self.model.eval()
                val_loss = 0.0
                val_steps = 0

                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation"):
                        loss_dict = self.compute_loss(
                            batch=batch, loss_type=loss_type, temperature=temperature
                        )
                        val_loss += loss_dict["loss"].item()
                        val_steps += 1

                avg_val_loss = val_loss / val_steps
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if save_dir:
                        best_model_path = os.path.join(save_dir, "best_model")
                        self.save(best_model_path)
                        logger.info(f"✓ Saved best model to {best_model_path}")

            if save_dir and (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                self.save(checkpoint_path)
                logger.info(f"✓ Saved checkpoint to {checkpoint_path}")

        logger.info("Training completed!")

        if save_dir:
            final_model_path = os.path.join(save_dir, "final_model")
            self.save(final_model_path)
            logger.info(f"✓ Saved final model to {final_model_path}")

    def save(self, save_directory: str, save_tokens: bool = True):
        """
        Saves UCE model configuration and weights.

        Args:
            save_directory: Directory to save the model.
            save_tokens: Whether to save token embeddings to tokens.pt.
        """
        os.makedirs(save_directory, exist_ok=True)

        config_path = os.path.join(save_directory, "config.json")
        self.config.save(config_path)
        logger.info(f"Saved config to {config_path}")

        if self.model_loaded:
            model_path = os.path.join(save_directory, "model.pt")
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model weights to {model_path}")
        else:
            logger.warning("Model weights not loaded, skipping model.pt save")

        if save_tokens and self.all_pe is not None:
            tokens_path = os.path.join(save_directory, "tokens.pt")
            torch.save(self.all_pe, tokens_path)
            logger.info(f"Saved token embeddings to {tokens_path}")

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[UCEModelConfig] = None,
        device: str = "cuda",
        **kwargs,
    ) -> "UCEModel":
        """
        Loads UCE model from pretrained weights.

        Args:
            model_name_or_path: Model name or path to directory.
            config: Optional configuration override.
            device: Device to load model on.

        Returns:
            Loaded UCEModel instance.
        """
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
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
                        model_name_or_path, organization="perturblab", **kwargs
                    )
                    logger.info(f"✓ Model cached at: {model_path}")
                except Exception as e:
                    raise ValueError(
                        f"Model not found: {model_name_or_path}. Error: {e}"
                    )

        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            loaded_config = UCEModelConfig.load(config_path)
            if config is not None:
                for key, value in config.__dict__.items():
                    if not key.startswith("_"):
                        setattr(loaded_config, key, value)
            config = loaded_config
        elif config is None:
            logger.warning(
                f"Config file not found at {config_path}, using default config"
            )
            config = UCEModelConfig(**kwargs)

        model = cls(config=config, device=device)

        model_weights_path = os.path.join(model_path, "model.pt")
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path)
        else:
            logger.warning(f"Model weights not found at {model_weights_path}")

        tokens_path = os.path.join(model_path, "tokens.pt")
        if os.path.exists(tokens_path):
            model.load_token_embeddings(tokens_path)
        else:
            if config.token_file and os.path.exists(config.token_file):
                model.load_token_embeddings(config.token_file)
            else:
                logger.info("Token embeddings not found, will be loaded on-demand")

        logger.info("✓ UCE model loaded successfully")
        return model
