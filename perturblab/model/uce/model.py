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
from .config import UCEConfig
from .source.data_proc.data_utils import (adata_path_to_prot_chrom_starts,
                                          get_spec_chrom_csv,
                                          get_species_to_pe,
                                          process_raw_anndata)
from .source.eval_data import (MultiDatasetSentenceCollator,
                               MultiDatasetSentences)
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

    @staticmethod
    def _get_token_embeddings_from_hub(model_name: str = "uce-4layer") -> str:
        """
        Download token embeddings from HuggingFace Hub.
        
        This uses HuggingFace's caching mechanism to download tokens.pt
        from the pretrained model repository.
        
        Args:
            model_name: Model name (e.g., 'uce-4layer', 'uce-33layer').
            
        Returns:
            Path to the cached tokens.pt file.
        """
        try:
            # Handle both 'uce-4layer' and 'perturblab/uce-4layer'
            if '/' not in model_name:
                hf_model_name = f"perturblab/{model_name}"
            else:
                hf_model_name = model_name
            
            logger.info(f"Downloading token embeddings from {hf_model_name}...")
            model_path = download_from_huggingface(hf_model_name, organization=None)
            
            token_file = os.path.join(model_path, "tokens.pt")
            if os.path.exists(token_file):
                logger.info(f"✓ Token embeddings cached at: {token_file}")
                return token_file
            else:
                raise FileNotFoundError(
                    f"tokens.pt not found in downloaded model: {model_path}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download token embeddings from HuggingFace: {e}"
            )

    def __init__(
        self,
        config: UCEConfig,
        device: str = "cuda",
        spec_chrom_csv_path: Optional[str] = None,
        token_file: Optional[str] = None,
        protein_embeddings_dir: Optional[str] = None,
        offset_pkl_path: Optional[str] = None,
    ):
        """
        Initializes the UCE model.

        Args:
            config: UCEConfig configuration object.
            device: Device to run the model on ('cuda' or 'cpu').
            spec_chrom_csv_path: Path to species chromosome CSV file.
            token_file: Path to token embeddings file.
            protein_embeddings_dir: Directory containing protein embeddings.
            offset_pkl_path: Path to species offsets pickle file.
        """
        super().__init__(config)

        self.config = config
        self.device = device
        
        self.spec_chrom_csv_path = spec_chrom_csv_path
        self.token_file = token_file
        self.protein_embeddings_dir = protein_embeddings_dir
        self.offset_pkl_path = offset_pkl_path

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

    def load_token_embeddings(self, token_file: Optional[str] = None, model_name: Optional[str] = None):
        """
        Loads protein/token embeddings from tokens.pt file.
        
        If token_file is not provided or doesn't exist, automatically downloads
        from HuggingFace Hub using the model's caching mechanism.

        Args:
            token_file: Path to tokens.pt file. If None, uses self.token_file or downloads from HF.
            model_name: Model name for downloading from HF (e.g., 'uce-4layer'). 
                       If None, uses self.config.model_name.
        """
        if token_file is None:
            token_file = self.token_file

        # If still None or doesn't exist, download from HuggingFace
        if token_file is None or not os.path.exists(token_file):
            if model_name is None:
                model_name = f"{self.config.model_series}-{self.config.model_name}"
            
            logger.info(f"Token file not found, downloading from HuggingFace ({model_name})...")
            token_file = self._get_token_embeddings_from_hub(model_name)

        logger.info(f"Loading token embeddings from {token_file}...")

        all_pe = torch.load(token_file, map_location=self.device, weights_only=False)

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
        if all_pe.shape[0] == ESM2_BASE_TOKEN_COUNT:
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
        config: UCEConfig,
        spec_chrom_csv_path: str,
        offset_pkl_path: str,
        protein_embeddings_dir: str,
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
            config: UCEConfig with model parameters.
            spec_chrom_csv_path: Path to species chromosome CSV file.
            offset_pkl_path: Path to species offsets pickle file.
            protein_embeddings_dir: Directory containing protein embeddings.
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

        species_to_pe = get_species_to_pe(protein_embeddings_dir)

        with open(offset_pkl_path, "rb") as f:
            species_to_offsets = pickle.load(f)

        gene_to_chrom_pos = get_spec_chrom_csv(spec_chrom_csv_path)

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
                "CHROM_TOKEN_OFFSET": config.chrom_token_offset,
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
            spec_chrom_csv_path=self.spec_chrom_csv_path,
            offset_pkl_path=self.offset_pkl_path,
            protein_embeddings_dir=self.protein_embeddings_dir,
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

    def save(self, save_directory: str, save_tokens: bool = True, save_auxiliary: bool = True):
        """
        Saves UCE model configuration and weights.

        Args:
            save_directory: Directory to save the model.
            save_tokens: Whether to save token embeddings to tokens.pt.
            save_auxiliary: Whether to save auxiliary files (species_chrom.csv, etc.).
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, "config.json")
        self.config.save(config_path)
        logger.info(f"✓ Saved config to {config_path}")

        # Save model weights
        if self.model_loaded:
            model_path = os.path.join(save_directory, "model.pt")
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"✓ Saved model weights to {model_path} ({len(self.model.state_dict())} parameters)")
        else:
            logger.warning("Model weights not loaded, skipping model.pt save")

        # Save token embeddings
        if save_tokens and self.all_pe is not None:
            tokens_path = os.path.join(save_directory, "tokens.pt")
            torch.save(self.all_pe, tokens_path)
            logger.info(f"✓ Saved token embeddings to {tokens_path} (shape: {self.all_pe.shape})")

        # Save auxiliary files if they exist
        if save_auxiliary:
            import shutil

            # Copy species_chrom.csv
            if self.spec_chrom_csv_path and os.path.exists(self.spec_chrom_csv_path):
                dest = os.path.join(save_directory, "species_chrom.csv")
                shutil.copy2(self.spec_chrom_csv_path, dest)
                logger.info(f"✓ Copied species_chrom.csv")
            
            # Copy species_offsets.pkl
            if self.offset_pkl_path and os.path.exists(self.offset_pkl_path):
                dest = os.path.join(save_directory, "species_offsets.pkl")
                shutil.copy2(self.offset_pkl_path, dest)
                logger.info(f"✓ Copied species_offsets.pkl")
            
            # Copy protein embeddings directory
            if self.protein_embeddings_dir and os.path.exists(self.protein_embeddings_dir):
                dest = os.path.join(save_directory, "protein_embeddings")
                if os.path.isdir(self.protein_embeddings_dir):
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(self.protein_embeddings_dir, dest)
                    logger.info(f"✓ Copied protein_embeddings/")

        logger.info(f"✓ Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[UCEConfig] = None,
        device: str = "cuda",
        load_tokens: bool = True,
        **kwargs,
    ) -> "UCEModel":
        """
        Loads UCE model from pretrained weights.

        Supports three loading modes:
        1. Local directory path: /path/to/model/
        2. Model name in weights folder: 'uce-4layer' or 'uce-33layer'
        3. HuggingFace model: 'perturblab/uce-4layer' or just 'uce-4layer'

        Args:
            model_name_or_path: Model name or path to directory.
                - Local path: '/home/user/models/my_uce_model'
                - Model name: 'uce-4layer', 'uce-33layer'
                - HuggingFace: 'perturblab/uce-4layer'
            config: Optional configuration override.
            device: Device to load model on ('cuda' or 'cpu').
            load_tokens: Whether to load token embeddings.
            **kwargs: Additional arguments passed to config.

        Returns:
            Loaded UCEModel instance.

        Examples:
            >>> # Load from HuggingFace
            >>> model = UCEModel.from_pretrained('perturblab/uce-4layer')
            >>> 
            >>> # Load from local weights folder
            >>> model = UCEModel.from_pretrained('uce-33layer')
            >>> 
            >>> # Load from custom path
            >>> model = UCEModel.from_pretrained('/path/to/my/model')
        """
        # Step 1: Resolve model path
        model_path = None
        
        # Check if it's a local directory
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            model_path = model_name_or_path
            logger.info(f"Loading model from local path: {model_path}")
        
        # Check if it's in the weights folder
        if model_path is None:
            weights_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "weights"
            )
            weights_dir = os.path.abspath(weights_dir)
            
            # Try direct name (e.g., 'uce-4layer')
            candidate = os.path.join(weights_dir, model_name_or_path)
            if os.path.isdir(candidate):
                model_path = candidate
                logger.info(f"Loading model from weights folder: {model_path}")
            else:
                # Try with 'uce-' prefix if not present
                if not model_name_or_path.startswith('uce-'):
                    candidate = os.path.join(weights_dir, f'uce-{model_name_or_path}')
                    if os.path.isdir(candidate):
                        model_path = candidate
                        logger.info(f"Loading model from weights folder: {model_path}")
        
        # Download from HuggingFace if not found locally
        if model_path is None:
            try:
                # Handle both 'uce-4layer' and 'perturblab/uce-4layer'
                if '/' not in model_name_or_path:
                    hf_model_name = f"perturblab/{model_name_or_path}"
                else:
                    hf_model_name = model_name_or_path
                
                logger.info(f"Downloading '{hf_model_name}' from HuggingFace...")
                model_path = download_from_huggingface(
                    hf_model_name, organization=None, **kwargs
                )
                logger.info(f"✓ Model cached at: {model_path}")
            except Exception as e:
                raise ValueError(
                    f"Model not found: {model_name_or_path}. "
                    f"Tried local path, weights folder, and HuggingFace. Error: {e}"
                )

        # Step 2: Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            loaded_config = UCEConfig.load(config_path)
            
            # Override with user-provided config
            if config is not None:
                for key, value in config.__dict__.items():
                    if not key.startswith("_") and value is not None:
                        setattr(loaded_config, key, value)
            config = loaded_config
            logger.info(f"✓ Loaded config from {config_path}")
        elif config is None:
            logger.warning(
                f"Config file not found at {config_path}, using default config"
            )
            config = UCEConfig(**kwargs)

        # Step 3: Prepare file paths from model directory
        spec_chrom_csv_path = os.path.join(model_path, "species_chrom.csv")
        offset_pkl_path = os.path.join(model_path, "species_offsets.pkl")
        protein_embeddings_dir = os.path.join(model_path, "protein_embeddings")
        token_file = os.path.join(model_path, "tokens.pt")
        
        # Check if paths exist, set to None if not
        if not os.path.exists(spec_chrom_csv_path):
            spec_chrom_csv_path = None
        if not os.path.exists(offset_pkl_path):
            offset_pkl_path = None
        if not os.path.exists(protein_embeddings_dir):
            protein_embeddings_dir = None
        if not os.path.exists(token_file):
            token_file = None

        # Step 4: Initialize model with paths
        model = cls(
            config=config,
            device=device,
            spec_chrom_csv_path=spec_chrom_csv_path,
            token_file=token_file,
            protein_embeddings_dir=protein_embeddings_dir,
            offset_pkl_path=offset_pkl_path,
        )

        # Step 5: Load model weights
        model_weights_path = os.path.join(model_path, "model.pt")
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path)
        else:
            logger.warning(f"Model weights not found at {model_weights_path}")

        # Step 6: Load token embeddings
        if load_tokens and model.token_file:
            model.load_token_embeddings(model.token_file)

        logger.info("✓ UCE model loaded successfully")
        logger.info(f"   Model: {config.model_series}-{config.model_name}")
        logger.info(f"   Layers: {config.nlayers}")
        logger.info(f"   Device: {device}")
        
        return model
