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
from .memory_dataset import InMemoryMultiDatasetSentences

# Constants (from UCE source: evaluate.py)
# These match the original UCE implementation and should not be changed
ESM2_BASE_TOKEN_COUNT = 143574  # ESM2-650M vocabulary size (without chromosome tokens)
CHROM_TOKEN_COUNT = 1895  # Number of chromosome identifier tokens (hardcoded in original)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class UCEModel(PerturbationModel):
    """UCE (Universal Cell Embeddings) Model.

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
    def verify_token_embeddings(token_file: str) -> Dict[str, any]:
        """Verify and inspect token embeddings file.
        
        This utility helps diagnose token embedding issues mentioned in code review.
        It checks whether the tokens.pt file contains base ESM2 tokens only,
        or includes pretrained chromosome tokens.
        
        Args:
            token_file: Path to tokens.pt file.
            
        Returns:
            Dictionary with verification information:
                - 'shape': Tuple of token embedding shape
                - 'has_chromosome_tokens': Boolean indicating if chromosome tokens are present
                - 'expected_shape': Expected shape for complete embeddings
                - 'device': Device where embeddings are stored
        """
        import torch
        
        all_pe = torch.load(token_file, map_location='cpu', weights_only=False)
        
        if isinstance(all_pe, dict):
            if "tokens" in all_pe:
                all_pe = all_pe["tokens"]
            elif "embeddings" in all_pe:
                all_pe = all_pe["embeddings"]
        
        shape = all_pe.shape
        has_chroms = shape[0] > ESM2_BASE_TOKEN_COUNT
        expected_complete = (ESM2_BASE_TOKEN_COUNT + CHROM_TOKEN_COUNT, shape[1])
        
        info = {
            'shape': shape,
            'has_chromosome_tokens': has_chroms,
            'expected_shape': expected_complete,
            'device': str(all_pe.device) if hasattr(all_pe, 'device') else 'unknown',
            'num_base_tokens': ESM2_BASE_TOKEN_COUNT,
            'num_chrom_tokens': CHROM_TOKEN_COUNT if has_chroms else 0,
        }
        
        return info

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
        device: str = 'cpu',
        spec_chrom_csv_path: Optional[str] = None,
        token_file: Optional[str] = None,
        protein_embeddings_dir: Optional[str] = None,
        offset_pkl_path: Optional[str] = None,
    ):
        """Initializes the UCE model.

        Args:
            config: UCE configuration object.
            device: Device to run the model on.
            spec_chrom_csv_path: Path to species chromosome CSV file.
            token_file: Path to token embeddings file.
            protein_embeddings_dir: Directory containing protein embeddings.
            offset_pkl_path: Path to species offsets pickle file.
        """
        super().__init__(config)

        self.config = config
        
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
    
    def to(self, device: str):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self
    
    def eval(self):
        self.model.eval()
        return self

    def load_weights(self, model_path: str, device: str = 'cpu'):
        """Loads model weights from a standard model.pt file.

        Args:
            model_path: Path to model.pt file containing state_dict.
            device: Device to load the model on.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        logger.info(f"Loading model weights from {model_path}...")

        state_dict = torch.load(model_path, map_location=device)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)
        self.model.eval()

        self.model_loaded = True
        logger.info("✓ Model weights loaded successfully")

    def load_token_embeddings(self, token_file: Optional[str] = None, model_name: Optional[str] = None, device: str = 'cpu'):
        """Loads protein/token embeddings from tokens.pt file.

        Args:
            token_file: Path to tokens.pt file.
            model_name: Model name for downloading from HF.
            device: Device to load the token embeddings on.
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

        all_pe = torch.load(token_file, map_location=device, weights_only=False)

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
        # This logic matches UCE source (evaluate.py:155-160)
        # If tokens.pt only contains ESM2 base tokens, we append chromosome tokens
        # using the same seed (23) as in the original implementation
        if all_pe.shape[0] == ESM2_BASE_TOKEN_COUNT:
            logger.info(
                f"Token embeddings contain only ESM2 base tokens ({ESM2_BASE_TOKEN_COUNT}). "
                f"Appending {CHROM_TOKEN_COUNT} chromosome tokens with seed=23 (matching UCE source)."
            )
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(
                mean=0, std=1, size=(CHROM_TOKEN_COUNT, self.config.token_dim)
            )
            all_pe = torch.vstack((all_pe, CHROM_TENSORS))
            logger.info(f"✓ Chromosome tokens appended. New shape: {all_pe.shape}")
        else:
            logger.info(
                f"Token embeddings already contain {all_pe.shape[0]} tokens "
                f"(expected {ESM2_BASE_TOKEN_COUNT + CHROM_TOKEN_COUNT} for pretrained models)."
            )

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
        use_memory: bool = True,
        cleanup_tempdir: bool = True,
    ) -> DataLoader:
        """
        Creates DataLoader for UCE model from PerturbationData or AnnData.

        This method handles the complete preprocessing pipeline:
        1. Data preprocessing (filtering, normalization).
        2. Gene-to-protein-embedding mapping.
        3. Chromosome and position information extraction.
        4. Sentence construction with sampling.
        5. Token indexing.

        Performance Optimization:
            - use_memory=True (default): Loads data into memory, avoids disk IO bottleneck.
              Best for: datasets < 1M cells, repeated inference, when speed is critical.
            - use_memory=False: Uses disk-based memmap (original UCE behavior).
              Best for: very large datasets, memory-constrained environments.

        Args:
            data: PerturbationData or AnnData object.
            species: Species name (e.g., 'human', 'mouse').
            config: UCEConfig with model parameters.
            spec_chrom_csv_path: Path to species chromosome CSV file.
            offset_pkl_path: Path to species offsets pickle file.
            protein_embeddings_dir: Directory containing protein embeddings.
            batch_size: Batch size for DataLoader.
            working_dir: Working directory for intermediate files. If None and use_memory=False,
                a temporary directory will be created and cleaned up automatically.
            filter_genes: Whether to filter genes based on protein embeddings availability.
            shuffle: Whether to shuffle data.
            num_workers: Number of workers for DataLoader.
            dataset_name: Name for this dataset (used as key in dicts).
            use_memory: If True, use in-memory dataset (faster). If False, use disk-based memmap.
            cleanup_tempdir: If True and working_dir is auto-created, register cleanup on exit.

        Returns:
            DataLoader: PyTorch DataLoader ready for model inference.
            
        Note:
            When use_memory=False and working_dir is auto-created, temporary files are 
            registered for cleanup but may persist until Python exit. For manual cleanup,
            specify your own working_dir and delete it when done.
        """
        if isinstance(data, PerturbationData):
            adata = data.adata
        else:
            adata = data

        # Manage temporary directory
        tempdir_obj = None
        user_provided_dir = working_dir is not None
        
        if working_dir is None:
            import tempfile
            if use_memory:
                # For memory mode, we still need tempdir for preprocessing
                # but data won't be repeatedly read from disk
                tempdir_obj = tempfile.TemporaryDirectory(prefix="uce_dataloader_")
                working_dir = tempdir_obj.name
            else:
                # For disk mode, create tempdir and optionally register cleanup
                working_dir = tempfile.mkdtemp(prefix="uce_dataloader_")
                if cleanup_tempdir:
                    import atexit
                    import shutil
                    atexit.register(lambda: shutil.rmtree(working_dir, ignore_errors=True))
                    logger.info(f"Temporary directory registered for cleanup: {working_dir}")
        
        os.makedirs(working_dir, exist_ok=True)

        logger.info(
            f"Preparing UCE DataLoader for {adata.n_obs} cells (species: {species}, "
            f"mode: {'memory' if use_memory else 'disk'})..."
        )

        # Step 1: Preprocess AnnData
        logger.info("Step 1/4: Preprocessing AnnData...")
        # Mock object to adapt UCE's argparse-based interface (from source/data_proc/data_utils.py)
        # UCE's process_raw_anndata expects an argparse Namespace-like object
        # We create a minimal mock object with required attributes
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

        if use_memory:
            # Memory mode: Store counts directly in memory
            logger.info("Using in-memory dataset (no disk IO during iteration)")
            if issparse(proc_adata.X):
                counts_data = proc_adata.X.toarray()
            else:
                counts_data = proc_adata.X
            counts_dict = {dataset_name: counts_data}
        else:
            # Disk mode: Write everything to disk for memmap access (original UCE behavior)
            # IO bottleneck: This disk write/read is required by UCE's MultiDatasetSentences
            # which uses np.memmap for memory-efficient loading (see source/eval_data.py:63)
            logger.info("Using disk-based dataset (memmap for memory efficiency)")
            
            # Save counts as npz
            counts_path = os.path.join(working_dir, f"{dataset_name}_counts.npz")
            if issparse(proc_adata.X):
                save_npz(counts_path, proc_adata.X)
            else:
                np.savez_compressed(counts_path, data=proc_adata.X)
            
            # Save metadata files that MultiDatasetSentences expects
            pe_path = os.path.join(working_dir, "dataset_to_pe.torch")
            chroms_path = os.path.join(working_dir, "dataset_to_chroms.pkl")
            starts_path = os.path.join(working_dir, "dataset_to_starts.pkl")
            
            torch.save(pe_idx_dict, pe_path)
            with open(chroms_path, "wb") as f:
                pickle.dump(chroms_dict, f)
            with open(starts_path, "wb") as f:
                pickle.dump(starts_dict, f)

        # Step 4: Create DataLoader
        logger.info("Step 4/4: Creating DataLoader...")

        # Mock object to adapt UCE's argparse-based interface (from source/eval_data.py)
        # MultiDatasetSentences and sample_cell_sentences expect an args object with these attributes
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

        if use_memory:
            # Use in-memory dataset for better performance
            dataset = InMemoryMultiDatasetSentences(
                sorted_dataset_names=[dataset_name],
                shapes_dict=shapes_dict,
                args=args,
                dataset_to_protein_embeddings_path=pe_idx_dict,
                datasets_to_chroms_path=chroms_dict,
                datasets_to_starts_path=starts_dict,
                counts_dict=counts_dict,
            )
        else:
            # Use disk-based dataset (original UCE behavior)
            # MultiDatasetSentences expects file paths, not dictionaries
            pe_path = os.path.join(working_dir, "dataset_to_pe.torch")
            chroms_path = os.path.join(working_dir, "dataset_to_chroms.pkl")
            starts_path = os.path.join(working_dir, "dataset_to_starts.pkl")
            
            dataset = MultiDatasetSentences(
                sorted_dataset_names=[dataset_name],
                shapes_dict=shapes_dict,
                args=args,
                dataset_to_protein_embeddings_path=pe_path,
                datasets_to_chroms_path=chroms_path,
                datasets_to_starts_path=starts_path,
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

        # Clean up temporary directory if it was auto-created and we're in memory mode
        if tempdir_obj is not None:
            tempdir_obj.cleanup()

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
        split: Optional[str] = None,
        use_memory: bool = True,
        **kwargs,
    ) -> Dict[str, Union[np.ndarray, List, Dict[str, np.ndarray]]]:
        """Generates cell embeddings and gene embeddings.

        Args:
            data: PerturbationData or AnnData object.
            species: Species name (e.g., 'human', 'mouse').
            batch_size: Batch size for inference.
            store_adata: Whether to store embeddings in adata.
            output_key: Key to store embeddings in adata.obsm.
            working_dir: Working directory for intermediate files.
            filter_genes: Whether to filter genes.
            return_gene_embeddings: Whether to return gene embeddings.
            split: Specific split to return, or None for all splits.
            use_memory: If True (default), use in-memory dataset for faster inference.
                If False, use disk-based memmap (slower but more memory efficient).

        Returns:
            Dictionary with split names as keys, each containing:
                - 'cell': np.ndarray of shape (n_cells, embedding_dim)
                - 'gene': np.ndarray of shape (n_cells, seq_len, token_dim) [optional]
                - 'gene_list': List[str] of gene names
            
            If split is specified, returns single dict for that split.
            If split is None, returns nested dict with split names as keys.
        """
        if not self.model_loaded:
            raise ValueError(
                "Model weights not loaded. Call load_weights() or from_pretrained() first."
            )

        if self.all_pe is None or self.model.pe_embedding is None:
            self.load_token_embeddings()

        if species is None:
            species = self.config.species

        # Extract AnnData and handle splits
        if isinstance(data, PerturbationData):
            adata = data.adata
        else:
            adata = data
        
        has_split = "split" in adata.obs
        
        if split is not None:
            if has_split:
                if split not in adata.obs["split"].values:
                    raise ValueError(f"Split '{split}' not found in dataset")
                adata = adata[adata.obs["split"] == split]
            # If no split in data, use all data for requested split
            return self._predict_embeddings_single(
                adata, species, batch_size, store_adata, output_key,
                working_dir, filter_genes, return_gene_embeddings, use_memory
            )
        elif has_split:
            # Process all splits separately
            split_names = adata.obs["split"].unique()
            result = {}
            for split_name in split_names:
                subset = adata[adata.obs["split"] == split_name]
                emb_result = self._predict_embeddings_single(
                    subset, species, batch_size, store_adata, output_key,
                    working_dir, filter_genes, return_gene_embeddings, use_memory
                )
                result[str(split_name)] = emb_result
            return result
        else:
            # No split, process all as 'train'
            emb_result = self._predict_embeddings_single(
                adata, species, batch_size, store_adata, output_key,
                working_dir, filter_genes, return_gene_embeddings, use_memory
            )
            return {'train': emb_result}
    
    def _predict_embeddings_single(
        self,
        adata: AnnData,
        species: str,
        batch_size: int,
        store_adata: bool,
        output_key: str,
        working_dir: Optional[str],
        filter_genes: bool,
        return_gene_embeddings: bool,
        use_memory: bool,
    ) -> Dict[str, np.ndarray]:
        """Internal method to process a single AnnData subset."""

        logger.info(
            f"Generating embeddings for {adata.n_obs} cells (species: {species})..."
        )

        # Create temporary AnnData wrapper for get_dataloader
        from anndata import AnnData as AnnDataType
        temp_data = AnnDataType(adata.X, obs=adata.obs, var=adata.var)
        
        dataloader = self.get_dataloader(
            data=temp_data,
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
            use_memory=use_memory,
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

        # Memory consideration: For very large datasets (>1M cells), this vstack 
        # may cause OOM. Consider batch writing to HDF5/Zarr for production use.
        cell_embeddings = np.vstack(cell_embeddings_list)

        # Unified dictionary format with 'cell' and 'gene' keys
        result = {"cell": cell_embeddings}

        if return_gene_embeddings and gene_embeddings_list:
            # Concatenate gene embeddings along batch dimension
            gene_embeddings = np.concatenate(gene_embeddings_list, axis=1)
            # Convert to batch_first
            gene_embeddings = np.transpose(gene_embeddings, (1, 0, 2))
            result["gene"] = gene_embeddings
            logger.info(f"✓ Gene embeddings shape: {gene_embeddings.shape}")

        if "gene_name" in adata.var.columns:
            result["gene_list"] = adata.var["gene_name"].tolist()
        else:
            result["gene_list"] = adata.var_names.tolist()

        if store_adata and output_key:
            adata.obsm[output_key] = cell_embeddings
            logger.info(f"Stored cell embeddings in adata.obsm['{output_key}']")

            if return_gene_embeddings and "gene" in result:
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
        """Forward pass of the UCE model.

        Args:
            batch: Tensor of token indices or DataLoader tuple.
            return_gene_embeddings: Whether to return gene embeddings.

        Returns:
            Dictionary with 'cell' key containing cell embeddings.
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

        # Unified dictionary format
        output = {"cell": cell_embeddings}

        if return_gene_embeddings:
            output["gene"] = gene_embeddings

        return output

    def compute_loss(
        self,
        batch: Union[torch.Tensor, tuple],
        labels: Optional[torch.Tensor] = None,
        loss_type: str = "contrastive",
        temperature: float = 0.07,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Computes loss for UCE model.

        Args:
            batch: Batch data.
            labels: Optional labels for supervised learning.
            loss_type: Loss function type ('contrastive', 'mse', 'cosine').
            temperature: Temperature for contrastive loss.

        Returns:
            Dictionary with 'loss' key containing the computed loss.
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
        """Trains the UCE model.
        
        Args:
            train_dataloader: Training DataLoader.
            val_dataloader: Validation DataLoader.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            warmup_steps: Number of warmup steps.
            max_grad_norm: Maximum gradient norm for clipping.
            loss_type: Type of loss function.
            temperature: Temperature parameter for contrastive loss.
            save_dir: Directory to save checkpoints.
            save_every: Save checkpoint every N epochs.
            log_every: Log every N steps.
            device: Device to use for training.
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

    def save(self, model_path: str, save_tokens: bool = True, save_auxiliary: bool = True):
        """Saves UCE model configuration, weights, and auxiliary files.

        Args:
            save_directory: Directory to save the model.
            save_tokens: Whether to save token embeddings (tokens.pt).
            save_auxiliary: Whether to save auxiliary data files (embeddings, offsets, etc.).
        """
        os.makedirs(model_path, exist_ok=True)

        # 1. Save config
        config_path = os.path.join(model_path, "config.json")
        self.config.save(config_path)
        logger.info(f"✓ Saved config to {config_path}")

        # 2. Save model weights
        if self.model_loaded:
            model_path = os.path.join(model_path, "model.pt")
            # Ensure we're saving the state dict of the inner model if wrapped
            state_dict = self.model.state_dict()
            torch.save(state_dict, model_path)
            logger.info(f"✓ Saved model weights to {model_path}")
        else:
            logger.warning("Model weights not loaded, skipping model.pt save")

        # 3. Save token embeddings (tokens.pt)
        if save_tokens and self.all_pe is not None:
            tokens_path = os.path.join(model_path, "tokens.pt")
            torch.save(self.all_pe, tokens_path)
            logger.info(f"✓ Saved token embeddings to {tokens_path}")

        # 4. Save auxiliary files (Mirroring the structure provided)
        if save_auxiliary:
            import shutil

            # Copy species_chrom.csv
            if self.spec_chrom_csv_path and os.path.exists(self.spec_chrom_csv_path):
                dest = os.path.join(model_path, "species_chrom.csv")
                shutil.copy2(self.spec_chrom_csv_path, dest)
                logger.info(f"✓ Copied species_chrom.csv")

            # Copy species_offsets.pkl
            if self.offset_pkl_path and os.path.exists(self.offset_pkl_path):
                dest = os.path.join(model_path, "species_offsets.pkl")
                shutil.copy2(self.offset_pkl_path, dest)
                logger.info(f"✓ Copied species_offsets.pkl")

            # Copy protein embeddings directory
            if self.protein_embeddings_dir and os.path.exists(self.protein_embeddings_dir):
                dest_dir = os.path.join(model_path, "protein_embeddings")
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)  # Remove existing to ensure clean copy
                shutil.copytree(self.protein_embeddings_dir, dest_dir)
                logger.info(f"✓ Copied protein_embeddings directory")

        logger.info(f"✓ Model saved to {model_path}")

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = 'cpu',
        load_tokens: bool = True,
    ) -> 'UCEModel':
        """
        Loads UCE model from a saved directory.

        Args:
            model_path: Path to the saved model directory.
            device: Device to load model on ('cpu' or 'cuda').
            load_tokens: Whether to load token embeddings.

        Returns:
            Loaded UCEModel instance.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # 1. Load config
        config_path = os.path.join(model_path, "config.json")
        
        # A. Start with file config if it exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = UCEConfig.load(config_path)
        logger.info(f"✓ Loaded config from {config_path}")
            
        # 2. Resolve auxiliary paths relative to model_path    
        spec_chrom_csv_path = os.path.join(model_path, "species_chrom.csv")
        offset_pkl_path = os.path.join(model_path, "species_offsets.pkl")
        protein_embeddings_dir = os.path.join(model_path, "protein_embeddings")
        token_file = os.path.join(model_path, "tokens.pt")

        # Validate existence (pass None if missing, so model doesn't crash on init)
        spec_chrom_csv_path = spec_chrom_csv_path if os.path.exists(spec_chrom_csv_path) else None
        offset_pkl_path = offset_pkl_path if os.path.exists(offset_pkl_path) else None
        protein_embeddings_dir = protein_embeddings_dir if os.path.exists(protein_embeddings_dir) else None
        
        # 3. Initialize model
        model = cls(
            config=config,
            device=device,
            spec_chrom_csv_path=spec_chrom_csv_path,
            token_file=token_file if os.path.exists(token_file) else None,
            protein_embeddings_dir=protein_embeddings_dir,
            offset_pkl_path=offset_pkl_path,
        )

        # 4. Load model weights
        model_weights_path = os.path.join(model_path, "model.pt")
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path, device=device)
        else:
            logger.warning(f"Model weights not found at {model_weights_path}")

        # 5. Load token embeddings
        if load_tokens and model.token_file:
            model.load_token_embeddings(model.token_file, device=device)

        logger.info(f"✓ UCE model loaded successfully from {model_path}")
        return model
