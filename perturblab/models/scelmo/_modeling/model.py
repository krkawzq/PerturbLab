"""scELMo: Embeddings from Language Models - Model Implementation.

This module implements the scELMo model logic. Unlike neural networks,
scELMo is a non-parametric model that performs weighted aggregation of
pre-computed gene embeddings.

Architecture:
    Cell_Embedding = Normalize( Expression_Matrix @ Gene_Embedding_Matrix )
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from perturblab.models.scelmo.config import scELMoConfig
from perturblab.models.scelmo.io import scELMoInput, scELMoOutput
from perturblab.types import GeneVocab
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["scELMoModel"]


class scELMoModel(nn.Module):
    """scELMo Model for generating cell embeddings.

    This model acts as a registry of gene embeddings. It uses the `gene_names`
    defined in the config to align inputs during the forward pass.

    Attributes:
        vocab (GeneVocab): Vocabulary constructed from config.gene_names.
        gene_embeddings (torch.Tensor): Dense embedding buffer [N_vocab, Dim].
    """

    def __init__(
        self,
        config: scELMoConfig,
        gene_embeddings: np.ndarray | torch.Tensor | None = None,
        device: str = "cpu",
    ):
        """Initialize scELMo model.

        Args:
            config (scELMoConfig): Configuration containing `gene_names` and hyperparameters.
            gene_embeddings (Optional[Union[np.ndarray, torch.Tensor]]):
                Pre-computed embedding matrix aligned with `config.gene_names`.
                Shape must be (len(gene_names), embedding_dim).
                If None, model initializes with zeros (must load weights later).
            device (str, optional): Device to place the embedding buffer. Defaults to "cpu".

        Raises:
            ValueError: If config.gene_names is empty.
            ValueError: If gene_embeddings shape doesn't match config.
        """
        super().__init__()
        self.config = config

        # 1. Validate and Initialize Vocabulary from Config
        if not config.gene_names:
            raise ValueError(
                "config.gene_names cannot be empty. "
                "scELMo requires a gene vocabulary to function."
            )

        # Use GeneVocab with duplicate handling
        # This automatically validates and handles duplicates according to policy
        try:
            self.vocab = GeneVocab(
                config.gene_names,
                default_token="<unk>",
                default_index=-1,
                duplicate_policy="first",  # Keep first occurrence if duplicates
            )
            logger.info(f"✓ Initialized scELMo vocabulary with {len(self.vocab)} unique genes")
        except ValueError as e:
            # If strict duplicate checking failed, provide helpful error
            raise ValueError(
                f"Failed to create gene vocabulary: {e}. "
                f"Consider cleaning gene_names in config or using duplicate_policy='first'."
            ) from e

        # 2. Initialize Embeddings Tensor
        num_genes = len(self.vocab)
        embed_dim = config.embedding_dim

        if gene_embeddings is not None:
            # Convert to Tensor
            if isinstance(gene_embeddings, np.ndarray):
                emb_tensor = torch.from_numpy(gene_embeddings).float()
            else:
                emb_tensor = gene_embeddings.float()

            # Validation
            if emb_tensor.ndim != 2:
                raise ValueError(f"gene_embeddings must be 2D, got shape {emb_tensor.shape}")
            if emb_tensor.shape[0] != num_genes:
                raise ValueError(
                    f"Shape Mismatch: config.gene_names has {num_genes} genes, "
                    f"but provided embeddings have {emb_tensor.shape[0]} rows."
                )
            if emb_tensor.shape[1] != embed_dim:
                raise ValueError(
                    f"Dim Mismatch: config.embedding_dim is {embed_dim}, "
                    f"but provided embeddings have dimension {emb_tensor.shape[1]}."
                )

            # Check for NaN or Inf
            if torch.isnan(emb_tensor).any():
                logger.warning("gene_embeddings contains NaN values!")
            if torch.isinf(emb_tensor).any():
                logger.warning("gene_embeddings contains Inf values!")

            logger.info(f"Loaded gene embeddings: {emb_tensor.shape}")
        else:
            # Initialize with zeros if no weights provided
            # (Allows loading state_dict or calling load_weights() later)
            emb_tensor = torch.zeros(num_genes, embed_dim)
            logger.warning(
                "Initialized gene_embeddings with zeros. "
                "Call load_weights() or load_state_dict() to load actual embeddings."
            )

        # Register as buffer (not a parameter, no gradient)
        self.register_buffer("gene_embeddings", emb_tensor.to(device))

    def forward(self, inputs: scELMoInput) -> scELMoOutput:
        """Forward pass: Gene Expression -> Cell Embeddings.

        Performs dynamic gene alignment and weighted aggregation to produce
        cell-level embeddings from gene expression profiles.

        Args:
            inputs (scELMoInput): Input container with:
                - expression: Gene expression matrix [batch_size, n_input_genes]
                - gene_names: List of gene symbols [n_input_genes]

        Returns:
            scELMoOutput: Output containing cell_embedding [batch_size, embedding_dim]

        Raises:
            ValueError: If expression shape doesn't match gene_names length.
            ValueError: If no valid genes found in vocabulary.
            RuntimeError: If gene_embeddings contains only zeros (not loaded).
        """
        expression = inputs.expression
        input_genes = inputs.gene_names
        device = expression.device
        batch_size = expression.shape[0]

        # =====================================================================
        # Input Validation
        # =====================================================================

        n_input = len(input_genes)

        if expression.shape[1] != n_input:
            raise ValueError(
                f"Expression matrix has {expression.shape[1]} columns "
                f"but gene_names has {n_input} elements. Shapes must match."
            )

        if n_input == 0:
            raise ValueError("gene_names cannot be empty.")

        # Check if embeddings are loaded (not all zeros)
        if torch.all(self.gene_embeddings == 0):
            logger.warning(
                "gene_embeddings appear to be all zeros. "
                "Have you called load_weights() or loaded a checkpoint?"
            )

        # =====================================================================
        # 1. Dynamic Gene Alignment
        # =====================================================================
        # Map input gene names -> model vocab indices -> embedding vectors

        # Get indices in model vocabulary using GeneVocab's lookup
        # Use warn_unknown=True to automatically log unknown genes
        model_indices = self.vocab.lookup_indices(
            input_genes,
            fallback=-1,
            warn_unknown=(n_input > 100),  # Only warn for larger gene sets to avoid spam
        )
        model_indices = torch.tensor(model_indices, device=device, dtype=torch.long)

        # Create mask for valid genes (known to the model)
        valid_mask = model_indices >= 0
        n_valid = valid_mask.sum().item()

        # Calculate alignment statistics
        alignment_rate = n_valid / n_input if n_input > 0 else 0.0

        if n_valid == 0:
            # No genes matched - provide detailed diagnostic
            sample_input = input_genes[:10]
            sample_vocab = list(self.vocab)[:10]
            raise ValueError(
                f"No genes from input matched the model vocabulary!\n"
                f"  Model vocabulary: {len(self.vocab)} genes (sample: {sample_vocab})\n"
                f"  Input genes: {n_input} genes (sample: {sample_input})\n"
                f"  Possible causes:\n"
                f"    - Gene name format mismatch (e.g., 'TP53' vs 'tp53')\n"
                f"    - Different gene ID systems (Ensembl vs Symbol)\n"
                f"    - Model trained on different species\n"
                f"  Suggestion: Check gene name standardization."
            )

        if n_valid < n_input:
            n_unknown = n_input - n_valid

            if alignment_rate < 0.5:
                # Low alignment rate - warn user
                logger.warning(
                    f"⚠️  Low gene alignment rate: {alignment_rate:.1%} ({n_valid}/{n_input}). "
                    f"{n_unknown} unknown genes will be zero vectors. "
                    f"Consider checking gene name formats."
                )
            else:
                logger.debug(
                    f"Gene alignment: {n_valid}/{n_input} ({alignment_rate:.1%}) genes matched. "
                    f"{n_unknown} unknown genes → zero vectors."
                )

        # Build projection matrix W: [n_input, embedding_dim]
        # Initialize with zeros (unknown genes remain zero)
        W = torch.zeros(n_input, self.config.embedding_dim, device=device, dtype=expression.dtype)

        # Fill in valid gene embeddings
        valid_indices = model_indices[valid_mask]
        W[valid_mask] = self.gene_embeddings[valid_indices].to(dtype=expression.dtype)

        # =====================================================================
        # 2. Weighted Aggregation
        # =====================================================================
        # Compute: cell_embedding = Expression @ Gene_Embeddings

        # Matrix multiplication: [batch, n_input] @ [n_input, dim] -> [batch, dim]
        raw_embedding = torch.mm(expression, W)

        # =====================================================================
        # 3. Normalization
        # =====================================================================

        if self.config.aggregation_mode == "wa":
            # Weighted Average: Normalize by total expression per cell
            cell_totals = expression.sum(dim=1, keepdim=True)  # [batch, 1]

            # Handle edge case: cells with zero total expression
            zero_cells = (cell_totals == 0).squeeze()
            if zero_cells.any():
                logger.warning(
                    f"{zero_cells.sum().item()} cells have zero total expression. "
                    f"Their embeddings will be zero vectors."
                )

            cell_totals = torch.clamp(cell_totals, min=1e-8)
            cell_embedding = raw_embedding / cell_totals

        elif self.config.aggregation_mode == "aa":
            # Arithmetic Average: Divide by number of valid genes
            # Better: only divide by number of expressed genes per cell
            n_expressed = (expression > 0).sum(dim=1, keepdim=True).float()  # [batch, 1]
            n_expressed = torch.clamp(n_expressed, min=1.0)  # At least 1 to avoid division by zero
            cell_embedding = raw_embedding / n_expressed

        else:
            raise ValueError(
                f"Unknown aggregation mode: {self.config.aggregation_mode}. "
                f"Supported modes: ['wa', 'aa']"
            )

        # Check output validity
        if torch.isnan(cell_embedding).any():
            logger.error("Output contains NaN values!")
        if torch.isinf(cell_embedding).any():
            logger.error("Output contains Inf values!")

        return scELMoOutput(cell_embedding=cell_embedding)

    def load_weights(self, path: str, strict: bool = True):
        """Load gene embeddings from a pickle file.

        The pickle file is expected to contain a dictionary:
        {
            'embeddings': np.ndarray or Tensor [num_genes, embedding_dim],
            'gene_list': List[str]  (Should match config.gene_names order)
        }

        Args:
            path (str): Path to the pickle file.
            strict (bool, optional): If True, raises error on gene list mismatch.
                If False, only logs warning. Defaults to True.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If pickle format is invalid.
            ValueError: If embeddings shape doesn't match.
            ValueError: If strict=True and gene lists don't match.
        """
        logger.info(f"Loading scELMo embeddings from {path}...")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load pickle file: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"Invalid pickle format. Expected dict, got {type(data)}.")

        if "embeddings" not in data:
            raise ValueError(
                "Invalid pickle format. Missing 'embeddings' key. "
                f"Found keys: {list(data.keys())}"
            )

        # Verify gene list consistency
        if "gene_list" in data:
            loaded_genes = data["gene_list"]

            if len(loaded_genes) != len(self.config.gene_names):
                msg = (
                    f"Gene list length mismatch: "
                    f"Loaded {len(loaded_genes)} genes, "
                    f"Config has {len(self.config.gene_names)} genes."
                )
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

            # Check order
            if loaded_genes != self.config.gene_names:
                mismatches = [
                    (i, l, c)
                    for i, (l, c) in enumerate(zip(loaded_genes, self.config.gene_names))
                    if l != c
                ]
                if mismatches:
                    msg = (
                        f"Gene list order mismatch at {len(mismatches)} positions. "
                        f"First mismatch at index {mismatches[0][0]}: "
                        f"'{mismatches[0][1]}' vs '{mismatches[0][2]}'. "
                        f"Embeddings might be misaligned!"
                    )
                    if strict:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
        else:
            logger.warning(
                "Pickle file doesn't contain 'gene_list' field. "
                "Cannot verify gene order alignment."
            )

        # Load embeddings
        embeddings = data["embeddings"]
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        else:
            embeddings = embeddings.float()

        # Shape validation
        expected_shape = self.gene_embeddings.shape
        if embeddings.shape != expected_shape:
            raise ValueError(f"Shape mismatch: Expected {expected_shape}, got {embeddings.shape}")

        # Check data validity
        if torch.isnan(embeddings).any():
            raise ValueError("Loaded embeddings contain NaN values!")
        if torch.isinf(embeddings).any():
            raise ValueError("Loaded embeddings contain Inf values!")

        # Update buffer
        self.gene_embeddings.copy_(embeddings.to(self.gene_embeddings.device))
        logger.info(
            f"✓ Embeddings loaded successfully: "
            f"{embeddings.shape[0]} genes × {embeddings.shape[1]} dims"
        )

    @classmethod
    def from_pretrained(
        cls,
        embedding_path: str,
        gene_list: list[str] | None = None,
        embedding_dim: int = 1536,
        aggregation_mode: str = "wa",
        device: str = "cpu",
    ) -> scELMoModel:
        """Create scELMo model from pretrained embedding file.

        This is a convenience method that loads embeddings and automatically
        extracts gene names if not provided.

        Args:
            embedding_path (str): Path to pickle file with embeddings.
            gene_list (Optional[List[str]]): Gene names. If None, extracted from pickle.
            embedding_dim (int, optional): Embedding dimension. Defaults to 1536.
            aggregation_mode (str, optional): Aggregation mode. Defaults to 'wa'.
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            scELMoModel: Initialized model with loaded embeddings.

        Raises:
            FileNotFoundError: If embedding_path doesn't exist.
            ValueError: If pickle doesn't contain gene_list and none provided.
        """
        # Load pickle
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        with open(embedding_path, "rb") as f:
            data = pickle.load(f)

        # Extract gene list
        if gene_list is None:
            if "gene_list" not in data:
                raise ValueError(
                    "gene_list not provided and pickle doesn't contain 'gene_list'. "
                    "Cannot determine gene vocabulary."
                )
            gene_list = data["gene_list"]
            logger.info(f"Extracted {len(gene_list)} genes from embedding file")

        # Extract embeddings
        embeddings = data["embeddings"]
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

        # Create config
        from perturblab.models.scelmo.config import scELMoConfig

        config = scELMoConfig(
            gene_names=gene_list,
            embedding_dim=embedding_dim,
            aggregation_mode=aggregation_mode,
        )

        # Create model
        model = cls(config, gene_embeddings=embeddings, device=device)

        return model

    def get_gene_embedding(self, gene_name: str) -> torch.Tensor | None:
        """Get embedding for a specific gene.

        Args:
            gene_name (str): Gene symbol to query.

        Returns:
            Optional[torch.Tensor]: Gene embedding vector [embedding_dim] or None if unknown.
        """
        if gene_name not in self.vocab:
            logger.warning(f"Gene '{gene_name}' not in vocabulary")
            return None

        idx = self.vocab[gene_name]
        return self.gene_embeddings[idx]

    def get_coverage(self, gene_list: list[str]) -> dict[str, float | int | list[str]]:
        """Calculate coverage statistics for a gene list.

        Useful for understanding how well a dataset's genes align with the model.

        Args:
            gene_list (List[str]): List of gene names to check.

        Returns:
            dict: Statistics including:
                - n_total (int): Total genes in input
                - n_matched (int): Genes found in vocab
                - n_unknown (int): Genes not in vocab
                - coverage_rate (float): n_matched / n_total
                - unknown_genes (List[str]): List of unknown genes (up to 20)
        """
        indices = self.vocab.lookup_indices(gene_list, fallback=-1)

        n_total = len(gene_list)
        n_matched = sum(1 for idx in indices if idx >= 0)
        n_unknown = n_total - n_matched

        unknown_genes = [gene for gene, idx in zip(gene_list, indices) if idx < 0]

        return {
            "n_total": n_total,
            "n_matched": n_matched,
            "n_unknown": n_unknown,
            "coverage_rate": n_matched / n_total if n_total > 0 else 0.0,
            "unknown_genes": unknown_genes[:20],  # Limit to 20 for display
        }
