"""Gene-specific vocabulary class with standardization support."""

from collections import Counter
from collections.abc import Callable
from typing import Literal

import pandas as pd
from anndata import AnnData

from ._vocab import Vocab


class GeneVocab(Vocab):
    """Vocabulary specialized for gene names with standardization utilities.

    Extends Vocab with gene-specific operations like name standardization
    and construction from AnnData objects.

    Args:
        genes: List of unique gene names.
        default_token: Token to return when index is out of bounds.
        default_index: Index to return when gene is not found.
        duplicate_policy: Strategy to handle duplicate gene names.
    """

    DuplicatePolicy = Literal["error", "first", "last", "remove"]

    def __init__(
        self,
        genes: list[str],
        default_token: str = "<unk>",
        default_index: int = -1,
        duplicate_policy: DuplicatePolicy = "error",
    ):
        # Clean genes before passing to parent constructor
        clean_genes = self._handle_duplicate_genes(genes, duplicate_policy)
        super().__init__(clean_genes, default_token, default_index)

    @staticmethod
    def _handle_duplicate_genes(
        genes: list[str],
        duplicate_policy: "GeneVocab.DuplicatePolicy" = "error",
    ) -> list[str]:
        """Handle duplicate genes according to policy.

        Args:
            genes: List of gene names (may contain duplicates).
            duplicate_policy: How to handle duplicates:
                - 'error': Raise ValueError if duplicates found.
                - 'first': Keep first occurrence of each duplicate.
                - 'last': Keep last occurrence of each duplicate.
                - 'remove': Remove all genes that have duplicates.

        Returns:
            List[str]: List of gene names with duplicates handled.
        """
        if duplicate_policy == "error":
            if len(genes) != len(set(genes)):
                # Optimization: Use Counter for O(N) duplicate detection instead of list.count() O(N^2)
                counts = Counter(genes)
                duplicates = [gene for gene, count in counts.items() if count > 1]
                raise ValueError(
                    f"Duplicate gene names found: {duplicates[:5]}... (Total {len(duplicates)})"
                )
            return genes

        elif duplicate_policy == "first":
            # Keep first occurrence, maintain order (Python 3.7+ guarantees dict order)
            return list(dict.fromkeys(genes))

        elif duplicate_policy == "last":
            # Keep last occurrence
            return list(dict.fromkeys(reversed(genes)))[::-1]

        elif duplicate_policy == "remove":
            # Remove all genes that appear more than once
            counts = Counter(genes)
            return [gene for gene in genes if counts[gene] == 1]

        else:
            raise ValueError(
                f"Invalid duplicate_policy: {duplicate_policy}. "
                "Must be 'error', 'first', 'last', or 'remove'."
            )

    @classmethod
    def from_anndata(
        cls,
        adata: AnnData,
        var_col: str | None = None,
        duplicate_policy: DuplicatePolicy = "error",
        default_token: str = "<unk>",
        default_index: int = -1,
        filter_empty: bool = True,
    ) -> "GeneVocab":
        """Construct GeneVocab from AnnData object with validation.

        Args:
            adata (AnnData): AnnData object.
            var_col (Optional[str]): Column name in adata.var to use as gene names.
                If None, uses adata.var_names.
            duplicate_policy (DuplicatePolicy): How to handle duplicate genes.
            default_token (str): Token to return when index is out of bounds.
            default_index (int): Index to return when gene is not found.
            filter_empty (bool, optional): If True, filters out empty gene names.
                Defaults to True.

        Returns:
            GeneVocab: New vocabulary instance.

        Raises:
            ValueError: If var_col doesn't exist or if no valid genes found.
        """
        if adata.n_vars == 0:
            raise ValueError("AnnData has no variables (genes)")

        if var_col is None:
            genes = adata.var_names.tolist()
        else:
            if var_col not in adata.var.keys():
                available = list(adata.var.keys())[:10]
                raise ValueError(
                    f"Column '{var_col}' not found in adata.var. "
                    f"Available columns: {available}..."
                )
            genes = adata.var[var_col].tolist()

        # Filter empty strings
        if filter_empty:
            original_len = len(genes)
            genes = [g for g in genes if g and isinstance(g, str) and g.strip()]
            if len(genes) < original_len:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Filtered out {original_len - len(genes)} empty/invalid gene names")

        if len(genes) == 0:
            raise ValueError("No valid gene names found after filtering")

        return cls(genes, default_token, default_index, duplicate_policy)

    @classmethod
    def from_index(
        cls,
        index: pd.Index,
        duplicate_policy: DuplicatePolicy = "error",
        default_token: str = "<unk>",
        default_index: int = -1,
    ) -> "GeneVocab":
        """Construct GeneVocab from pandas Index."""
        genes = index.tolist()
        return cls(genes, default_token, default_index, duplicate_policy)

    def standardize_gene_names(
        self,
        standardize_fn: Callable[[str], str] | None = None,
        duplicate_policy: DuplicatePolicy = "first",  # Changed default to 'first' to be safer
    ) -> None:
        """Standardize gene names in-place using the provided function.

        Warning:
            This operation changes the gene names and potentially the indices
            (if duplicates are merged/removed). Do not use this on a vocab
            that has already been used to tokenize data, unless you plan to
            re-tokenize.

        Args:
            standardize_fn: Function to standardize each gene name.
                If None, uses default human gene map standardization.
            duplicate_policy: How to handle duplicate genes created by standardization.
        """
        # Use default standardization if not provided
        if standardize_fn is None:
            from ._gene_map import get_default_human_gene_map

            standardize_fn = get_default_human_gene_map().standardize

        # Apply standardization function
        standardized = [standardize_fn(gene) for gene in self.itos]

        # Handle duplicates and update all structures
        # Note: This effectively rebuilds the vocab
        clean_genes = self._handle_duplicate_genes(standardized, duplicate_policy)

        # Re-initialize internal structures
        # Assuming Vocab uses standard self.itos/self.stoi/self.tokens attributes
        self.itos = clean_genes
        self.stoi = {gene: i for i, gene in enumerate(self.itos)}
        self.tokens = set(self.itos)

    def filter_genes(self, predicate: Callable[[str], bool]) -> "GeneVocab":
        """Create new GeneVocab with genes matching predicate."""
        filtered = [gene for gene in self.itos if predicate(gene)]
        return GeneVocab(filtered, self.default_token, self.default_index, duplicate_policy="error")

    def select_genes(
        self,
        genes_to_keep: set[str] | list[str],
        keep_order: bool = True,
        min_genes: int = 1,
    ) -> "GeneVocab":
        """Create new GeneVocab with specified genes.

        Args:
            genes_to_keep (Union[Set[str], List[str]]): Genes to select.
            keep_order (bool, optional): If True, preserves order from self.
                If False, uses order from genes_to_keep. Defaults to True.
            min_genes (int, optional): Minimum number of genes required.
                Raises error if fewer genes selected. Defaults to 1.

        Returns:
            GeneVocab: New vocabulary with selected genes.

        Raises:
            ValueError: If fewer than min_genes are found.
            ValueError: If no genes from genes_to_keep exist in vocabulary.
        """
        if not genes_to_keep:
            raise ValueError("genes_to_keep cannot be empty")

        genes_set = set(genes_to_keep)

        if keep_order:
            # Preserve order from self (Intersection logic)
            filtered = [gene for gene in self.itos if gene in genes_set]
        else:
            # Use order from input list (Reordering logic)
            # Only keep genes that actually exist in current vocab
            if not isinstance(genes_to_keep, list):
                genes_to_keep = list(genes_to_keep)
            filtered = [gene for gene in genes_to_keep if gene in self.tokens]

        if len(filtered) == 0:
            raise ValueError(
                f"No genes from genes_to_keep ({len(genes_to_keep)}) "
                f"exist in the current vocabulary ({len(self)})"
            )

        if len(filtered) < min_genes:
            raise ValueError(f"Only {len(filtered)} genes selected, but min_genes={min_genes}")

        return GeneVocab(filtered, self.default_token, self.default_index, duplicate_policy="first")

    def intersection(self, other: "GeneVocab") -> "GeneVocab":
        """Create vocabulary with genes present in both vocabularies."""
        common_genes = [gene for gene in self.itos if gene in other.tokens]
        return GeneVocab(
            common_genes, self.default_token, self.default_index, duplicate_policy="error"
        )

    def union(self, other: "GeneVocab") -> "GeneVocab":
        """Create vocabulary with genes from both vocabularies."""
        # self.itos order preserved, then append new genes from other
        all_genes = self.itos + [gene for gene in other.itos if gene not in self.tokens]
        return GeneVocab(
            all_genes, self.default_token, self.default_index, duplicate_policy="error"
        )

    def __repr__(self) -> str:
        return f"GeneVocab(size={len(self)}, default_token='{self.default_token}')"
