"""Gene-specific vocabulary class with standardization support."""

from ._vocab import Vocab
from typing import Callable, Literal
import pandas as pd


class GeneVocab(Vocab):
    """Vocabulary specialized for gene names with standardization utilities.
    
    Extends Vocab with gene-specific operations like name standardization
    and construction from AnnData objects.
    
    Args:
        genes: List of unique gene names.
        default_token: Token to return when index is out of bounds.
        default_index: Index to return when gene is not found.
    """
    
    DuplicatePolicy = Literal['error', 'first', 'last', 'remove']
    
    def __init__(
        self,
        genes: list[str],
        default_token: str = '<unk>',
        default_index: int = -1
    ):
        super().__init__(genes, default_token, default_index)
    
    @staticmethod
    def _handle_duplicate_genes(
        genes: list[str],
        duplicate_policy: 'GeneVocab.DuplicatePolicy' = 'error',
    ) -> list[str]:
        """Handle duplicate genes according to policy.
        
        Args:
            genes: List of gene names (may contain duplicates).
            duplicate_policy: How to handle duplicates:
                - 'error': Raise ValueError if duplicates found
                - 'first': Keep first occurrence of each duplicate
                - 'last': Keep last occurrence of each duplicate
                - 'remove': Remove all genes that have duplicates
        
        Returns:
            list[str]: List of gene names with duplicates handled.
        """
        if duplicate_policy == 'error':
            if len(genes) != len(set(genes)):
                duplicates = [
                    gene for gene in set(genes)
                    if genes.count(gene) > 1
                ]
                raise ValueError(f"Duplicate gene names found: {duplicates}")
            return genes
        
        elif duplicate_policy == 'first':
            # Keep first occurrence
            return list(dict.fromkeys(genes))
        
        elif duplicate_policy == 'last':
            # Keep last occurrence (reverse, dedupe, reverse back)
            return list(dict.fromkeys(reversed(genes)))[::-1]
        
        elif duplicate_policy == 'remove':
            # Remove all genes that appear more than once
            gene_counts = {}
            for gene in genes:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
            return [gene for gene in genes if gene_counts[gene] == 1]
        
        else:
            raise ValueError(
                f"Invalid duplicate_policy: {duplicate_policy}. "
                "Must be 'error', 'first', 'last', or 'remove'."
            )
    
    @classmethod
    def from_anndata(
        cls,
        adata,
        var_col: str | None = None,
        duplicate_policy: 'GeneVocab.DuplicatePolicy' = 'error',
        default_token: str = '<unk>',
        default_index: int = -1
    ) -> 'GeneVocab':
        """Construct GeneVocab from AnnData object.
        
        Args:
            adata: AnnData object.
            var_col: Column name in adata.var to use as gene names.
                If None, uses var_names.
            duplicate_policy: How to handle duplicate genes.
            default_token: Token to return when index is out of bounds.
            default_index: Index to return when gene is not found.
        
        Returns:
            GeneVocab: New vocabulary instance.
        """
        if var_col is None:
            genes = adata.var_names.tolist()
        else:
            if var_col not in adata.var_keys():
                raise ValueError(f"Column {var_col} not found in adata.var")
            genes = adata.var[var_col].tolist()
        
        genes = cls._handle_duplicate_genes(genes, duplicate_policy)
        return cls(genes, default_token, default_index)
    
    @classmethod
    def from_index(
        cls,
        index: pd.Index,
        duplicate_policy: 'GeneVocab.DuplicatePolicy' = 'error',
        default_token: str = '<unk>',
        default_index: int = -1
    ) -> 'GeneVocab':
        """Construct GeneVocab from pandas Index.
        
        Args:
            index: Pandas Index containing gene names.
            duplicate_policy: How to handle duplicate genes.
            default_token: Token to return when index is out of bounds.
            default_index: Index to return when gene is not found.
        
        Returns:
            GeneVocab: New vocabulary instance.
        """
        genes = index.tolist()
        genes = cls._handle_duplicate_genes(genes, duplicate_policy)
        return cls(genes, default_token, default_index)
    
    def standardize_gene_names(
        self,
        standardize_fn: Callable[[str], str] | None = None,
        duplicate_policy: 'GeneVocab.DuplicatePolicy' = 'error',
    ) -> None:
        """Standardize gene names using the provided function.
        
        Modifies the vocabulary in-place by applying standardize_fn to each
        gene name and handling any resulting duplicates.
        
        Args:
            standardize_fn: Function to standardize each gene name.
                If None, uses default human gene map standardization.
            duplicate_policy: How to handle duplicate genes after standardization.
        """
        # Use default standardization if not provided
        if standardize_fn is None:
            from ._gene_map import get_default_human_gene_map
            standardize_fn = get_default_human_gene_map().standardize
        
        # Apply standardization function
        standardized = [standardize_fn(gene) for gene in self.itos]
        
        # Handle duplicates and update all three data structures
        self.itos = self._handle_duplicate_genes(standardized, duplicate_policy)
        self.stoi = {gene: i for i, gene in enumerate(self.itos)}
        self.tokens = set(self.itos)
    
    def filter_genes(
        self,
        predicate: Callable[[str], bool]
    ) -> 'GeneVocab':
        """Create new GeneVocab with genes matching predicate.
        
        Args:
            predicate: Function that takes a gene name and returns True
                if the gene should be kept.
        
        Returns:
            GeneVocab: New vocabulary with filtered genes in original order.
        """
        filtered = [gene for gene in self.itos if predicate(gene)]
        return GeneVocab(filtered, self.default_token, self.default_index)
    
    def select_genes(
        self,
        genes_to_keep: set[str] | list[str],
        keep_order: bool = True
    ) -> 'GeneVocab':
        """Create new GeneVocab with specified genes.
        
        Args:
            genes_to_keep: Set or list of gene names to keep.
            keep_order: If True, maintain original order from this vocab.
                If False, use order from genes_to_keep.
        
        Returns:
            GeneVocab: New vocabulary with selected genes.
        """
        genes_set = set(genes_to_keep)
        
        if keep_order:
            filtered = [gene for gene in self.itos if gene in genes_set]
        else:
            filtered = [gene for gene in genes_to_keep if gene in self.stoi]
        
        return GeneVocab(filtered, self.default_token, self.default_index)
    
    def intersection(self, other: 'GeneVocab') -> 'GeneVocab':
        """Create vocabulary with genes present in both vocabularies.
        
        Args:
            other: Another GeneVocab instance.
        
        Returns:
            GeneVocab: New vocabulary with intersecting genes in this vocab's order.
        """
        common_genes = [gene for gene in self.itos if gene in other.tokens]
        return GeneVocab(common_genes, self.default_token, self.default_index)
    
    def union(self, other: 'GeneVocab') -> 'GeneVocab':
        """Create vocabulary with genes from both vocabularies.
        
        Args:
            other: Another GeneVocab instance.
        
        Returns:
            GeneVocab: New vocabulary with all genes (this vocab's genes first).
        """
        all_genes = self.itos + [gene for gene in other.itos if gene not in self.tokens]
        return GeneVocab(all_genes, self.default_token, self.default_index)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"GeneVocab(size={len(self)}, default_token='{self.default_token}')"
