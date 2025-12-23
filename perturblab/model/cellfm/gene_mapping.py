# modified from https://github.com/biomed-AI/CellFM
# Original source: biomed-AI/CellFM
# License: See original repository for details

"""
Gene mapping utilities for CellFM.

This module provides gene name standardization and mapping functions
using HGNC (HUGO Gene Nomenclature Committee) data.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


class CellFMGeneMapper:
    """
    Gene mapper for CellFM using HGNC gene nomenclature.
    
    This class handles:
    - Loading gene information and HGNC data
    - Mapping gene aliases and previous symbols to approved names
    - Standardizing gene lists for CellFM vocabulary
    """
    
    def __init__(self, csv_dir: Optional[str] = None):
        """
        Initialize gene mapper.
        
        Args:
            csv_dir: Directory containing CSV files. If None, uses default location.
        """
        if csv_dir is None:
            # Default to source/csv directory
            csv_dir = Path(__file__).parent / "source" / "csv"
        else:
            csv_dir = Path(csv_dir)
        
        self.csv_dir = csv_dir
        self.gene_info = None
        self.geneset = None
        self.map_dict = None
        
        self._load_data()
    
    def _load_data(self):
        """Load gene information and HGNC mapping data."""
        # Load gene info
        gene_info_path = self.csv_dir / "expand_gene_info.csv"
        if not gene_info_path.exists():
            logger.warning(
                f"Gene info file not found: {gene_info_path}. "
                "Gene mapping will not be available."
            )
            return
        
        self.gene_info = pd.read_csv(gene_info_path, index_col=0, header=0)
        self.geneset = set(self.gene_info.index)
        logger.info(f"Loaded gene info: {len(self.geneset)} genes")
        
        # Load HGNC data for alias mapping
        hgcn_path = self.csv_dir / "updated_hgcn.tsv"
        if not hgcn_path.exists():
            logger.warning(
                f"HGNC file not found: {hgcn_path}. "
                "Alias mapping will not be available."
            )
            self.map_dict = {}
            return
        
        hgcn = pd.read_csv(hgcn_path, index_col=1, header=0, sep='\t')
        hgcn = hgcn[hgcn['Status'] == 'Approved']
        
        # Build alias mapping dictionary
        self.map_dict = {}
        alias = hgcn['Alias symbols']
        prev = hgcn['Previous symbols']
        
        # Map aliases to approved names
        for gene_name in hgcn.index:
            if pd.notna(alias.loc[gene_name]):
                for alias_name in str(alias.loc[gene_name]).split(', '):
                    if alias_name not in hgcn.index:
                        self.map_dict[alias_name] = gene_name
        
        # Map previous symbols to approved names
        for gene_name in hgcn.index:
            if pd.notna(prev.loc[gene_name]):
                for prev_name in str(prev.loc[gene_name]).split(', '):
                    if prev_name not in hgcn.index:
                        self.map_dict[prev_name] = gene_name
        
        logger.info(f"Built alias mapping: {len(self.map_dict)} mappings")
    
    def map_gene_list(
        self,
        gene_list: List[str],
        verbose: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Map a gene list to standardized HGNC-approved names.
        
        This function:
        1. Checks if gene is already in the approved gene set
        2. If not, tries to map via alias/previous symbol dictionary
        3. Returns successfully mapped genes and failed genes
        
        Args:
            gene_list: List of gene names (may include aliases/old names).
            verbose: Whether to log mapping details.
            
        Returns:
            Tuple of (mapped_genes, failed_genes):
                - mapped_genes: List of successfully mapped gene names
                - failed_genes: List of genes that couldn't be mapped
        """
        if self.geneset is None or self.map_dict is None:
            logger.warning("Gene mapping data not loaded. Returning original list.")
            return list(gene_list), []
        
        gene_list = [str(g) for g in gene_list]
        mapped_genes = []
        failed_genes = []
        
        for gene in gene_list:
            if gene in self.geneset:
                # Already approved name
                mapped_genes.append(gene)
            elif gene in self.map_dict and self.map_dict[gene] in self.geneset:
                # Map via alias/previous symbol
                approved_name = self.map_dict[gene]
                mapped_genes.append(approved_name)
                if verbose:
                    logger.debug(f"Mapped {gene} -> {approved_name}")
            else:
                # Cannot map
                failed_genes.append(gene)
                if verbose:
                    logger.debug(f"Failed to map: {gene}")
        
        if verbose or len(failed_genes) > 10:
            logger.info(
                f"Gene mapping: {len(mapped_genes)} mapped, "
                f"{len(failed_genes)} failed"
            )
        
        return mapped_genes, failed_genes
    
    def prepare_adata_with_mapping(
        self,
        adata: AnnData,
        max_genes: int = 2048,
        min_cells: int = 1,
        inplace: bool = False,
    ) -> AnnData:
        """
        Prepare AnnData with gene name mapping.
        
        This function:
        1. Maps gene names to HGNC-approved names
        2. Filters to genes in CellFM vocabulary
        3. Selects top genes by expression if needed
        
        Args:
            adata: Input AnnData object.
            max_genes: Maximum number of genes to keep.
            min_cells: Minimum cells expressing each gene.
            inplace: Whether to modify adata in place.
            
        Returns:
            Processed AnnData with mapped gene names.
        """
        if not inplace:
            adata = adata.copy()
        
        logger.info(f"Input data shape: {adata.shape}")
        
        # Map gene names
        original_genes = adata.var_names.tolist()
        mapped_genes, failed_genes = self.map_gene_list(original_genes)
        
        if len(failed_genes) > 0:
            logger.info(
                f"Could not map {len(failed_genes)}/{len(original_genes)} genes"
            )
            if len(failed_genes) <= 20:
                logger.debug(f"Failed genes: {failed_genes}")
        
        # Filter to successfully mapped genes
        if len(mapped_genes) == 0:
            raise ValueError("No genes could be mapped to CellFM vocabulary!")
        
        # Create mapping from original to mapped names
        gene_mapping = {}
        for orig, gene in zip(original_genes, adata.var_names):
            if orig in mapped_genes:
                idx = mapped_genes.index(orig)
                gene_mapping[gene] = mapped_genes[idx]
        
        # Keep only mapped genes
        keep_genes = [g for g in adata.var_names if g in gene_mapping]
        adata = adata[:, keep_genes]
        
        # Update gene names to mapped names
        adata.var_names = [gene_mapping[g] for g in adata.var_names]
        
        logger.info(f"After mapping: {adata.shape}")
        
        # Filter by min_cells
        from scipy.sparse import issparse
        if issparse(adata.X):
            gene_counts = np.array((adata.X > 0).sum(axis=0)).flatten()
        else:
            gene_counts = (adata.X > 0).sum(axis=0)
        
        keep_mask = gene_counts >= min_cells
        adata = adata[:, keep_mask]
        
        logger.info(f"After filtering (min_cells={min_cells}): {adata.shape}")
        
        # Select top genes by total expression if needed
        if adata.n_vars > max_genes:
            if issparse(adata.X):
                gene_totals = np.array(adata.X.sum(axis=0)).flatten()
            else:
                gene_totals = adata.X.sum(axis=0)
            
            top_gene_idx = np.argsort(gene_totals)[-max_genes:]
            top_gene_idx = np.sort(top_gene_idx)  # Keep order
            adata = adata[:, top_gene_idx]
            
            logger.info(f"Selected top {max_genes} genes by expression")
        
        logger.info(f"✓ Final shape: {adata.shape}")
        
        return adata
    
    def get_gene_info(self, gene_name: str) -> Optional[pd.Series]:
        """
        Get gene information for a specific gene.
        
        Args:
            gene_name: Gene name (will be mapped if needed).
            
        Returns:
            Gene information as pandas Series, or None if not found.
        """
        if self.gene_info is None:
            return None
        
        # Try direct lookup
        if gene_name in self.gene_info.index:
            return self.gene_info.loc[gene_name]
        
        # Try mapping
        if gene_name in self.map_dict:
            mapped_name = self.map_dict[gene_name]
            if mapped_name in self.gene_info.index:
                return self.gene_info.loc[mapped_name]
        
        return None
    
    @property
    def available_genes(self) -> List[str]:
        """Get list of all available genes in vocabulary."""
        if self.gene_info is None:
            return []
        return list(self.gene_info.index)
    
    @property
    def n_genes(self) -> int:
        """Get number of genes in vocabulary."""
        return len(self.available_genes)


# Global instance for convenience
_global_mapper = None


def get_gene_mapper(csv_dir: Optional[str] = None) -> CellFMGeneMapper:
    """
    Get global gene mapper instance.
    
    Args:
        csv_dir: Directory containing CSV files. If None, uses default.
        
    Returns:
        CellFMGeneMapper instance.
    """
    global _global_mapper
    if _global_mapper is None or csv_dir is not None:
        _global_mapper = CellFMGeneMapper(csv_dir=csv_dir)
    return _global_mapper

