# modified from https://github.com/biomed-AI/CellFM
# Original source: biomed-AI/CellFM
# License: See original repository for details

"""Gene mapping utilities for CellFM.

Standardizes gene names and handles mapping via HGNC data for CellFM models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


class CellFMGeneMapper:
    """Gene mapper for CellFM using HGNC gene nomenclature.
    
    Handles loading gene info, mapping aliases/previous symbols to approved names,
    and standardizing AnnData objects for CellFM vocabulary.
    """
    
    def __init__(self, csv_dir: Optional[str] = None):
        """Initialize the gene mapper.
        
        Args:
            csv_dir: Directory containing CSV files. Defaults to source/csv.
        """
        self.csv_dir = Path(csv_dir) if csv_dir else Path(__file__).parent / "source" / "csv"
        self.gene_info = None
        self.geneset = None
        self.map_dict = None
        
        self._load_data()
    
    def _load_data(self):
        """Load gene information and HGNC mapping data from CSV files."""
        gene_info_path = self.csv_dir / "expand_gene_info.csv"
        if not gene_info_path.exists():
            logger.warning("Gene info file not found: %s. Mapping disabled.", gene_info_path)
            return
        
        self.gene_info = pd.read_csv(gene_info_path, index_col=0)
        self.geneset = set(self.gene_info.index)
        logger.info("Loaded gene info: %d genes", len(self.geneset))
        
        hgcn_path = self.csv_dir / "updated_hgcn.tsv"
        if not hgcn_path.exists():
            logger.warning("HGNC file not found: %s. Alias mapping disabled.", hgcn_path)
            self.map_dict = {}
            return
        
        hgcn = pd.read_csv(hgcn_path, index_col=1, sep='\t')
        hgcn = hgcn[hgcn['Status'] == 'Approved']
        
        self.map_dict = {}
        alias_series = hgcn['Alias symbols']
        prev_series = hgcn['Previous symbols']
        
        # Map aliases and previous symbols to approved names
        for gene_name in hgcn.index:
            for series in [alias_series, prev_series]:
                val = series.loc[gene_name]
                if pd.notna(val):
                    for name in str(val).split(', '):
                        if name not in hgcn.index:
                            self.map_dict[name] = gene_name
        
        logger.info("Built alias mapping: %d mappings", len(self.map_dict))
    
    def map_gene_list(
        self,
        gene_list: List[str],
        verbose: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Map a list of genes to standardized HGNC names.
        
        Args:
            gene_list: List of gene names or aliases.
            verbose: If True, log mapping details.
            
        Returns:
            Tuple containing mapped approved names and list of failed genes.
        """
        if self.geneset is None or self.map_dict is None:
            logger.warning("Mapping data not loaded. Returning original list.")
            return list(gene_list), []
        
        mapped_genes, failed_genes = [], []
        for gene in [str(g) for g in gene_list]:
            if gene in self.geneset:
                mapped_genes.append(gene)
            elif gene in self.map_dict and self.map_dict[gene] in self.geneset:
                approved_name = self.map_dict[gene]
                mapped_genes.append(approved_name)
            else:
                failed_genes.append(gene)
        
        if verbose or len(failed_genes) > 10:
            logger.info("Mapping results: %d success, %d failed", len(mapped_genes), len(failed_genes))
            
        return mapped_genes, failed_genes
    
    def prepare_adata_with_mapping(
        self,
        adata: AnnData,
        max_genes: int = 2048,
        min_cells: int = 1,
        inplace: bool = False,
    ) -> AnnData:
        """Map gene names and filter AnnData for model vocabulary.
        
        Args:
            adata: Input AnnData object.
            max_genes: Maximum number of genes to retain.
            min_cells: Minimum cells required for each gene.
            inplace: Whether to modify the object in place.
            
        Returns:
            Processed AnnData with updated gene names and filtered variables.
        """
        if not inplace:
            adata = adata.copy()
        
        logger.info("Input data shape: %s", str(adata.shape))
        original_genes = adata.var_names.tolist()
        mapped_genes, failed_genes = self.map_gene_list(original_genes)
        
        if not mapped_genes:
            raise ValueError("No genes could be mapped to CellFM vocabulary.")
        
        if failed_genes:
            logger.info("Failed to map %d/%d genes", len(failed_genes), len(original_genes))
        
        # Filter and rename variables
        gene_mapping = {orig: mapped for orig, mapped in zip(original_genes, original_genes) if orig in mapped_genes}
        keep_genes = [g for g in adata.var_names if g in gene_mapping]
        adata = adata[:, keep_genes]
        adata.var_names = [mapped_genes[original_genes.index(g)] for g in adata.var_names]
        
        # Filter by expression density
        gene_counts = np.array((adata.X > 0).sum(axis=0)).flatten() if issparse(adata.X) else (adata.X > 0).sum(axis=0)
        adata = adata[:, gene_counts >= min_cells]
        
        # Subsample to max_genes based on total expression
        if adata.n_vars > max_genes:
            gene_totals = np.array(adata.X.sum(axis=0)).flatten() if issparse(adata.X) else adata.X.sum(axis=0)
            top_idx = np.sort(np.argsort(gene_totals)[-max_genes:])
            adata = adata[:, top_idx]
            logger.info("Retained top %d genes by expression", max_genes)
        
        logger.info("Final processed shape: %s", str(adata.shape))
        return adata
    
    def get_gene_info(self, gene_name: str) -> Optional[pd.Series]:
        """Fetch metadata for a specific gene.
        
        Args:
            gene_name: Gene name to query.
            
        Returns:
            Metadata series or None if not found.
        """
        if self.gene_info is None:
            return None
        
        if gene_name in self.gene_info.index:
            return self.gene_info.loc[gene_name]
        
        if gene_name in self.map_dict:
            mapped_name = self.map_dict[gene_name]
            if mapped_name in self.gene_info.index:
                return self.gene_info.loc[mapped_name]
        
        return None
    
    @property
    def available_genes(self) -> List[str]:
        """List of approved genes in the vocabulary."""
        return list(self.gene_info.index) if self.gene_info is not None else []
    
    @property
    def n_genes(self) -> int:
        """Total count of approved genes."""
        return len(self.available_genes)


_global_mapper = None


def get_gene_mapper(csv_dir: Optional[str] = None) -> CellFMGeneMapper:
    """Access the global CellFMGeneMapper instance.
    
    Args:
        csv_dir: Optional custom directory for mapping files.
        
    Returns:
        The singleton gene mapper instance.
    """
    global _global_mapper
    if _global_mapper is None or csv_dir is not None:
        _global_mapper = CellFMGeneMapper(csv_dir=csv_dir)
    return _global_mapper
