"""Base dataset card classes.

Defines the core card types for different file formats.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import anndata as ad

from perturblab.data.downloader import download_from_url
from perturblab.utils import get_logger

logger = get_logger()


class PerturbationType(Enum):
    """Types of perturbations."""
    CRISPR = "CRISPR"
    CRISPRI = "CRISPRi"
    CRISPRA = "CRISPRa"
    CHEMICAL = "Chemical"
    GENETIC = "Genetic"
    ORF = "ORF overexpression"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class DatasetCard(ABC):
    """Base class for dataset cards.
    
    A dataset card holds all metadata about a dataset and knows how to
    download and load it. This separates data description from download logic.
    
    Attributes:
        name: Dataset identifier
        url: Download URL
        description: Brief description
        citation: Citation information
        source: Data source name
        perturbation_col: Column name for perturbation labels
        control_label: Label for control cells
        n_cells: Number of cells
        n_genes: Number of genes
        perturbation_type: Type of perturbation
        cell_type: Cell type(s) used
        file_size_mb: Approximate file size in MB
    """
    
    # Required metadata
    name: str
    url: str
    description: str
    citation: str
    
    # Source (can be set by subclass)
    source: str = ''
    
    # Optional metadata
    perturbation_col: str = 'perturbation'
    control_label: str = 'ctrl'
    n_cells: Optional[int] = None
    n_genes: Optional[int] = None
    perturbation_type: Optional[PerturbationType] = None
    cell_type: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    @abstractmethod
    def download(self, force: bool = False) -> Path:
        """Download the dataset and return path to cached file."""
        pass
    
    @abstractmethod
    def load(self, force_download: bool = False, **kwargs) -> Any:
        """Download (if needed) and load the dataset."""
        pass
    
    def __str__(self) -> str:
        """Format card information."""
        lines = [f"Dataset: {self.name}"]
        lines.append(f"Source: {self.source}")
        
        if self.n_cells:
            lines.append(f"Cells: {self.n_cells:,}")
        if self.n_genes:
            lines.append(f"Genes: {self.n_genes:,}")
        if self.file_size_mb:
            lines.append(f"Size: {self.file_size_mb:.1f} MB")
        if self.perturbation_type:
            lines.append(f"Type: {self.perturbation_type}")
        if self.cell_type:
            lines.append(f"Cell Type: {self.cell_type}")
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.citation:
            lines.append(f"Citation: {self.citation}")
        
        lines.append(f"Perturbation column: '{self.perturbation_col}'")
        lines.append(f"Control label: '{self.control_label}'")
        
        return "\n  ".join(lines)


@dataclass
class H5ADDatasetCard(DatasetCard):
    """Dataset card for AnnData h5ad files.
    
    Example:
        >>> card = H5ADDatasetCard(...)
        >>> adata = card.load()
    """
    
    def download(self, force: bool = False) -> Path:
        """Download h5ad file."""
        filename = f"{self.name}.h5ad"
        
        logger.info(f"Downloading {self.name} from {self.source}")
        
        path = download_from_url(
            url=self.url,
            filename=filename,
            force_download=force
        )
        
        return Path(path)
    
    def load(self, force_download: bool = False, **kwargs) -> ad.AnnData:
        """Download and load as AnnData."""
        cached_path = self.download(force=force_download)
        
        logger.info(f"Loading {self.name} from {cached_path}")
        adata = ad.read_h5ad(cached_path)
        logger.info(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        
        return adata


@dataclass
class PickleDatasetCard(DatasetCard):
    """Dataset card for Python pickle files."""
    
    def download(self, force: bool = False) -> Path:
        """Download pickle file."""
        filename = f"{self.name}.pkl"
        
        logger.info(f"Downloading {self.name} pickle file")
        
        path = download_from_url(
            url=self.url,
            filename=filename,
            force_download=force
        )
        
        return Path(path)
    
    def load(self, force_download: bool = False, **kwargs) -> Any:
        """Download and load pickle file."""
        import pickle
        
        cached_path = self.download(force=force_download)
        
        logger.info(f"Loading {self.name} from {cached_path}")
        with open(cached_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded pickle data")
        return data


@dataclass
class OBODatasetCard(DatasetCard):
    """Dataset card for OBO ontology files."""
    
    def download(self, force: bool = False) -> Path:
        """Download OBO file."""
        filename = f"{self.name}.obo"
        
        logger.info(f"Downloading {self.name} OBO file")
        
        path = download_from_url(
            url=self.url,
            filename=filename,
            force_download=force
        )
        
        return Path(path)
    
    def load(self, force_download: bool = False, load_obsolete: bool = False, **kwargs) -> tuple:
        """Download and parse OBO file."""
        from perturblab.utils import read_obo
        
        cached_path = self.download(force=force_download)
        
        logger.info(f"Parsing OBO file: {cached_path}")
        terms, dag = read_obo(str(cached_path), load_obsolete=load_obsolete)
        
        logger.info(f"Parsed {len(terms)} terms, {dag.n_edges} edges")
        return terms, dag

