"""PerturbBase dataset cards.

PerturbBase (http://www.perturbase.cn/) is a Chinese database containing 
122+ single-cell perturbation datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import anndata as ad

from ._base import DatasetCard
from perturblab.data.downloader import download_from_url
from perturblab.utils import get_logger

logger = get_logger()


@dataclass
class PerturbBaseCard(DatasetCard):
    """Dataset card for PerturbBase datasets.
    
    PerturbBase provides both raw and processed data.
    URLs are auto-constructed from data index and type.
    
    Attributes:
        ncbi_accession: NCBI project accession (e.g., 'PRJNA893678')
        data_index: Data index identifier (e.g., '201218_RNA')
        data_type: 'raw' or 'processed'
    
    Example:
        >>> card = PerturbBaseCard(
        ...     name='TF_atlas_2021',
        ...     ncbi_accession='PRJNA893678',
        ...     data_index='201218_RNA',
        ...     data_type='processed',
        ...     description='...',
        ...     citation='...',
        ... )
        >>> adata = card.load()
    """
    
    ncbi_accession: str = ''
    data_index: str = ''
    data_type: Literal['raw', 'processed'] = 'processed'
    
    def __post_init__(self):
        """Auto-construct URL and source from data_index."""
        # Set source
        object.__setattr__(self, 'source', 'PerturbBase')
        
        # Construct URL from data_index and type
        type_suffix = 'filter' if self.data_type == 'processed' else 'raw'
        url = f"http://www.perturbase.cn/download/{self.data_index}.{type_suffix}.tar.gz"
        object.__setattr__(self, 'url', url)
    
    def download(self, force: bool = False) -> Path:
        """Download PerturbBase dataset.
        
        Args:
            force: If True, re-download even if cached
        
        Returns:
            Path to cached tar.gz file
        """
        type_suffix = 'processed' if self.data_type == 'processed' else 'raw'
        filename = f"{self.name}_{type_suffix}.tar.gz"
        
        logger.info(f"Downloading {self.name} from PerturbBase ({self.data_type})")
        
        path = download_from_url(
            url=self.url,
            filename=filename,
            force_download=force
        )
        
        return Path(path)
    
    def load(self, force_download: bool = False, extract: bool = True, **kwargs) -> Path | ad.AnnData:
        """Download and optionally extract PerturbBase dataset.
        
        Args:
            force_download: If True, re-download even if cached
            extract: If True, extract tar.gz and load h5ad (if exists)
            **kwargs: Additional arguments
        
        Returns:
            If extract=False: Path to tar.gz file
            If extract=True: AnnData object (if h5ad found) or extracted directory path
        """
        cached_path = self.download(force=force_download)
        
        if not extract:
            return cached_path
        
        # Extract tar.gz
        import tarfile
        extract_dir = cached_path.parent / f"{cached_path.stem.replace('.tar', '')}_extracted"
        
        if not extract_dir.exists() or force_download:
            logger.info(f"Extracting {cached_path} to {extract_dir}")
            with tarfile.open(cached_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            logger.info(f"Extracted to {extract_dir}")
        else:
            logger.info(f"Using cached extraction: {extract_dir}")
        
        # Try to find and load h5ad file
        h5ad_files = list(extract_dir.glob('**/*.h5ad'))
        if h5ad_files:
            h5ad_path = h5ad_files[0]
            logger.info(f"Loading AnnData from {h5ad_path}")
            adata = ad.read_h5ad(h5ad_path)
            logger.info(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
            return adata
        
        logger.warning(f"No h5ad file found in extraction. Returning directory path.")
        return extract_dir

