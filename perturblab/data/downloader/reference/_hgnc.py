"""HGNC gene nomenclature database downloader.

Downloads gene metadata from HUGO Gene Nomenclature Committee (HGNC).
Supports custom column selection and automatic caching.
"""

import io
import time
from typing import Optional
from pathlib import Path

import pandas as pd
import requests

from .._base import BaseDownloader
from perturblab.utils import get_logger

logger = get_logger()


class HGNC:
    """HGNC API constants and column definitions."""
    
    # API endpoint
    BASE_URL = "https://www.genenames.org/cgi-bin/download/custom"
    
    # Column names (internal HGNC field IDs)
    COL_HGNC_ID = "gd_hgnc_id"
    COL_SYMBOL = "gd_app_sym"
    COL_NAME = "gd_app_name"
    COL_STATUS = "gd_status"
    COL_LOCUS_TYPE = "gd_locus_type"
    COL_LOCUS_GROUP = "gd_locus_group"
    COL_ALIASES = "gd_aliases"
    COL_ALIAS_NAMES = "gd_alias_name"
    COL_PREV_SYMBOLS = "gd_prev_sym"
    COL_PREV_NAMES = "gd_prev_name"
    COL_CHROMOSOME = "gd_pub_chrom_map"
    
    # External database IDs
    COL_ENSEMBL_ID = "gd_pub_ensembl_id"
    COL_REFSEQ_IDS = "gd_pub_refseq_ids"
    COL_NCBI_GENE_ID = "gd_pub_eg_id"
    
    # Default column set
    DEFAULT_COLS = [
        COL_HGNC_ID,
        COL_SYMBOL,
        COL_NAME,
        COL_STATUS,
        COL_LOCUS_TYPE,
        COL_ALIASES,
        COL_PREV_SYMBOLS,
        COL_ENSEMBL_ID,
        COL_REFSEQ_IDS,
    ]
    
    # Status options
    STATUS_APPROVED = "Approved"
    STATUS_WITHDRAWN = "Entry and symbol withdrawn"


class HGNCDownloader(BaseDownloader):
    """Downloader for HGNC gene nomenclature data.
    
    Downloads gene metadata from https://www.genenames.org
    
    Example:
        >>> downloader = HGNCDownloader()
        >>> df = downloader.download()
    """
    
    DEFAULT_CACHE_SUBDIR = 'reference/hgnc'
    
    def download(
        self,
        columns: Optional[list[str]] = None,
        status: list[str] | str = HGNC.STATUS_APPROVED,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download HGNC gene data.
        
        Args:
            columns: List of column IDs to download. Uses DEFAULT_COLS if None
            status: Gene status filter
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with gene data
        """
        cols = columns or HGNC.DEFAULT_COLS
        
        # Normalize status
        if isinstance(status, str):
            status = [status]
        
        # Generate cache filename
        status_str = "_".join(sorted(status)).replace(" ", "_")[:20]
        cache_filename = f"hgnc_{status_str}_{len(cols)}cols.tsv"
        
        if use_cache:
            # Check if cached
            cached_path = self.cache_dir / cache_filename
            if cached_path.exists():
                logger.info(f"Loading cached HGNC data: {cache_filename}")
                return self._parse_tsv(cached_path)
            
            # Download and cache
            logger.info("Downloading HGNC data...")
            self._download_hgnc(cols, status, cached_path)
            return self._parse_tsv(cached_path)
        else:
            # Download without cache
            return self._download_hgnc_direct(cols, status)
    
    def _download_hgnc(
        self,
        columns: list[str],
        status: list[str],
        target_path: Path
    ) -> None:
        """Download HGNC data to file.
        
        Args:
            columns: Column IDs
            status: Status filter
            target_path: Target file path
        """
        params = {
            "col": columns,
            "status": status,
            "hgnc_dbtag": "on",
            "order_by": "gd_app_sym_sort",
            "format": "text",
            "submit": "submit"
        }
        
        response = self._make_request(HGNC.BASE_URL, params)
        
        if not response.text.strip():
            raise ValueError("Downloaded HGNC data is empty")
        
        # Write to file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"Downloaded HGNC data to {target_path}")
    
    def _download_hgnc_direct(
        self,
        columns: list[str],
        status: list[str]
    ) -> pd.DataFrame:
        """Download HGNC data directly without caching.
        
        Args:
            columns: Column IDs
            status: Status filter
        
        Returns:
            DataFrame with gene data
        """
        params = {
            "col": columns,
            "status": status,
            "hgnc_dbtag": "on",
            "order_by": "gd_app_sym_sort",
            "format": "text",
            "submit": "submit"
        }
        
        logger.info(f"Downloading HGNC data (no cache)")
        response = self._make_request(HGNC.BASE_URL, params)
        
        if not response.text.strip():
            raise ValueError("Downloaded HGNC data is empty")
        
        df = pd.read_csv(
            io.StringIO(response.text),
            sep="\t",
            dtype=str,
            keep_default_na=False
        )
        df.columns = self._standardize_columns(df.columns)
        logger.info(f"Loaded {len(df)} genes from HGNC")
        return df
    
    def _parse_tsv(self, file_path: Path) -> pd.DataFrame:
        """Parse TSV file into DataFrame."""
        df = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            keep_default_na=False
        )
        df.columns = self._standardize_columns(df.columns)
        logger.info(f"Loaded {len(df)} genes from HGNC")
        return df
    
    @staticmethod
    def _standardize_columns(columns: pd.Index) -> list[str]:
        """Standardize column names."""
        return [col.lower().replace(" ", "_") for col in columns]
    
    def _make_request(
        self,
        url: str,
        params: dict,
        retries: int = 3,
    ) -> requests.Response:
        """Make HTTP request with retry logic."""
        for attempt in range(retries):
            try:
                return requests.get(url, params=params, timeout=60)
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)


# Singleton instance
_default_downloader: Optional[HGNCDownloader] = None


def get_default_downloader() -> HGNCDownloader:
    """Get default HGNC downloader (singleton).
    
    Returns:
        Default HGNCDownloader instance
    """
    global _default_downloader
    if _default_downloader is None:
        _default_downloader = HGNCDownloader()
    return _default_downloader


def download_hgnc(
    columns: Optional[list[str]] = None,
    status: list[str] | str = HGNC.STATUS_APPROVED,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download HGNC gene data using default downloader.
    
    Args:
        columns: List of column IDs. Uses DEFAULT_COLS if None
        status: Gene status filter
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with gene data
    
    Example:
        >>> from perturblab.data.downloader import download_hgnc, HGNC
        >>> 
        >>> # Use default columns
        >>> df = download_hgnc()
        >>> 
        >>> # Custom columns
        >>> df = download_hgnc(columns=[
        ...     HGNC.COL_SYMBOL,
        ...     HGNC.COL_ENSEMBL_ID,
        ...     HGNC.COL_ALIASES,
        ... ])
    """
    downloader = get_default_downloader()
    return downloader.download(
        columns=columns,
        status=status,
        use_cache=use_cache
    )
