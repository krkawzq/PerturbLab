"""Ensembl genome annotation database downloader.

Downloads GTF/GFF files and gene metadata from Ensembl FTP server.
Supports automatic version detection and caching.
"""

import re
import gzip
from typing import Optional, Literal
from pathlib import Path

import requests

from .._base import BaseDownloader
from perturblab.utils import get_logger

logger = get_logger()


class Ensembl:
    """Ensembl API constants and species definitions."""
    
    # Base URLs
    BASE_FTP = "https://ftp.ensembl.org/pub"
    CURRENT_GTF = f"{BASE_FTP}/current_gtf"
    
    # Common species
    SPECIES_HUMAN = "homo_sapiens"
    SPECIES_MOUSE = "mus_musculus"
    SPECIES_RAT = "rattus_norvegicus"
    
    # GTF file types
    GTF_FULL = "gtf"
    GTF_CHR = "chr.gtf"
    GTF_ABINITIO = "abinitio.gtf"


class EnsemblDownloader(BaseDownloader):
    """Downloader for Ensembl genome annotations.
    
    Downloads GTF files from Ensembl FTP.
    Automatically detects latest release version.
    
    Example:
        >>> downloader = EnsemblDownloader()
        >>> path = downloader.download(species='homo_sapiens')
    """
    
    DEFAULT_CACHE_SUBDIR = 'reference/ensembl'
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Ensembl downloader."""
        super().__init__(cache_dir=cache_dir)
        self._latest_release: Optional[int] = None
    
    def get_latest_release(self) -> int:
        """Get the latest Ensembl release number.
        
        Returns:
            Latest release version number
        """
        if self._latest_release is not None:
            return self._latest_release
        
        logger.info(f"Fetching latest Ensembl release")
        
        response = requests.get(Ensembl.BASE_FTP, timeout=30)
        response.raise_for_status()
        
        # Use regex to find release numbers
        releases = re.findall(r'release-(\d+)', response.text)
        
        if not releases:
            raise ValueError("No release directories found")
        
        self._latest_release = max(map(int, releases))
        logger.info(f"Latest Ensembl release: {self._latest_release}")
        return self._latest_release
    
    def get_gtf_url(
        self,
        species: str = Ensembl.SPECIES_HUMAN,
        gtf_type: str = Ensembl.GTF_CHR,
        release: int | Literal['current'] = 'current',
    ) -> str:
        """Construct GTF file URL.
        
        Args:
            species: Species name
            gtf_type: GTF type
            release: Release number or 'current'
        
        Returns:
            Full URL to GTF file
        """
        if release == 'current':
            release_str = 'current'
            release_num = self.get_latest_release()
        else:
            release_str = f'release-{release}'
            release_num = release
        
        # Get directory listing
        gtf_dir = f"{Ensembl.BASE_FTP}/{release_str}_gtf/{species}/"
        
        response = requests.get(gtf_dir, timeout=30)
        response.raise_for_status()
        
        # Find GTF file using regex
        escaped_gtf_type = re.escape(gtf_type)
        pattern = rf'{species.capitalize()}\.[^"<>\s]+\.{release_num}\.{escaped_gtf_type}\.gz'
        
        match = re.search(pattern, response.text, re.IGNORECASE)
        
        if match:
            filename = match.group(0)
            if '/' in filename:
                filename = filename.split('/')[-1]
            return gtf_dir + filename
        
        raise ValueError(f"No GTF file found matching pattern: {pattern}")
    
    def download(
        self,
        species: str = Ensembl.SPECIES_HUMAN,
        gtf_type: str = Ensembl.GTF_CHR,
        release: int | Literal['current'] = 'current',
        decompress: bool = True,
        force_download: bool = False,
    ) -> Path:
        """Download Ensembl GTF file.
        
        Args:
            species: Species name
            gtf_type: GTF type
            release: Release number or 'current'
            decompress: Whether to decompress .gz file
            force_download: Whether to force re-download
        
        Returns:
            Path to downloaded GTF file
        
        Example:
            >>> downloader = EnsemblDownloader()
            >>> path = downloader.download(species='homo_sapiens')
        """
        # Get URL
        url = self.get_gtf_url(species, gtf_type, release)
        filename = url.split('/')[-1]
        
        # Generate cache filename
        if decompress and filename.endswith('.gz'):
            cache_filename = filename[:-3]
        else:
            cache_filename = filename
        
        logger.info(f"Downloading Ensembl GTF: {cache_filename}")
        
        # Use parent's ensure_cached method
        return self.ensure_cached(
            url=url,
            filename=cache_filename,
            force_download=force_download
        )


# Singleton instance
_default_downloader: Optional[EnsemblDownloader] = None


def get_default_downloader() -> EnsemblDownloader:
    """Get default Ensembl downloader (singleton).
    
    Returns:
        Default EnsemblDownloader instance
    """
    global _default_downloader
    if _default_downloader is None:
        _default_downloader = EnsemblDownloader()
    return _default_downloader


def download_ensembl(
    species: str = Ensembl.SPECIES_HUMAN,
    gtf_type: str = Ensembl.GTF_CHR,
    release: int | Literal['current'] = 'current',
    decompress: bool = True,
    force_download: bool = False,
) -> Path:
    """Download Ensembl GTF file using default downloader.
    
    Args:
        species: Species name
        gtf_type: GTF type
        release: Release number or 'current'
        decompress: Whether to decompress .gz file
        force_download: Whether to force re-download
    
    Returns:
        Path to downloaded GTF file
    
    Example:
        >>> from perturblab.data.downloader import download_ensembl, Ensembl
        >>> 
        >>> # Download latest human GTF
        >>> path = download_ensembl(species=Ensembl.SPECIES_HUMAN)
    """
    downloader = get_default_downloader()
    return downloader.download(
        species=species,
        gtf_type=gtf_type,
        release=release,
        decompress=decompress,
        force_download=force_download,
    )
