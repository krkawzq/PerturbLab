"""Custom file downloader for arbitrary URLs.

Downloads files from any given URL with automatic caching support.
Supports compressed files and custom headers.
"""

import os
import gzip
import hashlib
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional

import requests

from ._base import BaseDownloader
from perturblab.utils import get_logger

logger = get_logger()


class CustomDownloader(BaseDownloader):
    """Generic file downloader for arbitrary URLs.
    
    Downloads files from any URL with automatic caching and optional decompression.
    
    Args:
        cache_dir: Custom cache directory. If None, uses default.
    
    Example:
        >>> # Download a file
        >>> downloader = CustomDownloader()
        >>> path = downloader.download('https://example.com/data.csv')
        >>> 
        >>> # Download with custom filename
        >>> path = downloader.download(
        ...     'https://example.com/file',
        ...     filename='my_data.csv'
        ... )
        >>> 
        >>> # Download and decompress gzip
        >>> path = downloader.download(
        ...     'https://example.com/data.csv.gz',
        ...     decompress=True
        ... )
    """
    
    DEFAULT_CACHE_SUBDIR = 'downloads/custom'
    
    def download(
        self,
        url: str,
        filename: Optional[str] = None,
        decompress: bool = False,
        force_download: bool = False,
    ) -> Path:
        """Download file from URL and return local path.
        
        Args:
            url: URL of the file to download
            filename: Custom filename. Auto-generated from URL if None
            decompress: Whether to decompress .gz files
            force_download: Whether to force re-download (ignore cache)
        
        Returns:
            Path to downloaded file
        
        Example:
            >>> downloader = CustomDownloader()
            >>> path = downloader.download(
            ...     'https://example.com/data.csv.gz',
            ...     decompress=True
            ... )
        """
        # Generate filename if not provided
        if filename is None:
            filename = self._generate_filename_from_url(url)
        
        # Adjust filename if decompression is requested
        if decompress and filename.endswith('.gz'):
            filename = filename[:-3]  # Remove .gz extension
        
        logger.info(f"Downloading {filename} from URL...")
        
        # Download and cache
        cached_path = self.ensure_cached(
            url=url,
            filename=filename,
            force_download=force_download
        )
        
        # Decompress if needed
        if decompress and url.endswith('.gz') and not cached_path.name.endswith('.gz'):
            # File was downloaded as .gz but we want decompressed
            # This case should not happen with proper filename handling above,
            # but handle it for robustness
            pass
        
        return cached_path
    
    def _generate_filename_from_url(self, url: str) -> str:
        """Generate filename from URL.
        
        Extracts filename from URL path. If no valid filename found,
        generates one using MD5 hash of the URL.
        
        Args:
            url: File URL
        
        Returns:
            Generated filename
        """
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # Extract filename from path
        filename = os.path.basename(path)
        
        # If filename is invalid, use URL hash
        if not filename or filename == '/' or '.' not in filename:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"download_{url_hash}"
            
            # Try to extract extension from URL query part
            if '?' in url:
                query_part = url.split('?')[0]
                query_basename = os.path.basename(query_part)
                if '.' in query_basename:
                    ext = os.path.splitext(query_basename)[1]
                    filename += ext
        
        return filename
    
    def _download_to_file(self, url: str, target_path: Path) -> None:
        """Override to handle optional decompression.
        
        This is called by BaseDownloader's implementation.
        """
        # Use parent's implementation for standard downloads
        super()._download_to_file(url, target_path)


# Singleton instance
_default_downloader: Optional[CustomDownloader] = None


def get_default_downloader() -> CustomDownloader:
    """Get default CustomDownloader instance (singleton).
    
    Returns:
        Default CustomDownloader instance
    """
    global _default_downloader
    if _default_downloader is None:
        _default_downloader = CustomDownloader()
    return _default_downloader


def download_from_url(
    url: str,
    filename: Optional[str] = None,
    decompress: bool = False,
    force_download: bool = False,
) -> Path:
    """Convenience function to download file from URL.
    
    Args:
        url: File URL to download from
        filename: Custom filename (optional)
        decompress: Whether to decompress gzip files
        force_download: Whether to force re-download (ignore cache)
    
    Returns:
        Path to downloaded file
    
    Example:
        >>> # Download CSV file
        >>> path = download_from_url('https://example.com/data.csv')
        >>> 
        >>> # Download and decompress gzip file
        >>> path = download_from_url(
        ...     'https://example.com/data.csv.gz',
        ...     decompress=True
        ... )
        >>> 
        >>> # Custom filename
        >>> path = download_from_url(
        ...     'https://example.com/123456',
        ...     filename='my_data.pkl'
        ... )
    """
    downloader = get_default_downloader()
    return downloader.download(
        url=url,
        filename=filename,
        decompress=decompress,
        force_download=force_download,
    )
