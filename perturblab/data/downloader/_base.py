"""Base downloader class for all downloaders.

Provides common functionality for downloading and caching files.
All specific downloaders should inherit from BaseDownloader.
"""

import requests
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

from ._cache import CacheManager
from perturblab.utils import get_logger

logger = get_logger()


class BaseDownloader(CacheManager, ABC):
    """Base class for all downloaders.
    
    Provides common download functionality with progress logging.
    Subclasses should implement _generate_cache_filename if needed.
    
    Args:
        cache_dir: Directory for caching downloaded files
        namespace: Subdirectory name for organizing caches
    """
    
    # Class-level constants for subclasses to override
    DEFAULT_CACHE_SUBDIR: str = "downloads"  # Default subdirectory under ~/.perturblab/
    CHUNK_SIZE: int = 8192  # 8KB chunks
    PROGRESS_LOG_INTERVAL: int = 100 * 1024 * 1024  # Log every 100MB
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        namespace: Optional[str] = None
    ):
        """Initialize base downloader.
        
        Args:
            cache_dir: Custom cache directory. If None, uses default.
            namespace: Namespace for cache organization. If None, uses class name.
        """
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()
        
        super().__init__(cache_dir=cache_dir, namespace=namespace or "")
    
    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory for this downloader.
        
        Returns:
            Default cache directory path
        """
        return Path.home() / '.perturblab' / self.DEFAULT_CACHE_SUBDIR
    
    def _download_to_file(self, url: str, target_path: Path) -> None:
        """Download file from URL with progress logging.
        
        Args:
            url: Download URL
            target_path: Target file path
        """
        logger.info(f"Downloading from {url}")
        logger.info(f"Saving to {target_path}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            if total_size == 0:
                # Unknown size, download entire content
                f.write(response.content)
            else:
                # Stream download with progress
                self._stream_download(response, f, total_size)
        
        file_size_mb = target_path.stat().st_size / (1024 ** 2)
        logger.info(f"Download complete: {file_size_mb:.1f} MB")
    
    def _stream_download(self, response, file, total_size: int) -> None:
        """Stream download with periodic progress logging.
        
        Args:
            response: Response object from requests
            file: File object to write to
            total_size: Total file size in bytes
        """
        downloaded = 0
        
        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
            if not chunk:
                continue
            
            file.write(chunk)
            downloaded += len(chunk)
            
            # Log progress periodically
            if downloaded % self.PROGRESS_LOG_INTERVAL < self.CHUNK_SIZE:
                progress = (downloaded / total_size) * 100
                mb_downloaded = downloaded / (1024 ** 2)
                logger.info(f"Progress: {progress:.1f}% ({mb_downloaded:.1f} MB)")
    
    @abstractmethod
    def download(self, *args, **kwargs) -> Path:
        """Download and cache a file.
        
        This method must be implemented by subclasses.
        Should return the path to the cached file.
        """
        pass

