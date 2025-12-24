"""HTTP/HTTPS downloader.

Pure namespace class providing HTTP download functionality with:
- Automatic retry with exponential backoff
- Progress tracking (tqdm integration)  
- Resumable downloads
- Streaming for large files
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from perturblab.utils import get_logger
from ._base import BaseDownloader, DownloadError, _prepare_path

# Try to import tqdm
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    _tqdm = None

logger = get_logger()

__all__ = ["HTTPDownloader"]


class HTTPDownloader(BaseDownloader):
    """HTTP/HTTPS downloader namespace.
    
    Provides HTTP download functionality with automatic retry, progress tracking,
    and resumable downloads. Cannot be instantiated - use static methods directly.
    
    Examples
    --------
    >>> from perturblab.io.download import HTTPDownloader
    >>> 
    >>> # Core download method
    >>> path = HTTPDownloader.download(
    ...     'https://example.com/data.csv',
    ...     '/tmp/data.csv',
    ...     resume=True,
    ...     show_progress=True
    ... )
    >>> 
    >>> # Get file info without downloading
    >>> info = HTTPDownloader.get_file_info('https://example.com/data.csv')
    >>> print(f"Size: {info['size_bytes'] / 1024 / 1024:.1f} MB")
    """
    
    @staticmethod
    def download(
        url: str,
        target_path: Path | str,
        *,
        resume: bool = False,
        show_progress: bool = True,
        chunk_size: int = 8192,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> Path:
        """Download file from HTTP/HTTPS URL.
        
        This is the core download method with all parameters.
        
        Parameters
        ----------
        url : str
            Download URL (required).
        target_path : Path or str
            Target file path (required).
        resume : bool, default=False
            Attempt to resume partial download if file exists.
        show_progress : bool, default=True
            Show progress bar (requires tqdm).
        chunk_size : int, default=8192
            Download chunk size in bytes.
        timeout : int, default=30
            Request timeout in seconds.
        max_retries : int, default=3
            Maximum number of retry attempts.
        backoff_factor : float, default=0.5
            Exponential backoff factor for retries.
        
        Returns
        -------
        Path
            Path to downloaded file.
        
        Raises
        ------
        DownloadError
            If download fails after all retries.
        
        Examples
        --------
        >>> # Basic download
        >>> path = HTTPDownloader.download(
        ...     'https://example.com/file.dat',
        ...     '/tmp/file.dat'
        ... )
        
        >>> # With resume and custom settings
        >>> path = HTTPDownloader.download(
        ...     'https://example.com/large.dat',
        ...     '/tmp/large.dat',
        ...     resume=True,
        ...     chunk_size=16384,
        ...     max_retries=5
        ... )
        """
        target_path = _prepare_path(target_path)
        
        logger.info(f"Downloading: {url}")
        logger.info(f"Target: {target_path}")
        
        try:
            # Create session with retry logic
            session = HTTPDownloader._create_session(max_retries, backoff_factor)
            
            # Determine start position for resume
            start_byte = 0
            if resume and target_path.exists():
                start_byte = target_path.stat().st_size
                logger.info(f"Resuming from byte {start_byte}")
            
            # Prepare headers
            headers = {}
            if start_byte > 0:
                headers['Range'] = f'bytes={start_byte}-'
            
            # Send request
            response = session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
            )
            response.raise_for_status()
            
            # Get total size
            total_size = None
            if 'Content-Length' in response.headers:
                total_size = int(response.headers['Content-Length'])
                if start_byte > 0:
                    total_size += start_byte
            
            # Check resume support
            if start_byte > 0 and response.status_code != 206:
                logger.warning("Server doesn't support resume, restarting download")
                start_byte = 0
            
            # Download with progress
            HTTPDownloader._download_stream(
                response,
                target_path,
                start_byte,
                total_size,
                chunk_size,
                show_progress and HAS_TQDM,
                url
            )
            
            logger.info(f"Download complete: {target_path}")
            return target_path
            
        except requests.RequestException as e:
            raise DownloadError(f"Failed to download from {url}: {e}")
        except Exception as e:
            raise DownloadError(f"Download error: {e}")
    
    @staticmethod
    def _create_session(max_retries: int, backoff_factor: float) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @staticmethod
    def _download_stream(
        response: requests.Response,
        target_path: Path,
        start_byte: int,
        total_size: int | None,
        chunk_size: int,
        show_progress: bool,
        url: str,
    ) -> None:
        """Download response stream to file with optional progress bar."""
        mode = 'ab' if start_byte > 0 else 'wb'
        
        start_time = time.time()
        downloaded_bytes = start_byte
        
        # Setup progress bar
        pbar = None
        if show_progress and total_size:
            pbar = _tqdm(
                total=total_size,
                initial=start_byte,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=Path(url).name,
            )
        
        try:
            with open(target_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    
                    if pbar:
                        pbar.update(len(chunk))
            
            # Log stats
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed_mbps = (downloaded_bytes - start_byte) / elapsed / (1024 * 1024)
                logger.info(
                    f"Downloaded {downloaded_bytes / 1024 / 1024:.1f} MB "
                    f"in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)"
                )
        finally:
            if pbar:
                pbar.close()
    
    @staticmethod
    def get_file_info(url: str, timeout: int = 30) -> dict:
        """Get file information without downloading.
        
        Parameters
        ----------
        url : str
            File URL.
        timeout : int, default=30
            Request timeout in seconds.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'size_bytes' (int): File size in bytes
            - 'content_type' (str): Content type
            - 'supports_resume' (bool): Whether server supports resume
            - 'url' (str): Original URL
        
        Examples
        --------
        >>> info = HTTPDownloader.get_file_info('https://example.com/file.dat')
        >>> print(f"Size: {info['size_bytes'] / 1024 / 1024:.1f} MB")
        """
        try:
            session = requests.Session()
            response = session.head(url, timeout=timeout)
            response.raise_for_status()
            
            return {
                'size_bytes': int(response.headers.get('Content-Length', 0)),
                'content_type': response.headers.get('Content-Type', 'unknown'),
                'supports_resume': 'Accept-Ranges' in response.headers,
                'url': url,
            }
        except requests.RequestException as e:
            logger.warning(f"Failed to get file info: {e}")
            return {
                'size_bytes': 0,
                'content_type': 'unknown',
                'supports_resume': False,
                'url': url,
            }
