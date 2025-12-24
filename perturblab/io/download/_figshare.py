"""Figshare downloader.

Provides download functionality for Figshare repository files with optional
token authentication for private files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from perturblab.utils import get_logger
from ._base import BaseDownloader, DownloadError, _prepare_path
from ._http import HTTPDownloader

logger = get_logger()

__all__ = ["FigshareDownloader"]

# Figshare API endpoints
FIGSHARE_API = "https://api.figshare.com/v2"
FIGSHARE_DOWNLOAD = "https://figshare.com/ndownloader/files"


class FigshareDownloader(BaseDownloader):
    """Figshare repository downloader namespace.
    
    Supports downloading files from Figshare with optional token authentication.
    Cannot be instantiated - use static methods directly.
    
    Examples
    --------
    >>> from perturblab.io.download import FigshareDownloader
    >>> 
    >>> # Download by file ID (public file)
    >>> path = FigshareDownloader.download(
    ...     file_id='12345678',
    ...     target_path='/tmp/data.h5ad'
    ... )
    >>> 
    >>> # Download by file ID (private file with token)
    >>> path = FigshareDownloader.download(
    ...     file_id='12345678',
    ...     target_path='/tmp/data.h5ad',
    ...     token='your_figshare_token'
    ... )
    >>> 
    >>> # Download by URL
    >>> path = FigshareDownloader.download_from_url(
    ...     'https://figshare.com/ndownloader/files/12345678',
    ...     '/tmp/data.h5ad'
    ... )
    """
    
    @staticmethod
    def download(
        file_id: str | int,
        target_path: Path | str,
        *,
        token: str | None = None,
        show_progress: bool = True,
    ) -> Path:
        """Download Figshare file by file ID.
        
        This is the core download method for Figshare.
        
        Parameters
        ----------
        file_id : str or int
            Figshare file ID (required).
        target_path : Path or str
            Target file path (required).
        token : str, optional
            Figshare API token for private file access.
        show_progress : bool, default=True
            Show download progress bar.
        
        Returns
        -------
        Path
            Path to downloaded file.
        
        Raises
        ------
        DownloadError
            If download fails.
        
        Examples
        --------
        >>> # Public file
        >>> path = FigshareDownloader.download(
        ...     '12345678',
        ...     '/tmp/data.h5ad'
        ... )
        
        >>> # Private file with token
        >>> path = FigshareDownloader.download(
        ...     '12345678',
        ...     '/tmp/data.h5ad',
        ...     token='your_token'
        ... )
        """
        target_path = _prepare_path(target_path)
        
        try:
            # Get file info
            file_info = FigshareDownloader.get_file_info(file_id, token=token)
            download_url = file_info['download_url']
            
            logger.info(f"Downloading Figshare file {file_id}: {file_info['name']}")
            
            # Download using HTTP downloader
            return HTTPDownloader.download(
                download_url,
                target_path,
                show_progress=show_progress
            )
            
        except Exception as e:
            raise DownloadError(
                f"Failed to download Figshare file {file_id}: {e}"
            )
    
    @staticmethod
    def download_from_url(
        url: str,
        target_path: Path | str,
        *,
        token: str | None = None,
        show_progress: bool = True,
    ) -> Path:
        """Download Figshare file from URL.
        
        Parameters
        ----------
        url : str
            Figshare download URL (required).
        target_path : Path or str
            Target file path (required).
        token : str, optional
            Figshare API token for private files.
        show_progress : bool, default=True
            Show download progress bar.
        
        Returns
        -------
        Path
            Path to downloaded file.
        
        Examples
        --------
        >>> path = FigshareDownloader.download_from_url(
        ...     'https://figshare.com/ndownloader/files/12345678',
        ...     '/tmp/data.h5ad'
        ... )
        """
        target_path = _prepare_path(target_path)
        
        logger.info(f"Downloading from Figshare URL: {url}")
        
        try:
            return HTTPDownloader.download(
                url,
                target_path,
                show_progress=show_progress
            )
        except Exception as e:
            raise DownloadError(f"Failed to download from {url}: {e}")
    
    @staticmethod
    def download_article(
        article_id: str | int,
        target_dir: Path | str,
        *,
        token: str | None = None,
        show_progress: bool = True,
    ) -> Path:
        """Download all files from a Figshare article.
        
        Parameters
        ----------
        article_id : str or int
            Figshare article ID (required).
        target_dir : Path or str
            Target directory for downloaded files (required).
        token : str, optional
            Figshare API token for private articles.
        show_progress : bool, default=True
            Show download progress bar.
        
        Returns
        -------
        Path
            Path to target directory containing downloaded files.
        
        Raises
        ------
        DownloadError
            If download fails.
        
        Examples
        --------
        >>> dir_path = FigshareDownloader.download_article(
        ...     '1234567',
        ...     '/tmp/article_files'
        ... )
        """
        target_dir = _prepare_path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get article files
            files = FigshareDownloader.get_article_files(article_id, token=token)
            
            if not files:
                raise DownloadError(f"No files found in article {article_id}")
            
            logger.info(
                f"Downloading Figshare article {article_id} ({len(files)} files)"
            )
            
            # Download each file
            for file_info in files:
                file_name = file_info['name']
                file_path = target_dir / file_name
                
                logger.info(f"  Downloading: {file_name}")
                
                HTTPDownloader.download(
                    file_info['download_url'],
                    file_path,
                    show_progress=show_progress
                )
            
            logger.info(f"Article download complete: {target_dir}")
            return target_dir
            
        except Exception as e:
            raise DownloadError(
                f"Failed to download article {article_id}: {e}"
            )
    
    @staticmethod
    def get_file_info(
        file_id: str | int,
        *,
        token: str | None = None
    ) -> dict[str, Any]:
        """Get file information from Figshare API.
        
        Parameters
        ----------
        file_id : str or int
            Figshare file ID.
        token : str, optional
            Figshare API token.
        
        Returns
        -------
        dict
            File information with keys:
            - 'id': File ID
            - 'name': File name
            - 'size': File size in bytes
            - 'download_url': Download URL
        
        Raises
        ------
        DownloadError
            If API request fails.
        
        Examples
        --------
        >>> info = FigshareDownloader.get_file_info('12345678')
        >>> print(f"Name: {info['name']}, Size: {info['size']} bytes")
        """
        url = f"{FIGSHARE_API}/file/{file_id}"
        
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return {
                'id': data['id'],
                'name': data['name'],
                'size': data['size'],
                'download_url': data['download_url'],
            }
            
        except requests.RequestException as e:
            raise DownloadError(f"Failed to get file info for {file_id}: {e}")
    
    @staticmethod
    def get_article_files(
        article_id: str | int,
        *,
        token: str | None = None
    ) -> list[dict[str, Any]]:
        """Get list of files in a Figshare article.
        
        Parameters
        ----------
        article_id : str or int
            Figshare article ID.
        token : str, optional
            Figshare API token.
        
        Returns
        -------
        list of dict
            List of file information dictionaries.
        
        Raises
        ------
        DownloadError
            If API request fails.
        
        Examples
        --------
        >>> files = FigshareDownloader.get_article_files('1234567')
        >>> for f in files:
        ...     print(f"File: {f['name']} ({f['size']} bytes)")
        """
        url = f"{FIGSHARE_API}/articles/{article_id}"
        
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return [
                {
                    'id': f['id'],
                    'name': f['name'],
                    'size': f['size'],
                    'download_url': f['download_url'],
                }
                for f in data.get('files', [])
            ]
            
        except requests.RequestException as e:
            raise DownloadError(
                f"Failed to get article files for {article_id}: {e}"
            )
