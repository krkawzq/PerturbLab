"""Base downloader interface and utilities.

This module defines the abstract base class for all downloaders with metaclass
enforcement. All downloaders are namespace classes that cannot be instantiated.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path

__all__ = ["BaseDownloader", "DownloadError", "_prepare_path"]


class DownloadError(Exception):
    """Exception raised when download fails."""
    pass


def _prepare_path(path: Path | str) -> Path:
    """Prepare target path for download.
    
    Expands user home (~), resolves relative paths, and creates parent directories.
    
    Parameters
    ----------
    path : Path or str
        Target file path.
    
    Returns
    -------
    Path
        Prepared absolute path with parent directories created.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class DownloaderMeta(ABCMeta):
    """Metaclass for downloaders.
    
    Enforces that:
    1. Downloaders cannot be instantiated
    2. Must implement abstract download method
    3. All methods should be static
    """
    
    def __call__(cls, *args, **kwargs):
        """Prevent instantiation of downloader classes."""
        raise TypeError(
            f"{cls.__name__} is a namespace class and cannot be instantiated. "
            f"Use {cls.__name__}.download(...) directly."
        )


class BaseDownloader(metaclass=DownloaderMeta):
    """Abstract base class for all downloaders.
    
    All downloaders are namespace classes providing static download methods.
    They cannot be instantiated and serve purely as namespaces for download functions.
    
    Requirements
    ------------
    1. Must implement abstract `download()` method as the core interface
    2. Can optionally provide specialized `download_xxx()` methods
    3. All methods should be @staticmethod
    4. Cannot be instantiated
    
    The core `download()` method should:
    - Accept positional args for required params (url, target_path, etc.)
    - Accept keyword-only args for optional params (use *, after required params)
    - Return Path to downloaded file/directory
    - Raise DownloadError on failure
    
    Design Pattern
    --------------
    ```python
    class MyDownloader(BaseDownloader):
        @staticmethod
        def download(
            source: str,              # Required positional
            target_path: Path | str,  # Required positional
            *,                        # Force keyword-only below
            option1: bool = False,    # Optional keyword-only
            option2: int = 10,        # Optional keyword-only
        ) -> Path:
            '''Core download method.'''
            # Implementation
            pass
        
        @staticmethod
        def download_from_api(api_key: str, target_path: Path | str) -> Path:
            '''Specialized download method.'''
            # Implementation - should call download() internally
            pass
    ```
    
    Usage
    -----
    >>> from perturblab.io.download import MyDownloader
    >>> 
    >>> # Core method
    >>> path = MyDownloader.download(
    ...     'https://example.com/file.dat',
    ...     '/tmp/file.dat',
    ...     option1=True,
    ...     option2=20
    ... )
    >>> 
    >>> # Specialized method
    >>> path = MyDownloader.download_from_api(
    ...     'my_api_key',
    ...     '/tmp/data.h5ad'
    ... )
    
    Examples
    --------
    >>> # Cannot instantiate
    >>> try:
    ...     downloader = MyDownloader()  # Raises TypeError
    ... except TypeError as e:
    ...     print(e)  # "MyDownloader is a namespace class..."
    """
    
    @staticmethod
    @abstractmethod
    def download(*args, **kwargs) -> Path:
        """Core download method (must be implemented by subclasses).
        
        This is the main download interface that all downloaders must provide.
        
        Signature Guidelines
        --------------------
        - First positional arg(s): Required source identifiers (url, file_id, etc.)
        - Last positional arg: target_path (Path | str)
        - After *: Keyword-only optional parameters
        - Returns: Path to downloaded file/directory
        
        Parameters
        ----------
        *args
            Required positional arguments (source, target_path, etc.).
        **kwargs
            Optional keyword-only arguments (resume, show_progress, etc.).
        
        Returns
        -------
        Path
            Path to downloaded file or directory.
        
        Raises
        ------
        DownloadError
            If download fails.
        
        Examples
        --------
        >>> class HTTPDownloader(BaseDownloader):
        ...     @staticmethod
        ...     def download(
        ...         url: str,
        ...         target_path: Path | str,
        ...         *,
        ...         resume: bool = False,
        ...         show_progress: bool = True
        ...     ) -> Path:
        ...         # Implementation
        ...         pass
        """
        pass
