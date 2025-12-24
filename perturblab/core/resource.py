"""
Resource abstraction for lazy loading local or remote data.

This module provides a unified interface for describing resources.
A Resource descriptor can simultaneously maintain a local path definition and a remote configuration.
Local paths take precedence, while remote configurations are used to fetch data into the cache manager if local data is missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from perturblab.io.cache import CacheManager, get_default_cache_manager
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["Resource"]

T = TypeVar("T")


class Resource(ABC, Generic[T]):
    """
    Abstract Descriptor for a Data Resource.

    A Resource describes *where* data is (Local Path and/or Remote Config) and
    *how* to load it into memory. It handles the lifecycle of checking local existence,
    downloading from remote if necessary, and loading into memory.

    Attributes:
        key (str): Unique identifier for the resource.
        local_path (Path | None): Explicit local path. If this exists, it takes precedence.
        remote_config (dict | None): Configuration for fetching the resource remotely.
        cache_manager (CacheManager): Manager for handling remote downloads.
    """

    def __init__(
        self,
        key: str,
        *,
        local_path: Union[str, Path, None] = None,
        remote_config: Optional[Dict[str, Any]] = None,
        is_directory: bool = False,
        cache_manager: Optional[CacheManager] = None,
    ):
        """
        Initialize the resource descriptor.

        Args:
            key: Unique identifier.
            local_path: Optional explicit path to the file on disk.
            remote_config: Optional dictionary configuring how to fetch the resource remotely.
                           (e.g. {'url': '...', 'token': '...'}).
            is_directory: Whether the resource is a directory (True) or file (False).
            cache_manager: The cache manager to use for remote downloads.
        """
        self._key = key
        self._local_path = Path(local_path).expanduser().resolve() if local_path else None
        self._remote_config = remote_config
        self._is_dir = is_directory
        self._cache_manager = cache_manager or get_default_cache_manager()

        # Validation: Must have at least one source
        if not self._local_path and not self._remote_config:
            raise ValueError(f"Resource '{key}' must have either a local_path or a remote_config.")

        # Internal state
        self._data: Optional[T] = None

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def key(self) -> str:
        return self._key

    @property
    def is_loaded(self) -> bool:
        """True if the data object is currently in memory."""
        return self._data is not None

    @property
    def is_materialized(self) -> bool:
        """True if the file exists locally (either explicit local path or cached)."""
        if self._local_path and self._local_path.exists():
            return True
        if self._remote_config:
            # Check if cache has it (without triggering download)
            return self._cache_manager.exists(self._get_cache_key())
        return False

    @property
    def path(self) -> Path:
        """
        Get the local filesystem path to the resource.

        Logic:
        1. If `local_path` is set and exists, validate and return it (calls _process_local).
        2. If `remote_config` is set, download via CacheManager (calls _process_remote).
        3. Raise FileNotFoundError if neither works.

        Returns
        -------
        Path
            Path to local file or directory.

        Raises
        ------
        FileNotFoundError
            If resource cannot be located.
        ValueError
            If file validation fails.
        """
        # 1. Explicit local path
        if self._local_path:
            if self._local_path.exists():
                # Validate local file
                if not self._check_file(self._local_path):
                    raise ValueError(f"Local file validation failed: {self._local_path}")

                # Call local processing hook
                self._process_local(self._local_path)

                return self._local_path
            elif not self._remote_config:
                # If we only have a local definition and it's missing, that's an error.
                raise FileNotFoundError(f"Local resource not found: {self._local_path}")
            else:
                logger.debug(f"Local path {self._local_path} missing, falling back to remote.")

        # 2. Remote resolution
        if self._remote_config:
            remote_path = self._resolve_remote()

            # Validate downloaded file
            if not self._check_file(remote_path):
                raise ValueError(f"Downloaded file validation failed: {remote_path}")

            # Call remote processing hook
            self._process_remote(remote_path)

            return remote_path

        raise FileNotFoundError(f"Resource '{self.key}' could not be located.")

    @property
    def data(self) -> T:
        """
        Get the data object in memory (Lazy Load).

        Triggers:
        1. `self.path` (which might trigger download).
        2. `self._load_from_disk`.

        Returns
        -------
        T
            Loaded data object.

        Examples
        --------
        >>> resource = AnnDataResource(key='data', local_path='file.h5ad')
        >>> adata = resource.data  # First access: loads from disk
        >>> adata2 = resource.data  # Second access: returns cached instance
        >>> assert adata is adata2  # Same object
        """
        if self._data is None:
            target_path = self.path  # Ensures file exists locally

            logger.info(f"Loading resource '{self.key}' from {target_path}...")
            try:
                self._data = self._load_from_disk(target_path)
            except Exception as e:
                logger.error(f"Failed to load resource '{self.key}': {e}")
                raise e

        return self._data

    def load(self) -> T:
        """
        Load the resource data (alias for .data property).

        This method provides a functional interface equivalent to accessing
        the .data property. Use whichever style you prefer.

        Returns
        -------
        T
            Loaded data object.

        Examples
        --------
        >>> resource = AnnDataResource(key='data', local_path='file.h5ad')
        >>> adata = resource.load()  # Functional style
        >>> # Equivalent to:
        >>> adata = resource.data  # Property style
        """
        return self.data

    # =========================================================================
    # Management Methods
    # =========================================================================

    def clear(self) -> None:
        """Clear the in-memory data object to free RAM.

        This only clears the cached Python object, not the file on disk.

        Examples
        --------
        >>> resource = AnnDataResource(key='data', local_path='file.h5ad')
        >>> adata = resource.load()  # Loads data
        >>> print(resource.is_loaded)  # True
        >>> resource.clear()  # Free memory
        >>> print(resource.is_loaded)  # False
        >>> adata2 = resource.load()  # Reloads from disk
        """
        if self._data is not None:
            logger.debug(f"Clearing in-memory data for: {self.key}")
        self._data = None

    def invalidate_cache(self) -> None:
        """
        Remove the downloaded file from the cache.
        Does NOT affect explicit `local_path` files.

        This is useful when you want to force a fresh download on next access.

        Examples
        --------
        >>> resource = AnnDataResource(
        ...     key='data',
        ...     download_config={'downloader': 'HTTPDownloader', 'url': '...'}
        ... )
        >>> path1 = resource.path  # Downloads and caches
        >>> resource.invalidate_cache()  # Removes from cache
        >>> path2 = resource.path  # Re-downloads
        """
        if self._remote_config:
            cache_key = self._get_cache_key()
            self._cache_manager.invalidate(cache_key)
            # Also clear memory since backing file was deleted
            self.clear()
            logger.info(f"Cache invalidated for: {self.key}")

    # =========================================================================
    # Abstract Methods (Subclass Implementation)
    # =========================================================================

    @abstractmethod
    def _fetch_remote(self, config: Dict[str, Any], target_path: Path) -> None:
        """
        Logic to download the resource.

        Args:
            config: The `remote_config` dictionary.
            target_path: The destination path (temp path provided by CacheManager).
        """
        pass

    @abstractmethod
    def _load_from_disk(self, path: Path) -> T:
        """
        Logic to load the file into memory.

        Args:
            path: The local path to the file/directory.
        """
        pass

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _resolve_remote(self) -> Path:
        """Interacts with CacheManager to fetch remote data."""
        cache_key = self._get_cache_key()

        # Define the callback for CacheManager
        def _download_callback(temp_path: Path):
            logger.info(f"Downloading remote resource '{self.key}'...")
            self._fetch_remote(self._remote_config, temp_path)

        # Atomic get or create
        return self._cache_manager.get_or_create(
            key=cache_key,
            create_fn=_download_callback,
            is_directory=self._is_dir,
            metadata=self._remote_config,
        )

    def _get_cache_key(self) -> str:
        """Generate a stable cache key based on the resource key.

        Subclasses can override to customize cache key generation.
        Default implementation uses the resource key directly.

        Returns
        -------
        str
            Cache key for this resource.

        Examples
        --------
        >>> class CustomResource(Resource[MyData]):
        ...     def _get_cache_key(self) -> str:
        ...         # Include version in cache key
        ...         version = self._remote_config.get('version', 'v1')
        ...         return f"{self.key}_{version}.dat"
        """
        return self.key

    def to_dict(self) -> dict[str, Any]:
        """Serialize resource descriptor to dictionary.

        Returns
        -------
        dict
            Serializable dictionary representation.

        Examples
        --------
        >>> resource = AnnDataResource(
        ...     key='data',
        ...     local_path='/data/file.h5ad',
        ...     download_config={'downloader': 'HTTPDownloader', 'url': '...'}
        ... )
        >>> config = resource.to_dict()
        >>> # Can be saved to JSON, YAML, etc.
        """
        return {
            "class": self.__class__.__name__,
            "key": self.key,
            "local_path": str(self._local_path) if self._local_path else None,
            "remote_config": self._remote_config,
            "is_directory": self._is_dir,
        }

    def get_info(self) -> dict[str, Any]:
        """Get resource information summary.

        Returns
        -------
        dict
            Dictionary with resource status and metadata.

        Examples
        --------
        >>> resource = AnnDataResource(key='data', local_path='file.h5ad')
        >>> info = resource.get_info()
        >>> print(info['is_loaded'])  # False
        >>> adata = resource.load()
        >>> info = resource.get_info()
        >>> print(info['is_loaded'])  # True
        """
        info = {
            "key": self.key,
            "is_loaded": self.is_loaded,
            "is_materialized": self.is_materialized,
            "has_local_path": self._local_path is not None,
            "has_remote_config": self._remote_config is not None,
        }

        if self._local_path:
            info["local_path"] = str(self._local_path)
            info["local_exists"] = self._local_path.exists()

        if self._remote_config:
            info["remote_config"] = self._remote_config
            info["cache_key"] = self._get_cache_key()

        if self.is_materialized:
            try:
                path = self.path
                info["path"] = str(path)
                if path.exists():
                    if path.is_file():
                        info["size_bytes"] = path.stat().st_size
                        info["size_mb"] = path.stat().st_size / (1024 * 1024)
                    elif path.is_dir():
                        info["is_directory"] = True
            except Exception:
                pass

        return info

    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded else "lazy"

        sources = []
        if self._local_path:
            exists = "✓" if (self._local_path.exists()) else "✗"
            sources.append(f"local={exists}")
        if self._remote_config:
            sources.append("remote=✓")

        source_str = ", ".join(sources)
        return f"<{self.__class__.__name__} '{self.key}' [{source_str}] ({status})>"

    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()

    def _check_file(self, path: Path) -> bool:
        """Hook to validate file/directory before loading.

        Subclasses can override this to add custom validation logic
        (e.g., check file format, verify integrity, etc.).

        Parameters
        ----------
        path : Path
            Path to validate.

        Returns
        -------
        bool
            True if file is valid, False otherwise.

        Examples
        --------
        >>> class MyResource(Resource[MyData]):
        ...     def _check_file(self, path: Path) -> bool:
        ...         # Check file extension
        ...         if path.suffix not in ['.h5ad', '.h5']:
        ...             return False
        ...         # Check file size
        ...         if path.stat().st_size == 0:
        ...             return False
        ...         return True
        """
        return path.exists()

    def _process_remote(self, path: Path) -> None:
        """Hook called after remote download completes.

        This hook is ONLY called after downloading from remote source.
        It is NOT called when loading from local_path.

        Use this for:
        - Post-download validation
        - File format conversion
        - Extraction/decompression
        - Metadata generation

        Parameters
        ----------
        path : Path
            Path to downloaded file/directory.

        Examples
        --------
        >>> class MyResource(Resource[MyData]):
        ...     def _process_remote(self, path: Path) -> None:
        ...         # Extract if it's an archive
        ...         if path.suffix == '.tar.gz':
        ...             import tarfile
        ...             with tarfile.open(path) as tar:
        ...                 tar.extractall(path.parent)
        """
        pass

    def _process_local(self, path: Path) -> None:
        """Hook called when loading from local_path.

        This hook is ONLY called when using explicit local_path.
        It is NOT called when loading from downloaded/cached files.

        Use this for:
        - Local file validation
        - Format checking
        - Preprocessing

        Parameters
        ----------
        path : Path
            Path to local file/directory.

        Examples
        --------
        >>> class MyResource(Resource[MyData]):
        ...     def _process_local(self, path: Path) -> None:
        ...         # Verify local file format
        ...         if not self._check_file(path):
        ...             raise ValueError(f"Invalid local file: {path}")
        """
        pass
