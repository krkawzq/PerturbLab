"""Common resource implementations for data files.

Provides concrete Resource implementations for common data types:
- File: Generic single file resource
- Files: Directory resource (collection of files)
- h5adFile: AnnData h5ad/h5 file resource
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from perturblab.core import Resource
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["File", "Files", "h5adFile"]


class File(Resource[Path]):
    """Generic file resource.

    A simple resource that represents a single file and returns its Path
    when loaded. Useful for lazy file downloading without immediate parsing.

    Examples
    --------
    >>> from perturblab.data.resources import File
    >>>
    >>> # Local file
    >>> file = File(
    ...     key='data.csv',
    ...     local_path='/data/file.csv'
    ... )
    >>> path = file.load()  # Returns Path

    >>> # Remote file (HTTP)
    >>> file = File(
    ...     key='model.pt',
    ...     remote_config={
    ...         'downloader': 'HTTPDownloader',
    ...         'url': 'https://example.com/model.pt'
    ...     }
    ... )
    >>> path = file.load()  # Downloads and returns cached Path
    """

    def _fetch_remote(self, config: dict[str, Any], target_path: Path) -> None:
        """Download file using configured downloader.

        Parameters
        ----------
        config : dict
            Remote configuration with 'downloader' and downloader-specific params.
        target_path : Path
            Where to save the downloaded file.
        """
        from perturblab.io import download

        downloader_name = config.get("downloader", "HTTPDownloader")

        try:
            downloader_cls = getattr(download, downloader_name)
        except AttributeError:
            raise RuntimeError(
                f"Downloader '{downloader_name}' not found. "
                f"Available: HTTPDownloader, FigshareDownloader, HGNCDownloader, "
                f"EnsemblDownloader, GODownloader"
            )

        # Build download kwargs
        download_kwargs = {k: v for k, v in config.items() if k != "downloader"}

        # Add target_path based on downloader
        if downloader_name in ["HGNCDownloader", "GODownloader"]:
            # target_path as first positional arg
            download_kwargs["target_path"] = target_path
        else:
            # target_path as keyword arg or second positional
            download_kwargs["target_path"] = target_path

        logger.debug(f"Downloading with {downloader_name}...")
        downloader_cls.download(**download_kwargs)

    def _load_from_disk(self, path: Path) -> Path:
        """Return the path itself (no parsing needed).

        Parameters
        ----------
        path : Path
            Path to file.

        Returns
        -------
        Path
            The same path.
        """
        logger.debug(f"File resource loaded: {path}")
        return path

    def _check_file(self, path: Path) -> bool:
        """Validate that path is a file (not a directory).

        Parameters
        ----------
        path : Path
            Path to validate.

        Returns
        -------
        bool
            True if path is a file.
        """
        if not path.exists():
            return False
        if not path.is_file():
            logger.warning(f"Expected file but got directory: {path}")
            return False
        if path.stat().st_size == 0:
            logger.warning(f"File is empty: {path}")
            return False
        return True


class Files(Resource[Path]):
    """Directory resource (collection of files).

    Represents a directory containing multiple files. Returns the directory
    Path when loaded.

    Examples
    --------
    >>> from perturblab.data.resources import Files
    >>>
    >>> # Local directory
    >>> files = Files(
    ...     key='dataset_dir',
    ...     local_path='/data/my_dataset/',
    ...     is_directory=True
    ... )
    >>> dir_path = files.load()  # Returns Path to directory
    >>> for file in dir_path.iterdir():
    ...     print(file)

    >>> # Remote directory (e.g., Figshare article)
    >>> files = Files(
    ...     key='article_files',
    ...     remote_config={
    ...         'downloader': 'FigshareDownloader',
    ...         'article_id': '1234567'
    ...     },
    ...     is_directory=True
    ... )
    >>> dir_path = files.load()  # Downloads all files to directory
    """

    def __init__(
        self,
        key: str,
        *,
        local_path: str | Path | None = None,
        remote_config: dict[str, Any] | None = None,
        cache_manager: Any | None = None,
    ):
        """Initialize directory resource.

        Parameters
        ----------
        key : str
            Unique identifier.
        local_path : str or Path, optional
            Local directory path.
        remote_config : dict, optional
            Remote download configuration.
        cache_manager : CacheManager, optional
            Custom cache manager.
        """
        super().__init__(
            key,
            local_path=local_path,
            remote_config=remote_config,
            is_directory=True,  # Always a directory
            cache_manager=cache_manager,
        )

    def _fetch_remote(self, config: dict[str, Any], target_path: Path) -> None:
        """Download directory using configured downloader.

        Parameters
        ----------
        config : dict
            Remote configuration.
        target_path : Path
            Target directory path.
        """
        from perturblab.io import download

        downloader_name = config.get("downloader", "HTTPDownloader")

        try:
            downloader_cls = getattr(download, downloader_name)
        except AttributeError:
            raise RuntimeError(f"Downloader '{downloader_name}' not found")

        # Build download kwargs
        download_kwargs = {k: v for k, v in config.items() if k != "downloader"}
        download_kwargs["target_path"] = target_path

        # Ensure target is a directory
        target_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Downloading directory with {downloader_name}...")
        downloader_cls.download(**download_kwargs)

    def _load_from_disk(self, path: Path) -> Path:
        """Return the directory path itself.

        Parameters
        ----------
        path : Path
            Path to directory.

        Returns
        -------
        Path
            The same directory path.
        """
        logger.debug(f"Directory resource loaded: {path}")
        return path

    def _check_file(self, path: Path) -> bool:
        """Validate that path is a directory.

        Parameters
        ----------
        path : Path
            Path to validate.

        Returns
        -------
        bool
            True if path is a directory.
        """
        if not path.exists():
            return False
        if not path.is_dir():
            logger.warning(f"Expected directory but got file: {path}")
            return False
        # Check that directory is not empty
        if not any(path.iterdir()):
            logger.warning(f"Directory is empty: {path}")
            return False
        return True


class h5adFile(Resource[Path]):
    """AnnData H5AD/H5 file resource.

    Specialized resource for AnnData files with format validation.
    Requires files to have .h5ad or .h5 extension.

    Examples
    --------
    >>> from perturblab.data.resources import h5adFile
    >>>
    >>> # Local h5ad file
    >>> file = h5adFile(
    ...     key='norman_2019',
    ...     local_path='/data/norman_2019.h5ad'
    ... )
    >>> path = file.load()  # Returns Path after validation

    >>> # Remote h5ad file
    >>> file = h5adFile(
    ...     key='adamson_2016',
    ...     remote_config={
    ...         'downloader': 'FigshareDownloader',
    ...         'file_id': '19667905'
    ...     }
    ... )
    >>> path = file.load()  # Downloads and validates

    >>> # Then load into AnnData
    >>> import anndata as ad
    >>> adata = ad.read_h5ad(path)
    """

    # Allowed file extensions
    VALID_EXTENSIONS = {".h5ad", ".h5"}

    def _fetch_remote(self, config: dict[str, Any], target_path: Path) -> None:
        """Download h5ad file using configured downloader.

        Parameters
        ----------
        config : dict
            Remote configuration.
        target_path : Path
            Target file path.
        """
        from perturblab.io import download

        downloader_name = config.get("downloader", "HTTPDownloader")

        try:
            downloader_cls = getattr(download, downloader_name)
        except AttributeError:
            raise RuntimeError(f"Downloader '{downloader_name}' not found")

        # Build download kwargs
        download_kwargs = {k: v for k, v in config.items() if k != "downloader"}
        download_kwargs["target_path"] = target_path

        logger.debug(f"Downloading h5ad with {downloader_name}...")
        downloader_cls.download(**download_kwargs)

    def _load_from_disk(self, path: Path) -> Path:
        """Return the h5ad file path.

        Parameters
        ----------
        path : Path
            Path to h5ad file.

        Returns
        -------
        Path
            The same path (after validation).
        """
        logger.debug(f"h5ad file resource loaded: {path}")
        return path

    def _check_file(self, path: Path) -> bool:
        """Validate that file is a valid h5ad/h5 file.

        Checks:
        1. File exists
        2. Is a file (not directory)
        3. Has .h5ad or .h5 extension
        4. Is not empty

        Parameters
        ----------
        path : Path
            Path to validate.

        Returns
        -------
        bool
            True if file is valid.
        """
        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return False

        if not path.is_file():
            logger.error(f"Expected file but got directory: {path}")
            return False

        if path.suffix not in self.VALID_EXTENSIONS:
            logger.error(
                f"Invalid file extension: {path.suffix}. "
                f"Expected one of: {self.VALID_EXTENSIONS}"
            )
            return False

        if path.stat().st_size == 0:
            logger.error(f"File is empty: {path}")
            return False

        # Optional: Quick h5 format check
        try:
            import h5py

            with h5py.File(path, "r") as f:
                # Basic check - h5ad files should have certain structure
                if "X" not in f and "obs" not in f:
                    logger.warning(
                        f"File {path} might not be a valid h5ad format "
                        f"(missing 'X' and 'obs' groups)"
                    )
        except Exception as e:
            logger.warning(f"Could not validate h5 format: {e}")

        return True

    def _get_cache_key(self) -> str:
        """Generate cache key with .h5ad extension.

        Returns
        -------
        str
            Cache key with .h5ad extension.
        """
        if self.key.endswith(".h5ad") or self.key.endswith(".h5"):
            return self.key
        return f"{self.key}.h5ad"

    def load_adata(self) -> "AnnData":  # type: ignore
        """Convenience method to directly load as AnnData object.

        Returns
        -------
        AnnData
            Loaded AnnData object.

        Examples
        --------
        >>> file = h5adFile(key='data', local_path='file.h5ad')
        >>> adata = file.load_adata()  # Directly returns AnnData
        >>> # Equivalent to:
        >>> import anndata as ad
        >>> path = file.load()
        >>> adata = ad.read_h5ad(path)
        """
        import anndata as ad

        path = self.load()
        logger.info(f"Loading AnnData from: {path}")
        return ad.read_h5ad(path)
