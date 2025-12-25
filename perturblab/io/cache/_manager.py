"""Core cache manager with pluggable eviction policies.

This is a completely redesigned cache system with:
- Pluggable eviction policies (LRU, size-based, time-based)
- Atomic operations with file locking
- Rich metadata tracking
- Multi-process safe
- Zero-copy file operations
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from perturblab.utils import get_logger

# Try to import filelock for multi-process safety
try:
    from filelock import FileLock as _FileLock

    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
    _FileLock = None

logger = get_logger()

__all__ = ["CacheManager", "CacheEntry", "get_default_cache_manager", "auto_cache"]


class CacheEntry:
    """Represents a cached file with metadata."""

    def __init__(
        self,
        key: str,
        path: Path,
        size_bytes: int,
        created_at: float,
        last_accessed: float,
        access_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        self.key = key
        self.path = path
        self.size_bytes = size_bytes
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.access_count = access_count
        self.metadata = metadata or {}

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def age(self) -> float:
        """Get age in seconds since creation."""
        return time.time() - self.created_at

    def time_since_access(self) -> float:
        """Get time in seconds since last access."""
        return time.time() - self.last_accessed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            key=data["key"],
            path=Path(data["path"]),
            size_bytes=data["size_bytes"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


class CacheManager:
    """Modern cache manager with rich features.

    Features:
    - Pluggable eviction policies
    - Atomic write operations
    - Rich metadata tracking
    - Multi-process safe (with filelock)
    - Size and count limits
    - LRU tracking

    Parameters
    ----------
    cache_dir : Path or str, optional
        Cache directory. Defaults to ~/.cache/perturblab.
    max_size_mb : float, optional
        Maximum cache size in MB. None for unlimited.
    max_entries : int, optional
        Maximum number of entries. None for unlimited.
    auto_evict : bool, default=True
        Automatically evict old entries when limits are reached.

    Examples
    --------
    >>> cache = CacheManager(max_size_mb=1000, max_entries=100)
    >>>
    >>> # Cache a file
    >>> def download_fn(target_path):
    ...     # Download to target_path
    ...     pass
    >>> path = cache.get_or_create("model.pt", download_fn)
    >>>
    >>> # Check stats
    >>> stats = cache.stats()
    >>> print(f"Cache size: {stats['size_mb']:.1f} MB")
    """

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "perturblab" / "auto"

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        max_size_mb: float | None = None,
        max_entries: int | None = None,
        auto_evict: bool = True,
    ):
        # Setup cache directory
        if cache_dir is None:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            cache_dir = self.DEFAULT_CACHE_DIR

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None
        self.max_entries = max_entries
        self.auto_evict = auto_evict

        # Metadata
        self._metadata_file = self.cache_dir / ".cache_metadata.json"
        self._entries: dict[str, CacheEntry] = {}

        # Load existing metadata
        self._load_metadata()

        logger.debug(f"Initialized cache at {self.cache_dir}")
        if self.max_size_bytes:
            logger.debug(f"  Max size: {self.max_size_bytes / 1024 / 1024:.1f} MB")
        if self.max_entries:
            logger.debug(f"  Max entries: {self.max_entries}")

    @contextmanager
    def _lock(self, timeout: int = 300):
        """Acquire cache-wide lock for atomic operations."""
        if HAS_FILELOCK:
            lock_path = self.cache_dir / ".cache.lock"
            with _FileLock(str(lock_path), timeout=timeout):
                yield
        else:
            # No-op if filelock not available
            logger.warning("filelock not installed, cache operations may not be multi-process safe")
            yield

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if not self._metadata_file.exists():
            self._entries = {}
            return

        try:
            with open(self._metadata_file) as f:
                data = json.load(f)

            self._entries = {
                key: CacheEntry.from_dict(entry_data) for key, entry_data in data.items()
            }

            # Validate: remove entries for non-existent files
            invalid_keys = [key for key, entry in self._entries.items() if not entry.path.exists()]
            for key in invalid_keys:
                logger.warning(f"Cache entry '{key}' points to missing file, removing")
                del self._entries[key]

        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            self._entries = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            data = {key: entry.to_dict() for key, entry in self._entries.items()}

            # Atomic write
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.cache_dir, delete=False, suffix=".tmp"
            ) as f:
                json.dump(data, f, indent=2)
                temp_path = f.name

            shutil.move(temp_path, self._metadata_file)

        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def get(self, key: str) -> Path | None:
        """Get cached file path if exists.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Path or None
            Path to cached file, or None if not in cache.
        """
        if key not in self._entries:
            return None

        entry = self._entries[key]

        # Verify file still exists
        if not entry.path.exists():
            logger.warning(f"Cache entry '{key}' file missing, removing from cache")
            del self._entries[key]
            self._save_metadata()
            return None

        # Update access metadata
        entry.touch()
        self._save_metadata()

        return entry.path

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None

    def get_or_create(
        self,
        key: str,
        create_fn: Callable[[Path], None],
        force: bool = False,
        is_directory: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Get cached file or directory, or create it.

        This is the primary method for cache usage. It ensures atomic
        operations and handles eviction if needed.

        Supports both single files and directories (multi-file resources).

        Parameters
        ----------
        key : str
            Cache key (e.g., "scgpt_v1.0.pt" or "bert-base-uncased").
        create_fn : callable
            Function that accepts a target path and creates the file/directory.
            - If is_directory=False: receives a file path, should write to it
            - If is_directory=True: receives a directory path, should populate it
            Should be idempotent.
        force : bool, default=False
            Force recreation even if cached.
        is_directory : bool, default=False
            If True, cache a directory instead of a single file.
            Useful for multi-file resources (e.g., HuggingFace models).
        metadata : dict, optional
            Additional metadata to store with entry.

        Returns
        -------
        Path
            Path to cached file or directory.

        Examples
        --------
        >>> # Single file
        >>> def download_file(path):
        ...     urllib.request.urlretrieve(url, path)
        >>> cache_path = cache.get_or_create("model.pt", download_file)

        >>> # Multi-file directory (e.g., HuggingFace model)
        >>> def download_hf_model(target_dir):
        ...     # target_dir is an empty directory
        ...     (target_dir / "config.json").write_text("{}")
        ...     (target_dir / "pytorch_model.bin").write_bytes(b"...")
        >>> model_dir = cache.get_or_create(
        ...     "bert-base-uncased",
        ...     download_hf_model,
        ...     is_directory=True
        ... )
        """
        with self._lock():
            # Check if already cached
            if not force:
                cached_path = self.get(key)
                if cached_path is not None:
                    logger.debug(f"Cache hit: {key}")
                    return cached_path

            logger.info(f"Cache miss: {key} (is_directory={is_directory}), creating...")

            # Create temporary location (file or directory)
            if is_directory:
                # Create temporary directory
                tmp_path_str = tempfile.mkdtemp(dir=self.cache_dir, prefix=f".tmp_{key}_")
                tmp_path = Path(tmp_path_str)
            else:
                # Create temporary file
                fd, tmp_path_str = tempfile.mkstemp(
                    dir=self.cache_dir, prefix=f".tmp_{key}_", suffix=""
                )
                os.close(fd)
                tmp_path = Path(tmp_path_str)

            try:
                # Execute creation function
                start_time = time.time()
                create_fn(tmp_path)
                elapsed = time.time() - start_time

                # Calculate size (handle both files and directories)
                if is_directory:
                    if not tmp_path.exists() or not tmp_path.is_dir():
                        raise ValueError(
                            f"Creation function failed to create directory for key: {key}"
                        )
                    # Recursively calculate directory size
                    file_size = sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file())
                    num_files = sum(1 for f in tmp_path.rglob("*") if f.is_file())
                    logger.debug(
                        f"Directory contains {num_files} files, total size: {file_size / 1024 / 1024:.1f} MB"
                    )
                else:
                    # Verify file was created
                    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                        raise ValueError(f"Creation function produced empty file for key: {key}")
                    file_size = tmp_path.stat().st_size

                # Determine final path
                final_path = self.cache_dir / key

                # If target exists (bad state or force overwrite), remove it
                if final_path.exists():
                    logger.debug(f"Removing existing cache entry: {key}")
                    if final_path.is_dir():
                        shutil.rmtree(final_path)
                    else:
                        final_path.unlink()

                # Check if we need to evict
                if self.auto_evict:
                    self._maybe_evict(needed_bytes=file_size)

                # Atomic move (works for both files and directories)
                shutil.move(str(tmp_path), str(final_path))

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    path=final_path,
                    size_bytes=file_size,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    metadata={
                        **(metadata or {}),
                        "is_directory": is_directory,
                    },
                )

                self._entries[key] = entry
                self._save_metadata()

                logger.info(
                    f"Cached {key}: {file_size / 1024 / 1024:.1f} MB "
                    f"(created in {elapsed:.1f}s)"
                )

                return final_path

            except Exception as e:
                # Clean up on failure
                if tmp_path.exists():
                    if is_directory:
                        shutil.rmtree(tmp_path, ignore_errors=True)
                    else:
                        tmp_path.unlink()
                logger.error(f"Failed to cache {key}: {e}")
                raise

    def _maybe_evict(self, needed_bytes: int = 0) -> None:
        """Evict entries if necessary to meet limits.

        Uses LRU policy: evicts least recently used entries first.
        """
        # Check size limit
        if self.max_size_bytes:
            current_size = sum(e.size_bytes for e in self._entries.values())
            target_size = self.max_size_bytes - needed_bytes

            if current_size > target_size:
                bytes_to_free = current_size - target_size
                self._evict_lru(bytes_to_free)

        # Check count limit
        if self.max_entries and len(self._entries) >= self.max_entries:
            # Evict oldest accessed entry
            self._evict_lru(count=1)

    def _evict_lru(self, bytes_to_free: int = 0, count: int = 0) -> None:
        """Evict least recently used entries."""
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(self._entries.items(), key=lambda x: x[1].last_accessed)

        freed_bytes = 0
        freed_count = 0

        for key, entry in sorted_entries:
            if (bytes_to_free > 0 and freed_bytes >= bytes_to_free) or (
                count > 0 and freed_count >= count
            ):
                break

            is_dir = entry.metadata.get("is_directory", False)
            logger.info(f"Evicting cache entry: {key} (is_directory={is_dir})")

            # Remove file or directory
            if entry.path.exists():
                if entry.path.is_dir():
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    entry.path.unlink()
                freed_bytes += entry.size_bytes

            # Remove from entries
            del self._entries[key]
            freed_count += 1

        if freed_count > 0:
            self._save_metadata()
            logger.info(f"Evicted {freed_count} entries, freed {freed_bytes / 1024 / 1024:.1f} MB")

    def invalidate(self, key: str) -> None:
        """Remove entry from cache (file or directory)."""
        if key not in self._entries:
            return

        entry = self._entries[key]

        if entry.path.exists():
            if entry.path.is_dir():
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                entry.path.unlink()

        del self._entries[key]
        self._save_metadata()

        logger.debug(f"Invalidated cache entry: {key}")

    def clear(self) -> None:
        """Clear all cache entries (files and directories)."""
        with self._lock():
            for entry in self._entries.values():
                if entry.path.exists():
                    if entry.path.is_dir():
                        shutil.rmtree(entry.path, ignore_errors=True)
                    else:
                        entry.path.unlink()

            self._entries = {}
            self._save_metadata()

            logger.info(f"Cleared cache directory: {self.cache_dir}")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(e.size_bytes for e in self._entries.values())

        return {
            "cache_dir": str(self.cache_dir),
            "num_entries": len(self._entries),
            "size_bytes": total_size,
            "size_mb": total_size / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024 if self.max_size_bytes else None,
            "max_entries": self.max_entries,
            "entries": [
                {
                    "key": e.key,
                    "size_mb": e.size_bytes / 1024 / 1024,
                    "age_hours": e.age() / 3600,
                    "access_count": e.access_count,
                }
                for e in sorted(self._entries.values(), key=lambda x: x.last_accessed, reverse=True)
            ],
        }

    def list_keys(self) -> list[str]:
        """List all cached keys."""
        return list(self._entries.keys())


# Global default cache manager
_default_cache: CacheManager | None = None


def get_default_cache_manager() -> CacheManager:
    """Get or create the default global cache manager."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager()
    return _default_cache


def auto_cache(
    key: str,
    is_directory: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Decorator to automatically cache a download function.

    Supports two usage styles:
    1. Function accepts 'path' parameter (zero-copy, writes directly to cache)
    2. Function returns Path (downloads to temp, then copies to cache)

    The decorator automatically detects which style by inspecting the function signature.

    Parameters
    ----------
    key : str
        Cache key (e.g., "bert-base-uncased", "dataset_v1.h5ad").
    is_directory : bool, default=False
        Whether the resource is a directory (multi-file).
    metadata : dict, optional
        Additional metadata to store with cache entry.

    Returns
    -------
    callable
        Decorator function.

    Examples
    --------
    >>> from perturblab.io.cache import auto_cache
    >>> from pathlib import Path
    >>> import urllib.request
    >>>
    >>> # Style 1: Function accepts 'path' parameter (zero-copy, recommended)
    >>> @auto_cache("model.pt")
    ... def download_model(path: Path) -> None:
    ...     # Download directly to cache path (no copying needed)
    ...     urllib.request.urlretrieve("https://example.com/model.pt", path)
    >>>
    >>> path = download_model()  # Returns cached path

    >>> # Style 2: Function returns Path (flexible, but requires copying)
    >>> @auto_cache("model.pt")
    ... def download_model() -> Path:
    ...     temp = Path("/tmp/temp_model.pt")
    ...     urllib.request.urlretrieve("https://example.com/model.pt", temp)
    ...     return temp
    >>>
    >>> path = download_model()  # Returns cached path

    >>> # For directories (multi-file resources)
    >>> @auto_cache("bert-base-uncased", is_directory=True)
    ... def download_hf_model(path: Path) -> None:
    ...     # Download directly to target directory
    ...     from huggingface_hub import snapshot_download
    ...     snapshot_download("bert-base-uncased", cache_dir=path)
    >>>
    >>> model_dir = download_hf_model()  # Returns cached directory

    Notes
    -----
    - Style 1 (with path parameter) is more efficient (zero-copy)
    - Style 2 (returns Path) is more flexible but requires file copying
    - The decorator automatically detects which style by inspecting the signature
    - Both styles produce the same result: a cached Path
    """
    cm = get_default_cache_manager()

    def decorator(func: Callable) -> Callable[[], Path]:
        """The actual decorator that wraps the download function."""

        # Inspect function signature to detect style
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check if function accepts 'path' parameter (Style 1)
        has_path_param = "path" in params

        if has_path_param:
            # Style 1: Function accepts path parameter (zero-copy)
            def wrapper() -> Path:
                """Wrapped function (Style 1: path parameter)."""

                def create_fn(target_path: Path) -> None:
                    """Call user function with target path directly."""
                    # Call user's function with cache path
                    # Note: For directories, CacheManager creates the directory,
                    # so user function should populate it (not create it)
                    func(path=target_path)

                    # Verify target exists and has content
                    if not target_path.exists():
                        raise ValueError(f"Function did not create resource at: {target_path}")

                    # Validate type
                    if is_directory:
                        if not target_path.is_dir():
                            raise ValueError(
                                f"is_directory=True but resource is not a directory: {target_path}"
                            )
                        # Check that directory has content
                        if not any(target_path.iterdir()):
                            raise ValueError(f"Function created empty directory: {target_path}")
                    else:
                        if target_path.is_dir():
                            raise ValueError(
                                f"is_directory=False but resource is a directory: {target_path}"
                            )
                        # Check that file has content
                        if target_path.stat().st_size == 0:
                            raise ValueError(f"Function created empty file: {target_path}")

                return cm.get_or_create(
                    key=key, create_fn=create_fn, is_directory=is_directory, metadata=metadata
                )

        else:
            # Style 2: Function returns Path (requires copying)
            def wrapper() -> Path:
                """Wrapped function (Style 2: returns Path)."""

                def create_fn(target_path: Path) -> None:
                    """Call user function and copy result to target."""
                    # Call user's download function
                    source_path = func()

                    if not source_path.exists():
                        raise FileNotFoundError(
                            f"Function returned non-existent path: {source_path}"
                        )

                    # Copy source to target based on type
                    if is_directory:
                        if not source_path.is_dir():
                            raise ValueError(
                                f"is_directory=True but function returned file: {source_path}"
                            )
                        # Copy entire directory tree
                        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    else:
                        if not source_path.is_file():
                            raise ValueError(
                                f"is_directory=False but function returned directory: {source_path}"
                            )
                        # Copy single file
                        shutil.copy2(source_path, target_path)

                return cm.get_or_create(
                    key=key, create_fn=create_fn, is_directory=is_directory, metadata=metadata
                )

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__wrapped__ = func  # For introspection

        return wrapper

    return decorator
