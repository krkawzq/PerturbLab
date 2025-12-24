"""Path-based caching system for large biological datasets and model weights.

Designed for bioinformatics and deep learning workflows where files are GB-scale.
Returns file paths instead of loading content into memory to prevent OOM errors.

Target Use Cases:
    - PyTorch model weights (scGPT, GEARS, CellFormer)
    - Single-cell datasets (AnnData h5ad, Loom files)
    - Reference databases (HGNC, Ensembl, UniProt)
    - Pre-computed embeddings (Gene2Vec, Cell2Vec)

Key Features:
    - Path-based API: Returns Path objects, not file contents
    - Atomic writes: Download to temp file, rename on success (crash-safe)
    - Semantic filenames: Human-readable names (model_v1.pt, not hash)
    - Multi-process safe: File locks prevent race conditions
    - Flexible decorators: Static, formatted, or hash-based filenames

Components:
    - CacheManager: Core cache management class
    - cached_path: Decorator with manual filename control
    - cached_auto: Decorator with automatic hash-based filenames
    
Example:
    >>> from perturblab.data.downloader._cache import cached_path
    >>> 
    >>> @cached_path(filename='scgpt_weights.pt')
    >>> def download_model(url, target_path=None):
    >>>     # Stream download to target_path
    >>>     response = requests.get(url, stream=True)
    >>>     with open(target_path, 'wb') as f:
    >>>         for chunk in response.iter_content(8192):
    >>>             f.write(chunk)
    >>> 
    >>> # Returns Path, can pass directly to torch.load
    >>> weights_path = download_model('https://...')
    >>> model = torch.load(weights_path)
"""

import os
import json
import shutil
import tempfile
import time
import hashlib
import inspect
from pathlib import Path
from typing import Callable, Dict, Any
from functools import wraps

# File locking for multi-process safety
try:
    from filelock import FileLock
except ImportError:
    # Fallback: no-op context manager
    import warnings
    warnings.warn(
        "filelock not installed. Multi-process cache access may be unsafe. "
        "Install with: pip install filelock"
    )
    from contextlib import contextmanager
    
    @contextmanager
    def FileLock(path):
        yield

from perturblab.utils import get_logger
logger = get_logger()


class CacheManager:
    """Path-based cache manager for large files.
    
    Returns file paths instead of loading content into memory.
    Ensures atomic writes to prevent corruption from interrupted downloads.
    
    Args:
        cache_dir: Cache directory path. Defaults to ~/.cache/perturblab
        namespace: Subdirectory name for organizing caches.
    """
    
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        namespace: str = "perturblab",
    ):
        if cache_dir is None:
            # Follow XDG Base Directory specification
            xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            cache_dir = Path(xdg_cache) / namespace
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata_file = self.cache_dir / ".metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self._metadata = {}
        else:
            self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def get_path(self, filename: str) -> Path:
        """Get target cache path (file may not exist yet).
        
        Args:
            filename: Target filename (should be semantic, not hash).
        
        Returns:
            Path: Full path to cached file location.
        """
        return self.cache_dir / filename
    
    def exists(self, filename: str, check_integrity: bool = True) -> bool:
        """Check if file exists in cache and is valid.
        
        Args:
            filename: Target filename.
            check_integrity: Whether to verify file size matches metadata.
        
        Returns:
            bool: True if file exists and is valid.
        """
        path = self.get_path(filename)
        
        if not path.exists():
            return False
        
        # Basic check: non-empty file
        if path.stat().st_size == 0:
            logger.warning(f"Empty cache file detected: {filename}")
            return False
        
        # Optional: verify size matches metadata
        if check_integrity and filename in self._metadata:
            expected_size = self._metadata[filename].get('size_bytes')
            actual_size = path.stat().st_size
            
            if expected_size and actual_size != expected_size:
                logger.warning(
                    f"Size mismatch for {filename}: "
                    f"expected {expected_size}, got {actual_size}"
                )
                return False
        
        return True
    
    def ensure_cached(
        self,
        filename: str,
        download_fn: Callable[[Path], None],
        force_redownload: bool = False,
    ) -> Path:
        """Ensure file exists in cache, downloading if necessary.
        
        Thread/process-safe with file locking. Uses atomic writes to prevent
        corruption from interrupted downloads.
        
        Args:
            filename: Target filename (semantic name recommended).
            download_fn: Function that accepts a target path and writes to it.
                Should handle streaming downloads for large files.
            force_redownload: If True, re-download even if cached.
        
        Returns:
            Path: Path to cached file.
        """
        target_path = self.get_path(filename)
        lock_path = self.cache_dir / f".{filename}.lock"
        
        # File lock prevents race conditions in multi-process scenarios
        with FileLock(str(lock_path), timeout=300):
            # Check cache after acquiring lock (another process may have downloaded)
            if not force_redownload and self.exists(filename):
                logger.debug(f"Cache hit: {filename}")
                return target_path
            
            logger.info(f"Downloading {filename}...")
            
            # Create temporary file in same directory (ensures same filesystem)
            fd, tmp_path_str = tempfile.mkstemp(
                dir=self.cache_dir,
                prefix=f".tmp_{filename}_",
                suffix=""
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)
            
            try:
                # Execute download to temporary path
                start_time = time.time()
                download_fn(tmp_path)
                elapsed = time.time() - start_time
                
                # Verify download succeeded
                if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise ValueError("Download produced empty file")
                
                # Atomic rename (crash-safe)
                shutil.move(str(tmp_path), str(target_path))
                
                # Update metadata
                file_size = target_path.stat().st_size
                self._metadata[filename] = {
                    'created_at': time.time(),
                    'size_bytes': file_size,
                    'download_time_sec': elapsed,
                }
                self._save_metadata()
                
                logger.info(
                    f"Downloaded {filename} "
                    f"({file_size / 1024 / 1024:.2f} MB in {elapsed:.1f}s)"
                )
                
            except Exception as e:
                # Clean up temporary file on failure
                if tmp_path.exists():
                    tmp_path.unlink()
                logger.error(f"Failed to cache {filename}: {e}")
                raise
            finally:
                # Clean up lock file
                if lock_path.exists():
                    try:
                        lock_path.unlink()
                    except:
                        pass
        
        return target_path
    
    def invalidate(self, filename: str) -> None:
        """Remove file from cache.
        
        Args:
            filename: Target filename to remove.
        """
        path = self.get_path(filename)
        
        if path.exists():
            path.unlink()
            logger.debug(f"Invalidated cache: {filename}")
        
        if filename in self._metadata:
            del self._metadata[filename]
            self._save_metadata()
    
    def clear(self) -> None:
        """Remove all cached files."""
        for file in self.cache_dir.glob('*'):
            if file.name.startswith('.'):
                # Skip metadata and lock files
                continue
            
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        
        self._metadata = {}
        self._save_metadata()
        
        logger.info(f"Cleared cache directory: {self.cache_dir}")
    
    def get_size_mb(self) -> float:
        """Get total cache size in MB.
        
        Returns:
            float: Total size in megabytes.
        """
        total_bytes = sum(
            f.stat().st_size
            for f in self.cache_dir.rglob('*')
            if f.is_file() and not f.name.startswith('.')
        )
        return total_bytes / (1024 * 1024)
    
    def list_cached_files(self) -> list[dict]:
        """List all cached files with metadata.
        
        Returns:
            list[dict]: List of file info dictionaries.
        """
        files = []
        for filename, metadata in self._metadata.items():
            path = self.get_path(filename)
            if path.exists():
                files.append({
                    'filename': filename,
                    'size_mb': metadata.get('size_bytes', 0) / 1024 / 1024,
                    'created_at': metadata.get('created_at'),
                    'exists': True,
                })
        return files
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            dict: Statistics dictionary.
        """
        return {
            'cache_dir': str(self.cache_dir),
            'n_files': len(self._metadata),
            'total_size_mb': self.get_size_mb(),
        }


# Global default cache manager
_default_manager: CacheManager | None = None


def get_default_manager() -> CacheManager:
    """Get or create the default cache manager.
    
    Returns:
        CacheManager: Default cache manager instance.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = CacheManager()
    return _default_manager


def _serialize_value(value: Any, max_depth: int = 10, _depth: int = 0) -> str:
    """Recursively serialize a value to stable string representation.
    
    Handles nested structures and avoids memory address in repr.
    
    Args:
        value: Value to serialize.
        max_depth: Maximum recursion depth to prevent infinite loops.
        _depth: Current recursion depth (internal).
    
    Returns:
        str: Stable string representation.
    """
    if _depth > max_depth:
        return "<max_depth_exceeded>"
    
    # None, bool, numbers
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    
    # Strings
    if isinstance(value, str):
        return f'"{value}"'
    
    # Bytes
    if isinstance(value, bytes):
        return f"bytes:{len(value)}"
    
    # Dict: sort keys and recursively serialize
    if isinstance(value, dict):
        items = []
        for k in sorted(value.keys(), key=str):
            k_str = _serialize_value(k, max_depth, _depth + 1)
            v_str = _serialize_value(value[k], max_depth, _depth + 1)
            items.append(f"{k_str}:{v_str}")
        return "{" + ",".join(items) + "}"
    
    # List/tuple: recursively serialize elements
    if isinstance(value, (list, tuple)):
        type_prefix = "list" if isinstance(value, list) else "tuple"
        items = [_serialize_value(item, max_depth, _depth + 1) for item in value]
        return f"{type_prefix}[{','.join(items)}]"
    
    # Set/frozenset: sort for stability
    if isinstance(value, (set, frozenset)):
        type_prefix = "set" if isinstance(value, set) else "frozenset"
        items = sorted(_serialize_value(item, max_depth, _depth + 1) for item in value)
        return f"{type_prefix}{{{','.join(items)}}}"
    
    # NumPy arrays
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            # Use shape and dtype, not actual data (data could be huge)
            return f"ndarray(shape={value.shape},dtype={value.dtype})"
    except ImportError:
        pass
    
    # Pandas DataFrames/Series
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return f"DataFrame(shape={value.shape})"
        if isinstance(value, pd.Series):
            return f"Series(len={len(value)})"
    except ImportError:
        pass
    
    # Path objects
    try:
        from pathlib import Path
        if isinstance(value, Path):
            return f"Path({str(value)})"
    except ImportError:
        pass
    
    # Fallback: use type name to avoid memory address
    # Avoid repr() which may include memory addresses like <object at 0x...>
    return f"<{type(value).__name__}>"


def _stable_hash(func_name: str, args_dict: Dict[str, Any]) -> str:
    """Generate stable hash from function arguments.
    
    Uses intelligent serialization to avoid memory addresses and handle
    nested structures. Hash is stable across runs for same logical arguments.
    
    Args:
        func_name: Function name to include in hash.
        args_dict: Dictionary of function arguments.
    
    Returns:
        str: MD5 hash string (32 characters).
    """
    # Remove injected parameter
    args_dict = {k: v for k, v in args_dict.items() if k != 'target_path'}
    
    # Build stable payload with smart serialization
    stable_list = [func_name]
    
    for key in sorted(args_dict.keys()):
        val = args_dict[key]
        val_str = _serialize_value(val)
        stable_list.append(f"{key}:{val_str}")
    
    payload = "|".join(stable_list)
    
    # Generate MD5 hash
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def cached_path(
    filename: str | Callable[..., str],
    manager: CacheManager | None = None,
    force_redownload: bool = False,
    auto_format: bool = False,
):
    """Decorator for caching downloaded files.
    
    The decorated function should accept a `target_path` parameter
    and write downloaded content to that path (streaming recommended).
    
    Returns the cached file path, NOT the content.
    
    Args:
        filename: Target filename. Can be:
            - Static string: "data.tsv"
            - Format string (if auto_format=True): "model_{version}.pt"
            - Callable: lambda ver: f"model_{ver}.pt"
        manager: Custom cache manager. Uses default if None.
        force_redownload: Force re-download even if cached.
        auto_format: If True, automatically formats filename using function args.
            Example: filename="gene_{symbol}.json" with func(symbol="TP53")
            becomes "gene_TP53.json"
    
    Returns:
        Decorator function.
    
    Example:
        >>> # Static filename
        >>> @cached_path(filename='hgnc_data.tsv')
        >>> def download_hgnc(url, target_path=None):
        >>>     ...
        
        >>> # Auto-format with function arguments
        >>> @cached_path(filename='scgpt_{version}.pt', auto_format=True)
        >>> def download_weights(version, url, target_path=None):
        >>>     ...
        >>> 
        >>> path = download_weights(version='human_v1', url='...')
        >>> # Saves to: scgpt_human_v1.pt
        
        >>> # Dynamic filename with callable
        >>> @cached_path(filename=lambda v: f'model_v{v}.pt')
        >>> def download_model(version, target_path=None):
        >>>     ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Path:
            mgr = manager or get_default_manager()
            
            # Determine final filename
            if isinstance(filename, str) and auto_format:
                # Auto-format: bind arguments and format string
                sig = inspect.signature(func)
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                
                try:
                    final_name = filename.format(**bound.arguments)
                except KeyError as e:
                    raise ValueError(
                        f"Failed to format filename '{filename}'. "
                        f"Missing argument: {e}"
                    )
            
            elif callable(filename):
                # Callable: execute to get filename
                final_name = filename(*args, **kwargs)
            
            else:
                # Static string: use as-is
                final_name = filename
            
            # Check for force_redownload override in kwargs
            force = kwargs.pop('force_redownload', force_redownload)
            
            # Define download action
            def download_action(tmp_path: Path):
                return func(*args, target_path=tmp_path, **kwargs)
            
            # Ensure cached and return path
            return mgr.ensure_cached(
                final_name,
                download_action,
                force_redownload=force
            )
        
        return wrapper
    return decorator


def cached_auto(
    manager: CacheManager | None = None,
    force_redownload: bool = False,
    extension: str = "",
    include_func_name: bool = True,
):
    """Hash-based automatic caching decorator.
    
    Automatically generates filename from function name and arguments hash.
    Useful when you don't want to manually specify filenames.
    
    Args:
        manager: Cache manager instance. Uses default if None.
        force_redownload: Force re-download even if cached.
        extension: File extension (e.g., ".pt", ".json").
        include_func_name: If True, prefix hash with function name.
            Recommended for debugging.
    
    Returns:
        Decorator function.
    
    Example:
        >>> @cached_auto(extension='.pt', include_func_name=True)
        >>> def download_model(url, version='1.0', target_path=None):
        >>>     # ... download logic ...
        >>>     pass
        >>> 
        >>> # Filename: download_model_7a8b9c0d1e2f3g4h5i6j7k8l9m0n1o2.pt
        >>> path = download_model(url='...', version='2.0')
        
        >>> # Different arguments = different cache file
        >>> path2 = download_model(url='...', version='3.0')
    
    Notes:
        The hash is stable and deterministic:
        - Same arguments always produce same hash
        - Order of kwargs doesn't matter
        - Useful for functions with many parameters
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Path:
            mgr = manager or get_default_manager()
            
            # Extract and normalize arguments
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            # Compute stable hash
            args_hash = _stable_hash(func.__name__, bound.arguments)
            
            # Construct filename
            if include_func_name:
                final_name = f"{func.__name__}_{args_hash}"
            else:
                final_name = args_hash
            
            if extension:
                if not extension.startswith('.'):
                    extension_str = f".{extension}"
                else:
                    extension_str = extension
                final_name += extension_str
            
            # Check for force override
            force = kwargs.pop('force_redownload', force_redownload)
            
            # Define download action
            def download_action(tmp_path: Path):
                return func(*args, target_path=tmp_path, **kwargs)
            
            # Ensure cached
            return mgr.ensure_cached(
                final_name,
                download_action,
                force_redownload=force
            )
        
        return wrapper
    return decorator
