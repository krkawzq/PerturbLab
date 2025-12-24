"""Cache management system for PerturbLab.

Provides robust caching for both single files and directories with:
- Single file and multi-file (directory) caching
- LRU eviction policies
- Size and count limits
- Multi-process safety (with filelock)
- Rich metadata tracking
- Atomic operations

Examples
--------
>>> from perturblab.io.cache import CacheManager
>>>
>>> # Create cache manager
>>> cache = CacheManager(max_size_mb=1000, max_entries=100)
>>>
>>> # Cache a single file
>>> def download_file(path):
...     # Download to path
...     pass
>>> file_path = cache.get_or_create("model.pt", download_file)
>>>
>>> # Cache a directory (e.g., HuggingFace model with multiple files)
>>> def download_model(target_dir):
...     # Populate target_dir with files
...     pass
>>> model_dir = cache.get_or_create(
...     "bert-base-uncased",
...     download_model,
...     is_directory=True
... )
"""

from ._manager import CacheEntry, CacheManager, auto_cache, get_default_cache_manager

__all__ = [
    "CacheManager",
    "CacheEntry",
    "get_default_cache_manager",
    "auto_cache",
]
