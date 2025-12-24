"""I/O utilities for PerturbLab.

This module provides a complete redesigned I/O system with:
- **Cache Management**: Robust caching with LRU eviction, size limits
- **HTTP Downloads**: Retry logic, progress tracking, resume support
- **Resource Management**: Unified interface combining cache + download

Quick Start
-----------
>>> from perturblab.io import get_resource, cached_download
>>> 
>>> # Simple download with automatic caching
>>> path = get_resource("https://example.com/model.pt")
>>> 
>>> # Or use more explicit naming
>>> path = cached_download(
...     "https://example.com/scgpt.pt",
...     filename="scgpt_v1.0.pt"
... )
>>> 
>>> # Use the cached file directly
>>> import torch
>>> model = torch.load(path)

Advanced Usage
--------------
>>> from perturblab.io import ResourceManager, CacheManager
>>> 
>>> # Custom cache with size limits
>>> cache = CacheManager(max_size_mb=1000, max_entries=100)
>>> manager = ResourceManager(cache=cache)
>>> 
>>> # Download with metadata
>>> path = manager.get_resource(
...     url="https://example.com/model.pt",
...     cache_key="my_model_v2.pt",
...     metadata={'version': '2.0', 'source': 'custom'}
... )
>>> 
>>> # Check cache stats
>>> stats = manager.cache_stats()
>>> print(f"Cache size: {stats['size_mb']:.1f} MB")

Components
----------
- **cache**: Cache management system
- **download**: HTTP download utilities
- **ResourceManager**: High-level resource management interface
"""

# Cache management
from .cache import CacheManager, CacheEntry, get_default_cache_manager

# Download utilities
from .download import (
    HTTPDownloader,
    download_http,
    # Repository downloaders
    FigshareDownloader,
    download_figshare,
    # Biological databases
    HGNCDownloader,
    download_hgnc,
    EnsemblDownloader,
    download_ensembl,
    GODownloader,
    download_go,
)

__all__ = [
    "CacheManager",
    "CacheEntry",
    "get_default_cache_manager",
    "auto_cache",
    "HTTPDownloader",
    "download_http",
    "FigshareDownloader",
    "download_figshare",
]
