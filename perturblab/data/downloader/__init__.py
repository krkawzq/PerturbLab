"""Data downloaders for biological databases and datasets.

Organized Structure:
    - Cache: Core caching system
    - Base: Base downloader class
    - Custom: Generic URL downloader
    - Reference: Reference databases (HGNC, Ensembl)

Note: Benchmark datasets are managed by DatasetCard system in dataset module.
"""

# Core cache system
from ._cache import (
    CacheManager,
    cached_path,
    cached_auto,
    get_default_manager,
)

# Base downloader
from ._base import BaseDownloader

# Custom downloader
from ._custom import (
    CustomDownloader,
    download_from_url,
    get_default_downloader as get_default_custom_downloader,
)

# Reference databases
from .reference import (
    HGNC,
    HGNCDownloader,
    download_hgnc,
    Ensembl,
    EnsemblDownloader,
    download_ensembl,
)

__all__ = [
    # Cache
    'CacheManager',
    'cached_path',
    'cached_auto',
    'get_default_manager',
    
    # Base
    'BaseDownloader',
    
    # Custom
    'CustomDownloader',
    'download_from_url',
    'get_default_custom_downloader',
    
    # Reference databases
    'HGNC',
    'HGNCDownloader',
    'download_hgnc',
    'Ensembl',
    'EnsemblDownloader',
    'download_ensembl',
]
