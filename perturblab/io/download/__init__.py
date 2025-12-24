"""Download utilities for PerturbLab.

Provides robust HTTP downloading with:
- Automatic retry with exponential backoff
- Progress tracking (tqdm integration)
- Resumable downloads
- Streaming for large files

Plus specialized downloaders:
- Figshare: Research data repository with optional token authentication
- HGNC: Official gene nomenclature
- Ensembl: Genome annotations and sequences
- Gene Ontology: Ontologies and gene-GO annotations

All downloaders inherit from BaseDownloader and support configuration-based downloads.

Note:
    Dataset-specific downloaders (e.g., scPerturb datasets, GEARS weights)
    are provided by their respective dataset/model classes in perturblab.data
    and perturblab.models, not here. This module only provides generic
    download utilities.
"""

# Base downloader and exceptions
from ._base import BaseDownloader, DownloadError

# Core HTTP downloader
from ._http import HTTPDownloader

download_http = HTTPDownloader.download

# Repository downloaders
from ._figshare import FigshareDownloader

download_figshare = FigshareDownloader.download

# Biological database downloaders
from ._hgnc import HGNCDownloader

download_hgnc = HGNCDownloader.download
from ._ensembl import EnsemblDownloader

download_ensembl = EnsemblDownloader.download
from ._go import GODownloader

download_go = GODownloader.download

__all__ = [
    # Base classes and utilities
    "BaseDownloader",
    "DownloadError",
    "prepare_path",
    # Downloaders (all are namespace classes)
    "HTTPDownloader",
    "FigshareDownloader",
    "HGNCDownloader",
    "EnsemblDownloader",
    "GODownloader",
    "download_http",
    "download_figshare",
    "download_hgnc",
    "download_ensembl",
    "download_go",
]
