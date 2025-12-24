"""Benchmark dataset registry and loaders.

Provides unified access to curated datasets from multiple sources:
- scPerturb: 31+ single-cell perturbation datasets
- PerturbBase: 122+ datasets (Chinese database)
- GO: Gene Ontology files and annotations
"""

from ._registry import (
    BenchmarkRegistry,
    list_benchmark,
    get_benchmark_info,
    list_sources,
    load_benchmark,
    load_dataset,
)
from ._scperturb import SCPERTURB_DATASETS
from ._perturbase import PERTURBASE_DATASETS
from ._go import GO_DATASETS


# Combine all datasets
ALL_DATASETS = SCPERTURB_DATASETS + PERTURBASE_DATASETS + GO_DATASETS

# Create global registry
_global_registry = BenchmarkRegistry(ALL_DATASETS)


# Public API (delegates to global registry)

def list_benchmark_datasets(source=None, perturbation_type=None, names_only=False):
    """List available benchmark datasets."""
    return list_benchmark(_global_registry, source, perturbation_type, names_only)


def get_dataset_info(dataset_name, source=None):
    """Get dataset metadata."""
    return get_benchmark_info(_global_registry, dataset_name, source)


def list_dataset_sources(dataset_name):
    """List available sources for a dataset."""
    return list_sources(_global_registry, dataset_name)


def load_benchmark_dataset(dataset_name, source=None, force_download=False):
    """Load a benchmark dataset (H5AD only)."""
    return load_benchmark(_global_registry, dataset_name, source, force_download)


def load_any_dataset(dataset_name, source=None, force_download=False, auto_fallback=True, **kwargs):
    """Load any dataset with auto-fallback."""
    return load_dataset(_global_registry, dataset_name, source, force_download, auto_fallback, **kwargs)


__all__ = [
    'list_benchmark_datasets',
    'get_dataset_info',
    'list_dataset_sources',
    'load_benchmark_dataset',
    'load_any_dataset',
]

