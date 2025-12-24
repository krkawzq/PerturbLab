"""Resource implementations and dataset registry.

Provides:
- Resource implementations (File, Files, h5adFile)
- Curated dataset resources (lazy loaded)
- Unified ResourceRegistry with tree structure

All dataset registries are lazy loaded - they are only imported when first accessed.
"""

from perturblab.core import ResourceRegistry
from perturblab.utils import get_logger

# Always export resource implementations
from ._resources import File, Files, h5adFile

logger = get_logger()

__all__ = [
    # Resource implementations
    "File",
    "Files",
    "h5adFile",
    # Dataset registry (lazy)
    "dataset_registry",
    # Convenience functions
    "list_datasets",
    "get_dataset",
    "load_dataset",
]


# ============================================================================
# Lazy Dataset Registry Builder
# ============================================================================

_registry_cache: ResourceRegistry | None = None


def _build_dataset_registry() -> ResourceRegistry:
    """Build the dataset registry (lazy - only called on first access).

    Returns
    -------
    ResourceRegistry
        Hierarchical registry with structure:
        - scperturb/norman_2019
        - scperturb/replogle_2022_k562_essential
        - go/go_basic
        - go/gene2go_gears
    """
    global _registry_cache

    if _registry_cache is not None:
        return _registry_cache

    logger.debug("Building dataset registry (first access)...")

    # Lazy import dataset lists
    from ._go import GO_RESOURCES
    from ._scperturb import SCPERTURB_DATASETS

    # Create sub-registries for each source
    scperturb_registry = ResourceRegistry(
        key="scperturb", resources=SCPERTURB_DATASETS, duplicate_policy="merge"
    )

    go_registry = ResourceRegistry(key="go", resources=GO_RESOURCES, duplicate_policy="merge")

    # Create main registry
    main_registry = ResourceRegistry(
        key="datasets",
        resources=[scperturb_registry, go_registry],
        duplicate_policy="error",  # Top level should not have duplicates
    )

    _registry_cache = main_registry

    logger.debug(
        f"Dataset registry built with {len(SCPERTURB_DATASETS) + len(GO_RESOURCES)} "
        f"resources in tree structure"
    )

    return _registry_cache


# Lazy property for dataset_registry
class _RegistryProxy:
    """Proxy object for lazy registry access."""

    @property
    def _registry(self) -> ResourceRegistry:
        return _build_dataset_registry()

    def __getitem__(self, key: str):
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        if _registry_cache is None:
            return "<DatasetRegistry (lazy - not yet built)>"
        return repr(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()

    def get(self, key, default=None):
        return self._registry.get(key, default)


# Export lazy proxy as dataset_registry
dataset_registry = _RegistryProxy()


# ============================================================================
# Convenience Functions (with tree path support)
# ============================================================================


def list_datasets(path: str | None = None, recursive: bool = True) -> list[str]:
    """List available datasets in tree structure.

    Supports querying at any level of the tree hierarchy.

    Parameters
    ----------
    path : str, optional
        Tree path to query (e.g., 'scperturb', 'scperturb/split_name').
        If None, lists from root.
    recursive : bool, default=True
        If True, returns full paths like 'scperturb/norman_2019'.
        If False, returns only immediate children.

    Returns
    -------
    list of str
        List of dataset paths.

    Raises
    ------
    KeyError
        If path does not exist in the tree.

    Examples
    --------
    >>> from perturblab.data.resources import list_datasets
    >>>
    >>> # List all datasets (full tree)
    >>> all_datasets = list_datasets()
    >>> # ['scperturb/norman_2019', 'scperturb/replogle_2022...', 'go/go_basic', ...]
    >>>
    >>> # List only categories (non-recursive)
    >>> categories = list_datasets(recursive=False)
    >>> # ['scperturb', 'go']
    >>>
    >>> # List only scPerturb datasets
    >>> scperturb_datasets = list_datasets(path='scperturb')
    >>> # ['scperturb/norman_2019', 'scperturb/dixit_2016...', ...]
    >>>
    >>> # List scPerturb datasets (non-recursive - immediate children only)
    >>> scperturb_keys = list_datasets(path='scperturb', recursive=False)
    >>> # ['norman_2019', 'replogle_2022_k562_essential', ...]
    >>>
    >>> # Query nested path (if exists)
    >>> # nested = list_datasets(path='scperturb/subgroup')
    """
    registry = _build_dataset_registry()

    # Navigate to target registry
    if path:
        parts = path.split("/")
        current = registry

        for part in parts:
            try:
                current = current[part]
            except KeyError:
                available = list(current.keys()) if hasattr(current, "keys") else []
                raise KeyError(
                    f"Path '{path}' not found. "
                    f"Available at '{'/'.join(parts[:parts.index(part)])}': {available}"
                )

            # If we hit a non-registry resource, can't go deeper
            if not isinstance(current, ResourceRegistry):
                raise KeyError(
                    f"Path '{path}' points to a resource, not a category. "
                    f"Cannot list children of a resource."
                )

        target_registry = current
        prefix = path + "/"
    else:
        target_registry = registry
        prefix = ""

    # List from target registry
    if not recursive:
        # Return only immediate children (no prefix)
        return sorted(list(target_registry.keys()))

    # Recursive listing with full paths
    paths = []

    def _collect_paths(reg: ResourceRegistry, current_prefix: str):
        """Recursively collect all paths."""
        for key in reg.keys():
            item = reg[key]
            full_path = f"{current_prefix}{key}"

            if isinstance(item, ResourceRegistry):
                # Recurse into nested registry
                _collect_paths(item, f"{full_path}/")
            else:
                # It's a resource, add the path
                paths.append(full_path)

    _collect_paths(target_registry, prefix)

    return sorted(paths)


def get_dataset(path: str) -> File | Files | h5adFile:
    """Get dataset resource by tree path.

    Parameters
    ----------
    path : str
        Dataset path in format 'category/dataset_key'.
        Examples: 'scperturb/norman_2019', 'go/go_basic'

    Returns
    -------
    File or Files or h5adFile
        Dataset resource.

    Raises
    ------
    KeyError
        If dataset not found.
    ValueError
        If path format is invalid.

    Examples
    --------
    >>> from perturblab.data.resources import get_dataset
    >>>
    >>> # Get by tree path
    >>> dataset = get_dataset('scperturb/norman_2019')
    >>> adata = dataset.load_adata()
    >>>
    >>> # Get GO resource
    >>> go_file = get_dataset('go/go_basic')
    >>> path = go_file.load()
    """
    registry = _build_dataset_registry()

    # Parse path
    parts = path.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Invalid dataset path '{path}'. "
            f"Expected format: 'category/dataset_key'. "
            f"Available categories: {list(registry.keys())}"
        )

    category = parts[0]
    dataset_key = "/".join(parts[1:])  # Support nested paths

    # Navigate to resource
    try:
        cat_registry = registry[category]

        if isinstance(cat_registry, ResourceRegistry):
            return cat_registry[dataset_key]
        else:
            # Category is a direct resource (edge case)
            if len(parts) == 1:
                return cat_registry
            raise KeyError(f"'{category}' is not a category")

    except KeyError as e:
        available = list_datasets(category=None, recursive=True)
        raise KeyError(
            f"Dataset '{path}' not found. "
            f"Available datasets: {available[:5]}... (total: {len(available)})"
        )


def load_dataset(path: str) -> "Path":  # type: ignore
    """Load dataset by tree path and return local Path.

    This function ensures the dataset file is materialized locally (downloading
    if necessary) and returns the Path. Users can then load the file themselves.

    Parameters
    ----------
    path : str
        Dataset path in format 'category/dataset_key'.
        Examples: 'scperturb/norman_2019', 'go/go_basic'

    Returns
    -------
    Path
        Path to local dataset file or directory.

    Examples
    --------
    >>> from perturblab.data.resources import load_dataset
    >>>
    >>> # Load scPerturb dataset (returns Path to .h5ad)
    >>> h5ad_path = load_dataset('scperturb/norman_2019')
    >>>
    >>> # Then load into AnnData yourself
    >>> import anndata as ad
    >>> adata = ad.read_h5ad(h5ad_path)
    >>> print(adata.shape)
    >>>
    >>> # Load GO file (returns Path to .obo)
    >>> go_path = load_dataset('go/go_basic')
    >>> print(go_path)
    """
    from pathlib import Path

    resource = get_dataset(path)

    # All resources should return Path via .load()
    return resource.load()


def get_datasets_dict(flat: bool = True):
    """Get all datasets as a dictionary.

    Parameters
    ----------
    flat : bool, default=True
        If True, returns flat dict with paths as keys ('scperturb/norman_2019').
        If False, returns nested dict structure.

    Returns
    -------
    dict
        Dictionary of datasets.

    Examples
    --------
    >>> from perturblab.data.resources import get_datasets_dict
    >>>
    >>> # Flat dictionary
    >>> datasets = get_datasets_dict(flat=True)
    >>> norman = datasets['scperturb/norman_2019']
    >>>
    >>> # Nested dictionary
    >>> info = get_datasets_dict(flat=False)
    >>> print(info['key'])
    >>> print(info['num_resources'])
    """
    registry = _build_dataset_registry()

    if flat:
        return registry.to_flat_dict()
    else:
        return registry.to_info_dict()


# Add to exports
__all__.extend(["print_dataset_tree", "get_datasets_dict"])


def print_dataset_tree():
    """Print dataset registry as a tree structure.

    Examples
    --------
    >>> from perturblab.data.resources import print_dataset_tree
    >>> print_dataset_tree()
    datasets/
    ├── scperturb/
    │   ├── norman_2019
    │   ├── replogle_2022_k562_essential
    │   └── ...
    └── go/
        ├── go_basic
        └── gene2go_gears
    """
    registry = _build_dataset_registry()

    print(f"{registry.key}/")

    categories = list(registry.keys())
    for i, cat_key in enumerate(categories):
        is_last_cat = i == len(categories) - 1
        cat_prefix = "└──" if is_last_cat else "├──"

        print(f"{cat_prefix} {cat_key}/")

        cat_registry = registry[cat_key]
        if isinstance(cat_registry, ResourceRegistry):
            datasets = list(cat_registry.keys())
            for j, ds_key in enumerate(datasets):
                is_last_ds = j == len(datasets) - 1
                ds_prefix = "    └──" if is_last_cat else "│   └──"
                mid_prefix = "    ├──" if is_last_cat else "│   ├──"

                prefix = ds_prefix if is_last_ds else mid_prefix
                print(f"{prefix} {ds_key}")
