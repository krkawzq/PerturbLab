"""Resource registry for managing collections of resources.

Provides a registry system for organizing and accessing multiple resources
with lazy initialization and flexible duplicate handling.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from perturblab.utils import get_logger

from .resource import Resource

logger = get_logger()

__all__ = ["ResourceRegistry"]


class ResourceRegistry:
    """Registry for managing collections of resources.

    ResourceRegistry provides a namespace for organizing multiple resources
    with lazy initialization - the internal resource dictionary is only built
    when first accessed.

    Features:
    - Lazy initialization (build dict only when queried)
    - Nested registries (registry of registries)
    - Duplicate key handling policies
    - Dict-like access interface

    Parameters
    ----------
    key : str
        Unique identifier for this registry.
    resources : list of Resource or ResourceRegistry, default=[]
        List of resources to register.
    duplicate_policy : {"error", "merge", "drop"}, default="merge"
        How to handle duplicate keys:
        - "error": Raise error on duplicate keys
        - "merge": Create nested registry with duplicates
        - "drop": Keep first resource, drop rest

    Examples
    --------
    >>> from perturblab.core import ResourceRegistry
    >>> from perturblab.data.resources import File
    >>>
    >>> # Create resources
    >>> file1 = File(key='data1', local_path='/data/file1.csv')
    >>> file2 = File(key='data2', local_path='/data/file2.csv')
    >>>
    >>> # Create registry (lazy - no dict built yet)
    >>> registry = ResourceRegistry(
    ...     key='my_datasets',
    ...     resources=[file1, file2]
    ... )
    >>>
    >>> # Access resource (triggers dict building)
    >>> data1 = registry['data1']
    >>> path = data1.load()

    >>> # Nested registries
    >>> sub_registry = ResourceRegistry(key='subgroup', resources=[file3, file4])
    >>> main_registry = ResourceRegistry(
    ...     key='main',
    ...     resources=[file1, file2, sub_registry]
    ... )
    >>> # Access nested: main_registry['subgroup']['file3']
    """

    def __init__(
        self,
        key: str,
        resources: list[Resource | ResourceRegistry] | None = None,
        duplicate_policy: Literal["error", "merge", "drop"] = "merge",
        replace_slash: bool = True,
    ):
        """Initialize resource registry.

        Note: No dict building happens here - purely descriptor creation.

        Parameters
        ----------
        key : str
            Registry identifier.
        resources : list, optional
            List of resources to register.
        duplicate_policy : {"error", "merge", "drop"}, default="merge"
            How to handle duplicate keys.
        replace_slash : bool, default=True
            If True, replace '/' in resource keys with '.'.
            This prevents path confusion in hierarchical registries.
        """
        self.key = key
        self.duplicate_policy = duplicate_policy
        self.replace_slash = replace_slash
        self._resources = resources or []

        # Lazy-initialized dict
        self._resources_dict: dict[str, Resource | ResourceRegistry] | None = None

        logger.debug(f"ResourceRegistry created: {key} ({len(self._resources)} resources)")

    def _ensure_built(self) -> None:
        """Build the internal resource dict if not already built (lazy initialization)."""
        if self._resources_dict is not None:
            return

        logger.debug(f"Building registry '{self.key}' with {len(self._resources)} resources")

        self._resources_dict = {}
        duplicates: dict[str, list] = {}

        # Build dict and track duplicates
        for resource in self._resources:
            res_key = resource.key

            # Handle slash in keys
            if "/" in res_key:
                if self.replace_slash:
                    # Replace / with .
                    original_key = res_key
                    res_key = res_key.replace("/", ".")
                    logger.debug(f"Replaced '/' in key: '{original_key}' -> '{res_key}'")
                else:
                    # Raise error if not allowing slash
                    raise ValueError(
                        f"Resource key '{res_key}' contains '/'. "
                        f"This is not allowed in flat registries. "
                        f"Set replace_slash=True to auto-replace with '.'."
                    )

            if res_key in self._resources_dict:
                # Track duplicate
                if res_key not in duplicates:
                    duplicates[res_key] = [self._resources_dict[res_key]]
                duplicates[res_key].append(resource)
            else:
                self._resources_dict[res_key] = resource

        # Handle duplicates according to policy
        if duplicates:
            logger.warning(
                f"Registry '{self.key}' has {len(duplicates)} duplicate keys: "
                f"{list(duplicates.keys())}"
            )

            for dup_key, dup_resources in duplicates.items():
                if self.duplicate_policy == "error":
                    raise ValueError(f"Duplicate resource key '{dup_key}' in registry '{self.key}'")
                elif self.duplicate_policy == "merge":
                    # Create nested registry for duplicates
                    self._resources_dict[dup_key] = self._merge_duplicates(dup_key, dup_resources)
                elif self.duplicate_policy == "drop":
                    # Keep first one (already in dict), drop the rest
                    logger.debug(
                        f"Dropping {len(dup_resources) - 1} duplicate(s) for key '{dup_key}'"
                    )

        logger.debug(f"Registry '{self.key}' built with {len(self._resources_dict)} entries")

    @staticmethod
    def _merge_duplicates(
        key: str, resources: list[Resource | ResourceRegistry]
    ) -> ResourceRegistry:
        """Merge duplicate resources into a nested registry.

        Creates a new registry where each duplicate is accessible by an index.

        Parameters
        ----------
        key : str
            The duplicated key.
        resources : list
            List of resources with the same key.

        Returns
        -------
        ResourceRegistry
            A new nested registry with indexed access.

        Examples
        --------
        If resources have duplicate key 'data', they become:
        - nested_registry['0'] -> first resource
        - nested_registry['1'] -> second resource
        - nested_registry['2'] -> third resource
        """
        # Create a wrapper registry with index-based access
        nested = ResourceRegistry(key=key, duplicate_policy="error")

        # Manually set the dict to avoid key conflicts
        nested._resources_dict = {}
        for i, resource in enumerate(resources):
            # Use index as key
            indexed_key = str(i)
            nested._resources_dict[indexed_key] = resource

        # Mark as built to avoid re-processing
        nested._resources = []  # Clear to prevent rebuilding

        logger.debug(
            f"Merged {len(resources)} duplicates for key '{key}' "
            f"with indexed keys: {list(nested._resources_dict.keys())}"
        )

        return nested

    def add(self, resource: Resource | ResourceRegistry) -> None:
        """Add a resource to the registry.

        If registry dict is already built, it will be invalidated and rebuilt
        on next access.

        Parameters
        ----------
        resource : Resource or ResourceRegistry
            Resource to add.

        Examples
        --------
        >>> registry = ResourceRegistry(key='datasets')
        >>> file = File(key='data1', local_path='file1.csv')
        >>> registry.add(file)
        >>> data = registry['data1']
        """
        self._resources.append(resource)

        # Invalidate built dict to force rebuild
        if self._resources_dict is not None:
            logger.debug(f"Registry '{self.key}' invalidated, will rebuild on next access")
            self._resources_dict = None

    def remove(self, key: str) -> None:
        """Remove a resource by key.

        Parameters
        ----------
        key : str
            Resource key to remove.

        Raises
        ------
        KeyError
            If key not found.
        """
        self._resources = [r for r in self._resources if r.key != key]

        # Invalidate dict
        if self._resources_dict is not None:
            self._resources_dict = None

    # =========================================================================
    # Dict-like Interface
    # =========================================================================

    def __getitem__(self, key: str) -> Resource | ResourceRegistry:
        """Get resource by key (dict-like access).

        Triggers lazy dict building on first access.

        Parameters
        ----------
        key : str
            Resource key.

        Returns
        -------
        Resource or ResourceRegistry
            The requested resource.

        Raises
        ------
        KeyError
            If key not found.

        Examples
        --------
        >>> registry = ResourceRegistry(key='datasets', resources=[...])
        >>> data = registry['norman_2019']
        >>> adata = data.load()
        """
        self._ensure_built()
        return self._resources_dict[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in registry.

        Parameters
        ----------
        key : str
            Resource key.

        Returns
        -------
        bool
            True if key exists.

        Examples
        --------
        >>> if 'norman_2019' in registry:
        ...     data = registry['norman_2019']
        """
        self._ensure_built()
        return key in self._resources_dict

    def get(
        self, key: str, default: Resource | ResourceRegistry | None = None
    ) -> Resource | ResourceRegistry | None:
        """Get resource by key with default fallback.

        Parameters
        ----------
        key : str
            Resource key.
        default : Resource or ResourceRegistry or None, optional
            Default value if key not found.

        Returns
        -------
        Resource or ResourceRegistry or None
            The requested resource or default.

        Examples
        --------
        >>> data = registry.get('norman_2019')
        >>> if data:
        ...     adata = data.load()
        """
        self._ensure_built()
        return self._resources_dict.get(key, default)

    def keys(self) -> Iterator[str]:
        """Get all resource keys.

        Yields
        ------
        str
            Resource keys.

        Examples
        --------
        >>> for key in registry.keys():
        ...     print(key)
        """
        self._ensure_built()
        return iter(self._resources_dict.keys())

    def values(self) -> Iterator[Resource | ResourceRegistry]:
        """Get all resources.

        Yields
        ------
        Resource or ResourceRegistry
            Resources.

        Examples
        --------
        >>> for resource in registry.values():
        ...     print(resource.key)
        """
        self._ensure_built()
        return iter(self._resources_dict.values())

    def items(self) -> Iterator[tuple[str, Resource | ResourceRegistry]]:
        """Get all (key, resource) pairs.

        Yields
        ------
        tuple
            (key, resource) pairs.

        Examples
        --------
        >>> for key, resource in registry.items():
        ...     print(f"{key}: {resource}")
        """
        self._ensure_built()
        return iter(self._resources_dict.items())

    def __len__(self) -> int:
        """Get number of resources in registry."""
        self._ensure_built()
        return len(self._resources_dict)

    def __iter__(self) -> Iterator[str]:
        """Iterate over resource keys."""
        return self.keys()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_keys(self, recursive: bool = False) -> list[str]:
        """List all resource keys.

        Parameters
        ----------
        recursive : bool, default=False
            If True, recursively list keys from nested registries.

        Returns
        -------
        list of str
            List of resource keys.

        Examples
        --------
        >>> keys = registry.list_keys()
        >>> print(keys)
        ['norman_2019', 'adamson_2016', 'dixit_2016']

        >>> # Recursive listing (includes nested registries)
        >>> keys = registry.list_keys(recursive=True)
        >>> print(keys)
        ['norman_2019', 'subgroup.file1', 'subgroup.file2']
        """
        self._ensure_built()

        if not recursive:
            return list(self._resources_dict.keys())

        # Recursive listing
        keys = []
        for key, resource in self._resources_dict.items():
            if isinstance(resource, ResourceRegistry):
                # Add nested keys with prefix
                nested_keys = resource.list_keys(recursive=True)
                keys.extend([f"{key}.{nk}" for nk in nested_keys])
            else:
                keys.append(key)

        return keys

    def to_dict(self) -> dict:
        """Serialize registry to dictionary.

        Returns
        -------
        dict
            Serialized registry structure.

        Examples
        --------
        >>> config = registry.to_dict()
        >>> # Can be saved to JSON
        >>> import json
        >>> json.dump(config, open('registry.json', 'w'))
        """
        self._ensure_built()

        return {
            "key": self.key,
            "duplicate_policy": self.duplicate_policy,
            "replace_slash": self.replace_slash,
            "resources": {
                k: v.to_dict() if hasattr(v, "to_dict") else str(v)
                for k, v in self._resources_dict.items()
            },
        }

    def to_flat_dict(self, prefix: str = "") -> dict[str, Resource]:
        """Convert registry to flat dictionary with full paths as keys.

        Recursively flattens nested registries into a single-level dict
        with keys like 'category/dataset'.

        Parameters
        ----------
        prefix : str, default=''
            Prefix for keys (used internally for recursion).

        Returns
        -------
        dict[str, Resource]
            Flat dictionary mapping full paths to resources.

        Examples
        --------
        >>> registry = ResourceRegistry(key='main', resources=[...])
        >>> flat = registry.to_flat_dict()
        >>> # {'scperturb/norman_2019': <Resource>, 'go/go_basic': <Resource>, ...}
        >>>
        >>> # Access resource directly
        >>> resource = flat['scperturb/norman_2019']
        """
        self._ensure_built()

        result = {}

        for key, item in self._resources_dict.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(item, ResourceRegistry):
                # Recursively flatten nested registry
                nested = item.to_flat_dict(prefix=f"{full_key}/")
                result.update(nested)
            else:
                # It's a resource
                result[full_key] = item

        return result

    def to_info_dict(self, recursive: bool = True) -> dict:
        """Get registry information as nested dictionary.

        Parameters
        ----------
        recursive : bool, default=True
            If True, includes nested registry information.

        Returns
        -------
        dict
            Information dictionary with keys:
            - 'key': Registry key
            - 'num_resources': Number of resources
            - 'keys': List of resource keys
            - 'resources': Resource info (if recursive)

        Examples
        --------
        >>> info = registry.to_info_dict()
        >>> print(info['num_resources'])
        >>> print(info['keys'])
        """
        self._ensure_built()

        info = {
            "key": self.key,
            "num_resources": len(self._resources_dict),
            "keys": list(self._resources_dict.keys()),
        }

        if recursive:
            info["resources"] = {}
            for key, item in self._resources_dict.items():
                if isinstance(item, ResourceRegistry):
                    info["resources"][key] = item.to_info_dict(recursive=True)
                elif hasattr(item, "get_info"):
                    info["resources"][key] = item.get_info()
                else:
                    info["resources"][key] = str(item)

        return info

    def __repr__(self) -> str:
        """String representation."""
        if self._resources_dict is None:
            status = "lazy"
            count = len(self._resources)
        else:
            status = "built"
            count = len(self._resources_dict)

        return f"<ResourceRegistry '{self.key}' ({count} resources, {status})>"

    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()
