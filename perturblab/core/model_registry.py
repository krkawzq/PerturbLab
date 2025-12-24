"""Model registry for managing model classes and architectures.

Provides a decorator-based registration system for organizing model classes
with hierarchical namespaces. Commonly used in deep learning frameworks like
Hugging Face Transformers, OpenMMLab, and Detectron2.

Key Features:
- Decorator-based registration (@registry.register())
- Hierarchical organization (GEARS.gnn, GEARS.transformer)
- Config-driven instantiation (model = MODELS.build("GEARS.gnn", **config))
- Lazy vs eager loading support
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional, Type, TypeVar

from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["ModelRegistry"]

T = TypeVar("T")


class ModelRegistry:
    """Registry for managing model classes with decorator support.

    Allows registering model classes via decorators and accessing them
    through a dict-like interface or a build method. Supports nested
    registries for hierarchical organization (e.g., GEARS.gnn).

    This follows the same design philosophy as ResourceRegistry but is
    optimized for model class management with decorator registration.

    Parameters
    ----------
    name : str
        Name of this registry (e.g., "MODELS", "GEARS", "optimizers").
    parent : ModelRegistry, optional
        Parent registry for hierarchical organization.
    auto_import : bool, default=False
        If True, automatically import all model modules to trigger
        decorator registration. Requires setting up import paths.

    Examples
    --------
    >>> # 1. Create global registry
    >>> MODELS = ModelRegistry("MODELS")
    >>>
    >>> # 2. Register models via decorator
    >>> @MODELS.register()
    >>> class MLP(nn.Module):
    ...     def __init__(self, hidden_dim):
    ...         super().__init__()
    ...         self.linear = nn.Linear(hidden_dim, hidden_dim)
    >>>
    >>> @MODELS.register("Transformer")  # Custom name
    >>> class MyTransformer(nn.Module):
    ...     pass
    >>>
    >>> # 3. Create sub-registry for specific method
    >>> GEARS_MODELS = ModelRegistry("GEARS")
    >>> MODELS.add_child(GEARS_MODELS)
    >>>
    >>> @GEARS_MODELS.register("gnn")
    >>> class GearsGNN(nn.Module):
    ...     pass
    >>>
    >>> # 4. Build models
    >>> model = MODELS.build("MLP", hidden_dim=64)
    >>> gnn = MODELS.build("GEARS.gnn", hidden_dim=128)
    >>>
    >>> # 5. Check available models
    >>> print(MODELS.list_keys())
    >>> print(MODELS.list_keys(recursive=True))  # Include nested

    Notes
    -----
    **Import Order Issue**: This is the biggest pitfall of decorator-based
    registration. If a Python file containing model definitions is never
    imported, the decorator won't execute and the model won't be registered.

    **Solutions**:
    1. Explicitly import all model files in `__init__.py`
    2. Use `auto_import=True` with auto-discovery
    3. Document which modules need to be imported

    **Design Philosophy**:
    - ResourceRegistry: Lazy initialization (build dict on first access)
    - ModelRegistry: Eager registration (register immediately on import)

    This difference is intentional:
    - Resources are external (files, URLs) - lazy loading saves startup time
    - Models are classes in memory - eager registration ensures availability
    """

    def __init__(
        self,
        name: str,
        parent: Optional[ModelRegistry] = None,
        auto_import: bool = False,
    ):
        """Initialize model registry.

        Parameters
        ----------
        name : str
            Registry identifier (e.g., "MODELS", "GEARS").
        parent : ModelRegistry, optional
            Parent registry for hierarchical organization.
        auto_import : bool, default=False
            Whether to automatically import model modules.
        """
        self._name = name
        self._parent = parent
        self._auto_import = auto_import

        # Eagerly initialized (unlike ResourceRegistry)
        self._obj_map: Dict[str, Type[Any]] = {}
        self._child_registries: Dict[str, ModelRegistry] = {}

        logger.debug(f"ModelRegistry created: {name}")

    @property
    def name(self) -> str:
        """Registry name."""
        return self._name

    @property
    def parent(self) -> Optional[ModelRegistry]:
        """Parent registry."""
        return self._parent

    # =========================================================================
    # Decorator Registration
    # =========================================================================

    def register(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[str] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class.

        Parameters
        ----------
        name : str, optional
            Custom name for the model. If None, uses class.__name__.
        force : bool, default=False
            If True, allow overwriting existing registrations.
        module : str, optional
            Module name for documentation (auto-detected if None).

        Returns
        -------
        Callable
            Decorator function.

        Examples
        --------
        >>> @MODELS.register()
        >>> class MyModel(nn.Module):
        ...     pass
        >>>
        >>> @MODELS.register("CustomName")
        >>> class MyModel2(nn.Module):
        ...     pass
        >>>
        >>> @MODELS.register(force=True)  # Allow overwrite
        >>> class MyModel(nn.Module):
        ...     pass
        """

        def _register_wrapper(cls: Type[T]) -> Type[T]:
            # Use provided name or class name
            key = name if name is not None else cls.__name__

            # Check for duplicates
            if key in self._obj_map:
                if not force:
                    existing = self._obj_map[key]
                    raise KeyError(
                        f"Model '{key}' is already registered in registry '{self._name}'. "
                        f"Existing: {existing.__module__}.{existing.__name__}, "
                        f"New: {cls.__module__}.{cls.__name__}. "
                        f"Use force=True to overwrite."
                    )
                else:
                    logger.warning(f"Overwriting model '{key}' in registry '{self._name}'")

            # Register the class
            self._obj_map[key] = cls

            # Log with module info
            module_name = module or cls.__module__
            logger.debug(
                f"Registered model '{key}' in registry '{self._name}' "
                f"({module_name}.{cls.__name__})"
            )

            return cls

        return _register_wrapper

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        """Alias for register() with clearer name for modules.

        This is just an alias for consistency with frameworks like MMDetection
        that use @MODELS.register_module().
        """
        return self.register(name=name, force=force)

    # =========================================================================
    # Manual Registration
    # =========================================================================

    def register_class(
        self,
        cls: Type[T],
        name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Manually register a class (non-decorator style).

        Parameters
        ----------
        cls : Type
            Class to register.
        name : str, optional
            Custom name. If None, uses cls.__name__.
        force : bool, default=False
            Allow overwriting existing registrations.

        Examples
        --------
        >>> class MyModel(nn.Module):
        ...     pass
        >>> MODELS.register_class(MyModel)
        >>> MODELS.register_class(MyModel, name="CustomName")
        """
        key = name if name is not None else cls.__name__

        if key in self._obj_map and not force:
            raise KeyError(f"Model '{key}' already registered in '{self._name}'")

        self._obj_map[key] = cls
        logger.debug(f"Manually registered model '{key}' in registry '{self._name}'")

    # =========================================================================
    # Hierarchical Organization
    # =========================================================================

    def child(self, name: str, force: bool = False) -> ModelRegistry:
        """Create and return a child registry.

        This is the primary method for creating hierarchical namespaces.
        It creates a new ModelRegistry, attaches it as a child, and returns it.

        Parameters
        ----------
        name : str
            Name of the child registry.
        force : bool, default=False
            If True, allow overwriting existing child registries.

        Returns
        -------
        ModelRegistry
            The newly created child registry.

        Examples
        --------
        >>> # Create child registry in one line
        >>> GEARS_MODELS = MODELS.child("GEARS")
        >>>
        >>> # Now register models to the child
        >>> @GEARS_MODELS.register("gnn")
        >>> class GearsGNN(nn.Module):
        ...     pass
        >>>
        >>> # Access via parent: MODELS.build("GEARS.gnn", ...)
        >>> model = MODELS.build("GEARS.gnn", hidden_dim=128)
        >>>
        >>> # Chain multiple levels
        >>> variants = GEARS_MODELS.child("variants")
        >>> @variants.register("v1")
        >>> class GearsGNNV1(nn.Module):
        ...     pass
        """
        if name in self._child_registries:
            if not force:
                raise KeyError(
                    f"Child registry '{name}' already exists in '{self._name}'. "
                    f"Use force=True to overwrite or use a different name."
                )
            logger.warning(f"Overwriting child registry '{name}' in '{self._name}'")

        # Create new child registry
        child_registry = ModelRegistry(name, parent=self)

        # Attach to parent
        self._child_registries[name] = child_registry

        logger.debug(f"Created child registry '{name}' under '{self._name}'")

        return child_registry

    def add_child(self, registry: ModelRegistry, name: Optional[str] = None) -> None:
        """Add an existing registry as a child.

        This is an advanced method for attaching pre-existing registries.
        For most use cases, prefer the `.child()` method instead.

        Parameters
        ----------
        registry : ModelRegistry
            Child registry to add.
        name : str, optional
            Name to use for the child. If None, uses registry.name.

        Examples
        --------
        >>> # Advanced: Add existing registry
        >>> external_registry = ModelRegistry("external")
        >>> # ... register some models ...
        >>> MODELS.add_child(external_registry)
        >>>
        >>> # Standard usage (prefer this):
        >>> GEARS_MODELS = MODELS.child("GEARS")
        """
        child_name = name if name is not None else registry.name

        if child_name in self._child_registries:
            logger.warning(f"Overwriting child registry '{child_name}' in '{self._name}'")

        self._child_registries[child_name] = registry
        registry._parent = self

        logger.debug(f"Added existing child registry '{child_name}' to '{self._name}'")

    def remove_child(self, name: str) -> None:
        """Remove a child registry.

        Parameters
        ----------
        name : str
            Name of child registry to remove.
        """
        if name in self._child_registries:
            del self._child_registries[name]
            logger.debug(f"Removed child registry '{name}' from '{self._name}'")
        else:
            raise KeyError(f"Child registry '{name}' not found in '{self._name}'")

    # =========================================================================
    # Access Methods
    # =========================================================================

    def get(
        self, key: str, default: Optional[Type[Any]] = None
    ) -> Type[Any] | ModelRegistry | None:
        """Get a model class or sub-registry by key.

        Supports dot notation for nested access (e.g., 'GEARS.gnn').
        Returns child registries for intermediate paths (e.g., 'GEARS').

        Parameters
        ----------
        key : str
            Model key or path (e.g., "MLP", "GEARS", or "GEARS.gnn").
        default : Type, optional
            Default value if key not found.

        Returns
        -------
        Type or ModelRegistry or None
            Model class, sub-registry, or default.

        Examples
        --------
        >>> cls = MODELS.get("MLP")
        >>> model = cls(hidden_dim=64)
        >>>
        >>> # Get sub-registry (for chaining)
        >>> gears = MODELS.get("GEARS")  # Returns ModelRegistry
        >>> gnn_cls = gears.get("gnn")
        >>>
        >>> # Or use dot notation directly
        >>> gnn_cls = MODELS.get("GEARS.gnn")
        """
        try:
            return self._get_no_default(key)
        except KeyError:
            return default

    def _get_no_default(self, key: str) -> Type[Any] | ModelRegistry:
        """Internal get without default (raises KeyError)."""
        if "." in key:
            # Handle nested lookup: "GEARS.gnn"
            root, sub_key = key.split(".", 1)

            if root not in self._child_registries:
                raise KeyError(
                    f"Registry '{root}' not found in '{self._name}'. "
                    f"Available children: {list(self._child_registries.keys())}"
                )

            return self._child_registries[root]._get_no_default(sub_key)

        # Check child registries first (priority to namespaces)
        if key in self._child_registries:
            return self._child_registries[key]

        # Check models
        if key not in self._obj_map:
            raise KeyError(
                f"Model '{key}' not found in registry '{self._name}'. "
                f"Available: {list(self._obj_map.keys())}"
            )

        return self._obj_map[key]

    def build(self, key: str, *args, **kwargs) -> Any:
        """Instantiate a model from the registry.

        This is the primary method for config-driven model creation.

        Parameters
        ----------
        key : str
            Model key or path (e.g., "MLP" or "GEARS.gnn").
        *args
            Positional arguments for model __init__.
        **kwargs
            Keyword arguments for model __init__.

        Returns
        -------
        Any
            Instantiated model.

        Raises
        ------
        KeyError
            If model not found.
        TypeError
            If key points to a registry instead of a model class.

        Examples
        --------
        >>> model = MODELS.build("MLP", hidden_dim=64, num_layers=3)
        >>> gnn = MODELS.build("GEARS.gnn", hidden_dim=128)
        >>>
        >>> # Config-driven
        >>> config = {"model": "GEARS.gnn", "params": {"hidden_dim": 128}}
        >>> model = MODELS.build(config["model"], **config["params"])
        """
        obj = self._get_no_default(key)

        if isinstance(obj, ModelRegistry):
            raise TypeError(
                f"Key '{key}' points to a registry, not a model class. "
                f"Cannot instantiate. Available models in '{key}': "
                f"{list(obj._obj_map.keys())}"
            )

        try:
            return obj(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to build model '{key}' from registry '{self._name}': {e}")
            raise

    # =========================================================================
    # Dict-like Interface
    # =========================================================================

    def __getitem__(self, key: str) -> Type[Any] | ModelRegistry:
        """Get model class or sub-registry (dict-like access)."""
        return self._get_no_default(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in registry."""
        try:
            self._get_no_default(key)
            return True
        except KeyError:
            return False

    def keys(self) -> Iterator[str]:
        """Iterate over model keys (excludes child registries)."""
        return iter(self._obj_map.keys())

    def values(self) -> Iterator[Type[Any]]:
        """Iterate over model classes."""
        return iter(self._obj_map.values())

    def items(self) -> Iterator[tuple[str, Type[Any]]]:
        """Iterate over (key, model_class) pairs."""
        return iter(self._obj_map.items())

    def __len__(self) -> int:
        """Number of models (excludes child registries)."""
        return len(self._obj_map)

    def __iter__(self) -> Iterator[str]:
        """Iterate over model keys."""
        return self.keys()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_keys(self, recursive: bool = False, include_children: bool = False) -> list[str]:
        """List all model keys.

        Parameters
        ----------
        recursive : bool, default=False
            If True, include keys from nested registries with dot notation.
        include_children : bool, default=False
            If True and recursive=False, include child registry names.

        Returns
        -------
        list of str
            List of model keys.

        Examples
        --------
        >>> keys = MODELS.list_keys()
        >>> print(keys)
        ['MLP', 'Transformer']
        >>>
        >>> # Include nested models
        >>> keys = MODELS.list_keys(recursive=True)
        >>> print(keys)
        ['MLP', 'Transformer', 'GEARS.gnn', 'GEARS.transformer']
        """
        if not recursive:
            if include_children:
                return list(self._obj_map.keys()) + list(self._child_registries.keys())
            return list(self._obj_map.keys())

        # Recursive listing
        keys = list(self._obj_map.keys())
        for child_name, child_registry in self._child_registries.items():
            nested_keys = child_registry.list_keys(recursive=True)
            keys.extend([f"{child_name}.{nk}" for nk in nested_keys])

        return keys

    def list_modules(self) -> dict[str, str]:
        """List all registered models with their module paths.

        Returns
        -------
        dict[str, str]
            Mapping of model keys to module paths.

        Examples
        --------
        >>> modules = MODELS.list_modules()
        >>> print(modules)
        {'MLP': 'mypackage.models.mlp.MLP',
         'GEARS.gnn': 'perturblab.methods.gears.models.GearsGNN'}
        """
        modules = {}

        for key, cls in self._obj_map.items():
            modules[key] = f"{cls.__module__}.{cls.__name__}"

        # Recursively add child registries
        for child_name, child_registry in self._child_registries.items():
            child_modules = child_registry.list_modules()
            for key, module in child_modules.items():
                modules[f"{child_name}.{key}"] = module

        return modules

    def get_info(self) -> dict:
        """Get registry information.

        Returns
        -------
        dict
            Information dictionary with structure and contents.

        Examples
        --------
        >>> info = MODELS.get_info()
        >>> print(info['num_models'])
        >>> print(info['models'])
        >>> print(info['children'])
        """
        return {
            "name": self._name,
            "num_models": len(self._obj_map),
            "num_children": len(self._child_registries),
            "models": list(self._obj_map.keys()),
            "children": list(self._child_registries.keys()),
            "modules": {
                key: f"{cls.__module__}.{cls.__name__}" for key, cls in self._obj_map.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ModelRegistry '{self._name}' "
            f"models={len(self._obj_map)} "
            f"children={list(self._child_registries.keys())}>"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()


# =============================================================================
# Note on Global Registry
# =============================================================================

# The global MODELS instance is created in perturblab.models module
# with smart lazy loading capabilities. Import it from there:
#     from perturblab.models import MODELS
#
# This separation keeps core as pure type definitions while models
# handles the actual instance with intelligent loading logic.
