"""Model registry for managing model classes and architectures.

This module provides a registry system for organizing model classes with hierarchical
namespaces. It supports both eager registration (via decorators) and lazy registration
(via config), making it suitable for large-scale deep learning frameworks.

Key Features:
    - **Decorator-based Registration**: @registry.register()
    - **Hierarchical Organization**: Access models via paths like "GEARS.gnn".
    - **Lazy Loading**: Define models in config without importing modules until use.
    - **Config-driven Instantiation**: model = registry.build("name", **config).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from perturblab.utils import DependencyError, check_dependencies, get_logger

logger = get_logger()

__all__ = ["ModelRegistry", "register_lazy_models"]

T = TypeVar("T")


class ModelRegistry:
    """Registry for managing model classes with support for hierarchy and lazy loading.

    This class allows registering model classes via decorators and accessing them
    through a dictionary-like interface or a `build` factory method. It supports
    nested registries for hierarchical organization (e.g., `GEARS.gnn`, `GEARS.transformer`).

    Attributes:
        name (str): The name of the registry (e.g., "MODELS").
        parent (Optional[ModelRegistry]): The parent registry if this is a child.
    """

    def __init__(
        self,
        name: str,
        parent: ModelRegistry | None = None,
        auto_import: bool = False,
    ):
        """Initializes the ModelRegistry.

        Args:
            name: The registry identifier (e.g., "MODELS", "GEARS").
            parent: Parent registry for hierarchical organization. Defaults to None.
            auto_import: Whether to automatically import model modules to trigger
                registration. Defaults to False.
        """
        self._name = name
        self._parent = parent
        self._auto_import = auto_import

        # Eagerly initialized models (class objects)
        self._obj_map: dict[str, type[Any]] = {}

        # Child registries for hierarchy
        self._child_registries: dict[str, ModelRegistry] = {}

        # Lazy registration metadata: stores info to import models later.
        # Format: {model_name: {'module': '...', 'class': '...', 'dependencies': [...]}}
        self._lazy_map: dict[str, dict[str, Any]] = {}

        logger.debug(f"ModelRegistry created: {name}")

    @property
    def name(self) -> str:
        """Returns the registry name."""
        return self._name

    @property
    def parent(self) -> ModelRegistry | None:
        """Returns the parent registry."""
        return self._parent

    # =========================================================================
    # Registration Methods (Eager & Lazy)
    # =========================================================================

    def register(
        self,
        name: str | None = None,
        force: bool = False,
        module: str | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class.

        Args:
            name: Custom name for the model. If None, uses the class name.
            force: If True, allows overwriting existing registrations.
            module: Module name for documentation (auto-detected if None).

        Returns:
            A decorator function that registers the class.

        Raises:
            KeyError: If the name is already registered and `force` is False.

        Examples:
            >>> @MODELS.register()
            >>> class MyModel(nn.Module): ...
            >>>
            >>> @MODELS.register("CustomName")
            >>> class MyModel2(nn.Module): ...
        """

        def _register_wrapper(cls: type[T]) -> type[T]:
            key = name if name is not None else cls.__name__

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

            self._obj_map[key] = cls

            module_name = module or cls.__module__
            logger.debug(
                f"Registered model '{key}' in registry '{self._name}' "
                f"({module_name}.{cls.__name__})"
            )
            return cls

        return _register_wrapper

    def register_class(
        self,
        cls: type[T],
        name: str | None = None,
        force: bool = False,
    ) -> None:
        """Manually registers a class without using a decorator.

        Args:
            cls: The class to register.
            name: Custom name. If None, uses `cls.__name__`.
            force: Allow overwriting existing registrations.
        """
        key = name if name is not None else cls.__name__

        if key in self._obj_map and not force:
            raise KeyError(f"Model '{key}' already registered in '{self._name}'")

        self._obj_map[key] = cls
        logger.debug(f"Manually registered model '{key}' in registry '{self._name}'")

    def register_lazy(
        self,
        name: str,
        module: str,
        class_name: str,
        requirements: list[str] | None = None,
        dependencies: list[str] | None = None,
        force: bool = False,
    ) -> None:
        """Registers a model lazily without importing the module immediately.

        This is the recommended way to register built-in models to reduce startup time
        and circular import issues. The model module is only imported when the model
        is accessed via `build()` or `get()`.

        Args:
            name: The registration name (e.g., 'default', 'gnn').
            module: The full module path (e.g., 'perturblab.models.gears.model').
            class_name: The name of the class inside the module (e.g., 'GEARS').
            requirements: List of required package dependencies (e.g., ['torch_geometric']).
                If any are missing, raises DependencyError when loading.
            dependencies: List of optional package dependencies (e.g., ['accelerate']).
                If any are missing, logs info message when loading.
            force: Whether to overwrite existing registrations.

        Examples:
            >>> # In perturblab/models/gears/__init__.py
            >>> GEARS_REGISTRY.register_lazy(
            ...     name="default",
            ...     module="perturblab.models.gears.model",
            ...     class_name="GEARS",
            ...     requirements=['torch_geometric'],
            ...     dependencies=[]
            ... )
        """
        if name in self._obj_map:
            if not force:
                raise KeyError(
                    f"Model '{name}' is already eagerly registered in '{self._name}'. "
                    f"Use force=True to overwrite."
                )
            logger.warning(f"Overwriting eager model '{name}' with lazy registration.")

        if name in self._lazy_map:
            if not force:
                raise KeyError(
                    f"Model '{name}' is already lazily registered in '{self._name}'. "
                    f"Use force=True to overwrite."
                )
            logger.warning(f"Overwriting lazy model '{name}' in registry '{self._name}'.")

        self._lazy_map[name] = {
            "module": module,
            "class": class_name,
            "requirements": requirements or [],
            "dependencies": dependencies or [],
        }

        logger.debug(
            f"Lazy registered model '{name}' in registry '{self._name}' "
            f"(module: {module}, class: {class_name})"
        )

    # =========================================================================
    # Factory & Retrieval Methods
    # =========================================================================

    def build(self, key: str, *args, **kwargs) -> Any:
        """Instantiates a model from the registry using the provided configuration.

        Args:
            key: Model key or dot-path (e.g., "MLP" or "GEARS.gnn").
            *args: Positional arguments passed to the model constructor.
            **kwargs: Keyword arguments passed to the model constructor.

        Returns:
            Any: The instantiated model object.

        Raises:
            KeyError: If the model key is not found.
            TypeError: If the key points to a sub-registry instead of a class.
            Exception: Any exception raised during model instantiation.

        Examples:
            >>> model = MODELS.build("MLP", hidden_dim=64)
            >>> gnn = MODELS.build("GEARS.gnn", hidden_dim=128)
        """
        obj = self._get_no_default(key)

        if isinstance(obj, ModelRegistry):
            raise TypeError(
                f"Key '{key}' points to a registry, not a model class. "
                f"Cannot instantiate. Available models inside: "
                f"{list(obj.keys())}"
            )

        try:
            return obj(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to build model '{key}' from registry '{self._name}': {e}")
            raise

    def get(
        self, key: str, default: type[Any] | None = None
    ) -> type[Any] | ModelRegistry | None:
        """Retrieves a model class or sub-registry by key.

        Args:
            key: Model key or path (e.g., "MLP", "GEARS.gnn").
            default: Value to return if key is not found. Defaults to None.

        Returns:
            The model class, sub-registry, or the default value.
        """
        try:
            return self._get_no_default(key)
        except KeyError:
            return default

    def _get_no_default(self, key: str) -> type[Any] | ModelRegistry:
        """Internal retrieval method that raises KeyError if not found.

        Handles nested lookups and triggers lazy loading if necessary.
        """
        if "." in key:
            # Handle nested lookup: "GEARS.gnn" -> root="GEARS", sub_key="gnn"
            root, sub_key = key.split(".", 1)

            if root not in self._child_registries:
                raise KeyError(
                    f"Registry '{root}' not found in '{self._name}'. "
                    f"Available children: {list(self._child_registries.keys())}"
                )

            return self._child_registries[root]._get_no_default(sub_key)

        # 1. Check child registries (Namespaces have priority)
        if key in self._child_registries:
            return self._child_registries[key]

        # 2. Check eagerly registered models
        if key in self._obj_map:
            return self._obj_map[key]

        # 3. Try lazy loading
        if key in self._lazy_map:
            model_class = self._load_lazy_model(key)
            if model_class is not None:
                return model_class
            # If loading failed, fall through to raise KeyError

        # 4. Not found
        available = list(self._obj_map.keys()) + list(self._lazy_map.keys())
        raise KeyError(
            f"Model '{key}' not found in registry '{self._name}'. " f"Available models: {available}"
        )

    # =========================================================================
    # Hierarchy Management
    # =========================================================================

    def child(self, name: str, force: bool = False) -> ModelRegistry:
        """Creates and returns a child registry.

        Args:
            name: Name of the child registry.
            force: If True, overwrite existing child registry.

        Returns:
            ModelRegistry: The created child registry.
        """
        if name in self._child_registries:
            if not force:
                raise KeyError(
                    f"Child registry '{name}' already exists in '{self._name}'. "
                    f"Use force=True to overwrite."
                )
            logger.warning(f"Overwriting child registry '{name}' in '{self._name}'")

        child_registry = ModelRegistry(name, parent=self)
        self._child_registries[name] = child_registry
        logger.debug(f"Created child registry '{name}' under '{self._name}'")
        return child_registry

    def add_child(self, registry: ModelRegistry, name: str | None = None) -> None:
        """Attaches an existing registry as a child.

        Args:
            registry: The registry object to attach.
            name: Optional name for the child. If None, uses `registry.name`.
        """
        child_name = name if name is not None else registry.name

        if child_name in self._child_registries:
            logger.warning(f"Overwriting child registry '{child_name}' in '{self._name}'")

        self._child_registries[child_name] = registry
        registry._parent = self
        logger.debug(f"Added existing child registry '{child_name}' to '{self._name}'")

    # =========================================================================
    # Internal Lazy Loading Logic
    # =========================================================================

    def _load_lazy_model(self, name: str) -> type[Any] | None:
        """Internal method to import and cache a lazily registered model.

        Args:
            name: The model key.

        Returns:
            The imported class object, or None if import failed.
        """
        if name not in self._lazy_map:
            return None

        metadata = self._lazy_map[name]
        module_path = metadata["module"]
        class_name = metadata["class"]
        requirements = metadata["requirements"]
        dependencies = metadata["dependencies"]

        # Check dependencies using the utility function
        if requirements or dependencies:
            try:
                check_dependencies(
                    requirements=requirements,
                    dependencies=dependencies,
                    package_name=f"Model '{name}'",
                )
            except DependencyError:
                # Re-raise to prevent model loading if required deps are missing
                raise

        try:
            if module_path not in sys.modules:
                logger.info(f"Loading module '{module_path}' for model '{name}'...")

            module = importlib.import_module(module_path)

            if not hasattr(module, class_name):
                logger.error(
                    f"Module '{module_path}' does not contain class '{class_name}'. "
                    f"Available contents: {dir(module)}"
                )
                return None

            model_class = getattr(module, class_name)

            # Promote from lazy map to eager map (cache result)
            self._obj_map[name] = model_class
            del self._lazy_map[name]

            logger.info(f"✓ Successfully loaded model '{name}' from '{module_path}'")
            return model_class

        except ImportError as e:
            logger.error(
                f"Failed to import module '{module_path}' for model '{name}': {e}\n"
                f"This may indicate missing dependencies or incorrect paths."
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading model '{name}': {e}")
            return None

    # =========================================================================
    # Dict-like Interface & Introspection
    # =========================================================================

    def keys(self) -> Iterator[str]:
        """Iterates over all model keys (eager and lazy, excluding child registries)."""
        yield from self._obj_map.keys()
        yield from self._lazy_map.keys()

    def values(self) -> Iterator[type[Any]]:
        """Iterates over all model classes (triggers loading of lazy models)."""
        for value in self._obj_map.values():
            yield value

        # Snapshot keys to avoid runtime error during iteration if map changes
        lazy_keys = list(self._lazy_map.keys())
        for key in lazy_keys:
            model_class = self._load_lazy_model(key)
            if model_class is not None:
                yield model_class

    def items(self) -> Iterator[tuple[str, type[Any]]]:
        """Iterates over (key, class) pairs (triggers loading of lazy models)."""
        yield from self._obj_map.items()

        lazy_keys = list(self._lazy_map.keys())
        for key in lazy_keys:
            model_class = self._load_lazy_model(key)
            if model_class is not None:
                yield key, model_class

    def list_keys(self, recursive: bool = False) -> list[str]:
        """Lists all registered model keys.

        Args:
            recursive: If True, includes keys from nested child registries using
                dot notation (e.g., 'GEARS.gnn').

        Returns:
            List[str]: A list of available model keys.
        """
        current_keys = list(self._obj_map.keys()) + list(self._lazy_map.keys())

        if not recursive:
            return current_keys

        # Recursive listing
        for child_name, child_registry in self._child_registries.items():
            nested_keys = child_registry.list_keys(recursive=True)
            current_keys.extend([f"{child_name}.{nk}" for nk in nested_keys])

        return current_keys

    def __getitem__(self, key: str) -> type[Any] | ModelRegistry:
        return self._get_no_default(key)

    def __contains__(self, key: str) -> bool:
        try:
            self._get_no_default(key)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._obj_map) + len(self._lazy_map)

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry '{self._name}' "
            f"models={len(self)} "
            f"children={list(self._child_registries.keys())}>"
        )

    def __str__(self) -> str:
        return self.__repr__()


# =============================================================================
# Utility Functions
# =============================================================================


def register_lazy_models(
    registry: ModelRegistry,
    models: dict[str, str],
    base_module: str,
    requirements: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> None:
    """Batch registers multiple models lazily to a registry.

    This utility function simplifies the common pattern of registering multiple
    model variants or components with the same dependencies.

    Args:
        registry: The ModelRegistry instance to register models to.
        models: Dictionary mapping model names to class names.
            Example: {"default": "GEARSModel", "GEARSModel": "GEARSModel"}
        base_module: Base module path where models are located.
            Example: "perturblab.models.gears._modeling.model"
        requirements: List of required package dependencies.
        dependencies: List of optional package dependencies.

    Examples:
        >>> # Register multiple GEARS models
        >>> register_lazy_models(
        ...     registry=GEARS_REGISTRY,
        ...     models={
        ...         "default": "GEARSModel",
        ...         "GEARSModel": "GEARSModel",
        ...     },
        ...     base_module="perturblab.models.gears._modeling.model",
        ...     requirements=["torch_geometric"],
        ...     dependencies=[]
        ... )

        >>> # Register scGPT components from _modeling/__init__.py exports
        >>> from perturblab.models.scgpt._modeling import __all__ as scgpt_models
        >>> register_lazy_models(
        ...     registry=SCGPT_REGISTRY,
        ...     models={name: name for name in scgpt_models},
        ...     base_module="perturblab.models.scgpt._modeling",
        ...     requirements=[],
        ...     dependencies=["flash_attn"]
        ... )
    """
    requirements = requirements or []
    dependencies = dependencies or []

    for model_name, class_name in models.items():
        registry.register_lazy(
            name=model_name,
            module=base_module,
            class_name=class_name,
            requirements=requirements,
            dependencies=dependencies,
        )
