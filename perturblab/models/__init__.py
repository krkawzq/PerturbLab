"""Model registry with smart lazy loading.

This module provides the MODELS registry with intelligent on-demand loading:
- First access: Scans models directory structure (lightweight, no imports).
- Model lookup: Loads only the specific model's modules (targeted loading).
- Not found: Loads all models as a fallback.

Usage:
    >>> from perturblab.models import MODELS
    >>>
    >>> # Smart loading - only loads GEARS modules
    >>> model = MODELS.build("GEARS.gnn", hidden_dim=128)
    >>>
    >>> # List all available models (triggers full load if needed)
    >>> print(MODELS.list_keys(recursive=True))

Environment Variables:
    PERTURBLAB_DISABLE_AUTO_LOAD: Set to 'TRUE' to disable auto-loading.

Global Override:
    import perturblab
    perturblab._disable_auto_load = True  # Disable auto-loading
"""

import importlib
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, Union

from perturblab.core import ModelRegistry
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["MODELS", "Model"]


# =============================================================================
# Create Base Registry
# =============================================================================

# Create the base ModelRegistry instance.
# This is the actual global MODELS registry that the package uses internally.
_MODELS_BASE = ModelRegistry("MODELS")


# =============================================================================
# Smart Lazy Loading Utilities
# =============================================================================

_module_tree_cache: Optional[Dict[str, List[str]]] = None  # Cache: {model_name: [module_paths]}
_fully_loaded: bool = False  # Flag to indicate if all modules have been loaded


def _scan_model_tree() -> Dict[str, List[str]]:
    """Scans the perturblab.models directory structure without importing.

    Builds a lightweight cache of model names to their module paths.
    This allows targeted loading instead of loading everything at startup.

    Returns:
        Dict[str, List[str]]: A mapping of model names to a list of module paths.
            Example:
            {'gears': ['perturblab.models.gears', 'perturblab.models.gears._modeling.model']}
            {'scgpt': ['perturblab.models.scgpt', 'perturblab.models.scgpt._modeling.model']}
    """
    tree = {}

    try:
        # Use __file__ from this module to find models directory
        # This approach works even when perturblab.__file__ is None (namespace package)
        current_file = Path(__file__).resolve()
        models_path = current_file.parent

        if not models_path.exists():
            logger.debug(f"Models directory not found: {models_path}")
            return tree

        # Walk through the models directory
        for model_dir in models_path.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue

            model_name = model_dir.name
            modules = []

            # Collect all Python modules within this model directory
            for py_file in model_dir.rglob("*.py"):
                # Skip private files unless it's __init__.py
                if py_file.name.startswith("_") and py_file.name != "__init__.py":
                    continue

                # Convert file path to dotted module name
                rel_path = py_file.relative_to(models_path.parent)
                module_parts = list(rel_path.parts[:-1])  # Remove file extension from path
                if py_file.name != "__init__.py":
                    module_parts.append(py_file.stem)

                module_name = "perturblab." + ".".join(module_parts)
                modules.append(module_name)

            tree[model_name] = modules
            logger.debug(f"Scanned model '{model_name}': {len(modules)} modules found.")

    except Exception as e:
        logger.warning(f"Failed to scan models tree: {e}")

    return tree


def _load_model_modules(model_name: str) -> bool:
    """加载特定模型的所有模块。

    此函数采用容错策略：单个模块加载失败不会影响其他模块的加载。
    
    Args:
        model_name (str): 模型名称（例如 'gears', 'scgpt'）

    Returns:
        bool: 如果至少有一个模块加载成功则返回 True，否则返回 False
    """
    global _module_tree_cache

    # 确保树缓存存在
    if _module_tree_cache is None:
        _module_tree_cache = _scan_model_tree()

    if model_name not in _module_tree_cache:
        logger.debug(f"Model '{model_name}' not found in module tree cache.")
        return False

    modules = _module_tree_cache[model_name]
    loaded_any = False
    failed_modules = []

    for module_name in modules:
        # 跳过已导入的模块
        if module_name in sys.modules:
            logger.debug(f"Module already loaded: {module_name}")
            loaded_any = True
            continue

        try:
            importlib.import_module(module_name)
            logger.info(f"✓ Loaded module: {module_name}")
            loaded_any = True
        except Exception as e:
            # 记录失败的模块，但继续尝试加载其他模块
            failed_modules.append((module_name, e))
            logger.debug(f"✗ Failed to import {module_name}: {e}")

    # 如果有模块加载失败，提供汇总信息
    if failed_modules:
        error_summary = "\n".join([f"  - {mod}: {str(err)[:100]}" for mod, err in failed_modules])
        if loaded_any:
            logger.warning(
                f"Model '{model_name}' partially loaded: "
                f"{len(modules) - len(failed_modules)}/{len(modules)} modules succeeded.\n"
                f"Failed modules:\n{error_summary}"
            )
        else:
            logger.error(
                f"Model '{model_name}' failed to load: all {len(modules)} modules failed.\n"
                f"Errors:\n{error_summary}\n"
                f"This model will not be available in the registry."
            )

    return loaded_any


def _load_all_models() -> None:
    """Loads all model modules (fallback mechanism).

    This function is only called when a model cannot be found after targeted loading.
    """
    global _module_tree_cache, _fully_loaded

    if _fully_loaded:
        return

    logger.debug("Loading all model modules...")

    # Ensure tree cache exists
    if _module_tree_cache is None:
        _module_tree_cache = _scan_model_tree()

    # Load all models found in the tree
    for model_name in _module_tree_cache:
        _load_model_modules(model_name)

    _fully_loaded = True
    total_models = len(_MODELS_BASE.list_keys(recursive=True))
    logger.debug(f"✓ Loaded all models: {total_models} total registered.")


class _SmartLazyModelRegistry:
    """Smart lazy-loading wrapper for the MODELS registry.

    Implements intelligent on-demand model loading:
    1. First access: Scans models tree structure (lightweight, no imports).
    2. Model lookup: Tries targeted loading of specific model modules.
    3. Not found: Loads all models as a fallback.

    This ensures fast startup time while guaranteeing that all models are eventually available.
    """

    def __init__(self, registry: ModelRegistry):
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_initialized", False)

    def _check_user_override(self) -> bool:
        """Checks if the user has disabled auto-loading via global state or environment variable.

        Returns:
            bool: True if auto-loading should be disabled.
        """
        # Check environment variable
        env_disable = os.environ.get("PERTURBLAB_DISABLE_AUTO_LOAD", "").upper()
        if env_disable in ("TRUE", "1", "YES"):
            return True

        # Check if user has set a global flag in their code
        import perturblab

        if hasattr(perturblab, "_disable_auto_load"):
            return bool(perturblab._disable_auto_load)

        return False

    def _smart_load_model(self, key: str) -> bool:
        """Intelligently loads modules to find a model.

        Loading strategy:
        1. Check if model is already in registry
        2. Parse key to identify model (e.g. 'GEARS.gnn' -> 'gears')
        3. Load only that model's modules
        4. If still not found, load all models

        Args:
            key (str): Model key to find (e.g. 'GEARS.gnn' or 'MLP')

        Returns:
            bool: True if model was found or successfully loaded
        """
        # Check if user disabled auto-loading
        if self._check_user_override():
            logger.debug("Smart loading disabled by user configuration.")
            return False

        # Helper to check if key exists in registry
        def key_exists(k: str) -> bool:
            return (k in self._registry._child_registries or 
                    k in self._registry._obj_map)

        # Parse key to get model name
        if "." in key or "/" in key:
            # Support both formats: 'GEARS.gnn' and 'GEARS/gnn'
            separator = "." if "." in key else "/"
            model_name = key.split(separator)[0].lower()
        else:
            # Top-level model, use key itself as model name
            model_name = key.lower()

        # First try targeted loading
        logger.debug(f"Attempting targeted load for model: {model_name}")
        try:
            if _load_model_modules(model_name):
                # Check if model now exists in registry
                if key_exists(key):
                    logger.info(f"✓ Model '{key}' found via targeted loading.")
                    return True
                else:
                    logger.debug(
                        f"Modules for '{model_name}' loaded, but '{key}' not found in registry. "
                        f"Available: {list(self._registry._obj_map.keys())[:5]} / "
                        f"{list(self._registry._child_registries.keys())[:5]}"
                    )
        except Exception as e:
            logger.warning(f"Error during targeted loading of '{model_name}': {e}")

        # Fallback: load all models
        logger.debug(f"Model '{key}' not found via targeted loading, attempting full load...")
        try:
            _load_all_models()
        except Exception as e:
            logger.error(f"Failed to load all models: {e}")
            return False

        found = key_exists(key)
        if not found:
            logger.debug(
                f"Model '{key}' not found even after loading all modules. "
                f"Available models: {self._registry.list_keys(recursive=True)[:20]}"
            )
        
        return found

    def _initialize_if_needed(self):
        """Initializes the tree cache on first access (if not disabled)."""
        if object.__getattribute__(self, "_initialized"):
            return

        if not self._check_user_override():
            global _module_tree_cache
            if _module_tree_cache is None:
                logger.debug("Initializing models tree cache...")
                _module_tree_cache = _scan_model_tree()

        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name: str):
        # Do not intercept private attributes
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # Initialize lazy loading on first real access
        object.__getattribute__(self, "_initialize_if_needed")()

        # Delegate to the actual registry
        registry = object.__getattribute__(self, "_registry")
        attr = getattr(registry, name)

        # If the attribute is a child registry, wrap it with smart loading too
        if isinstance(attr, ModelRegistry):
            return _SmartLazyModelRegistry(attr)

        return attr

    def __getattr__(self, name: str):
        """Supports dot notation access (e.g., MODELS.GEARS.gnn)."""
        self._initialize_if_needed()

        # Try to get from child registries first
        if name in self._registry._child_registries:
            child = self._registry._child_registries[name]
            return _SmartLazyModelRegistry(child)

        # Try to get from registered objects
        if name in self._registry._obj_map:
            return self._registry._obj_map[name]

        # Not found locally - try smart loading
        # Try variations: exact, uppercase, lowercase
        for key_variant in [name, name.upper(), name.lower()]:
            if self._smart_load_model(key_variant):
                # Check child registries again after loading
                if key_variant in self._registry._child_registries:
                    child = self._registry._child_registries[key_variant]
                    return _SmartLazyModelRegistry(child)
                # Also check with original name (case-sensitive)
                if name in self._registry._child_registries:
                    child = self._registry._child_registries[name]
                    return _SmartLazyModelRegistry(child)
                # Check registered objects again
                if key_variant in self._registry._obj_map:
                    return self._registry._obj_map[key_variant]
                if name in self._registry._obj_map:
                    return self._registry._obj_map[name]

        raise AttributeError(
            f"'{self._registry.name}' registry has no model or child registry '{name}'. "
            f"Available: {list(self._registry._obj_map.keys()) + list(self._registry._child_registries.keys())}"
        )

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            registry = object.__getattribute__(self, "_registry")
            setattr(registry, name, value)

    def __getitem__(self, key: str):
        """Supports dict-style access (e.g., MODELS['GEARS']['gnn'])."""
        self._initialize_if_needed()

        # Try getting from registry first
        result = self._registry.get(key)
        if result is not None:
            if isinstance(result, ModelRegistry):
                return _SmartLazyModelRegistry(result)
            return result

        # Not found - trigger smart loading
        self._smart_load_model(key)

        # Retry after loading
        result = self._registry.get(key)
        if result is not None:
            if isinstance(result, ModelRegistry):
                return _SmartLazyModelRegistry(result)
            return result

        raise KeyError(
            f"Model or registry '{key}' not found. "
            f"Available: {self._registry.list_keys(recursive=False, include_children=True)}"
        )

    def __contains__(self, key: str) -> bool:
        """Checks if a model exists, triggering smart loading if necessary."""
        self._initialize_if_needed()

        if key in self._registry:
            return True

        return self._smart_load_model(key)

    def __dir__(self):
        """Enables auto-completion for IDEs/shells."""
        self._initialize_if_needed()
        registry = object.__getattribute__(self, "_registry")
        return list(registry._obj_map.keys()) + list(registry._child_registries.keys()) + dir(super())

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a model or registry with a default fallback.

        Args:
            key (str): The model key.
            default (Any, optional): Value to return if key is not found. Defaults to None.

        Returns:
            Any: The model, registry, or default value.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def build(self, key: str, *args, **kwargs) -> Any:
        """Builds a model instance with smart loading support.

        Args:
            key (str): The model key.
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.

        Returns:
            Any: The constructed model instance.
        """
        self._initialize_if_needed()

        # Try to build directly first
        if key in self._registry:
            return self._registry.build(key, *args, **kwargs)

        # Not found - try smart loading
        self._smart_load_model(key)

        # Try build again after loading
        return self._registry.build(key, *args, **kwargs)

    def __repr__(self):
        self._initialize_if_needed()
        return repr(self._registry)

    def __str__(self):
        self._initialize_if_needed()
        return str(self._registry)


# Create smart lazy-loading MODELS registry
MODELS = _SmartLazyModelRegistry(_MODELS_BASE)


# =============================================================================
# Helper Class and Function for Elegant Access
# =============================================================================

class _ModelBuilder:
    """Builder object to construct models in a flexible way.

    This class handles the URL-style path parsing, registry traversal, and
    instantiation of models.
    """

    def __init__(self, model_key: str):
        """Initializes the builder.

        Args:
            model_key (str): Model identifier in URL-style format.
        """
        self._key = model_key
        self._model_class: Optional[Type] = None
        self._resolved_path: Optional[List[str]] = None

    def _parse_path(self) -> List[str]:
        """Parses the URL-style path into components.

        Returns:
            List[str]: List of path components (e.g., ['GEARS', 'default']).
        """
        if self._resolved_path:
            return self._resolved_path

        # Split by '/' and filter empty parts
        parts = [p for p in self._key.split("/") if p]
        if not parts:
            parts = ["default"]  # Default to root default

        # Normalize first part to uppercase (convention for top-level model types)
        # But preserve mixed case for names like "scGPT" that contain lowercase
        if len(parts) > 0 and not any(c.islower() for c in parts[0]):
            parts[0] = parts[0].upper()

        self._resolved_path = parts
        logger.debug(f"Parsed path: '{self._key}' -> {parts}")
        return parts

    def _find_in_registry(self, registry: Any, path_parts: List[str], case_sensitive: bool = True) -> Optional[Any]:
        """Manually traverses the registry tree to find a model.

        Args:
            registry (Any): The registry object to search in.
            path_parts (List[str]): List of path components to traverse.
            case_sensitive (bool): Whether to use case-sensitive matching.

        Returns:
            Optional[Any]: The found model class or None.
        """
        current = registry
        remaining_parts = path_parts.copy()

        # Unwrap SmartLazyModelRegistry to access internals
        if hasattr(current, "_initialize_if_needed"):
            current._initialize_if_needed()
        if hasattr(current, "_registry"):
            current = current._registry

        while remaining_parts:
            part = remaining_parts.pop(0)
            found = None

            # 1. Search in Child Registries
            if hasattr(current, "_child_registries"):
                for name, child in current._child_registries.items():
                    if (case_sensitive and name == part) or (
                        not case_sensitive and name.upper() == part.upper()
                    ):
                        found = child
                        break

            # 2. Search in Registered Objects (Models)
            if found is None and hasattr(current, "_obj_map"):
                for name, obj in current._obj_map.items():
                    if (case_sensitive and name == part) or (
                        not case_sensitive and name.upper() == part.upper()
                    ):
                        found = obj
                        break

            # 3. Not found: Attempt Lazy Loading (if case_sensitive is True)
            if found is None and case_sensitive:
                try:
                    # Attempt to trigger loading for this part
                    if hasattr(MODELS, "_smart_load_model"):
                        MODELS._smart_load_model(part)
                        # Construct a partial dot-path for more precise loading
                        full_key = ".".join(self._parse_path()[: len(self._parse_path()) - len(remaining_parts)])
                        MODELS._smart_load_model(full_key)

                        # Retry search in this node after loading
                        # (Recursion logic simplified by checking current maps again)
                        if hasattr(current, "_child_registries"):
                            found = current._child_registries.get(part)
                        if found is None and hasattr(current, "_obj_map"):
                            found = current._obj_map.get(part)
                except Exception as e:
                    logger.debug(f"Lazy loading failed for part '{part}': {e}")

            # 4. Handle Not Found
            if found is None:
                if case_sensitive:
                    # Retry entire path case-insensitively
                    return self._find_in_registry(registry, path_parts, case_sensitive=False)
                return None

            # 5. Move to next level
            if remaining_parts:
                # If we have more parts, 'found' must be a registry-like object
                if isinstance(found, ModelRegistry):
                    current = found
                elif hasattr(found, "_registry"):
                    current = found._registry
                else:
                    # Found a leaf (Model) but path expects more depth
                    return None
            else:
                # Path consumed, return the result (unwrap if necessary)
                if hasattr(found, "_registry"):
                    found = found._registry
                return found

        # If loop finishes (e.g. empty path parts?), return default
        if hasattr(current, "_obj_map") and "default" in current._obj_map:
            return current._obj_map["default"]

        return None

    def _get_model_class(self) -> Type:
        """Resolves the model class by traversing the registry.

        Returns:
            Type: The resolved model class.

        Raises:
            KeyError: If the model class cannot be resolved after all attempts.
        """
        if self._model_class is not None:
            return self._model_class

        path_parts = self._parse_path()

        # 1. Try finding in registry
        found = self._find_in_registry(MODELS, path_parts, case_sensitive=True)

        # 2. If not found, force load module corresponding to the top-level type
        if found is None:
            model_type = path_parts[0].lower()
            try:
                _load_model_modules(model_type)
                # Also explicitly import the top-level module to trigger registration
                # (e.g., perturblab.models.scgpt for components registration)
                top_level_module = f"perturblab.models.{model_type}"
                if top_level_module not in sys.modules:
                    try:
                        importlib.import_module(top_level_module)
                        logger.debug(f"Loaded top-level module: {top_level_module}")
                    except Exception as e:
                        logger.warning(f"Could not import top-level module '{top_level_module}': {e}")
            except Exception as e:
                logger.warning(f"Failed to load model modules for '{model_type}': {e}")
            
            # Retry finding
            found = self._find_in_registry(MODELS, path_parts, case_sensitive=True)

        # 3. If still not found, load everything (last resort)
        if found is None:
            _load_all_models()
            found = self._find_in_registry(MODELS, path_parts, case_sensitive=True)

        # 4. If still not found, try appending 'default' (e.g., GEARS -> GEARS/default)
        if found is None and len(path_parts) == 1:
            found = self._find_in_registry(MODELS, path_parts + ["default"], case_sensitive=True)

        # 5. Final not found - provide detailed diagnostic information
        if found is None:
            available = MODELS.list_keys(recursive=True)[:20]
            
            # Provide helpful error message
            error_msg = (
                f"Model '{self._key}' (path: {path_parts}) not found in registry.\n"
                f"\n"
                f"Available models (sample):\n"
            )
            
            if available:
                for model_key in available:
                    error_msg += f"  - {model_key}\n"
            else:
                error_msg += "  (No models registered - check import errors)\n"
            
            # Provide specific suggestions for scGPT
            if path_parts and path_parts[0].lower() == 'scgpt':
                error_msg += (
                    f"\n"
                    f"Note: scGPT models may require additional dependencies or may have import errors.\n"
                    f"Check the logs above for any import warnings or errors.\n"
                )
            
            # Suggest similar model names
            if len(path_parts) > 0:
                similar_models = [
                    k for k in MODELS.list_keys(recursive=True)
                    if path_parts[0].lower() in k.lower()
                ]
                if similar_models:
                    error_msg += f"\nDid you mean one of these?\n"
                    for model_key in similar_models[:5]:
                        error_msg += f"  - {model_key}\n"
            
            raise KeyError(error_msg)

        # 6. Handle case where a Registry is found instead of a Model
        if isinstance(found, ModelRegistry):
            if "default" in found._obj_map:
                found = found._obj_map["default"]
            else:
                raise KeyError(
                    f"Model '{self._key}' resolved to a registry without a 'default' model. "
                    f"Available in registry: {list(found._obj_map.keys())}"
                )

        # Unwrap if needed
        if hasattr(found, "_registry"):
            if "default" in found._registry._obj_map:
                found = found._registry._obj_map["default"]

        self._model_class = found
        logger.debug(f"Found model class for '{self._key}': {found}")
        return found

    @property
    def class_(self) -> Type:
        """Returns the resolved model class.

        Returns:
            Type: The class of the model.
        """
        return self._get_model_class()

    def build(self, *args, **kwargs) -> Any:
        """Constructs a model instance.

        Args:
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.

        Returns:
            Any: The initialized model instance.

        Raises:
            ValueError: If instantiation fails due to TypeError.
        """
        model_class = self._get_model_class()
        try:
            return model_class(*args, **kwargs)
        except TypeError as e:
            raise ValueError(
                f"Failed to build model '{self._key}' with provided arguments. Error: {e}"
            )

    def __call__(self, *args, **kwargs) -> Any:
        """Calls the builder to directly construct the model.

        Enables usage like: `Model("GEARS/default")(num_genes=1000)`

        Args:
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.

        Returns:
            Any: The initialized model instance.
        """
        return self.build(*args, **kwargs)

    def __repr__(self) -> str:
        path = self._parse_path()
        return f"ModelBuilder(key='{self._key}', path={path})"


def Model(key: str) -> _ModelBuilder:
    """Elegant model loader function as an alternative to MODELS.xxx.xxx access.

    This function provides a concise way to access and build models using URL-style
    paths with case-insensitive matching. It manually traverses the registry tree
    and handles lazy imports.

    Args:
        key (str): Model identifier in URL-style format:
            - Path format: `GEARS/default`, `GEARS/component/sss`, `scGPT/default`.
            - Case-insensitive matching is supported.
            - Falls back to case-insensitive search if an exact match fails.

    Returns:
        _ModelBuilder: A callable builder object supporting multiple usage patterns:
            - `Model("GEARS/default")(num_genes=1000)`  # Direct construction
            - `Model("GEARS/default").build(...)`        # Explicit build method
            - `Model("GEARS/default").class_`            # Access raw model class

    Examples:
        >>> # Instead of MODELS.GEARS.default
        >>> model = Model("GEARS/default")(num_genes=1000, num_perts=50)
        >>>
        >>> # Case-insensitive matching
        >>> model = Model("gears/default")(num_genes=1000)
        >>>
        >>> # Access nested components
        >>> component = Model("GEARS/component/sss")
        >>>
        >>> # Get model class
        >>> model_class = Model("GEARS/default").class_
        >>> model = model_class(num_genes=1000)
    """
    return _ModelBuilder(key)