"""Model registry with smart lazy loading.

This module provides the MODELS registry with intelligent on-demand loading:
- First access: Scans methods directory structure (lightweight, no imports)
- Model lookup: Loads only the specific method's modules (targeted loading)
- Not found: Loads all methods as fallback

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

import os
import sys
import importlib
import pkgutil
from pathlib import Path

from perturblab.core import ModelRegistry
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["MODELS"]


# =============================================================================
# Create Base Registry
# =============================================================================

# Create the base ModelRegistry instance
# This is the actual global MODELS that the package uses
_MODELS_BASE = ModelRegistry("MODELS")


# =============================================================================
# Smart Lazy Loading for Models
# =============================================================================

_module_tree_cache = None  # Cache of module structure: {method_name: [module_paths]}
_fully_loaded = False  # Whether all modules have been loaded


def _scan_methods_tree() -> dict[str, list[str]]:
    """Scan perturblab.methods directory structure without importing.
    
    Builds a lightweight cache of method names to their module paths.
    This allows targeted loading instead of loading everything.
    
    Returns
    -------
    dict[str, list[str]]
        Mapping of method names to list of module paths.
        Example: {'gears': ['perturblab.methods.gears', 
                            'perturblab.methods.gears.models']}
    """
    tree = {}
    
    try:
        import perturblab
        perturblab_path = Path(perturblab.__file__).parent
        methods_path = perturblab_path / 'methods'
        
        if not methods_path.exists():
            logger.debug(f"Methods directory not found: {methods_path}")
            return tree
        
        # Walk through methods directory
        for method_dir in methods_path.iterdir():
            if not method_dir.is_dir() or method_dir.name.startswith('_'):
                continue
            
            method_name = method_dir.name
            modules = []
            
            # Collect all Python modules in this method
            for py_file in method_dir.rglob('*.py'):
                if py_file.name.startswith('_') and py_file.name != '__init__.py':
                    continue
                
                # Convert path to module name
                rel_path = py_file.relative_to(perturblab_path)
                module_parts = list(rel_path.parts[:-1])  # Remove .py
                if py_file.name != '__init__.py':
                    module_parts.append(py_file.stem)
                
                module_name = 'perturblab.' + '.'.join(module_parts)
                modules.append(module_name)
            
            tree[method_name] = modules
            logger.debug(f"Scanned method '{method_name}': {len(modules)} modules")
        
    except Exception as e:
        logger.warning(f"Failed to scan methods tree: {e}")
    
    return tree


def _load_method_modules(method_name: str) -> bool:
    """Load all modules for a specific method.
    
    Parameters
    ----------
    method_name : str
        Name of the method (e.g., 'gears').
    
    Returns
    -------
    bool
        True if any modules were loaded successfully.
    """
    global _module_tree_cache
    
    # Ensure tree cache exists
    if _module_tree_cache is None:
        _module_tree_cache = _scan_methods_tree()
    
    if method_name not in _module_tree_cache:
        return False
    
    modules = _module_tree_cache[method_name]
    loaded_any = False
    
    for module_name in modules:
        # Skip if already imported
        if module_name in sys.modules:
            continue
        
        try:
            importlib.import_module(module_name)
            logger.debug(f"Loaded module: {module_name}")
            loaded_any = True
        except Exception as e:
            logger.warning(
                f"Failed to import {module_name}: {e}. "
                f"Models in this module won't be registered."
            )
    
    return loaded_any


def _load_all_methods() -> None:
    """Load all method modules (fallback when model not found).
    
    This is only called when a model cannot be found after targeted loading.
    """
    global _module_tree_cache, _fully_loaded
    
    if _fully_loaded:
        return
    
    logger.debug("Loading all method modules...")
    
    # Ensure tree cache exists
    if _module_tree_cache is None:
        _module_tree_cache = _scan_methods_tree()
    
    # Load all methods
    for method_name in _module_tree_cache:
        _load_method_modules(method_name)
    
    _fully_loaded = True
    logger.debug(f"✓ Loaded all models: {len(_MODELS_BASE.list_keys(recursive=True))} total")


class _SmartLazyModelRegistry:
    """Smart lazy-loading wrapper for MODELS registry.
    
    Implements intelligent on-demand model loading:
    1. First access: Scans methods tree structure (lightweight, no imports)
    2. Model lookup: Tries targeted loading of specific method modules
    3. Not found: Loads all methods as fallback
    
    This ensures fast startup while guaranteeing all models are eventually available.
    """
    
    def __init__(self, registry: ModelRegistry):
        object.__setattr__(self, '_registry', registry)
        object.__setattr__(self, '_initialized', False)
    
    def _check_user_override(self) -> bool:
        """Check if user has disabled auto-loading via global state or env var.
        
        Returns
        -------
        bool
            True if auto-loading should be disabled.
        """
        # Check environment variable
        env_disable = os.environ.get('PERTURBLAB_DISABLE_AUTO_LOAD', '').upper()
        if env_disable in ('TRUE', '1', 'YES'):
            return True
        
        # Check if user has set a global flag in their code
        # This allows: perturblab._disable_auto_load = True
        import perturblab
        if hasattr(perturblab, '_disable_auto_load'):
            return bool(perturblab._disable_auto_load)
        
        return False
    
    def _smart_load_model(self, key: str) -> bool:
        """Intelligently load modules to find a model.
        
        Strategy:
        1. Check if already in registry (no loading needed)
        2. Parse key to identify method (e.g., 'GEARS.gnn' -> 'gears')
        3. Load only that method's modules
        4. If still not found, load all methods
        
        Parameters
        ----------
        key : str
            Model key to find (e.g., 'GEARS.gnn' or 'MLP').
        
        Returns
        -------
        bool
            True if model was found or loaded.
        """
        # Check if user disabled auto-loading
        if self._check_user_override():
            logger.debug("Smart loading disabled by user")
            return False
        
        # Parse key to get method name
        if '.' in key:
            method_name = key.split('.')[0].lower()
        else:
            # Top-level model, need to search all methods
            method_name = None
        
        # Try targeted loading first
        if method_name:
            logger.debug(f"Attempting targeted load for method: {method_name}")
            if _load_method_modules(method_name):
                # Check if model now exists
                if key in self._registry:
                    logger.debug(f"✓ Model '{key}' found via targeted loading")
                    return True
        
        # Fallback: load all methods
        logger.debug(f"Model '{key}' not found via targeted loading, loading all methods...")
        _load_all_methods()
        
        return key in self._registry
    
    def _initialize_if_needed(self):
        """Initialize tree cache on first access (if not disabled)."""
        if object.__getattribute__(self, '_initialized'):
            return
        
        if not self._check_user_override():
            global _module_tree_cache
            if _module_tree_cache is None:
                logger.debug("Initializing methods tree cache...")
                _module_tree_cache = _scan_methods_tree()
        
        object.__setattr__(self, '_initialized', True)
    
    def __getattribute__(self, name: str):
        # Don't intercept private/internal attributes
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        # Initialize on first real access
        object.__getattribute__(self, '_initialize_if_needed')()
        
        # Delegate to actual registry
        registry = object.__getattribute__(self, '_registry')
        attr = getattr(registry, name)
        
        # If it's a child registry, wrap it with smart loading too
        if isinstance(attr, ModelRegistry):
            return _SmartLazyModelRegistry(attr)
        
        return attr
    
    def __getattr__(self, name: str):
        """Support dot notation access: MODELS.GEARS.gnn
        
        This enables edict-style access where you can navigate through
        the registry hierarchy using dot notation.
        """
        self._initialize_if_needed()
        
        # Try to get from child registries first
        if name in self._registry._child_registries:
            child = self._registry._child_registries[name]
            return _SmartLazyModelRegistry(child)
        
        # Try to get from models
        if name in self._registry._obj_map:
            return self._registry._obj_map[name]
        
        # Not found - try smart loading
        # Convert attribute name to potential model key
        # e.g., MODELS.GEARS -> try loading 'GEARS' or 'gears'
        for key_variant in [name, name.upper(), name.lower()]:
            if self._smart_load_model(key_variant):
                # Check child registries again
                if key_variant in self._registry._child_registries:
                    child = self._registry._child_registries[key_variant]
                    return _SmartLazyModelRegistry(child)
                # Check models
                if key_variant in self._registry._obj_map:
                    return self._registry._obj_map[key_variant]
        
        raise AttributeError(
            f"'{self._registry.name}' registry has no model or child registry '{name}'. "
            f"Available: {list(self._registry._obj_map.keys()) + list(self._registry._child_registries.keys())}"
        )
    
    def __setattr__(self, name: str, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            registry = object.__getattribute__(self, '_registry')
            setattr(registry, name, value)
    
    def __getitem__(self, key: str):
        """Support dict-style access: MODELS['GEARS']['gnn'] or MODELS['GEARS.gnn']
        
        This enables flexible dictionary-style access with both nested keys
        and dot-notation keys.
        """
        self._initialize_if_needed()
        
        # Try to get from registry first
        result = self._registry.get(key)
        if result is not None:
            # If it's a child registry, wrap it
            if isinstance(result, ModelRegistry):
                return _SmartLazyModelRegistry(result)
            return result
        
        # Not found - try smart loading
        self._smart_load_model(key)
        
        # Try again after loading
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
        """Check if model exists with smart loading."""
        self._initialize_if_needed()
        
        # Check registry first
        if key in self._registry:
            return True
        
        # Try smart loading
        return self._smart_load_model(key)
    
    def get(self, key: str, default=None):
        """Get model or registry with default fallback.
        
        Supports both 'GEARS.gnn' and 'GEARS' as keys.
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def build(self, key: str, *args, **kwargs):
        """Build model with smart loading."""
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

