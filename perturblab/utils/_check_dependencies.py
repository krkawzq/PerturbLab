"""Dependency checking and lazy loading utilities.

This module provides utilities for checking package dependencies and
implementing lazy loading to defer imports until actual usage.
"""

import importlib
from typing import Any, Callable

from .logging import get_logger

logger = get_logger()


__all__ = [
    "check_dependencies",
    "create_lazy_loader",
    "DependencyError",
]


class DependencyError(ImportError):
    """Raised when required dependencies are not installed."""

    pass


def check_dependencies(dependencies: list[str]) -> tuple[bool, list[str]]:
    """Check if all required dependencies are installed.

    Parameters
    ----------
    dependencies : list[str]
        List of package names to check.

    Returns
    -------
    tuple[bool, list[str]]
        (all_satisfied, missing_packages)
        - all_satisfied: True if all dependencies are installed
        - missing_packages: List of missing package names

    Examples
    --------
    >>> satisfied, missing = check_dependencies(['torch', 'numpy'])
    >>> if not satisfied:
    ...     print(f"Missing: {missing}")
    """
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)

    return len(missing) == 0, missing


def create_lazy_loader(
    dependencies: list[str],
    lazy_modules: dict[str, str],
    package_name: str,
    install_hint: str | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Create lazy loading functions for a module.

    This function returns `__getattr__` and `__dir__` functions that implement
    lazy loading with dependency checking.

    Parameters
    ----------
    dependencies : list[str]
        List of required package names.
    lazy_modules : dict[str, str]
        Mapping from attribute names to module paths.
        Example: {'Model': '.model', 'Config': '.config'}
    package_name : str
        Name of the package (for error messages).
    install_hint : str, optional
        Custom installation hint. If None, generates a default hint.

    Returns
    -------
    tuple[Callable, Callable]
        (__getattr__, __dir__) functions for the module.

    Examples
    --------
    >>> # In your __init__.py:
    >>> dependencies = ['torch_geometric']
    >>> lazy_modules = {'Model': '.model', 'Config': '.config'}
    >>> __getattr__, __dir__ = create_lazy_loader(
    ...     dependencies, lazy_modules, __package__
    ... )
    """
    # Cache dependency check result
    _cache = {"checked": False, "satisfied": False, "missing": []}

    def _ensure_dependencies():
        """Ensure dependencies are satisfied, raise DependencyError if not."""
        if not _cache["checked"]:
            _cache["satisfied"], _cache["missing"] = check_dependencies(dependencies)
            _cache["checked"] = True

        if not _cache["satisfied"]:
            if install_hint:
                hint = install_hint
            else:
                # Generate default hint
                dep_str = " ".join(_cache["missing"])
                hint = f"pip install {dep_str}"

            raise DependencyError(
                f"{package_name} requires the following packages that are not installed: "
                f"{', '.join(_cache['missing'])}.\n"
                f"Install them with: {hint}"
            )

    def __getattr__(name: str) -> Any:
        """Lazy load module attributes.

        Parameters
        ----------
        name : str
            Name of the attribute to load.

        Returns
        -------
        Any
            The loaded attribute.

        Raises
        ------
        DependencyError
            If required dependencies are not installed.
        AttributeError
            If the attribute doesn't exist.
        """
        if name in lazy_modules:
            # Check dependencies before loading
            _ensure_dependencies()

            # Import the module
            module_path = lazy_modules[name]
            try:
                # Get the module's globals to cache the attribute
                import sys

                caller_module = sys.modules[package_name]

                module = importlib.import_module(module_path, package=package_name)
                attr = getattr(module, name)

                # Cache it in the caller module's namespace
                setattr(caller_module, name, attr)

                logger.debug(f"Lazily loaded {name} from {module_path}")
                return attr
            except (ImportError, AttributeError) as e:
                raise AttributeError(f"Module '{package_name}' has no attribute '{name}'") from e

        raise AttributeError(f"Module '{package_name}' has no attribute '{name}'")

    def __dir__() -> list[str]:
        """Return list of available attributes for autocomplete."""
        return list(lazy_modules.keys())

    return __getattr__, __dir__


def format_install_command(
    missing_packages: list[str], extra_name: str | None = None, base_package: str = "perturblab"
) -> str:
    """Format installation command for missing packages.

    Parameters
    ----------
    missing_packages : list[str]
        List of missing package names.
    extra_name : str, optional
        Name of the extras group (e.g., 'gears', 'scgpt').
    base_package : str, default='perturblab'
        Base package name.

    Returns
    -------
    str
        Formatted installation command.

    Examples
    --------
    >>> format_install_command(['torch_geometric'], 'gears')
    'pip install perturblab[gears]'
    >>> format_install_command(['torch_geometric'])
    'pip install torch_geometric'
    """
    if extra_name:
        return f"pip install {base_package}[{extra_name}]"
    else:
        return f"pip install {' '.join(missing_packages)}"
