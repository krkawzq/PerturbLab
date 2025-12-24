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
    requirements: list[str] | None = None,
    dependencies: list[str] | None = None,
    lazy_modules: dict[str, str] | None = None,
    package_name: str | None = None,
    install_hint: str | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Create lazy loading functions for a module.

    This function returns `__getattr__` and `__dir__` functions that implement
    lazy loading with dependency checking.

    Parameters
    ----------
    requirements : list[str], optional
        List of required (mandatory) package names. If missing, raises DependencyError.
    dependencies : list[str], optional
        List of optional package names. If missing, logs info message recommending installation.
    lazy_modules : dict[str, str]
        Mapping from attribute names to module paths.
        Example: {'Model': '.model', 'Config': '.config'}
    package_name : str, optional
        Name of the package (for error messages). If None, uses "unknown".
    install_hint : str, optional
        Custom installation hint. If None, generates a default hint.

    Returns
    -------
    tuple[Callable, Callable]
        (__getattr__, __dir__) functions for the module.

    Examples
    --------
    >>> # Separate requirements (mandatory) and dependencies (optional)
    >>> requirements = ['torch_geometric']  # Mandatory
    >>> dependencies = ['accelerate']  # Optional
    >>> lazy_modules = {'Model': '.model'}
    >>> __getattr__, __dir__ = create_lazy_loader(
    ...     requirements=requirements,
    ...     dependencies=dependencies,
    ...     lazy_modules=lazy_modules,
    ...     package_name=__package__,
    ...     install_hint="pip install perturblab[gears]"
    ... )
    """
    # Normalize to lists
    requirements = requirements or []
    dependencies = dependencies or []
    lazy_modules = lazy_modules or {}
    package_name = package_name or "unknown"

    # Cache dependency check result
    _cache = {
        "requirements_checked": False,
        "requirements_satisfied": False,
        "requirements_missing": [],
        "dependencies_checked": False,
        "dependencies_satisfied": False,
        "dependencies_missing": [],
    }

    def _ensure_requirements():
        """Ensure required dependencies are satisfied, raise DependencyError if not."""
        if not _cache["requirements_checked"]:
            _cache["requirements_satisfied"], _cache["requirements_missing"] = check_dependencies(
                requirements
            )
            _cache["requirements_checked"] = True

        if not _cache["requirements_satisfied"]:
            if install_hint:
                hint = install_hint
            else:
                # Generate default hint
                dep_str = " ".join(_cache["requirements_missing"])
                hint = f"pip install {dep_str}"

            raise DependencyError(
                f"{package_name} requires the following packages that are not installed: "
                f"{', '.join(_cache['requirements_missing'])}.\n"
                f"Install them with: {hint}"
            )

    def _check_optional_dependencies():
        """Check optional dependencies and log recommendation if missing."""
        if not _cache["dependencies_checked"]:
            _cache["dependencies_satisfied"], _cache["dependencies_missing"] = check_dependencies(
                dependencies
            )
            _cache["dependencies_checked"] = True

        if not _cache["dependencies_satisfied"] and _cache["dependencies_missing"]:
            if install_hint:
                hint = install_hint
            else:
                dep_str = " ".join(_cache["dependencies_missing"])
                hint = f"pip install {dep_str}"

            logger.info(
                f"[{package_name}] Optional dependencies are not installed: "
                f"{', '.join(_cache['dependencies_missing'])}. "
                f"For enhanced functionality, install with: {hint}"
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
            # Check required dependencies before loading (raises if missing)
            _ensure_requirements()

            # Check optional dependencies (logs recommendation if missing)
            _check_optional_dependencies()

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
