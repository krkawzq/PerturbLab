"""Dependency checking and lazy loading utilities.

This module provides utilities for checking package dependencies and implementing
lazy loading to defer imports until actual usage. This is critical for avoiding
import-time dependency errors in large frameworks with optional components.

Key Features:
    - **Dependency Checking**: Verify if required packages are installed
    - **Lazy Loading**: Defer module imports until first access
    - **Graceful Degradation**: Allow partial functionality when optional deps are missing
    - **User-Friendly Errors**: Provide clear installation instructions

Example:
    >>> from perturblab.utils import check_dependencies
    >>>
    >>> # Check if dependencies are satisfied
    >>> satisfied, missing = check_dependencies(['torch', 'numpy'])
    >>> if not satisfied:
    ...     print(f"Missing packages: {missing}")
    ...     print("Install with: pip install torch numpy")
"""


from .logging import get_logger

logger = get_logger()

__all__ = [
    "check_dependencies",
    "check_packages_installed",
    "create_lazy_loader",
    "format_install_command",
    "DependencyError",
]


class DependencyError(ImportError):
    """Raised when required dependencies are not installed.

    This exception is raised when attempting to import a module that requires
    packages that are not currently installed. It provides clear error messages
    with installation instructions.

    Example:
        >>> try:
        ...     from perturblab.models.gears import GEARSModel
        ... except DependencyError as e:
        ...     print(e)
        GEARS requires: torch, torch_geometric, networkx
        Install with: pip install perturblab[gears]
    """

    pass


def check_dependencies(
    requirements: list[str] | None = None,
    dependencies: list[str] | None = None,
    package_name: str | None = None,
    install_hint: str | None = None,
) -> None:
    """Checks if required and optional dependencies are installed.

    This function checks two types of dependencies:
    - **requirements** (mandatory): Raises DependencyError if any are missing
    - **dependencies** (optional): Logs info message if any are missing

    Uses try-except style for checking imports to avoid importlib overhead.

    Args:
        requirements: List of required package names. If any are missing,
            raises DependencyError. Package names should be importable module
            names (e.g., 'sklearn' not 'scikit-learn').
        dependencies: List of optional package names. If any are missing,
            logs an info message recommending installation.
        package_name: Name of the package/module for error messages.
            If None, uses "this module".
        install_hint: Custom installation command. If None, generates a
            default hint like "pip install package1 package2".

    Raises:
        DependencyError: If any required packages are missing.

    Examples:
        >>> # Check only required dependencies
        >>> check_dependencies(
        ...     requirements=['torch', 'numpy'],
        ...     package_name='my_model'
        ... )

        >>> # Check both required and optional dependencies
        >>> check_dependencies(
        ...     requirements=['torch'],
        ...     dependencies=['accelerate', 'flash_attn'],
        ...     package_name='scGPT',
        ...     install_hint='pip install perturblab[scgpt]'
        ... )

        >>> # Handle missing requirements
        >>> try:
        ...     check_dependencies(
        ...         requirements=['torch_geometric'],
        ...         package_name='GEARS'
        ...     )
        ... except DependencyError as e:
        ...     print(f"Installation required: {e}")

    Note:
        This function only checks if packages can be imported, not their versions.
        For version-specific checks, use `importlib.metadata.version()`.
    """
    requirements = requirements or []
    dependencies = dependencies or []
    package_name = package_name or "this module"

    # Check required dependencies
    missing_requirements = []
    for pkg in requirements:
        try:
            __import__(pkg)
        except ImportError:
            missing_requirements.append(pkg)

    if missing_requirements:
        if install_hint:
            hint = install_hint
        else:
            hint = f"pip install {' '.join(missing_requirements)}"

        raise DependencyError(
            f"{package_name} requires the following packages that are not installed: "
            f"{', '.join(missing_requirements)}.\n"
            f"Install them with: {hint}"
        )

    # Check optional dependencies
    missing_dependencies = []
    for pkg in dependencies:
        try:
            __import__(pkg)
        except ImportError:
            missing_dependencies.append(pkg)

    if missing_dependencies:
        if install_hint:
            hint = install_hint
        else:
            hint = f"pip install {' '.join(missing_dependencies)}"

        logger.info(
            f"[{package_name}] Optional dependencies are not installed: "
            f"{', '.join(missing_dependencies)}. "
            f"For enhanced functionality, install with: {hint}"
        )


def check_packages_installed(packages: list[str]) -> tuple[bool, list[str]]:
    """Checks if packages are installed without raising errors.

    This is a simpler utility function that just checks package availability
    without any error handling or logging. Useful for conditional feature checks.

    Args:
        packages: List of package names to check (e.g., ['torch', 'numpy']).
            Package names should be importable module names, not PyPI names.

    Returns:
        A tuple of (all_satisfied, missing_packages):
            - all_satisfied (bool): True if all packages are installed
            - missing_packages (list[str]): List of missing package names

    Examples:
        >>> # Check if optional feature is available
        >>> satisfied, missing = check_packages_installed(['flash_attn'])
        >>> if satisfied:
        ...     print("Flash attention available")
        ... else:
        ...     print(f"Flash attention not available: {missing}")

        >>> # Check multiple packages
        >>> satisfied, missing = check_packages_installed(['torch', 'numpy'])
        >>> if not satisfied:
        ...     print(f"Missing: {', '.join(missing)}")

    Note:
        This function only checks if packages can be imported, not their versions.
        For enforcing requirements, use `check_dependencies()` instead.
    """
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    return len(missing) == 0, missing


def format_install_command(
    missing_packages: list[str], extra_name: str | None = None, base_package: str = "perturblab"
) -> str:
    """Formats installation command for missing packages.

    This utility function generates user-friendly pip install commands for
    missing dependencies. It can generate commands for either extras groups
    or individual packages.

    Args:
        missing_packages: List of missing package names.
        extra_name: Name of the extras group (e.g., 'gears', 'scgpt').
            If provided, generates command like "pip install perturblab[gears]".
        base_package: Base package name. Defaults to 'perturblab'.

    Returns:
        Formatted installation command string.

    Examples:
        >>> # With extras group
        >>> format_install_command(['torch_geometric'], 'gears')
        'pip install perturblab[gears]'

        >>> # Without extras group
        >>> format_install_command(['torch', 'numpy'])
        'pip install torch numpy'

        >>> # Custom base package
        >>> format_install_command(['pandas'], 'data', 'mypackage')
        'pip install mypackage[data]'

    Note:
        When using extras groups, the actual packages installed are defined
        in the package's `setup.py` or `pyproject.toml` extras_require section.
    """
    if extra_name:
        return f"pip install {base_package}[{extra_name}]"
    else:
        return f"pip install {' '.join(missing_packages)}"
