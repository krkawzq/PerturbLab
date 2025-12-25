"""Custom exceptions for PerturbLab.

This module defines all custom exceptions used throughout the framework.
"""


class PerturbLabError(Exception):
    """Base exception class for all PerturbLab errors."""

    pass


class DependencyError(PerturbLabError):
    """Raised when required dependencies are missing.

    This exception is raised when attempting to use functionality that
    requires external packages that are not installed.

    Examples
    --------
    >>> from perturblab.core.exceptions import DependencyError
    >>> raise DependencyError("torch-geometric is required for GEARS model")
    """

    pass


class ConfigurationError(PerturbLabError):
    """Raised when there are configuration errors.

    This exception is raised when configuration parameters are invalid
    or incompatible.
    """

    pass


class DataError(PerturbLabError):
    """Raised when there are data-related errors.

    This exception is raised for issues with data loading, processing,
    or validation.
    """

    pass


class RegistryError(PerturbLabError):
    """Raised when there are registry-related errors.

    This exception is raised when there are issues with model or resource
    registration.
    """

    pass


class ValidationError(PerturbLabError):
    """Raised when validation fails.

    This exception is raised when input validation or data validation fails.
    """

    pass

