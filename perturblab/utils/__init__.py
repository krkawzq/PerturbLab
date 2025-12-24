"""Utility functions for PerturbLab."""

from ._check_dependencies import (
    DependencyError,
    check_dependencies,
    create_lazy_loader,
    format_install_command,
)
from .logging import (
    disable_logging,
    enable_logging,
    get_distributed_logger,
    get_logger,
    get_rank,
    is_distributed,
    is_main_process,
    set_log_level,
    setup_logger,
)


def __getattr__(name: str):
    """Lazy import for read_obo to avoid circular import."""
    if name == "read_obo":
        from ._read_obo import read_obo
        return read_obo
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "setup_logger",
    "get_logger",
    "get_distributed_logger",
    "set_log_level",
    "disable_logging",
    "enable_logging",
    "is_distributed",
    "get_rank",
    "is_main_process",
    "read_obo",
    "check_dependencies",
    "create_lazy_loader",
    "DependencyError",
    "format_install_command",
]
