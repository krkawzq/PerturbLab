"""Utility functions for PerturbLab."""

# Import order matters to avoid circular imports!
# 1. First import logging (no dependencies, needed by many modules)
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

# 2. Then import dependency checking (no circular deps)
from ._check_dependencies import (
    check_dependencies,
    create_lazy_loader,
    DependencyError,
    format_install_command,
)

# 3. Finally import modules that create circular dependencies
# _read_obo → types.math.DAG → types._vocab → kernels.mapping → utils.get_logger
# By importing it last, get_logger is already available
from ._read_obo import read_obo

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
