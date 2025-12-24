"""Unified logging system for PerturbLab.

Provides colored console output, file logging, and distributed training support.
Compatible with tqdm progress bars and Jupyter notebooks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class Colors:
    """ANSI escape codes for colored terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels in console output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_BLUE,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }
    
    def __init__(self, fmt: str = None, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors and self._supports_color()
    
    @staticmethod
    def _supports_color() -> bool:
        """Check if the terminal supports color output."""
        # Check if running in terminal
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Windows terminal color support
        if sys.platform == 'win32':
            try:
                import colorama
                colorama.init()
                return True
            except ImportError:
                return False
        
        # Unix-like systems
        return True
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors if enabled."""
        if self.use_colors:
            levelname = record.levelname
            color = self.LEVEL_COLORS.get(record.levelno, '')
            record.levelname = f"{color}{levelname}{Colors.RESET}"
        
        return super().format(record)


def setup_logger(
    name: str = 'perturblab',
    level: str | int = logging.INFO,
    log_file: Optional[str | Path] = None,
    console: bool = True,
    use_colors: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup a logger with console and/or file output.
    
    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for logging to file.
        console: Whether to log to console.
        use_colors: Whether to use colored output in console.
        format_string: Custom format string. If None, uses default.
    
    Returns:
        logging.Logger: Configured logger instance.
    
    Example:
        >>> logger = setup_logger('myapp', level='DEBUG')
        >>> logger.info("Application started")
        >>> logger.warning("This is a warning")
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = '[%(name)s] [%(levelname)s] %(message)s'
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if use_colors:
            console_formatter = ColoredFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        
        # File output without colors
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(
    name: str = 'perturblab',
    level: Optional[str | int] = None,
) -> logging.Logger:
    """Get or create a logger.
    
    If the logger doesn't exist, creates it with default settings.
    
    Args:
        name: Logger name.
        level: Optional logging level. If None, uses existing level or INFO.
    
    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger not configured, set it up with defaults
    if not logger.handlers:
        setup_logger(name, level=level or logging.INFO)
    elif level is not None:
        # Update level if specified
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    
    return logger


def set_log_level(level: str | int, name: str = 'perturblab') -> None:
    """Set logging level for existing logger.
    
    Args:
        level: New logging level.
        name: Logger name.
    """
    logger = logging.getLogger(name)
    
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging(name: str = 'perturblab') -> None:
    """Disable logging for specified logger.
    
    Args:
        name: Logger name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL + 1)


def enable_logging(name: str = 'perturblab', level: str | int = logging.INFO) -> None:
    """Enable logging for specified logger.
    
    Args:
        name: Logger name.
        level: Logging level to enable.
    """
    set_log_level(level, name)


def is_distributed() -> bool:
    """Check if running in distributed training mode.
    
    Returns:
        bool: True if in distributed mode.
    """
    return (
        os.environ.get('RANK') is not None or
        os.environ.get('LOCAL_RANK') is not None
    )


def get_rank() -> int:
    """Get current process rank in distributed training.
    
    Returns:
        int: Process rank (0 if not distributed).
    """
    if not is_distributed():
        return 0
    
    return int(os.environ.get('RANK', 0))


def is_main_process() -> bool:
    """Check if current process is the main process.
    
    In distributed training, only rank 0 is the main process.
    
    Returns:
        bool: True if main process.
    """
    return get_rank() == 0


class DistributedLogger:
    """Logger wrapper that only logs on main process in distributed training.
    
    Useful for distributed training to avoid duplicate logs from all processes.
    
    Args:
        logger: Base logger instance.
        force: If True, always log regardless of rank.
    
    Example:
        >>> logger = get_logger()
        >>> dist_logger = DistributedLogger(logger)
        >>> dist_logger.info("This only prints on rank 0")
    """
    
    def __init__(self, logger: logging.Logger, force: bool = False):
        self.logger = logger
        self.force = force
    
    def _should_log(self) -> bool:
        """Check if should log based on rank and force flag."""
        return self.force or is_main_process()
    
    def debug(self, msg, *args, **kwargs):
        """Log debug message on main process only."""
        if self._should_log():
            self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """Log info message on main process only."""
        if self._should_log():
            self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log warning message on main process only."""
        if self._should_log():
            self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log error message (always logs, even on non-main process)."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Log critical message (always logs, even on non-main process)."""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        """Log exception (always logs, even on non-main process)."""
        self.logger.exception(msg, *args, **kwargs)


def get_distributed_logger(
    name: str = 'perturblab',
    force: bool = False,
) -> DistributedLogger:
    """Get a distributed-aware logger.
    
    Args:
        name: Logger name.
        force: If True, always log regardless of rank.
    
    Returns:
        DistributedLogger: Logger that respects distributed training.
    """
    base_logger = get_logger(name)
    return DistributedLogger(base_logger, force=force)


# Initialize default logger on import
_default_logger = None


def init_default_logger(
    level: str | int = None,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Initialize the default PerturbLab logger.
    
    Args:
        level: Logging level. If None, uses INFO or PERTURBLAB_LOG_LEVEL env var.
        log_file: Optional log file path.
    
    Returns:
        logging.Logger: Initialized logger.
    """
    global _default_logger
    
    # Get level from environment or use default
    if level is None:
        level_str = os.environ.get('PERTURBLAB_LOG_LEVEL', 'INFO')
        level = getattr(logging, level_str.upper(), logging.INFO)
    
    _default_logger = setup_logger(
        name='perturblab',
        level=level,
        log_file=log_file,
        console=True,
        use_colors=True,
    )
    
    return _default_logger


# Auto-initialize on import
if _default_logger is None:
    init_default_logger()

