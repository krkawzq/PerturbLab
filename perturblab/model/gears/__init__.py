import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

def __getattr__(name: str):
    """Lazy import for GEARS model classes.

    Args:
        name (str): The name of the attribute to fetch.

    Returns:
        type: The imported class or configuration.
    """
    if name == 'GearsConfig':
        from .config import GearsConfig
        return GearsConfig

    if name == 'GearsModel':
        from .model import GearsModel
        return GearsModel

    logger.error("Attribute '%s' not found in %s", name, __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['GearsConfig', 'GearsModel']
