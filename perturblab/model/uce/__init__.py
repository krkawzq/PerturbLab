import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

def __getattr__(name):
    """Lazy import for UCE model classes.

    Args:
        name (str): The name of the attribute to fetch.

    Returns:
        type: The imported class or configuration.
    """
    if name == 'UCEConfig':
        from .config import UCEConfig
        return UCEConfig
    
    if name == 'UCEModel':
        from .model import UCEModel
        return UCEModel

    logger.error("Attribute '%s' not found in %s", name, __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['UCEModel', 'UCEConfig', 'UCEModelConfig']  # UCEModelConfig for backward compatibility
