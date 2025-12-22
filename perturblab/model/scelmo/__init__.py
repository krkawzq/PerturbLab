import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

def __getattr__(name: str):
    """Lazy import for scELMo model classes.

    Args:
        name (str): The name of the attribute to fetch.

    Returns:
        type: The imported class or configuration.
    """
    if name == 'scELMoConfig':
        from .config import scELMoConfig
        return scELMoConfig

    if name == 'scELMoModel':
        from .model import scELMoModel
        return scELMoModel

    logger.error("Attribute '%s' not found in %s", name, __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['scELMoModel', 'scELMoConfig']
