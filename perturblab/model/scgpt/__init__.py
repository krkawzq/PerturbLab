import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

def __getattr__(name):
    """Lazy import for scGPT model classes.

    Args:
        name (str): The name of the attribute to fetch.

    Returns:
        type: The imported class.
    """
    if name == 'scGPTConfig':
        from .config import scGPTConfig
        return scGPTConfig
        
    if name == 'scGPTModel':
        from .model import scGPTModel
        return scGPTModel
        
    if name == 'scGPTPerturbationModel':
        from .model import scGPTPerturbationModel
        return scGPTPerturbationModel
    
    logger.error("Attribute '%s' not found in %s", name, __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['scGPTConfig', 'scGPTModel', 'scGPTPerturbationModel']
