import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

def __getattr__(name):
    """Lazy import for scFoundation model classes and utility functions.

    Args:
        name (str): The name of the attribute to fetch.

    Returns:
        Any: The imported class or function.
    """
    if name == 'scFoundationConfig':
        from .config import scFoundationConfig
        return scFoundationConfig

    if name == 'scFoundationModel':
        from .model import scFoundationModel
        return scFoundationModel

    if name == 'scFoundationPerturbationModel':
        from .perturbation_model import scFoundationPerturbationModel
        return scFoundationPerturbationModel

    if name == 'load_gene_list':
        from .config import load_gene_list
        return load_gene_list

    if name == 'load_training_config':
        from .config import load_training_config
        return load_training_config

    logger.error("Attribute '%s' not found in %s", name, __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'scFoundationConfig',
    'scFoundationModel',
    'scFoundationPerturbationModel',
    'load_gene_list',
    'load_training_config',
]
