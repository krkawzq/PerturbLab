import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

def __getattr__(name):
    """Lazy import for CellFM model classes.

    Args:
        name (str): The name of the attribute to fetch.

    Returns:
        type: The imported class or configuration.
    """
    if name == 'CellFMConfig':
        from .config import CellFMConfig
        return CellFMConfig

    if name == 'CellFMModel':
        from .model import CellFMModel
        return CellFMModel
    
    if name == 'CellFMPerturbationModel':
        from .model import CellFMPerturbationModel
        return CellFMPerturbationModel
    
    if name == 'CellFMGeneMapper':
        from .gene_mapping import CellFMGeneMapper
        return CellFMGeneMapper
    
    if name == 'get_gene_mapper':
        from .gene_mapping import get_gene_mapper
        return get_gene_mapper

    logger.error("Attribute '%s' not found in %s", name, __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'CellFMModel',
    'CellFMPerturbationModel',
    'CellFMConfig',
    'CellFMGeneMapper',
    'get_gene_mapper',
]

