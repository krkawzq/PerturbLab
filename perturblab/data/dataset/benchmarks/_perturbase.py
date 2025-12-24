"""PerturbBase dataset cards.

PerturbBase (http://www.perturbase.cn/) is a Chinese database containing
122+ single-cell perturbation datasets with both raw and processed data.

This module creates DatasetCards from configurations defined in _perturbase_config.py
"""

from ..cards import PerturbBaseCard
from ._perturbase_config import get_perturbase_configs


def create_perturbase_cards():
    """Create PerturbBaseCard instances from configurations.
    
    Returns:
        List of PerturbBaseCard instances
    """
    cards = []
    
    for config in get_perturbase_configs():
        card = PerturbBaseCard(
            name=config['name'],
            url='',  # Auto-constructed from ncbi_accession and data_index
            description=config['description'],
            citation=config['ncbi'],  # Use NCBI as citation
            ncbi_accession=config['ncbi'],
            data_index=config['index'],
            data_type='processed',  # Default to processed
            perturbation_type=config['type'],
            cell_type=config.get('cell_type', None),
        )
        cards.append(card)
    
    return cards


# Create all PerturbBase dataset cards
PERTURBASE_DATASETS = create_perturbase_cards()
