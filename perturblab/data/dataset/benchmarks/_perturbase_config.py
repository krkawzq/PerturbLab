"""PerturbBase dataset configurations.

Complete list of datasets from PerturbBase (http://www.perturbase.cn/).
This configuration file stores all dataset metadata separately from the card creation logic.

PerturbBase is a comprehensive Chinese database of single-cell perturbation screens,
containing 122+ datasets with both raw and processed data.

Data structure:
    {
        'name': str,              # Dataset identifier
        'ncbi': str,              # NCBI accession
        'index': str,             # Data index
        'description': str,       # Description
        'type': PerturbationType, # Perturbation type
        'cell_type': str,         # Cell type (optional)
    }

Note: This is a foundational set. To add all 122+ datasets:
1. Visit http://www.perturbase.cn/download
2. Extract NCBI accession and data index for each dataset
3. Add configuration entries following the pattern below
"""

from ..cards import PerturbationType


# PerturbBase dataset configurations
# Based on: http://www.perturbase.cn/download
PERTURBASE_CONFIGS = [
    # ========================================================================
    # RNA Screens
    # ========================================================================
    
    {
        'name': 'TF_atlas_directed_diff_RNA',
        'ncbi': 'PRJNA893678',
        'index': '201218_RNA',
        'description': 'A transcription factor atlas of directed differentiation (RNA)',
        'type': PerturbationType.CRISPR,
    },
    
    # ========================================================================
    # CRISPRa Screens
    # ========================================================================
    
    {
        'name': 'Neuron_CRISPRa_lysosome',
        'ncbi': 'PRJNA641125',
        'index': 'CRISPRa',
        'description': 'Genome-wide CRISPRa screens in human neurons link lysosomal failure to ferroptosis',
        'type': PerturbationType.CRISPRA,
        'cell_type': 'Human neuron',
    },
    
    # ========================================================================
    # ECCITE-seq (Multimodal)
    # ========================================================================
    
    {
        'name': 'ECCITE_immune_checkpoints',
        'ncbi': 'PRJNA641353',
        'index': 'ECCITE',
        'description': 'Molecular regulation of inhibitory immune checkpoints with multimodal screens (ECCITE-seq)',
        'type': PerturbationType.CRISPR,
        'cell_type': 'T cell',
    },
    
    # ========================================================================
    # Oncogenicity Screens in Teratomas
    # ========================================================================
    
    {
        'name': 'Oncogenicity_Driver_library',
        'ncbi': 'PRJNA715235',
        'index': 'Driver_lib',
        'description': 'Charting oncogenicity of genes and variants - Driver library',
        'type': PerturbationType.GENETIC,
        'cell_type': 'Teratoma',
    },
    
    {
        'name': 'Oncogenicity_Driver_sub_library',
        'ncbi': 'PRJNA715235',
        'index': 'Driver_sub_lib',
        'description': 'Charting oncogenicity of genes and variants - Driver sub-library',
        'type': PerturbationType.GENETIC,
        'cell_type': 'Teratoma',
    },
    
    # ========================================================================
    # Combinatorial CRISPR Screens
    # ========================================================================
    
    {
        'name': 'Combinatorial_CRISPR_exp10',
        'ncbi': 'PRJNA609688',
        'index': 'exp10',
        'description': 'Combinatorial single-cell CRISPR screens by direct guide RNA capture - Experiment 10',
        'type': PerturbationType.CRISPR,
        'cell_type': 'K562',
    },
    
    {
        'name': 'Combinatorial_CRISPR_exp6',
        'ncbi': 'PRJNA609688',
        'index': 'exp6',
        'description': 'Combinatorial single-cell CRISPR screens by direct guide RNA capture - Experiment 6',
        'type': PerturbationType.CRISPR,
        'cell_type': 'K562',
    },
    
    {
        'name': 'Combinatorial_CRISPR_exp8',
        'ncbi': 'PRJNA609688',
        'index': 'exp8',
        'description': 'Combinatorial single-cell CRISPR screens by direct guide RNA capture - Experiment 8',
        'type': PerturbationType.CRISPR,
        'cell_type': 'K562',
    },
    
    {
        'name': 'Combinatorial_CRISPR_exp9',
        'ncbi': 'PRJNA609688',
        'index': 'exp9',
        'description': 'Combinatorial single-cell CRISPR screens by direct guide RNA capture - Experiment 9',
        'type': PerturbationType.CRISPR,
        'cell_type': 'K562',
    },
    
    # ========================================================================
    # CRISPRi - Host-Pathogen Interactions
    # ========================================================================
    
    {
        'name': 'CRISPRi_CMV_infection_host',
        'ncbi': 'PRJNA693896',
        'index': 'CRISPRi_perturb_host',
        'description': 'Functional single-cell genomics of human cytomegalovirus infection - Host perturbation',
        'type': PerturbationType.CRISPRI,
        'cell_type': 'Human fibroblast',
    },
    
    # ========================================================================
    # Additional Placeholder for Remaining ~112 Datasets
    # ========================================================================
    
    # Note: PerturbBase contains 122+ total datasets
    # Current count: 10 datasets integrated
    # Remaining: ~112 datasets
    #
    # To complete the integration:
    # 1. Visit http://www.perturbase.cn/download
    # 2. Browse through all pages (122 items, ~13 pages at 10 items/page)
    # 3. For each dataset, extract:
    #    - Title/Description
    #    - NCBI Accession (PRJNAxxxxxx)
    #    - Data Index (shown in "Data Index" column)
    #    - Perturbation type (infer from description)
    # 4. Add configuration entry following the pattern above
    #
    # Helper tools available:
    # - scripts/add_perturbase_datasets.py (interactive)
    # - scripts/scrape_perturbase.py (automated scraping)
    #
    # Example entry:
    # {
    #     'name': 'Descriptive_Name',
    #     'ncbi': 'PRJNAxxxxxx',
    #     'index': 'data_index_value',
    #     'description': 'Full description from website',
    #     'type': PerturbationType.CRISPR,  # or CRISPRI, CRISPRA, CHEMICAL, GENETIC
    #     'cell_type': 'Cell type if available',
    # },
]


def get_perturbase_configs():
    """Get all PerturbBase dataset configurations.
    
    Returns:
        List of dataset configuration dictionaries
    """
    return PERTURBASE_CONFIGS


def count_by_type():
    """Count datasets by perturbation type.
    
    Returns:
        Dictionary mapping perturbation type to count
    """
    counts = {}
    for config in PERTURBASE_CONFIGS:
        ptype = str(config['type'])
        counts[ptype] = counts.get(ptype, 0) + 1
    return counts


def count_by_ncbi():
    """Count datasets by NCBI project.
    
    Returns:
        Dictionary mapping NCBI accession to count
    """
    counts = {}
    for config in PERTURBASE_CONFIGS:
        ncbi = config['ncbi']
        counts[ncbi] = counts.get(ncbi, 0) + 1
    return counts


def search_configs(
    ncbi_accession: str = None,
    perturbation_type: PerturbationType = None,
    cell_type: str = None,
):
    """Search configurations by criteria.
    
    Args:
        ncbi_accession: Filter by NCBI accession
        perturbation_type: Filter by perturbation type
        cell_type: Filter by cell type (substring match)
    
    Returns:
        List of matching configurations
    """
    results = PERTURBASE_CONFIGS
    
    if ncbi_accession:
        results = [c for c in results if c['ncbi'] == ncbi_accession]
    
    if perturbation_type:
        results = [c for c in results if c['type'] == perturbation_type]
    
    if cell_type:
        results = [
            c for c in results 
            if 'cell_type' in c and cell_type.lower() in c['cell_type'].lower()
        ]
    
    return results
