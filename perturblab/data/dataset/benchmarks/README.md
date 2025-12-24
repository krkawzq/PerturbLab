# Benchmark Dataset System

## Overview

This module provides unified access to curated perturbation datasets from multiple sources:
- **scPerturb**: 31+ datasets from Zenodo
- **PerturbBase**: 122+ datasets from Chinese database
- **GO/GEARS**: Gene Ontology and annotation resources

## Architecture

```
benchmark/
├── __init__.py                 # Public API
├── _registry.py                # Core registry system
├── _scperturb.py               # scPerturb dataset cards (8 curated)
├── _perturbase.py              # PerturbBase loader
├── _perturbase_config.py       # PerturbBase configurations (10+ so far)
├── _go.py                      # GO resources (6 resources)
└── README.md                   # This file
```

## Current Status

### Integrated Datasets

| Source | Datasets | Status |
|--------|----------|--------|
| scPerturb | 8 | ✅ Curated selection |
| PerturbBase | 10 | 🔄 Partial (122+ available) |
| GO Consortium | 2 | ✅ With mirrors |
| GEARS | 1 | ✅ Complete |

### scPerturb Datasets (8/31+)

1. Norman2019 - CRISPRi screen (105,942 cells)
2. Replogle2022_K562_essential - Essential genes
3. Replogle2022_K562_gwps - Genome-wide
4. Replogle2022_RPE1 - RPE1 cells
5. Dixit2016_K562_TFs - Transcription factors
6. Adamson2016 - K562 screen
7. Srivatsan2020_sciplex3 - Chemical perturbations
8. Frangieh2021_RNA - Melanoma

### PerturbBase Datasets (10/122+)

Currently integrated:
1. TF_atlas_directed_diff (PRJNA893678)
2. Neuron_CRISPRa_lysosome_ferroptosis (PRJNA641125)
3. ECCITE_immune_checkpoints (PRJNA641353)
4. Oncogenicity_Driver_library (PRJNA715235)
5. Oncogenicity_Driver_sub_library (PRJNA715235)
6. Combinatorial_CRISPR_exp10 (PRJNA609688)
7. Combinatorial_CRISPR_exp6 (PRJNA609688)
8. Combinatorial_CRISPR_exp8 (PRJNA609688)
9. Combinatorial_CRISPR_exp9 (PRJNA609688)
10. CRISPRi_CMV_infection_host (PRJNA693896)

## How to Add More Datasets

### Option 1: Manual Addition

Edit `_perturbase_config.py` and add entries:

```python
{
    'name': 'Your_Dataset_Name',
    'ncbi': 'PRJNAxxxxxx',
    'index': 'data_index_from_website',
    'description': 'Description from PerturbBase',
    'type': PerturbationType.CRISPR,
    'cell_type': 'Cell type (optional)',
},
```

### Option 2: Use Helper Script

```bash
python scripts/add_perturbase_datasets.py
```

Follow the prompts to batch add datasets.

### Option 3: Web Scraping

```bash
python scripts/scrape_perturbase.py > new_datasets.txt
```

This will attempt to scrape http://www.perturbase.cn/download and generate configuration entries.

## Usage Examples

### List All Datasets

```python
from perturblab.data.dataset import list_benchmark_datasets

# List all
datasets = list_benchmark_datasets()

# Filter by source
perturbase = list_benchmark_datasets(source='PerturbBase')

# Filter by type
crispr_i = list_benchmark_datasets(perturbation_type=PerturbationType.CRISPRI)
```

### Load a Dataset

```python
from perturblab.data.dataset import load_any_dataset, get_dataset_info

# Get metadata
info = get_dataset_info('Norman2019')
print(info)

# Load dataset (auto-download + cache)
adata = load_any_dataset('Norman2019')

# Load with specific source
adata = load_any_dataset('Norman2019', source='scPerturb')
```

### Auto-Fallback to Mirrors

```python
# GO ontology with multiple mirrors
# If main source fails, automatically tries EBI and NCBI mirrors
terms, dag = load_any_dataset('go_basic', auto_fallback=True)
```

## Extending with New Sources

To add a new data source (e.g., CELLxGENE, HuggingFace):

1. Create a new Card class in `cards/`
2. Create a dataset list file in `benchmark/`
3. Import and add to `ALL_DATASETS` in `benchmark/__init__.py`

Example:

```python
# cards/_cellxgene.py
@dataclass
class CELLxGENECard(H5ADDatasetCard):
    collection_id: str = ''
    
    def __post_init__(self):
        object.__setattr__(self, 'source', 'CELLxGENE')
        if not self.url:
            url = f"https://cellxgene.cziscience.com/collections/{self.collection_id}"
            object.__setattr__(self, 'url', url)

# benchmark/_cellxgene.py
CELLXGENE_DATASETS = [
    CELLxGENECard(
        name='...',
        collection_id='...',
        ...
    ),
]
```

## Notes

- **PerturbBase** provides both raw and processed data (tar.gz format)
- **scPerturb** datasets are h5ad files directly
- **GO resources** have multiple mirrors for reliability
- All downloads are automatically cached in `~/.perturblab/downloads/`

## Contributing

To complete the PerturbBase integration (122+ datasets):

1. Visit http://www.perturbase.cn/download
2. Extract dataset metadata (NCBI, Index, Description)
3. Add to `_perturbase_config.py`
4. Submit PR

Target: Add all 122 datasets for comprehensive coverage!

