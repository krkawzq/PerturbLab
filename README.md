# PerturbLab

A unified Python library for single-cell foundation models and perturbation prediction.

## Overview

PerturbLab provides a standardized interface for working with state-of-the-art single-cell foundation models. It simplifies the process of loading pre-trained models, generating cell embeddings, and predicting perturbation effects.

## Features

- 🧬 **Unified Interface**: Consistent API across different foundation models (scGPT, scFoundation, UCE, CellFM, scELMo)
- 🚀 **Pre-trained Models**: 25+ pre-trained models available via HuggingFace Hub with automatic loading
- 🔬 **Perturbation Prediction**: Hybrid foundation + GNN models for genetic perturbation analysis
- 📊 **Rich Data Support**: Compatible with AnnData and custom PerturbationData formats
- ⚡ **Efficient Processing**: Optimized data loading, batching, and GPU acceleration
- 🎯 **Flexible Fine-tuning**: Support for downstream tasks (classification, perturbation, etc.)
- 🧪 **Gene Mapping**: Built-in HGNC gene name standardization for CellFM
- 📦 **Easy Installation**: Simple installation with uv or pip

## Supported Models

### Foundation Models (Embedding Generation)

| Model | Parameters | Pre-trained Weights | Description |
|-------|-----------|---------------------|-------------|
| **scGPT** | ~100M | 9 variants (human, blood, brain, heart, kidney, lung, pan-cancer, continual-pretrained) | Transformer-based foundation model for single-cell analysis |
| **scFoundation** | ~100M | 3 variants (cell, gene, rde) | Large-scale foundation model with multi-task capabilities |
| **UCE** | 4-layer, 33-layer | 2 variants | Universal Cell Embedding with zero-shot capabilities |
| **CellFM** | 80M, 800M | 2 variants | Retention-based foundation model for transcriptomics |
| **scELMo** | - | 8 variants (gene, protein, perturbation, drugs, celltypes, celllines with different LLM backbones) | ELMo-inspired contextualized gene embeddings |

### Perturbation Prediction Models

| Model | Base Architecture | Description | Status |
|-------|------------------|-------------|--------|
| **GEARS** | GNN | Graph-based perturbation prediction using gene co-expression and GO networks | ✅ Ready |
| **scGPT + GEARS** | scGPT + GNN | Combines scGPT embeddings with GEARS perturbation head | ✅ Ready |
| **scFoundation + GEARS** | scFoundation + GNN | Hybrid foundation model + GNN for perturbation prediction | ✅ Ready |
| **CellFM + GEARS** | CellFM + GNN | Retention-based encoder + GNN perturbation head | ✅ Ready |

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/PerturbLab.git
cd PerturbLab

# Install with uv
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Generate Cell Embeddings

```python
from perturblab.model.scgpt import scGPTModel
import scanpy as sc

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Load pre-trained model
model = scGPTModel.from_pretrained('scgpt')

# Generate embeddings
embeddings = model.predict_embeddings(adata, batch_size=32)
cell_embeddings = embeddings['cell_embeddings']
```

### Perturbation Prediction

```python
from perturblab.model.scfoundation import scFoundationPerturbationModel
from perturblab.data import PerturbationData

# Load perturbation data (adata should have perturbation info in obs)
data = PerturbationData(
    adata,
    perturb_col='perturbation',  # Column name in adata.obs
    control_tag='control',        # Tag for control cells
)
data.set_gears_format(fallback_cell_type='unknown')
data.split_data(split_type='simple', split_ratio=(0.7, 0.15, 0.15))

# Initialize model with pre-trained foundation model
model = scFoundationPerturbationModel.from_pretrained('scfoundation-cell')

# Initialize perturbation head from dataset
model.init_perturbation_head_from_dataset(data)

# Train on your data
model.train_model(data, epochs=20, lr=1e-4)

# Predict perturbation effects
predictions = model.predict_perturbation(data, split='test')

# Evaluate
metrics = model.evaluate(data, split='test')
print(f"Pearson correlation: {metrics['test_pearson']:.4f}")
```

### Using CellFM with Gene Mapping

```python
from perturblab.model.cellfm import CellFMModel, CellFMGeneMapper
import scanpy as sc

# Load data
adata = sc.read_h5ad('your_data.h5ad')

# Map gene names to CellFM vocabulary (optional)
mapper = CellFMGeneMapper()
adata = mapper.prepare_adata_with_mapping(adata, max_genes=2048)

# Prepare data
adata = CellFMModel.prepare_data(adata)

# Load model (supports both 80M and 800M versions)
model = CellFMModel.from_pretrained('cellfm-80m')  # or 'cellfm-800m'

# Generate embeddings
embeddings = model.predict_embeddings(adata, batch_size=16)
```

## Model Zoo

### Available Pre-trained Weights

All models are available on HuggingFace Hub under the [`perturblab`](https://huggingface.co/perturblab) organization:

#### scGPT Models
- [`perturblab/scgpt-human`](https://huggingface.co/perturblab/scgpt-human) - General human tissues
- [`perturblab/scgpt-blood`](https://huggingface.co/perturblab/scgpt-blood) - Blood cells
- [`perturblab/scgpt-brain`](https://huggingface.co/perturblab/scgpt-brain) - Brain tissues
- [`perturblab/scgpt-heart`](https://huggingface.co/perturblab/scgpt-heart) - Heart tissues
- [`perturblab/scgpt-kidney`](https://huggingface.co/perturblab/scgpt-kidney) - Kidney tissues
- [`perturblab/scgpt-lung`](https://huggingface.co/perturblab/scgpt-lung) - Lung tissues
- [`perturblab/scgpt-pan-cancer`](https://huggingface.co/perturblab/scgpt-pan-cancer) - Pan-cancer
- [`perturblab/scgpt-continual-pretrained`](https://huggingface.co/perturblab/scgpt-continual-pretrained) - Continual learning

#### scFoundation Models
- [`perturblab/scfoundation-cell`](https://huggingface.co/perturblab/scfoundation-cell) - Cell-level embeddings
- [`perturblab/scfoundation-gene`](https://huggingface.co/perturblab/scfoundation-gene) - Gene-level embeddings
- [`perturblab/scfoundation-rde`](https://huggingface.co/perturblab/scfoundation-rde) - RDE variant

#### UCE Models
- [`perturblab/uce-4-layer`](https://huggingface.co/perturblab/uce-4-layer) - 4-layer model
- [`perturblab/uce-33-layer`](https://huggingface.co/perturblab/uce-33-layer) - 33-layer model

#### CellFM Models
- [`perturblab/cellfm-80m`](https://huggingface.co/perturblab/cellfm-80m) - 80M parameters
- [`perturblab/cellfm-800m`](https://huggingface.co/perturblab/cellfm-800m) - 800M parameters

#### scELMo Models
- [`perturblab/scelmo-gene-gpt-4o`](https://huggingface.co/perturblab/scelmo-gene-gpt-4o) - Gene embeddings (GPT-4o)
- [`perturblab/scelmo-gene-gpt-3.5`](https://huggingface.co/perturblab/scelmo-gene-gpt-3.5) - Gene embeddings (GPT-3.5)
- [`perturblab/scelmo-gene-ncbi`](https://huggingface.co/perturblab/scelmo-gene-ncbi) - Gene embeddings (NCBI)
- [`perturblab/scelmo-protein-gpt-3.5`](https://huggingface.co/perturblab/scelmo-protein-gpt-3.5) - Protein embeddings
- [`perturblab/scelmo-perturbation-gpt-3.5`](https://huggingface.co/perturblab/scelmo-perturbation-gpt-3.5) - Perturbation embeddings
- [`perturblab/scelmo-drugs-gpt-3.5`](https://huggingface.co/perturblab/scelmo-drugs-gpt-3.5) - Drug embeddings
- [`perturblab/scelmo-celltypes-gpt-3.5`](https://huggingface.co/perturblab/scelmo-celltypes-gpt-3.5) - Cell type embeddings
- [`perturblab/scelmo-celllines-gpt-3.5`](https://huggingface.co/perturblab/scelmo-celllines-gpt-3.5) - Cell line embeddings

## Advanced Usage

### Fine-tuning scGPT for Cell Type Classification

```python
from perturblab.model.scgpt import scGPTModel
from perturblab.data import PerturbationData

# Load pre-trained model
model = scGPTModel.from_pretrained('scgpt-human')

# Prepare your labeled data
data = PerturbationData(
    adata,
    perturb_col='cell_type',  # For classification, use cell type column
    control_tag='control',
)
data.split_data(split_type='simple', split_ratio=(0.7, 0.15, 0.15))

# Fine-tune on labeled data
model.train_model(
    dataset=data,
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
)

# Generate embeddings after fine-tuning
embeddings = model.predict_embeddings(data.adata)
```

### Using GEARS for Perturbation Prediction

```python
from perturblab.model.gears import GearsModel
from perturblab.data import PerturbationData
import scanpy as sc

# Load perturbation data
adata = sc.read_h5ad('perturbation_data.h5ad')

# Create PerturbationData object
data = PerturbationData(
    adata,
    perturb_col='perturbation',  # Column in adata.obs with perturbation info
    control_tag='ctrl',           # Tag for control cells
)

# Set GEARS format (required for GEARS-based models)
data.set_gears_format(fallback_cell_type='unknown')

# Compute DE genes (required for GEARS)
data.compute_de_genes(n_top_genes=20)

# Split data
data.split_data(split_type='simple', split_ratio=(0.7, 0.15, 0.15))

# Initialize GEARS model
model = GearsModel.from_pretrained('gears', device='cuda')

# Or initialize from dataset
# model = GearsModel(device='cuda')
# model.init_perturbation_head_from_dataset(data)

# Train model
model.train_model(data, epochs=20, lr=1e-3)

# Predict and evaluate
predictions = model.predict_perturbation(data, split='test')
metrics = model.evaluate(data, split='test')
print(f"Pearson: {metrics['test_pearson']:.4f}")
```

### Batch Processing Large Datasets

```python
from perturblab.model.uce import UCEModel
import scanpy as sc

# Load model
model = UCEModel.from_pretrained('uce-4-layer')

# Process in batches
batch_size = 64
all_embeddings = []

for i in range(0, len(adata), batch_size):
    batch = adata[i:i+batch_size]
    embeddings = model.predict_embeddings(batch, batch_size=32)
    all_embeddings.append(embeddings['cell_embeddings'])

# Combine results
import numpy as np
final_embeddings = np.vstack(all_embeddings)
```

## Architecture

```
PerturbLab/
├── perturblab/
│   ├── data/              # Data structures and loaders
│   │   ├── perturbation.py
│   │   └── graph.py
│   ├── model/             # Model implementations
│   │   ├── scgpt/
│   │   ├── scfoundation/
│   │   ├── uce/
│   │   ├── cellfm/
│   │   ├── scelmo/
│   │   └── gears/
│   ├── utils/             # Utility functions
│   └── configuration.py   # Base configuration classes
├── weights/               # Pre-trained model weights
└── tests/                 # Unit tests
```

## Data Format

PerturbLab works with standard single-cell data formats:

### AnnData Format

```python
adata.X                    # Expression matrix (cells × genes)
adata.obs                  # Cell metadata
adata.var                  # Gene metadata
adata.obs['cell_type']     # Cell type annotations (optional)
```

### PerturbationData Format

The `PerturbationData` class wraps AnnData for perturbation experiments:

```python
from perturblab.data import PerturbationData
import scanpy as sc

# Load your data
adata = sc.read_h5ad('perturbation_data.h5ad')
# adata.obs should contain:
# - 'perturbation': perturbation conditions (e.g., 'GENE1', 'GENE1+GENE2', 'control')
# - 'cell_type': cell type information (optional)

# Initialize PerturbationData
data = PerturbationData(
    adata,
    perturb_col='perturbation',  # Column in adata.obs with perturbation info
    control_tag='control',        # Tag(s) for control cells (can be list)
    ignore_tags=['unknown'],      # Optional: tags to ignore
)

# For GEARS-based models, apply GEARS format
data.set_gears_format(
    fallback_cell_type='unknown',  # Default cell type if not specified
)
# This standardizes column names:
# - 'perturbation' → 'condition'
# - 'control' → 'ctrl'

# Compute DE genes (required for GEARS models)
data.compute_de_genes(
    n_top_genes=20,
    method='t-test_overestim_var',
    use_hpdex=False,  # Set True for faster computation with hpdex
)

# Split data
data.split_data(
    split_type='simple',           # 'simple', 'simulation', 'combo_seen0', etc.
    split_ratio=(0.7, 0.15, 0.15), # (train, val, test)
    seed=1,
)

# Access data
data.adata                       # Underlying AnnData object
data.adata.obs['condition']      # Perturbation conditions (after GEARS format)
data.adata.obs['split']          # Train/val/test splits
data.adata.obs['cell_type']      # Cell type information
data.perturb_col                 # Current perturbation column name
data.control_tags                # Set of control tags
data.gears_format                # Boolean: whether GEARS format is applied

# Convert to GEARS PertData (if needed)
pert_data = data.to_gears(data_path='./data', check_de_genes=True)
```

### Split Types

PerturbationData supports multiple split strategies:

- **`simple`**: Random split by cells (train/val/test)
- **`simulation`**: Gene-level split (seen/unseen genes)
- **`simulation_single`**: Single perturbation gene-level split
- **`combo_seen0`**: Combo perturbations with 0 seen genes
- **`combo_seen1`**: Combo perturbations with 1 seen gene
- **`combo_seen2`**: Combo perturbations with 2 seen genes
- **`no_test`**: Only train/val split (no test set)

## Performance Tips

1. **Use GPU**: All models support GPU acceleration
   ```python
   model = Model.from_pretrained('model-name', device='cuda')
   ```

2. **Optimize Batch Size**: Larger batches = faster processing
   ```python
   embeddings = model.predict_embeddings(adata, batch_size=64)
   ```

3. **Enable Mixed Precision**: For faster training
   ```python
   model.train_model(train_loader, use_amp=True)
   ```

4. **Cache Preprocessed Data**: Save preprocessing time
   ```python
   adata_processed = model.prepare_data(adata)
   adata_processed.write_h5ad('processed_data.h5ad')
   ```

## Citation

If you use PerturbLab in your research, please cite the original papers for the models you use:

### scGPT
```bibtex
@article{cui2024scgpt,
  title={scGPT: toward building a foundation model for single-cell multi-omics using generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Duan, Nan and Wang, Bo},
  journal={Nature Methods},
  year={2024}
}
```

### scFoundation
```bibtex
@article{hao2024large,
  title={Large-scale foundation model on single-cell transcriptomics},
  author={Hao, Minsheng and Gong, Jing and Zeng, Xin and Liu, Chiming and Guo, Jianzhu and Cheng, Xingyi and Wang, Taifeng and Ma, Jianzhu and Song, Le and Zhang, Xuegong},
  journal={Nature Methods},
  year={2024}
}
```

### UCE
```bibtex
@article{rosen2024universal,
  title={Universal Cell Embeddings: A Foundation Model for Cell Biology},
  author={Rosen, Yanay and Roohani, Yusuf and Agarwal, Ayush and Samotorčan, Leon and Consortium, Tabula Sapiens and Quake, Stephen R and Leskovec, Jure},
  journal={bioRxiv},
  year={2024}
}
```

### CellFM
```bibtex
@article{zhao2024cellfm,
  title={CellFM: A Large-Scale Foundation Model for Single-Cell Transcriptomics},
  author={Zhao, Shuai and others},
  journal={bioRxiv},
  year={2024}
}
```

### GEARS
```bibtex
@article{roohani2023predicting,
  title={Predicting transcriptional outcomes of novel multigene perturbations with GEARS},
  author={Roohani, Yusuf and Huang, Kexin and Leskovec, Jure},
  journal={Nature Biotechnology},
  year={2023}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Individual models may have their own licenses. Please refer to the original repositories:

- [scGPT](https://github.com/bowang-lab/scGPT)
- [scFoundation](https://github.com/biomap-research/scFoundation)
- [UCE](https://github.com/snap-stanford/UCE)
- [CellFM](https://github.com/biomed-AI/CellFM)
- [scELMo](https://github.com/HelloWorldLTY/scELMo)
- [GEARS](https://github.com/snap-stanford/GEARS)

## Acknowledgments

PerturbLab builds upon the excellent work of many researchers in the single-cell genomics community. We are grateful to the authors of the original models for making their work available.

## Contact

- **Issues**: Please report bugs and request features via [GitHub Issues](https://github.com/yourusername/PerturbLab/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/PerturbLab/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Note**: This is a research tool. Please validate results carefully before using in production or clinical settings.
