# PerturbLab

> **A unified Python library for single-cell perturbation analysis and foundation models**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PerturbLab is a modular and high-performance library for single-cell analysis, featuring:
- 🚀 **High-performance kernels** - C++/Cython accelerated statistical operators
- 🎯 **Unified model registry** - 6 foundation models with consistent interface
- 🧬 **GEARS integration** - Graph-based perturbation prediction
- 📊 **Complete analysis toolkit** - Preprocessing, DE, HVG with optimized kernels
- 📦 **Minimal core dependencies** - Only numpy, scipy, torch required

---

## ✨ Key Features

### 🔥 High-Performance Statistical Kernels

Accelerated implementations with automatic backend selection (C++ > Cython > Numba > Python):

```python
import perturblab.preprocessing as pp
import anndata as ad

# Load data
adata = ad.read_h5ad('data.h5ad')

# High-performance preprocessing (auto-selects fastest backend)
pp.normalize_total(adata, target_sum=1e4)  # C++ accelerated
pp.scale(adata, max_value=10)              # Cython/OpenMP optimized

# Backend hierarchy: C++ > Cython > Numba > Python
```

**Performance**: Significantly faster than pure Python/NumPy implementations

### 🎯 Unified Model Registry

6 foundation models with intelligent lazy loading and hierarchical organization:

```python
from perturblab.models import MODELS

# Access models via registry (multiple styles supported)
model = MODELS.GEARS.GEARSModel(config, ...)          # Dot notation
model = MODELS['scGPT']['scGPTModel'](config, vocab)  # Dict style
model = MODELS.build("UCE.UCEModel", config)          # Config-driven

# Only loads model dependencies when accessed - fast startup!
```

**Available Models**:
- **GEARS**: Graph-based perturbation prediction
- **UCE**: Universal cell embeddings (Transformer)
- **scGPT**: Generative pretrained transformer (3 variants)
- **scFoundation**: Large-scale MAE with auto-binning
- **CellFM**: Retention-based foundation model
- **scELMo**: Non-parametric LLM embeddings

### 📊 Complete Analysis Toolkit

```python
from perturblab.analysis import highly_variable_genes, rank_genes_groups

# HVG detection with C++ kernels (optimized)
highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

# Differential expression with optimized kernels
rank_genes_groups(adata, groupby='perturbation', method='wilcoxon')
```

### 🧬 GEARS Perturbation Prediction

```python
from perturblab.methods import gears
from perturblab.models import MODELS

# Build perturbation graph from GO annotations
gene_list = adata.var_names
pert_graph = gears.build_perturbation_graph(
    gene_list,
    similarity='jaccard',
    threshold=0.1
)

# Load GEARS model
config = gears.GEARSConfig(num_genes=5000, num_perts=100)
model = MODELS.GEARS.GEARSModel(config, G_coexpress=..., G_go=...)
```

---

## 📦 Installation

### Requirements

- Python ≥ 3.11
- Core: numpy, scipy, torch, anndata, scikit-learn

### Quick Install

```bash
pip install perturblab
```

### Install with Specific Models

```bash
# Install with GEARS support
pip install perturblab[gears]

# Install with all foundation models
pip install perturblab[scgpt,uce,scfoundation,cellfm]

# Install with acceleration
pip install perturblab[accelerate]

# Install everything
pip install perturblab[all]
```

### Development Install

```bash
git clone https://github.com/krkawzq/PerturbLab.git
cd PerturbLab
pip install -e .
```

---

## 🚀 Quick Start

### 1. Download Datasets

PerturbLab provides easy access to benchmark datasets with automatic caching:

```python
from perturblab.data.resources import load_dataset, list_datasets
import anndata as ad

# List available datasets
print(list_datasets())
# ['go/go_basic', 'scperturb/norman_2019', 'scperturb/dixit_2016', ...]

# Download and load scPerturb benchmark dataset
h5ad_path = load_dataset('scperturb/norman_2019')
adata = ad.read_h5ad(h5ad_path)
print(f"Loaded dataset: {adata.shape}")

# Download GO ontology file
go_path = load_dataset('go/go_basic')
# File is automatically cached - subsequent calls are instant
```

**Available Datasets**:
- **scPerturb**: 55+ benchmark datasets from Zenodo
- **GO**: Gene Ontology files (basic, full)

All datasets are automatically cached in `~/.cache/perturblab/` for fast subsequent access.

### 2. Load Models

PerturbLab provides multiple ways to load models with intelligent dependency management:

#### Method 1: Using `Model()` Function (Recommended)

```python
from perturblab.models import Model
from perturblab.models.gears import GEARSConfig

# URL-style path with case-insensitive matching
model = Model("GEARS/default")(
    config=GEARSConfig(num_genes=1000, num_perts=50),
    G_coexpress=coexpress_graph,
    G_go=go_graph
)

# Access nested components
encoder = Model("scGPT/components/GeneEncoder")(vocab_size=5000, dim=512)

# Get model class directly
model_class = Model("UCE/default").class_
model = model_class(config=UCEConfig(token_dim=512))
```

#### Method 2: Using `MODELS` Registry

```python
from perturblab.models import MODELS
from perturblab.models.uce import UCEConfig

# Dot notation (IDE-friendly)
config = UCEConfig(token_dim=512, d_model=1280, nlayers=4)
model = MODELS.UCE.UCEModel(config)

# Dictionary-style access
model = MODELS['scGPT']['scGPTModel'](config, vocab)

# Build from config dict
model = MODELS.build("GEARS.default", num_genes=1000, num_perts=50)
```

**Dependency Management**:
- Required dependencies (`requirements`) are checked and raise `DependencyError` if missing
- Optional dependencies (`dependencies`) log info messages recommending installation
- Install missing dependencies: `pip install perturblab[model_name]`

### 3. High-Performance Preprocessing

```python
import perturblab.preprocessing as pp
import anndata as ad

# Load data
adata = ad.read_h5ad('data.h5ad')

# High-performance preprocessing
pp.normalize_total(adata, target_sum=1e4)  # C++ kernel
pp.scale(adata, max_value=10)              # Cython kernel

# Seamlessly compatible with scanpy
import scanpy as sc
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
```

### 4. Foundation Models

```python
from perturblab.models import Model, MODELS

# List available models
print(MODELS.list_keys(recursive=True))
# ['GEARS.GEARSModel', 'UCE.UCEModel', 'scGPT.scGPTModel', ...]

# Method 1: Using Model() function (recommended)
from perturblab.models.uce import UCEConfig, UCEInput
config = UCEConfig(token_dim=512, d_model=1280, nlayers=4)
model = Model("UCE/default")(config)

# Method 2: Using MODELS registry
model = MODELS.UCE.UCEModel(config)

# Forward pass with typed I/O
inputs = UCEInput(src=tokens, mask=padding_mask)
outputs = model(inputs)
embeddings = outputs.cell_embedding  # Type-safe access
```

### 5. GEARS Perturbation Prediction

```python
from perturblab.methods import gears
from perturblab.models.gears import GEARSConfig, GEARSInput

# Build gene similarity graph
pert_graph = gears.build_perturbation_graph(
    gene_vocab=adata.var_names,
    similarity='jaccard',
    threshold=0.1,
    num_workers=4  # parallel computation
)

# Create and use model
config = GEARSConfig(num_genes=5000, num_perts=100)
model = MODELS.GEARS.default(config, G_coexpress=..., G_go=...)

inputs = GEARSInput(
    gene_expression=expr,
    pert_idx=[[0, 1], [2]],  # Multi-gene perturbations
    graph_batch_indices=batch
)
outputs = model(inputs)
predictions = outputs.predictions  # Type-safe
```

### 6. Highly Variable Genes

```python
from perturblab.analysis import highly_variable_genes

# Using PerturbLab's optimized kernels (2-5x faster)
highly_variable_genes(
    adata, 
    n_top_genes=2000,
    flavor="seurat_v3",
    batch_key="batch"
)

# Or use as method on PerturbationData
from perturblab.types import PerturbationData
dataset = PerturbationData(adata, perturbation_col='condition')
hvg_genes = dataset.calculate_hvg(n_top_genes=2000)
```

---

## 🏗️ Architecture

```
PerturbLab/
├── perturblab/
│   ├── core/                 # Core infrastructure
│   │   ├── config.py        # Config base class
│   │   ├── model_io.py      # ModelIO base class
│   │   ├── model_registry.py # Model registry
│   │   └── dataset.py       # Dataset base classes
│   ├── models/               # Foundation models (6 models)
│   │   ├── gears/           # Graph-based perturbation prediction
│   │   ├── uce/             # Universal cell embeddings
│   │   ├── scgpt/           # Generative pretrained transformer
│   │   ├── scfoundation/    # Large-scale MAE
│   │   ├── cellfm/          # Retention-based model
│   │   └── scelmo/          # LLM-based embeddings
│   ├── methods/              # Analysis methods
│   │   └── gears/           # GEARS utilities
│   ├── kernels/              # High-performance kernels
│   │   ├── statistics/      # Statistical operators (C++/Cython)
│   │   │   ├── backends/    # C++, Cython, Python implementations
│   │   │   └── ops/         # Unified operator interface
│   │   └── mapping/         # Mapping kernels
│   ├── preprocessing/        # Preprocessing functions
│   │   ├── _normalization.py
│   │   └── _scale.py
│   ├── analysis/             # Analysis tools
│   │   ├── _de.py           # Differential expression
│   │   └── _hvg.py          # Highly variable genes
│   ├── tools/                # General-purpose tools
│   │   ├── _bipartite.py    # Graph projection
│   │   ├── _gene_similarity.py
│   │   └── _split_*.py      # Data splitting
│   ├── types/                # Type definitions
│   │   ├── _vocab.py        # Generic vocabulary
│   │   ├── _gene_vocab.py   # Gene-specific vocabulary
│   │   ├── _cell.py         # Cell dataset
│   │   ├── _perturbation.py # Perturbation dataset
│   │   └── math/            # Graph types
│   └── utils/                # Utilities
├── forks/                    # Original implementations (reference)
├── pyproject.toml
└── README.md
```

---

## ⚡ Performance Features

### Optimized Kernels

PerturbLab includes high-performance statistical kernels with automatic backend selection:

- **C++ Backend**: SIMD vectorization + OpenMP parallelization
- **Cython Backend**: Compiled Python extensions
- **Numba Backend**: JIT compilation (fallback)
- **Python Backend**: Pure NumPy (universal fallback)

The library automatically selects the fastest available backend at import time.

### Lazy Model Loading

PerturbLab uses intelligent lazy loading to minimize startup time:

- **Scan Phase**: Lightweight directory scan (no imports)
- **Load Phase**: Only loads models when accessed
- **Cache Phase**: Subsequent access is instant

This enables fast startup even with many models registered.

---

## 🎓 Advanced Usage

### Model Loading Patterns

```python
from perturblab.models import Model, MODELS

# Method 1: Model() function (URL-style, case-insensitive)
model = Model("GEARS/default")(config=GEARSConfig(...), G_coexpress=..., G_go=...)
model = Model("scGPT/default")(config=scGPTConfig(...), vocab=vocab)
model = Model("gears/default")  # Case-insensitive

# Method 2: MODELS registry (dot notation, IDE-friendly)
model = MODELS.scGPT.scGPTModel(config, vocab)

# Method 3: Dictionary access (dynamic)
model = MODELS['scFoundation']['scFoundationModel'](config)

# Method 4: Build from config
model = MODELS.build("GEARS.default", num_genes=1000, num_perts=50)

# Access components
encoder = Model("scGPT/components/GeneEncoder")(vocab_size=5000, dim=512)
encoder = MODELS.scGPT.components.GeneEncoder(vocab_size, dim)
```

### Dataset Download Patterns

```python
from perturblab.data.resources import load_dataset, get_dataset, list_datasets

# List all available datasets
datasets = list_datasets()
print(f"Available datasets: {len(datasets)}")

# Download dataset (returns Path, downloads if needed)
h5ad_path = load_dataset('scperturb/norman_2019')
import anndata as ad
adata = ad.read_h5ad(h5ad_path)

# Get dataset resource object (for metadata)
resource = get_dataset('scperturb/norman_2019')
print(f"Resource key: {resource.key}")
print(f"Has remote: {resource._remote_config is not None}")

# All datasets are cached automatically
# Second call is instant (uses cache)
h5ad_path_2 = load_dataset('scperturb/norman_2019')  # Fast!
```

### Custom Preprocessing Pipeline

```python
import perturblab.preprocessing as pp

def preprocess_pipeline(adata):
    # High-performance kernels
    pp.normalize_total(adata, target_sum=1e4)
    pp.scale(adata, max_value=10)
    
    # Analysis
    from perturblab.analysis import highly_variable_genes
    highly_variable_genes(adata, n_top_genes=2000)
    
    return adata

adata = preprocess_pipeline(adata)
```

### Bipartite Graph Projection

```python
from perturblab.tools import compute_gene_similarity_from_go

# Build gene-gene similarity from GO annotations
gene2go = {
    'TP53': {'GO:0001', 'GO:0002'},
    'KRAS': {'GO:0002', 'GO:0003'},
}
similarity_df = compute_gene_similarity_from_go(
    gene2go,
    similarity='jaccard',
    threshold=0.1,
    num_workers=4
)
```

---

## 📚 API Reference

### Preprocessing

- `perturblab.preprocessing.normalize_total(adata, target_sum)` - Normalize counts per cell
- `perturblab.preprocessing.scale(adata, max_value)` - Scale to unit variance

### Analysis

- `perturblab.analysis.highly_variable_genes(adata, n_top_genes, flavor)` - HVG detection
- `perturblab.analysis.rank_genes_groups(adata, groupby, method)` - Differential expression

### GEARS Methods

- `perturblab.methods.gears.build_perturbation_graph(genes, similarity, threshold)` - Build GO-based gene graph
- `perturblab.methods.gears.filter_perturbations_in_go(perturbations, go_genes)` - Filter valid perturbations

### Tools

- `perturblab.tools.project_bipartite_graph(graph, similarity)` - Project bipartite graph
- `perturblab.tools.split_cells(adata, split_ratio)` - Split cells for train/val/test
- `perturblab.tools.split_perturbations_simple(perturbations, split_ratio)` - Split perturbations

### Model Loading

- `Model(key)` - Load model using URL-style path (e.g., `"GEARS/default"`)
  - Case-insensitive matching
  - Supports nested paths (e.g., `"scGPT/components/GeneEncoder"`)
  - Returns `ModelBuilder` with `.build()`, `.class_`, and `__call__()` methods
- `MODELS.{Model}.{Variant}(config, ...)` - Create model from registry (dot notation)
- `MODELS.build(key, **params)` - Build model from config string
- `MODELS.list_keys(recursive)` - List available models
- `MODELS.{Model}.components.{Component}(...)` - Access model components

### Dataset Resources

- `load_dataset(path)` - Download and return Path to dataset file
  - Automatically caches in `~/.cache/perturblab/`
  - Supports paths like `'scperturb/norman_2019'`, `'go/go_basic'`
- `get_dataset(path)` - Get dataset resource object (for metadata)
- `list_datasets()` - List all available datasets

---

## 🔧 Configuration

### Logging

```python
import os
# Set log level
os.environ['PERTURBLAB_LOG_LEVEL'] = 'DEBUG'  # or 'INFO', 'WARNING'

# Or programmatically
from perturblab.utils import set_log_level
set_log_level('INFO')  # Default: no DEBUG messages
```

### Disable Auto-Loading

```python
# Environment variable
os.environ['PERTURBLAB_DISABLE_AUTO_LOAD'] = 'TRUE'

# Or global flag
import perturblab
perturblab._disable_auto_load = True
```

---

## 📝 Supported Models

| Model | Type | Architecture | Dependencies |
|-------|------|--------------|--------------|
| **GEARS** | Perturbation | Graph Neural Network | torch-geometric |
| **UCE** | Embedding | Transformer | accelerate |
| **scGPT** | Foundation | GPT-style Transformer | - |
| **scFoundation** | Foundation | MAE + Auto-binning | local-attention* |
| **CellFM** | Foundation | Retention mechanism | - |
| **scELMo** | Embedding | Non-parametric | - |

\* Optional dependency

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/krkawzq/PerturbLab.git
cd PerturbLab
pip install -e ".[accelerate]"
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

Individual models may have their own licenses:
- scGPT: MIT License
- scFoundation: MIT License
- CellFM: CC BY-NC-ND 4.0
- See respective forks/ directories for details

---

## 🙏 Acknowledgments

PerturbLab builds upon excellent work from:

- **GEARS**: [Roohani et al., Nature Biotechnology 2023](https://www.nature.com/articles/s41587-023-01905-6)
- **scGPT**: [Cui et al., Nature Methods 2024](https://doi.org/10.1038/s41592-024-02201-0)
- **UCE**: [Rosen et al., bioRxiv 2023](https://doi.org/10.1101/2023.11.28.568918)
- **scFoundation**: [Wang et al., bioRxiv 2023](https://doi.org/10.1101/2023.05.29.542705v3)
- **CellFM**: CellFM Authors

Special thanks to:
- OpenMMLab for registry design inspiration
- scanpy/anndata for single-cell ecosystem
- PyTorch and NumPy communities

---

## 📧 Contact

- **Repository**: [https://github.com/krkawzq/PerturbLab](https://github.com/krkawzq/PerturbLab)
- **Issues**: [https://github.com/krkawzq/PerturbLab/issues](https://github.com/krkawzq/PerturbLab/issues)
- **Author**: Wang Zhongqi (2868116803@qq.com)

---

**Built with ❤️ for single-cell genomics research**
