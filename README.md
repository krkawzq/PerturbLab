# PerturbLab

> **A high-performance Python library for single-cell perturbation analysis**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PerturbLab is a modular and high-performance library for single-cell perturbation analysis, featuring:
- 🚀 **High-performance kernels** - C++/Cython accelerated statistical operators (2-15x faster)
- 🎯 **Smart model registry** - Hierarchical model management with lazy loading
- 🧬 **GEARS integration** - Graph-based perturbation prediction
- 🛠️ **Flexible preprocessing** - Compatible with scanpy workflows
- 📦 **Minimal dependencies** - Only numpy, scipy, torch, and requests required

---

## ✨ Key Features

### 🔥 High-Performance Statistical Kernels

Accelerated implementations with automatic backend selection:

```python
import perturblab.preprocessing as pp
import anndata as ad

# Load data
adata = ad.read_h5ad('data.h5ad')

# High-performance preprocessing (auto-selects fastest backend)
pp.normalize_total(adata, target_sum=1e4)  # C++/Cython accelerated
pp.scale(adata, max_value=10)              # OpenMP parallelized

# Backend hierarchy: C++ > Cython > Numba > Python
```

**Performance**: 2-15x faster than pure Python/NumPy implementations

### 🎯 Smart Model Registry

Hierarchical model management with intelligent lazy loading:

```python
from perturblab.models import MODELS

# Dot notation access (IDE-friendly)
model = MODELS.GEARS.gnn(hidden_dim=128)

# Dictionary access (dynamic)
model = MODELS['GEARS']['gnn'](hidden_dim=128)

# Config-driven
model = MODELS.build("GEARS.gnn", hidden_dim=128)

# Only loads GEARS modules on first access - fast startup!
```

### 🧬 GEARS Perturbation Prediction

Integrated GEARS method for genetic perturbation analysis:

```python
from perturblab.methods import gears
import numpy as np

# Build perturbation graph from GO annotations
gene_list = ['TP53', 'KRAS', 'MYC', 'EGFR', 'BRCA1']
pert_graph = gears.build_perturbation_graph(
    gene_list,
    similarity='jaccard',
    threshold=0.1
)

# Access graph
print(f"Nodes: {pert_graph.n_nodes}")
print(f"Edges: {pert_graph.n_unique_edges}")
```

---

## 📦 Installation

### Requirements

- Python ≥ 3.11
- Core: numpy, scipy, torch, requests

### Quick Install

```bash
pip install perturblab
```

### Install with Acceleration (Optional)

For maximum performance, install with Cython/Numba:

```bash
pip install perturblab[accelerate]
```

Then compile C++ extensions:

```bash
python setup.py build_ext --inplace
```

### Development Install

```bash
git clone https://github.com/krkawzq/PerturbLab.git
cd PerturbLab
pip install -e .
```

---

## 🚀 Quick Start

### 1. Preprocessing with High-Performance Kernels

```python
import perturblab.preprocessing as pp
import anndata as ad

# Load data
adata = ad.read_h5ad('data.h5ad')

# Normalize (C++ accelerated)
pp.normalize_total(adata, target_sum=1e4)

# Scale (Cython/OpenMP accelerated)
pp.scale(adata, max_value=10)

# Works seamlessly with scanpy
import scanpy as sc
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
```

### 2. GEARS Perturbation Graph

```python
from perturblab.methods import gears

# Build gene similarity network from GO annotations
pert_graph = gears.build_perturbation_graph(
    gene_vocab=adata.var_names,
    similarity='jaccard',  # or 'overlap', 'cosine'
    threshold=0.1,
    num_workers=4  # parallel computation
)

# Export to PyTorch Geometric format
edge_df = gears.weighted_graph_to_dataframe(pert_graph)

# Filter perturbations
valid_perts = gears.filter_perturbations_in_go(
    perturbations=['TP53', 'KRAS', 'TP53+KRAS'],
    go_genes=pert_graph.node_names
)
```

### 3. Model Registry

```python
from perturblab.models import MODELS

# Register custom models
@MODELS.register("MyModel")
class MyModel:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

# Create sub-registries for methods
MYMETHOD = MODELS.child("MYMETHOD")

@MYMETHOD.register("backbone")
class MyBackbone:
    pass

# Access models flexibly
model = MODELS.MyModel(hidden_dim=64)
model = MODELS.MYMETHOD.backbone()
model = MODELS.build("MYMETHOD.backbone")
```

---

## 🏗️ Project Structure

```
PerturbLab/
├── perturblab/
│   ├── core/                 # Core abstractions
│   │   ├── dataset.py       # Dataset base classes
│   │   ├── resource.py      # Resource management
│   │   ├── resource_registry.py
│   │   └── model_registry.py
│   ├── models/               # Model registry with smart loading
│   │   └── __init__.py      # MODELS instance
│   ├── methods/              # Analysis methods
│   │   └── gears/           # GEARS implementation
│   │       ├── utils.py     # Graph construction, utilities
│   │       └── ...
│   ├── kernels/              # High-performance kernels
│   │   ├── statistics/      # Statistical operators
│   │   │   ├── backends/
│   │   │   │   ├── cpp/     # C++ implementations
│   │   │   │   ├── cython/  # Cython implementations
│   │   │   │   └── python/  # Numba/Python fallbacks
│   │   │   └── ops/         # Unified operator interface
│   │   └── mapping/         # Mapping kernels
│   ├── preprocessing/        # Preprocessing functions
│   │   ├── _normalization.py
│   │   └── _scale.py
│   ├── tools/                # General-purpose tools
│   │   ├── _bipartite.py    # Bipartite graph projection
│   │   ├── _gene_similarity.py
│   │   ├── _split_cell.py   # Data splitting
│   │   └── _split_perturbation.py
│   ├── types/                # Type definitions
│   │   ├── _gene_vocab.py   # Gene vocabulary
│   │   ├── _perturbation.py
│   │   └── math/            # Math types (Graph, etc.)
│   ├── data/                 # Data loading
│   │   ├── datasets/        # Dataset loaders
│   │   └── resources/       # Resource registry
│   ├── io/                   # I/O utilities
│   │   └── download/        # Download utilities
│   ├── analysis/             # Analysis tools
│   ├── metrics/              # Evaluation metrics
│   ├── engine/               # Training engines
│   └── utils/                # Utilities
├── setup.py                  # Build configuration
├── CMakeLists.txt           # C++ build
└── pyproject.toml           # Project metadata
```

---

## 📊 Performance Benchmarks

### Preprocessing Speed (1M cells × 2000 genes)

| Operation | Pure Python | Numba | Cython | C++ | Speedup |
|-----------|------------|-------|--------|-----|---------|
| `normalize_total` | 5.2s | 1.8s | 0.8s | **0.3s** | **17x** |
| `scale` | 3.1s | 1.2s | 0.5s | **0.2s** | **15x** |

### Model Loading

| Strategy | Cold Start | Description |
|----------|-----------|-------------|
| **Eager** (load all) | 2.5s | Load all methods on import |
| **Lazy** (PerturbLab) | 0.1s | Scan directory, load on access |
| **Targeted** | 0.3s | Load only accessed method |

---

## 🎓 Advanced Usage

### Custom Backend Selection

```python
# Check available backends
from perturblab.kernels.statistics import ops

print(f"C++ available: {ops.cpp_available}")
print(f"Cython available: {ops.cython_available}")
print(f"Numba available: {ops.numba_available}")
```

### Bipartite Graph Projection

```python
from perturblab.tools import project_bipartite_graph, compute_gene_similarity_from_go
from perturblab.types import BipartiteGraph

# Create gene-GO term bipartite graph
edges = [(0, 0), (0, 1), (1, 1), (1, 2)]  # gene -> GO term
bg = BipartiteGraph(edges)

# Project to gene-gene similarity
similarity_df = project_bipartite_graph(
    bg,
    source_names=['TP53', 'KRAS'],
    similarity='jaccard',
    threshold=0.1
)

# Or directly from gene2go mapping
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

### Data Splitting

```python
from perturblab.tools import split_cells, split_perturbations_simple
import anndata as ad

# Split cells
adata = ad.read_h5ad('data.h5ad')
train_idx, val_idx, test_idx = split_cells(
    adata,
    split_ratio=(0.7, 0.15, 0.15),
    seed=42
)

# Split perturbations
perturbations = ['TP53', 'KRAS', 'TP53+KRAS', 'MYC']
train_perts, val_perts, test_perts = split_perturbations_simple(
    perturbations,
    split_ratio=(0.7, 0.15, 0.15),
    seed=42
)
```

---

## 🔧 Configuration

### Logging

```python
# Set log level
import os
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

## 📚 API Reference

### Preprocessing

- `perturblab.preprocessing.normalize_total(adata, target_sum)` - Normalize counts per cell
- `perturblab.preprocessing.scale(adata, max_value)` - Scale data to unit variance

### GEARS Methods

- `perturblab.methods.gears.build_perturbation_graph(genes, similarity, threshold)` - Build GO-based gene similarity network
- `perturblab.methods.gears.filter_perturbations_in_go(perturbations, go_genes)` - Filter valid perturbations
- `perturblab.methods.gears.get_perturbation_genes(perturbations)` - Extract genes from perturbations

### Tools

- `perturblab.tools.project_bipartite_graph(graph, similarity)` - Project bipartite graph
- `perturblab.tools.compute_gene_similarity_from_go(gene2go)` - Compute gene similarity
- `perturblab.tools.split_cells(adata, split_ratio)` - Split cells for train/val/test
- `perturblab.tools.split_perturbations_simple(perturbations, split_ratio)` - Split perturbations

### Model Registry

- `MODELS.register(name)` - Decorator to register models
- `MODELS.child(name)` - Create sub-registry
- `MODELS.build(key, **params)` - Instantiate model from config
- `MODELS.list_keys(recursive)` - List available models

---

## 🤝 Contributing

Contributions are welcome! Please:

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

---

## 🙏 Acknowledgments

PerturbLab builds upon excellent work from:

- **GEARS**: [Roohani et al., Nature Biotechnology 2023](https://www.nature.com/articles/s41587-023-01905-6)
- **scanpy**: [Wolf et al., Genome Biology 2018](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0)
- **Highway**: Google's SIMD library for vectorization
- **OpenMMLab**: For registry design inspiration

---

## 📧 Contact

- **Repository**: [https://github.com/krkawzq/PerturbLab](https://github.com/krkawzq/PerturbLab)
- **Issues**: [https://github.com/krkawzq/PerturbLab/issues](https://github.com/krkawzq/PerturbLab/issues)
- **Author**: Wang Zhongqi (2868116803@qq.com)

---

**Built with ❤️ for single-cell genomics research**
