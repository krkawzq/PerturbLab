# PerturbLab

A unified Python library for single-cell foundation models and perturbation prediction.

## Overview

PerturbLab provides a standardized interface for working with state-of-the-art single-cell foundation models. It simplifies the process of loading pre-trained models, generating cell embeddings, and predicting perturbation effects.

## Features

- 🧬 **Unified Interface**: Consistent API across different foundation models
- 🚀 **Pre-trained Models**: Easy access to models via HuggingFace Hub
- 🔬 **Perturbation Prediction**: Built-in support for genetic perturbation analysis
- 📊 **Rich Data Support**: Compatible with AnnData and custom PerturbationData formats
- ⚡ **Efficient Processing**: Optimized data loading and batching
- 🎯 **Flexible Fine-tuning**: Support for downstream task adaptation

## Supported Models

### Foundation Models

| Model | Parameters | Description | Status |
|-------|-----------|-------------|--------|
| **scGPT** | 100M | Transformer-based model for single-cell analysis | ✅ Ready |
| **scFoundation** | 100M | Foundation model with perturbation capabilities | ✅ Ready |
| **UCE** | 4-layer, 33-layer | Universal Cell Embedding model | ✅ Ready |
| **CellFM** | 80M, 800M | Retention-based foundation model | ✅ Ready |
| **scELMo** | - | ELMo-inspired single-cell model | ✅ Ready |

### Perturbation Models

| Model | Base | Description | Status |
|-------|------|-------------|--------|
| **GEARS** | GNN | Graph-based perturbation prediction | ✅ Ready |
| **scFoundation + GEARS** | scFoundation | Hybrid foundation + GNN model | ✅ Ready |
| **CellFM + GEARS** | CellFM | Retention-based + GNN model | ✅ Ready |

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
from perturblab.model.gears import GearsModel
from perturblab.data import PerturbationData

# Load perturbation data
data = PerturbationData.from_anndata(adata)
data.split_data(train=0.7, val=0.15, test=0.15)

# Initialize model
model = GearsModel.from_pretrained('gears')

# Train on your data
model.train_model(data, epochs=20)

# Predict perturbation effects
predictions = model.predict_perturbation(data, split='test')

# Evaluate
metrics = model.evaluate(data, split='test')
print(f"Pearson correlation: {metrics['pearson']:.4f}")
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

All models are available on HuggingFace Hub under the `perturblab` organization:

- [`perturblab/scgpt`](https://huggingface.co/perturblab/scgpt)
- [`perturblab/scfoundation`](https://huggingface.co/perturblab/scfoundation)
- [`perturblab/uce-4-layer`](https://huggingface.co/perturblab/uce-4-layer)
- [`perturblab/uce-33-layer`](https://huggingface.co/perturblab/uce-33-layer)
- [`perturblab/cellfm-80m`](https://huggingface.co/perturblab/cellfm-80m)
- [`perturblab/cellfm-800m`](https://huggingface.co/perturblab/cellfm-800m)
- [`perturblab/scelmo`](https://huggingface.co/perturblab/scelmo)
- [`perturblab/gears`](https://huggingface.co/perturblab/gears)

## Advanced Usage

### Fine-tuning for Cell Type Classification

```python
from perturblab.model.scgpt import scGPTModel, scGPTConfig

# Initialize model with classification head
config = scGPTConfig(num_classes=10)  # 10 cell types
model = scGPTModel(config, for_finetuning=True)

# Load pre-trained weights
model.load_weights('path/to/pretrained/weights.pt')

# Fine-tune on labeled data
model.train_model(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=10,
    learning_rate=1e-4,
)
```

### Custom Data Processing

```python
from perturblab.data import PerturbationData
import scanpy as sc

# Load and preprocess
adata = sc.read_h5ad('perturbation_data.h5ad')

# Create PerturbationData object
data = PerturbationData.from_anndata(
    adata,
    condition_key='perturbation',
    control_key='ctrl',
)

# Set GEARS format for perturbation models
data.set_gears_format(cell_type_key='cell_type')

# Split data
data.split_data(train=0.7, val=0.15, test=0.15)

# Use with any perturbation model
from perturblab.model.scfoundation import scFoundationPerturbationModel

model = scFoundationPerturbationModel.from_pretrained('scfoundation')
model.init_perturbation_head_from_dataset(data)
model.train_model(data, epochs=20)
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

```python
data.adata                 # Underlying AnnData object
data.adata.obs['condition'] # Perturbation conditions
data.adata.obs['split']    # Train/val/test splits
data.pert_names            # List of perturbation names
data.ctrl_expression       # Control expression profiles
```

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
