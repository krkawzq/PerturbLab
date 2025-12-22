# scGPT Model Wrapper

This module provides a clean, user-friendly wrapper around the original [scGPT](https://github.com/bowang-lab/scGPT) implementation for single-cell perturbation prediction tasks.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model Classes](#model-classes)
- [API Reference](#api-reference)
- [Migration Details](#migration-details)
- [Usage Examples](#usage-examples)

---

## Overview

The wrapper provides two main model classes:

1. **`scGPTModel`**: General-purpose single-cell transformer for embedding, classification, and integration tasks
2. **`scGPTPerturbationModel`**: Specialized model for predicting perturbation effects on gene expression

Both models follow a unified API design and integrate seamlessly with HuggingFace Hub for model distribution.

### Key Features

- ✅ **HuggingFace Integration**: Automatic model download and caching
- ✅ **Unified API**: Consistent interfaces across all models
- ✅ **PyTorch-Native**: Full PyTorch compatibility with no external dependencies
- ✅ **Training Support**: Built-in training loops with AMP, gradient clipping, and checkpointing
- ✅ **Flexible Data Handling**: Automatic dataloader preparation from AnnData objects
- ✅ **Custom Vocab**: Pure Python vocabulary implementation (no torchtext dependency)

---

## Architecture

### Directory Structure

```
scgpt/
├── __init__.py          # Public API exports
├── config.py            # scGPTConfig class
├── model.py             # scGPTModel and scGPTPerturbationModel
└── source/              # Original scGPT implementation (modified)
    ├── model/           # TransformerModel, TransformerGenerator
    ├── tokenizer/       # GeneVocab, tokenization utilities
    ├── loss/            # Loss functions
    ├── utils/           # Utility functions (load_pretrained, etc.)
    └── torch_vocab.py   # Custom Vocab implementation
```

### Key Modifications

1. **Removed torchtext dependency**: Implemented custom `Vocab` class in pure Python
2. **Hardcoded file naming**: `args.json` for config, `model.pt`/`best_model.pt` for weights
3. **Lazy ntoken loading**: Config doesn't load vocab; model computes ntoken from `vocab.json`
4. **Weight compatibility**: `load_pretrained()` handles flash-attention ↔ PyTorch transformer conversion

---

## Model Classes

### scGPTModel

General-purpose transformer model for single-cell analysis.

**Use Cases:**
- Cell embedding generation
- Gene embedding extraction
- Cell type annotation (with CLS head)
- Batch integration
- Masked gene expression prediction

**Architecture:**
- Base: `TransformerModel` (encoder-only)
- Supports: MLM, MVC (GEPC), ECS, DAB, CLS objectives

### scGPTPerturbationModel

Specialized model for perturbation prediction tasks.

**Use Cases:**
- Predicting gene expression after genetic perturbations
- In-silico perturbation screening
- Perturbation effect analysis

**Architecture:**
- Base: `TransformerGenerator` (encoder-decoder style)
- Inputs: Control cell expression + perturbation flags
- Outputs: Predicted perturbed cell expression

---

## API Reference

### Configuration

```python
from perturblab.model.scgpt import scGPTConfig

config = scGPTConfig(
    ntoken=60697,           # Vocabulary size (loaded from vocab.json if None)
    d_model=512,            # Model dimension
    nhead=8,                # Number of attention heads
    nlayers=12,             # Number of transformer layers
    dropout=0.2,            # Dropout rate
    max_seq_len=None,       # Max sequence length (None = no limit)
    # ... more parameters
)
```

#### Key Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ntoken` | int \| None | None | Vocabulary size (auto-computed if None) |
| `d_model` | int | 512 | Model dimension |
| `nhead` | int | 8 | Number of attention heads |
| `nlayers` | int | 12 | Number of transformer layers |
| `dropout` | float | 0.2 | Dropout rate |
| `max_seq_len` | int \| None | None | Maximum sequence length |
| `use_fast_transformer` | bool | False | Use flash-attention if available |
| `do_mvc` | bool | False | Enable MVC (masked value prediction for cell embedding) |
| `do_dab` | bool | False | Enable DAB (domain adaptation by reverse backprop) |
| `batch_label_key` | str | 'batch' | Key in adata.obs for batch labels |
| `dab_weight` | float | 1.0 | Weight for DAB loss |

### Loading Pretrained Models

```python
from perturblab.model.scgpt import scGPTModel

# From HuggingFace (automatic download)
model = scGPTModel.from_pretrained("scgpt-human")

# Tissue-specific models
model = scGPTModel.from_pretrained("scgpt-brain")
model = scGPTModel.from_pretrained("scgpt-blood")

# From local directory
model = scGPTModel.from_pretrained("/path/to/model")

# With custom parameters
model = scGPTModel.from_pretrained(
    "scgpt-human",
    device='cuda',
    gene_list=my_custom_genes,  # Optional: use custom gene list
    revision="v1.0",             # HuggingFace revision
    cache_dir="./cache"          # Custom cache directory
)
```

**Available Pretrained Models** (HuggingFace: `perturblab` organization):
- `scgpt-human`: General human single-cell model
- `scgpt-blood`: Blood cell specialized
- `scgpt-brain`: Brain cell specialized
- `scgpt-heart`: Heart cell specialized
- `scgpt-kidney`: Kidney cell specialized
- `scgpt-lung`: Lung cell specialized
- `scgpt-pan-cancer`: Pan-cancer model
- `scgpt-continual-pretrained`: Continual pretrained model

### Training

```python
from perturblab.data import PerturbationData

# Prepare dataset (AnnData with 'split' column)
dataset = PerturbationData(adata)

# Train model
history = model.train_model(
    dataset=dataset,
    epochs=10,
    batch_size=32,
    lr=1e-4,
    mask_ratio=0.4,        # Masking ratio for MLM
    use_amp=True,          # Automatic mixed precision
    grad_clip=1.0,         # Gradient clipping
    log_interval=100,      # Log every N batches
    save_dir="./ckpts",    # Save checkpoints
    MVC=True,              # Enable MVC objective
    ECS=False,             # Enable ECS objective
    scheduler_gamma=0.99   # Learning rate decay
)

# Training history
print(history['train_loss'])
print(history['valid_loss'])
print(history['epoch_times'])
```

### Inference

#### Cell Embeddings

```python
import scanpy as sc
from perturblab.data import PerturbationData

# Load data
adata = sc.read_h5ad("data.h5ad")
dataset = PerturbationData(adata)

# Set model to evaluation mode
model.eval()

# Generate cell embeddings
cell_embs = model.predict_embeddings(
    dataset,
    batch_size=64,
    embedding_type="cell"
)

# Add to AnnData
adata.obsm['X_scGPT'] = cell_embs

# Downstream analysis
sc.pp.neighbors(adata, use_rep='X_scGPT')
sc.tl.umap(adata)
```

#### Gene Embeddings

```python
# Extract gene embeddings (directly from encoder)
gene_embs = model.predict_embeddings(
    dataset,
    embedding_type="gene"
)

# Shape: (n_genes, d_model)
```

#### Perturbation Prediction

```python
from perturblab.model.scgpt import scGPTPerturbationModel

# Load perturbation model
pert_model = scGPTPerturbationModel.from_pretrained("./finetuned_model")

# Predict perturbation effects
# Dataset should have 'condition' column with perturbation names
predictions = pert_model.predict_perturbation(
    dataset,
    batch_size=32,
    return_numpy=True
)

# Shape: (n_cells, n_genes)
```

### Forward Pass (Custom)

```python
# Prepare dataloader
dataloaders = model.prepare_dataloader(
    dataset,
    batch_size=32,
    mask_ratio=0.4,
    return_split='train'
)

# Forward pass
model.train()
for batch in dataloaders:
    # Forward
    output_dict = model.forward(
        batch,
        CLS=True,   # Include CLS prediction
        MVC=True,   # Include MVC prediction
        ECS=False   # Exclude ECS
    )
    
    # Compute loss
    losses = model.compute_loss(
        batch,
        output_dict=output_dict,
        CLS=True,
        MVC=True,
        mask_value=-1
    )
    
    total_loss = losses['loss']
    # ... backward pass ...
```

### Saving Models

```python
# Save model (saves both weights and config)
model.save("./my_model")

# Files created:
# - ./my_model/model.pt     (weights)
# - ./my_model/args.json    (config)
```

---

## Migration Details

### From Original scGPT

This wrapper maintains full compatibility with original scGPT checkpoints while providing a cleaner API.

#### Key Changes

1. **Vocabulary Handling**
   - **Original**: Uses `torchtext.vocab.Vocab` (deprecated, requires torchtext)
   - **Wrapper**: Custom `Vocab` class in pure Python (`source/torch_vocab.py`)
   - **Impact**: Removed torchtext dependency completely

2. **Configuration Loading**
   - **Original**: Config includes training parameters and model architecture
   - **Wrapper**: `scGPTConfig` only stores model architecture
   - **Impact**: Cleaner config, no training clutter

3. **Weight Loading**
   - **Original**: Manual handling of flash-attention vs PyTorch attention
   - **Wrapper**: `load_pretrained()` automatically converts weight formats
   - **Impact**: Seamless loading across different attention backends

4. **File Naming Convention**
   - **Original**: Flexible naming (model.pt, best_model.pt, checkpoint.pt, etc.)
   - **Wrapper**: Hardcoded to `model.pt`/`best_model.pt` and `args.json`
   - **Impact**: Consistent file structure, easier distribution

5. **Dataloader Preparation**
   - **Original**: Manual tokenization and batching
   - **Wrapper**: Automatic dataloader creation from AnnData
   - **Impact**: Much simpler data pipeline

#### Compatibility Matrix

| Feature | Original scGPT | Wrapper | Compatible |
|---------|---------------|---------|------------|
| Model weights | ✅ | ✅ | ✅ Yes |
| Config files | ✅ | ✅ | ✅ Yes (with key mapping) |
| Vocab files | ✅ | ✅ | ✅ Yes |
| Flash-attention | ✅ | ✅ | ✅ Auto-convert |
| Training scripts | ✅ | ✅ | ⚠️ Different API |
| Inference | ✅ | ✅ | ✅ Yes |

### Checkpoint Conversion

Original scGPT checkpoints work directly with this wrapper:

```python
# Original checkpoint directory structure:
# model_dir/
#   ├── best_model.pt    (weights)
#   ├── args.json        (config)
#   └── vocab.json       (optional)

# Load directly
model = scGPTModel.from_pretrained("./model_dir")
```

### Important Notes

1. **ntoken handling**: The wrapper computes `ntoken` from `vocab.json` if not in config
2. **Pad token**: Automatically added to vocab if missing, set as default index
3. **max_seq_len**: `None` means no length limit (uses all genes in data)
4. **Weight format**: Automatically handles Wqkv (flash-attn) ↔ in_proj_weight (PyTorch)

---

## Usage Examples

### Example 1: Cell Type Annotation

```python
from perturblab.model.scgpt import scGPTModel
from perturblab.data import PerturbationData
import scanpy as sc

# Load data
adata = sc.read_h5ad("pbmc.h5ad")
dataset = PerturbationData(adata)

# Load pretrained model
model = scGPTModel.from_pretrained("scgpt-human")

# Generate embeddings
embeddings = model.predict_embeddings(dataset, embedding_type="cell")

# Add to anndata
adata.obsm['X_scGPT'] = embeddings

# Clustering
sc.pp.neighbors(adata, use_rep='X_scGPT')
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden')
```

### Example 2: Perturbation Prediction

```python
from perturblab.model.scgpt import scGPTModel, scGPTPerturbationModel
from perturblab.data import PerturbationData

# Step 1: Fine-tune from pretrained model
base_model = scGPTModel.from_pretrained("scgpt-human")

# Initialize perturbation model with pretrained weights
config = base_model.config
pert_model = scGPTPerturbationModel(config, device='cuda')
pert_model.model.load_state_dict(
    base_model.model.state_dict(),
    strict=False  # Some layers may not match
)

# Step 2: Prepare perturbation dataset
# Dataset should have:
# - 'condition' column: perturbation names (e.g., 'GENE1', 'GENE1+GENE2', 'ctrl')
# - Paired control cells via dataset.pair_cells()
pert_dataset = PerturbationData(adata)
pert_dataset.pair_cells()  # Automatically pairs perturbed with control cells

# Step 3: Train perturbation model
history = pert_model.train_model(
    dataset=pert_dataset,
    epochs=15,
    batch_size=16,
    lr=5e-5,
    save_dir="./pert_model"
)

# Step 4: Predict perturbations
predictions = pert_model.predict_perturbation(
    pert_dataset,
    batch_size=32
)
```

### Example 3: Custom Training Loop

```python
import torch
from perturblab.model.scgpt import scGPTModel

model = scGPTModel.from_pretrained("scgpt-human")
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)

# Prepare data
dataloaders = model.prepare_dataloader(
    dataset,
    batch_size=32,
    mask_ratio=0.4,
    return_split=None  # Returns dict with train/valid/test
)

# Custom training loop
model.train()
for epoch in range(10):
    for batch in dataloaders['train']:
        # Forward pass
        output = model.forward(batch, MVC=True, ECS=True)
        
        # Compute loss
        losses = model.compute_loss(
            batch,
            output_dict=output,
            MVC=True,
            ECS=True,
            mask_value=-1
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        
        print(f"Loss: {losses['loss'].item():.4f}")
```

---

## Advanced Topics

### Custom Gene List

```python
# Use custom gene list instead of default vocabulary
my_genes = ["GENE1", "GENE2", ..., "GENEN"]

config = scGPTConfig(
    ntoken=len(my_genes),
    use_default_gene_vocab=False,
    # ... other params
)

model = scGPTModel(
    config,
    gene_list=my_genes,
    device='cuda'
)
```

### Mixed Precision Training

```python
# Automatically handled in train_model
history = model.train_model(
    dataset,
    use_amp=True,  # Enable automatic mixed precision
    grad_clip=1.0  # Gradient clipping for stability
)
```

### Distributed Training

```python
# Wrap model with DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")

# Wrap model
model = scGPTModel.from_pretrained("scgpt-human")
model = DDP(model.model, device_ids=[local_rank])

# Train normally (dataloader should use DistributedSampler)
```

---

## Troubleshooting

### Issue: "Token '<pad>' not found in vocabulary"

**Solution**: The wrapper automatically adds pad token if missing. If you see this error, ensure you're using the latest version.

### Issue: Flash-attention weight loading errors

**Solution**: The `load_pretrained()` function handles this automatically. Weights are converted between Wqkv (flash-attn) and in_proj_weight (PyTorch) formats.

### Issue: Out of memory during training

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing (not yet implemented)
3. Use mixed precision (`use_amp=True`)
4. Limit sequence length via `max_seq_len` in config

### Issue: Model loading timeout from HuggingFace

**Solution**: Models are cached after first download. Use `cache_dir` parameter to specify custom cache location.

---

## Citation

If you use this wrapper in your research, please cite both PerturbLab and the original scGPT paper:

```bibtex
@article{cui2024scgpt,
  title={scGPT: toward building a foundation model for single-cell multi-omics using generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Duan, Nan and Wang, Bo},
  journal={Nature Methods},
  volume={21},
  number={8},
  pages={1470--1480},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```

---

## License

This wrapper follows the same license as the original scGPT implementation (MIT License).

## Links

- **Original scGPT**: https://github.com/bowang-lab/scGPT
- **PerturbLab**: https://github.com/perturblab/perturblab
- **HuggingFace Models**: https://huggingface.co/perturblab

