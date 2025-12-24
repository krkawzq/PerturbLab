## Perturbation Prediction Metrics

This module provides comprehensive evaluation metrics for perturbation prediction models, migrated and enhanced from scPerturbBench.

### Overview

The metrics module evaluates prediction quality across four key dimensions:

1. **Expression Accuracy**: How well predicted expression matches ground truth
2. **Distribution Similarity**: How well distributions match at the population level  
3. **Direction Consistency**: Whether perturbation effects have correct direction (up/down regulation)
4. **DEG Overlap**: How well predicted differentially expressed genes match true DEGs

### Quick Start

```python
import perturblab as pl
import numpy as np

# Create or load your data
pred = ...  # Predicted expression [cells × genes]
true = ...  # True expression [cells × genes]
ctrl = ...  # Control expression [cells × genes]

# Compute all metrics at once
metrics = pl.metrics.evaluate_prediction(pred, true, ctrl)

# Or compute individual metric categories
expr_metrics = pl.metrics.compute_expression_metrics(pred, true, ctrl)
dist_metrics = pl.metrics.compute_distribution_metrics(pred, true)
dir_metrics = pl.metrics.compute_direction_metrics(pred, true, ctrl)
```

### Available Metrics

#### 1. Expression Metrics (`_expression.py`)

Evaluate absolute expression and perturbation effect (delta) accuracy:

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| `r_squared` | R² coefficient | [-∞, 1] | Higher |
| `pearson_correlation` | Pearson correlation | [-1, 1] | Higher |
| `mse` | Mean squared error | [0, ∞] | Lower |
| `rmse` | Root mean squared error | [0, ∞] | Lower |
| `mae` | Mean absolute error | [0, ∞] | Lower |
| `cosine_similarity_score` | Cosine similarity | [-1, 1] | Higher |
| `l2_distance` | Euclidean distance | [0, ∞] | Lower |

All metrics computed for both:
- **Absolute expression**: `metric(pred_mean, true_mean)`
- **Delta expression**: `metric(pred_mean - ctrl_mean, true_mean - ctrl_mean)`

```python
# Individual metrics
r2 = pl.metrics.r_squared(pred, true, ctrl=None)  # Absolute
r2_delta = pl.metrics.r_squared(pred, true, ctrl=ctrl)  # Delta

# All metrics at once
metrics = pl.metrics.compute_expression_metrics(pred, true, ctrl)
# Returns: {
#     'R_squared': ...,
#     'R_squared_delta': ...,
#     'Pearson_Correlation': ...,
#     'Pearson_Correlation_delta': ...,
#     ...
# }
```

#### 2. Distribution Metrics (`_distribution.py`)

Evaluate how well predicted distributions match true distributions:

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| `mmd` | Maximum Mean Discrepancy | [0, ∞] | Lower |
| `wasserstein_distance` | Earth Mover's Distance | [0, ∞] | Lower |

```python
# Individual metrics
mmd_val = pl.metrics.mmd(pred, true, kernel='rbf', gamma=1.0)
ws_val = pl.metrics.wasserstein_distance(pred, true)

# Per-gene option
mmd_per_gene = pl.metrics.mmd(pred, true, per_gene=True)  # Returns array

# All metrics at once
metrics = pl.metrics.compute_distribution_metrics(pred, true)
# Returns: {'MMD': ..., 'Wasserstein': ...}
```

**MMD (Maximum Mean Discrepancy):**
- Measures distribution distance in RKHS (Reproducing Kernel Hilbert Space)
- RBF kernel: k(x,y) = exp(-γ||x-y||²)
- Smaller γ = smoother kernel, less sensitive to outliers
- Computed per gene then averaged

**Wasserstein Distance:**
- Optimal transport distance between distributions
- Also known as Earth Mover's Distance (EMD)
- Intuitive interpretation: minimum cost to transform one distribution into another

#### 3. Direction Consistency (`_direction.py`)

Evaluate whether perturbation effects have the correct direction:

| Metric | Description | Range |
|--------|-------------|-------|
| `delta_agreement_acc` | Fraction of genes with correct direction | [0, 1] |

```python
# Direction accuracy
acc = pl.metrics.delta_direction_accuracy(pred, true, ctrl)
# Returns: 0.85 (85% of genes have correct up/down direction)

# Per-gene consistency
per_gene = pl.metrics.delta_direction_accuracy(pred, true, ctrl, per_gene=True)
# Returns: boolean array [True, False, True, ...]

# Comprehensive direction metrics
metrics = pl.metrics.compute_direction_metrics(pred, true, ctrl)
# Returns: {
#     'delta_agreement_acc': 0.85,
#     'n_genes_up_pred': 120,
#     'n_genes_down_pred': 80,
#     'n_genes_up_true': 100,
#     'n_genes_down_true': 100,
#     'n_genes_agree': 170,
# }
```

**Interpretation:**
- 1.0 = Perfect direction agreement for all genes
- 0.5 = Random guessing performance
- Useful when magnitude is less important than regulation pattern

#### 4. DEG Overlap Metrics (`_deg_overlap.py`)

Evaluate overlap between predicted and true differentially expressed genes:

| Metric Type | Description | Range |
|-------------|-------------|-------|
| `deg_overlap_topn` | Overlap for top-N ranked genes | [0, 1] |
| `deg_overlap_pvalue` | Overlap for p < threshold genes | [0, 1] |
| `deg_overlap_fdr` | Overlap for FDR < threshold genes | [0, 1] |

```python
# Requires DEG DataFrames from differential_expression
pred_degs = pl.data.process.differential_expression(
    pred_adata, groupby_key='condition', reference='ctrl'
)
true_degs = pl.data.process.differential_expression(
    true_adata, groupby_key='condition', reference='ctrl'
)

# Individual metrics
overlap_20 = pl.metrics.deg_overlap_topn(pred_degs, true_degs, top_n=20)
overlap_p005 = pl.metrics.deg_overlap_pvalue(pred_degs, true_degs, p_threshold=0.05)
overlap_fdr = pl.metrics.deg_overlap_fdr(pred_degs, true_degs, fdr_threshold=0.05)

# All DEG overlap metrics
metrics = pl.metrics.compute_deg_overlap_metrics(
    pred_degs, true_degs,
    top_n_list=[20, 50, 100, 200],
    p_thresholds=[0.05, 0.01],
    fdr_thresholds=[0.05, 0.01],
)
# Returns: {
#     'Top20_DEG_Overlap': 0.85,
#     'Top50_DEG_Overlap': 0.76,
#     'Top100_DEG_Overlap': 0.68,
#     'P<005_DEG_Overlap': 0.72,
#     'FDR<005_DEG_Overlap': 0.65,
#     ...
# }
```

**Overlap Calculation:**
- Top-N: `|pred_top_N ∩ true_top_N| / N`
- P-value: `|pred_sig ∩ true_sig| / |true_sig|`
- FDR: `|pred_sig ∩ true_sig| / |true_sig|`

### Comprehensive Evaluation

Use `evaluate_prediction()` to compute all metrics at once:

```python
# Without DEG overlap
metrics = pl.metrics.evaluate_prediction(
    pred, true, ctrl,
    include_deg_overlap=False,
)
# Returns 22 metrics (7 expression × 2 + 2 distribution + 6 direction)

# With DEG overlap
metrics = pl.metrics.evaluate_prediction(
    pred, true, ctrl,
    pred_degs=pred_degs,
    true_degs=true_degs,
    deg_top_n=[20, 50, 100],
    deg_p_thresholds=[0.05, 0.01],
    deg_fdr_thresholds=[0.05],
)
# Returns 30+ metrics (includes DEG overlaps)

# Customize which metrics to compute
metrics = pl.metrics.evaluate_prediction(
    pred, true, ctrl,
    include_expression=True,
    include_distribution=True,
    include_direction=True,
    include_deg_overlap=False,
    mmd_gamma=0.5,  # Adjust MMD kernel bandwidth
)
```

### Sparse Matrix Support

All metrics automatically handle sparse matrices:

```python
from scipy import sparse

# Sparse matrices are automatically converted
pred_sparse = sparse.csr_matrix(pred)
true_sparse = sparse.csr_matrix(true)
ctrl_sparse = sparse.csr_matrix(ctrl)

# Works seamlessly
metrics = pl.metrics.evaluate_prediction(pred_sparse, true_sparse, ctrl_sparse)
```

### Interpretation Guide

**Good Prediction Indicators:**
- R² > 0.8, Pearson > 0.9 (expression)
- R²_delta > 0.7, Pearson_delta > 0.8 (perturbation effect)
- Direction accuracy > 0.7 (most genes correct)
- Top-20 DEG overlap > 0.5 (at least half of key genes)
- Low MMD and Wasserstein distance

**Model Comparison:**
- Higher R², Pearson, Cosine, DEG overlaps, direction accuracy = Better
- Lower MSE, RMSE, MAE, L2, MMD, Wasserstein = Better

### Performance

Metrics are optimized for large-scale evaluation:

| Dataset Size | Compute Time |
|--------------|--------------|
| 100 cells × 50 genes | ~0.1s |
| 1000 cells × 500 genes | ~1s |
| 10000 cells × 2000 genes | ~10s |

### Examples

#### Example 1: Basic Evaluation

```python
import perturblab as pl
import numpy as np

# Load data
adata = pl.datasets.load_dataset("norman_2019")
pred_adata = ...  # Your model predictions
ctrl_adata = adata[adata.obs['perturbation'] == 'control']

# Extract matrices
pred = pred_adata.X
true = adata[adata.obs['perturbation'] == 'IRF1'].X
ctrl = ctrl_adata.X

# Evaluate
metrics = pl.metrics.evaluate_prediction(pred, true, ctrl)

# Print results
print(f"R² = {metrics['R_squared']:.3f}")
print(f"R² (delta) = {metrics['R_squared_delta']:.3f}")
print(f"Direction accuracy = {metrics['delta_agreement_acc']:.2%}")
```

#### Example 2: Model Comparison

```python
import perturblab as pl
import pandas as pd

models = ['ModelA', 'ModelB', 'ModelC']
results = []

for model_name in models:
    pred = load_predictions(model_name)
    metrics = pl.metrics.evaluate_prediction(pred, true, ctrl)
    metrics['model'] = model_name
    results.append(metrics)

# Compare models
df = pd.DataFrame(results)
print(df[['model', 'R_squared_delta', 'Pearson_Correlation_delta', 
          'delta_agreement_acc']].sort_values('R_squared_delta', ascending=False))
```

#### Example 3: Per-Gene Analysis

```python
# Get per-gene metrics
mmd_per_gene = pl.metrics.mmd(pred, true, per_gene=True)
dir_per_gene = pl.metrics.delta_direction_accuracy(pred, true, ctrl, per_gene=True)

# Find problematic genes
import pandas as pd
gene_analysis = pd.DataFrame({
    'gene': gene_names,
    'mmd': mmd_per_gene,
    'direction_correct': dir_per_gene,
})

# Worst performing genes (high MMD, wrong direction)
worst_genes = gene_analysis[
    (gene_analysis['mmd'] > gene_analysis['mmd'].quantile(0.9)) &
    (gene_analysis['direction_correct'] == False)
]
print(f"Problematic genes:\n{worst_genes}")
```

### References

- scPerturbBench: https://github.com/wxicu/scPerturbBench
- MMD: Gretton et al. (2012). "A Kernel Two-Sample Test"
- Wasserstein: Villani (2009). "Optimal Transport: Old and New"

### Spatial Autocorrelation Metrics (from scanpy)

#### Moran's I

Moran's I measures global spatial autocorrelation on a graph. High positive values indicate that similar values cluster together spatially.

```python
import perturblab as pl
from scipy import sparse

# Requires a neighbor graph (e.g., from KNN or spatial neighbors)
morans = pl.metrics.morans_i(graph, expression)

# For multiple genes
morans_per_gene = pl.metrics.morans_i(graph, expression_matrix)  # [genes × cells]

# Interpretation:
# I > 0: Positive autocorrelation (similar values cluster)
# I ≈ 0: Random pattern
# I < 0: Negative autocorrelation (dissimilar values cluster)
```

#### Geary's C

Geary's C measures local spatial autocorrelation. Lower values indicate stronger positive autocorrelation.

```python
gearys = pl.metrics.gearys_c(graph, expression)

# Interpretation:
# C < 1: Positive autocorrelation
# C = 1: Random pattern
# C > 1: Negative autocorrelation
```

**Use cases:**
- Spatial transcriptomics: Identify spatially coherent gene expression patterns
- Cell-cell communication: Test if ligand-receptor pairs show coordinated spatial patterns
- Perturbation spreading: Evaluate if perturbation effects propagate to neighboring cells

#### Confusion Matrix

Evaluate classification or clustering accuracy:

```python
# Compare predicted to true cell types
cm = pl.metrics.confusion_matrix(
    adata.obs['true_cell_type'],
    adata.obs['predicted_cell_type'],
    normalize=True
)

# Visualize
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')

# Compute accuracy
import numpy as np
accuracy = np.diag(cm.values).mean()
print(f"Overall accuracy: {accuracy:.2%}")
```

**Use cases:**
- Label transfer evaluation
- Clustering quality assessment
- Cell type annotation validation
- Perturbation response classification

### Copyright and Attribution

This module contains code from multiple sources:

**scPerturbBench metrics** (_expression.py, _distribution.py, _direction.py, _deg_overlap.py):
- Original implementation for PerturbLab
- References: https://github.com/wxicu/scPerturbBench

**scanpy metrics** (_spatial.py, _classification.py):
- Adapted from scanpy (https://github.com/scverse/scanpy)
- Original authors: F. Alexander Wolf, P. Angerer, Theis Lab
- License: BSD-3-Clause
- Modifications: Simplified API, enhanced documentation, removed AnnData dependencies

Please cite the original papers when using these metrics in publications.

### Troubleshooting

**Q: Why is my R² negative?**

A: R² can be negative when predictions are worse than just predicting the mean. Check if your model is actually learning patterns.

**Q: DEG overlap is very low, but other metrics look good?**

A: This can happen if your model predicts similar overall patterns but misses specific gene signatures. Try:
- Using more flexible DEG overlap thresholds
- Checking if the ranking of genes is still correlated (Spearman)
- Focusing on top-N overlap (which is often more forgiving)

**Q: MMD seems unstable across runs?**

A: Try adjusting the `gamma` parameter. Smaller values (0.1-0.5) give more stable estimates but may miss fine details.

**Q: Getting NaN values?**

A: Common causes:
- Insufficient variance in data (all genes have same expression)
- No significant DEGs (for DEG overlap metrics)
- Numerical issues with very small values

Use logger output to debug:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

