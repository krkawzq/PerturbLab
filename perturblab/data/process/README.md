# Data Processing Module

This module provides high-performance data processing and analysis tools for single-cell data, leveraging optimized C++/Cython kernels.

## Differential Expression Analysis

### Quick Start

```python
import perturblab as pl

# Load data
adata = pl.datasets.load_dataset("adamson_2016")

# Run differential expression
results = pl.data.process.differential_expression(
    adata,
    groupby_key='perturbation',
    reference='non-targeting',
    method='wilcoxon',
)

# Or use scanpy-compatible interface
pl.data.process.rank_genes_groups(
    adata,
    groupby_key='perturbation',
    reference='non-targeting',
    method='wilcoxon',
    n_top=100,
)
```

### Available Methods

- **`wilcoxon`**: Mann-Whitney U test (rank-based, non-parametric)
  - Most robust for non-normal distributions
  - Recommended for most single-cell analyses
  - Fastest with C++ backend

- **`t-test`**: Student's t-test (assumes equal variance)
  - Parametric test for normally distributed data
  - Good for well-behaved data with similar variances

- **`t-test_overestim_var`**: Student's t-test with overestimated variance
  - Scanpy's default method
  - More conservative than standard t-test

- **`welch`**: Welch's t-test (does not assume equal variance)
  - Recommended when groups have different variances
  - More robust than Student's t-test

### Performance

The module uses high-performance statistical kernels with automatic backend selection:

1. **C++ Backend** (fastest, 100-300x speedup)
   - Optimized with SIMD (Highway library)
   - Multi-threaded with OpenMP
   - Sparse matrix optimized

2. **Cython Backend** (fast, 5-10x speedup)
   - Used as fallback if C++ is unavailable
   - Still significantly faster than pure Python

3. **SciPy/NumPy Fallback** (always available)
   - Pure Python implementation
   - Ensures compatibility

**Benchmark Results** (2000 cells × 1000 genes, 70% sparsity):
```
Method          Time       vs scipy
───────────────────────────────────
wilcoxon        0.007s     ~140x
t-test          0.002s     ~500x
welch           0.002s     ~500x
log_fold_change 0.001s     instant
```

### API Reference

#### `differential_expression()`

Full differential expression analysis with detailed results.

```python
results_df = pl.data.process.differential_expression(
    adata,
    groupby_key='perturbation',
    reference='control',
    groups=None,              # Compare all groups
    method='wilcoxon',
    min_samples=2,
    threads=-1,               # Use all cores
    clip_value=20.0,
    fdr_method='bh',
    layer=None,
    use_raw=False,
)
```

**Returns**: `pd.DataFrame` with columns:
- `target`: Target group name
- `feature`: Gene name
- `p_value`: P-value from statistical test
- `statistic`: Test statistic (U for Wilcoxon, t for t-tests)
- `fold_change`: Fold change (target_mean / reference_mean)
- `log2_fold_change`: Log2 fold change
- `fdr`: FDR-corrected p-value

#### `rank_genes_groups()`

Scanpy-compatible interface that stores results in `adata.uns`.

```python
pl.data.process.rank_genes_groups(
    adata,
    groupby_key='perturbation',
    reference='control',
    groups=None,
    method='wilcoxon',
    n_top=100,                # Top genes per group
    min_samples=2,
    threads=-1,
    key_added='rank_genes_groups',
)
```

**Results stored in**:
- `adata.uns['rank_genes_groups']`: Scanpy format (arrays)
- `adata.uns['rank_genes_groups_df']`: Full DataFrame

### Examples

#### Basic Usage

```python
import perturblab as pl

# Load data
adata = pl.datasets.load_dataset("norman_2019")

# Run DE analysis
results = pl.data.process.differential_expression(
    adata,
    groupby_key='perturbation',
    reference='non-targeting',
    method='wilcoxon',
)

# Filter significant genes
sig_genes = results[results['fdr'] < 0.05]
print(f"Found {len(sig_genes)} significant genes")

# Top DE genes per group
for target in results['target'].unique():
    top = results[results['target'] == target].nsmallest(10, 'p_value')
    print(f"\nTop 10 DE genes for {target}:")
    print(top[['feature', 'p_value', 'log2_fold_change']])
```

#### Compare Multiple Methods

```python
methods = ['wilcoxon', 't-test', 'welch']
results_dict = {}

for method in methods:
    results_dict[method] = pl.data.process.differential_expression(
        adata,
        groupby_key='perturbation',
        reference='control',
        method=method,
    )

# Compare results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, method in enumerate(methods):
    df = results_dict[method]
    axes[i].scatter(df['log2_fold_change'], -np.log10(df['p_value']))
    axes[i].set_title(f'{method.capitalize()}')
    axes[i].set_xlabel('Log2 Fold Change')
    axes[i].set_ylabel('-log10(p-value)')
plt.tight_layout()
```

#### Handle NA Values

```python
# Treat NA values as reference group
results = pl.data.process.differential_expression(
    adata,
    groupby_key='perturbation',
    reference=None,  # NA → 'non-targeting'
    method='wilcoxon',
)
```

#### Use Custom Layers

```python
# Use raw counts
results = pl.data.process.differential_expression(
    adata,
    groupby_key='perturbation',
    reference='control',
    use_raw=True,
)

# Use specific layer
results = pl.data.process.differential_expression(
    adata,
    groupby_key='perturbation',
    reference='control',
    layer='log1p_norm',
)
```

### Notes

- **Sparse Data**: The module automatically handles sparse matrices efficiently. Dense matrices are converted to sparse format internally.
- **Thread Safety**: All backends are thread-safe and support multi-threading.
- **Memory Efficiency**: Uses zero-copy operations where possible to minimize memory usage.
- **NA Handling**: NA values in the groupby column can be treated as a reference group.
- **FDR Correction**: Uses Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) methods.

### Troubleshooting

**Q: "ValueError: sparse_matrix must be CSR or CSC format"**

A: This error occurs when passing an unsupported matrix format. The module now automatically converts dense matrices to sparse format, so this should not occur in normal usage.

**Q: "No significant genes found"**

A: Try:
- Reducing `min_samples` threshold
- Using a different statistical method
- Checking if your data is properly normalized
- Increasing the number of cells per group

**Q: "C++ backend not available"**

A: Install C++ kernels:
```bash
./scripts/setup_cpp_deps.sh
./scripts/build_cpp_kernels.sh
```

The module will automatically fall back to Cython or SciPy if C++ is unavailable.

## Future Enhancements

- [ ] Pseudo-bulk differential expression
- [ ] Mixed-effects models
- [ ] Batch effect correction
- [ ] Streaming analysis for large datasets
- [ ] GPU acceleration

## References

- Mann-Whitney U test: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
- Welch's t-test: https://en.wikipedia.org/wiki/Welch%27s_t-test
- Benjamini-Hochberg: https://en.wikipedia.org/wiki/False_discovery_rate

