# Statistics Kernels

High-performance statistical computing kernels for single-cell analysis, with C++ and Cython implementations.

## 🎯 Overview

This module provides optimized statistical algorithms for differential expression analysis:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Mann-Whitney U** | Wilcoxon rank-sum test | Non-parametric, robust to outliers |
| **Welch's t-test** | Unequal variance t-test | Parametric, recommended for log-transformed data |
| **Student's t-test** | Equal variance t-test | Parametric, faster but less robust |
| **Log Fold Change** | Effect size measure | Quick screening, no p-values |

**Performance:** 100-300x faster than scipy on sparse matrices (C++ backend with multi-threading).

## 🚀 Quick Start

```python
import numpy as np
import scipy.sparse as sp
from perturblab.kernels.statistics import mannwhitneyu, ttest, group_mean

# Create sparse data (cells × genes)
X = sp.random(500, 200, density=0.1, format='csc')
group_id = np.array([0]*200 + [1]*150 + [2]*150, dtype=np.int32)

# Mann-Whitney U test (non-parametric)
U1, U2, P = mannwhitneyu(X, group_id, n_targets=2, threads=8)

# Welch's t-test (parametric, recommended)
t_stat, p_val, mean_diff, log2_fc = ttest(
    X, group_id, n_targets=2, method='welch', threads=8
)

# Group means
means = group_mean(X, group_id, n_groups=3)
```

## 📦 Installation

### Basic Installation (scipy fallback)

```bash
pip install -e .
```

Works out-of-the-box but uses slower scipy fallback.

### Compile C++ Backend (Recommended)

For **100-300x speedup**, compile the C++ library:

```bash
cd perturblab/kernels/statistics

# Install dependencies (Highway SIMD + OpenMP)
./setup_deps.sh

# Compile C++ library
./build.sh
```

This creates `libmwu_kernel.so` with all optimized kernels.

### Compile Cython Backend (Alternative)

For **5-10x speedup** without C++ complexity:

```bash
# From project root
python setup.py build_ext --inplace
```

## 🏗️ Architecture

### Backend Selection

The package automatically selects the best available backend:

1. **C++ Backend** (highest performance)
   - 100-300x faster than scipy
   - SIMD optimizations (Highway library)
   - Multi-threading (OpenMP)
   - File: `libmwu_kernel.so`

2. **Cython Backend** (good performance)
   - 5-10x faster than scipy
   - Multi-threading (OpenMP)
   - File: `_mannwhitneyu_kernel.*.so`

3. **SciPy/NumPy Fallback** (always available)
   - Pure Python, slower
   - No compilation needed

### File Structure

```
perturblab/kernels/statistics/
├── __init__.py                    # Public API
├── _mannwhitneyu.py               # Mann-Whitney U wrapper
├── _mannwhitneyu_capi.py          # C API interface (ctypes)
├── _mannwhitneyu_kernel.pyx       # Cython implementation
├── _ttest.py                      # T-test wrapper
├── _ttest_capi.py                 # T-test C API interface
├── src/                           # C++ source code
│   ├── CMakeLists.txt             # Build configuration
│   ├── mannwhitneyu.cpp           # ⚠️  Manually optimized (do not auto-refactor)
│   ├── ttest.cpp                  # 🤖 AI-generated (can be optimized)
│   ├── capi_wrapper.cpp           # 🤖 AI-generated C API
│   └── include/                   # Header files
│       ├── mannwhitneyu.hpp
│       ├── ttest.hpp
│       ├── sparse.hpp
│       └── ...
├── lib/highway/                   # Highway SIMD library (git submodule)
├── build.sh                       # C++ build script
├── setup_deps.sh                  # Dependency installation
├── test_all.py                    # Comprehensive tests
└── README.md                      # This file
```

## 📚 API Reference

### `mannwhitneyu`

```python
mannwhitneyu(
    sparse_matrix: csr_matrix | csc_matrix,
    group_id: ndarray[int32],
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = False,
    zero_handling: Literal["none", "min", "max", "mix"] = "min",
    threads: int = -1,
) -> tuple[ndarray, ndarray, ndarray]
```

**Returns:** `(U1, U2, P)` where each has shape `(n_targets, n_features)`
- `U1`: U statistic for target group
- `U2`: U statistic for reference group (= n_ref × n_tar - U1)
- `P`: Two-sided p-values

### `ttest`

```python
ttest(
    sparse_matrix: csr_matrix | csc_matrix,
    group_id: ndarray[int32],
    n_targets: int,
    method: Literal["student", "welch"] = "welch",
    threads: int = -1,
) -> tuple[ndarray, ndarray, ndarray, ndarray]
```

**Returns:** `(t_statistic, p_value, mean_diff, log2_fc)` each with shape `(n_targets, n_features)`

### `log_fold_change`

```python
log_fold_change(
    sparse_matrix: csr_matrix | csc_matrix,
    group_id: ndarray[int32],
    n_targets: int,
    pseudocount: float = 1e-9,
    threads: int = -1,
) -> tuple[ndarray, ndarray]
```

**Returns:** `(mean_diff, log2_fc)` each with shape `(n_targets, n_features)`

### `group_mean`

```python
group_mean(
    sparse_matrix: csr_matrix | csc_matrix,
    group_id: ndarray[int32],
    n_groups: int,
    include_zeros: bool = True,
    threads: int = -1,
) -> ndarray
```

**Returns:** Group means with shape `(n_groups, n_features)`

## 🧪 Testing

```bash
cd perturblab/kernels/statistics
python test_all.py
```

Tests include:
- Numerical validation against scipy
- Performance benchmarks
- Backend comparison
- Edge case handling

## 📊 Performance

Benchmark on AMD EPYC (64 cores), sparse CSC (10k cells × 2k genes, 10% density):

| Algorithm | scipy | Cython (4T) | C++ (4T) | C++ (64T) | Speedup |
|-----------|-------|-------------|----------|-----------|---------|
| Mann-Whitney U | 45.2s | 8.3s | 0.42s | 0.15s | **301x** |
| Welch's t-test | 12.5s | 3.1s | 0.18s | 0.08s | **156x** |
| Group mean | 2.3s | 0.6s | 0.05s | 0.02s | **115x** |

## 🔧 Building for Distribution

### Pre-compiled Wheels (Recommended)

For distribution, build platform-specific wheels with pre-compiled C++ libraries:

```bash
# Install build dependencies
pip install build wheel

# Build wheel
python -m build

# The wheel will include:
# - Python code
# - Compiled Cython extensions (if available)
# - Pre-compiled C++ library (if built)
```

### Multi-platform Build

Use GitHub Actions or cibuildwheel for automated multi-platform builds:

```yaml
# .github/workflows/build.yml
- name: Build wheels
  uses: pypa/cibuildwheel@v2
  env:
    CIBW_BUILD: cp310-* cp311-* cp312-*
    CIBW_BEFORE_BUILD: |
      cd perturblab/kernels/statistics
      ./setup_deps.sh
      ./build.sh
```

## 📝 Development Notes

### Code Optimization Levels

1. **mannwhitneyu.cpp** - ⚠️  **MANUALLY OPTIMIZED**
   - Extensively profiled and optimized
   - Do NOT use AI tools to refactor
   - Any changes should be benchmarked

2. **ttest.cpp** - 🤖 **AI-GENERATED**
   - Functional but has optimization potential
   - Can be improved with profiling
   - Consider SIMD vectorization

3. **capi_wrapper.cpp** - 🤖 **AI-GENERATED**
   - Simple C API wrapper
   - Room for optimization in memory handling

### Adding New Algorithms

To add a new statistical test:

1. Implement in `src/new_algorithm.cpp`
2. Add header to `src/include/new_algorithm.hpp`
3. Add C API wrapper in `src/capi_wrapper.cpp`
4. Create Python wrapper `_new_algorithm.py`
5. Update `src/CMakeLists.txt` to include new source
6. Export in `__init__.py`

## 🙏 Credits

- **Mann-Whitney U implementation**: Based on [hpdex](https://github.com/AI4Cell/hpdex) by Wang Zhongqi
- **T-test implementation**: AI-generated by Cursor (2025-12-24)
- **Highway SIMD library**: Google Highway team

## 📄 License

MIT License - see project root for details.

