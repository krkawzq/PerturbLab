## PerturbLab Test Suite

This directory contains the complete test suite for the PerturbLab framework.

### Test Files

#### 1. `test_imports.py`
Tests core imports and model registry system.

**Test Coverage:**
- Core module imports (`perturblab`, `__version__`, `MODELS`)
- Model registry functionality (`list_keys()`, `keys()`, `values()`, `items()`, `__dir__()`)
- Lazy loading mechanism
- Model registry hierarchical structure

**Run:**
```bash
make test-imports
```

#### 2. `test_types.py`
Tests correctness and consistency of data structures.

**Test Coverage:**
- `Vocab` and `GeneVocab`: vocabulary basics, serialization
- `WeightedGraph` and `GeneGraph`: graph structure, sparse matrix, PyG conversion
- `CellData`: basic operations, slicing, layer switching, virtual genes, caching
- `PerturbationData`: perturbation-specific functionality inherited from `CellData`

**Run:**
```bash
make test-types
```

#### 3. `test_download.py`
Tests data download and resource registry system.

**Test Coverage:**
- Resource registry (`list_datasets()`, `list_resources()`)
- URL accessibility checks (using HEAD requests)
- Partial download verification (streaming, avoids full file downloads)
- Network resilience (handling timeouts, HTTP errors, proxy issues)

**Features:**
- Uses `@pytest.mark.network` to mark tests requiring network access
- Network errors trigger warnings instead of failures
- Partial downloads (only first 50KB) to speed up tests

**Run:**
```bash
make test-download              # includes network tests
make test-download-no-network   # skips network tests
```

#### 4. `test_metrics.py`
Tests all evaluation metric functions.

**Test Coverage:**
- **Expression metrics**: R², Pearson, MSE, RMSE, MAE, Cosine, L2
- **Distribution metrics**: MMD, Wasserstein distance
- **Direction consistency**: Delta direction accuracy
- **DEG overlap**: Top-N, p-value, FDR thresholds
- **Spatial metrics**: Moran's I, Geary's C
- **Classification metrics**: Confusion matrix
- **Comprehensive evaluation**: `evaluate_prediction()` function

**Run:**
```bash
make test-metrics
```

**Status:** ✅ 48/48 tests passing

#### 5. `test_preprocessing.py`
Tests preprocessing functions.

**Test Coverage:**
- **normalize_total**: CPM/TPM normalization, sparse/dense matrices, auto target, layer support
- **scale**: Z-score normalization, zero-centering, max value clipping, constant gene handling
- **Integration tests**: normalization + scaling pipeline
- **Performance tests**: Large-scale data benchmarks (marked with `@pytest.mark.slow`)

**Run:**
```bash
make test-preprocessing
make test-pp  # alias
```

**Status:** ⚠️ C++ extension has Segmentation fault, needs fixing

### Test Commands

#### Basic Commands
```bash
make test                 # run all tests
make test-fast            # fast-fail mode
make test-verbose         # verbose output
make test-coverage        # generate coverage report
make test-quick           # quick import sanity check
```

#### Module Tests
```bash
make test-imports         # import tests
make test-types           # type tests
make test-metrics         # metrics tests
make test-preprocessing   # preprocessing tests
make test-download        # download tests (with network)
make test-download-no-network  # download tests (skip network)
```

#### Other Commands
```bash
make test-models          # test model registry
```

### Test Markers

#### `@pytest.mark.network`
Marks tests that require network access.

**Skip network tests:**
```bash
pytest tests/ -m "not network"
```

#### `@pytest.mark.slow`
Marks slow-running tests (e.g., performance benchmarks).

**Skip slow tests:**
```bash
pytest tests/ -m "not slow"
```

### Test Statistics

| Test File | Test Count | Status | Notes |
|-----------|-----------|--------|-------|
| `test_imports.py` | 12 | ✅ Passing | Core imports and registry |
| `test_types.py` | 40+ | ✅ Passing | Data structure integrity |
| `test_download.py` | 34 | ✅ 32 pass, 2 skip | Network limitations |
| `test_metrics.py` | 48 | ✅ Passing | All metric functions |
| `test_preprocessing.py` | 29 | ⚠️ C++ bug | Segmentation fault |

**Total:** 163+ tests

### Network Test Notes

Network tests in `test_download.py` are designed to be resilient:

1. **URL accessibility checks**: Uses HEAD requests instead of full downloads
2. **Partial download verification**: Only downloads first 50KB to verify download mechanism
3. **Error handling**: Network errors (timeouts, HTTP 4xx/5xx) trigger warnings instead of failures
4. **Skip strategy**: Temporary network issues skip tests without failing CI

**Known Issues:**
- Zenodo API may trigger rate limiting (HTTP 429)
- Harvard Dataverse doesn't support HEAD requests (HTTP 403), but GET requests work

### CI/CD Integration

**Recommended CI configuration:**
```yaml
# Quick tests (no network)
- make test-imports
- make test-types
- make test-metrics
- make test-download-no-network

# Full tests (with network, allows partial skips)
- make test
```

### Development Guidelines

1. **When writing new tests:**
   - Use Google-style English comments
   - Add `@pytest.mark.network` for tests requiring network
   - Add `@pytest.mark.slow` for slow tests
   - Ensure tests have descriptive names and docstrings

2. **When running tests:**
   - Use `make test-fast` during development for quick validation
   - Run `make test` before committing to ensure all tests pass
   - Use `make test-coverage` to check code coverage

3. **When debugging test failures:**
   - Use `make test-verbose` for detailed output
   - Use `pytest tests/test_xxx.py::TestClass::test_method -vv` to run single tests
   - Check `.pytest_cache` to understand cache state

### TODO

- [ ] Fix C++ Segmentation fault in `test_preprocessing.py`
- [ ] Add more edge case tests
- [ ] Add integration tests (end-to-end training pipeline)
- [ ] Add performance regression tests
- [ ] Increase code coverage to 90%+

### Contributing

Contributions of new tests are welcome! Please ensure:

1. Tests have clear docstrings
2. Tests cover both normal and edge cases
3. Tests run reasonably fast (< 5 seconds/test)
4. Use appropriate pytest markers
5. Follow existing test code style
