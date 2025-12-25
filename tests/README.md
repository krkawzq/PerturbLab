# PerturbLab Test Suite

This directory contains the test suite for PerturbLab.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_imports.py

# Run with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_imports.py::TestCoreImports

# Run specific test
pytest tests/test_imports.py::TestCoreImports::test_import_perturblab
```

## Test Structure

### `test_imports.py`

Tests for import system and model registry functionality:

- **TestCoreImports**: Core package imports (perturblab, types, MODELS)
- **TestModelRegistry**: Model registry functionality (lazy loading, hierarchical access)
- **TestDirectModelImports**: Direct model imports (GEARS, scGPT, etc.)
- **TestSubmoduleImports**: Submodule imports (methods, tools, metrics, data, engine)
- **TestUtilityImports**: Utility function imports (check_dependencies)
- **TestModelRegistryEdgeCases**: Edge cases and error handling

### `test_types.py`

Tests for type classes and data structures:

- **TestVocab**: Vocabulary class (name-to-index mapping)
- **TestGeneVocab**: Gene vocabulary functionality
- **TestWeightedGraph**: Weighted graph data structure
- **TestGeneGraph**: Gene graph with vocabulary
- **TestCellDataBasic**: Basic CellData functionality
- **TestCellDataLayers**: Layer management (X, layers, switching)
- **TestCellDataCache**: Caching mechanism for performance
- **TestCellDataVirtualGenes**: Virtual gene alignment
- **TestCellDataConsistency**: Data consistency across operations
- **TestPerturbationDataBasic**: Basic PerturbationData functionality
- **TestPerturbationDataInheritance**: Inheritance from CellData
- **TestPerturbationDataAdvanced**: Advanced perturbation features
- **TestDataConsistencyAcrossTypes**: Consistency across type conversions

## Test Coverage

Current test coverage includes:

### Import & Registry Tests (26 tests)
1. ✅ Core package imports
2. ✅ Model registry with lazy loading
3. ✅ Hierarchical model organization
4. ✅ Dependency checking and error handling
5. ✅ Direct model imports with __getattr__ lazy loading
6. ✅ Submodule imports (methods, tools, metrics, etc.)
7. ✅ Registry methods (keys, values, items, list_keys, __dir__)

### Type Tests (50 tests)
1. ✅ Vocab and GeneVocab (name-to-index mapping)
2. ✅ WeightedGraph and GeneGraph (graph structures)
3. ✅ CellData basic operations (initialization, properties, slicing)
4. ✅ CellData layers (switching, consistency)
5. ✅ CellData cache (enable/disable, performance, consistency)
6. ✅ CellData virtual genes (alignment, reordering, backed mode)
7. ✅ PerturbationData (control cells, perturbation labels)
8. ✅ PerturbationData inheritance (cache, layers, virtual genes)
9. ✅ Data consistency across operations and type conversions

**Total: 76 tests, all passing** ✨

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use descriptive test names starting with `test_`
2. Add docstrings explaining what is being tested
3. Group related tests into test classes
4. Use pytest fixtures for common setup
5. Handle optional dependencies gracefully with `pytest.skip()`

Example:

```python
def test_my_feature(self):
    """Test that my feature works correctly."""
    from perturblab import my_feature
    
    result = my_feature()
    assert result is not None
```

## Continuous Integration

Tests are run automatically on:
- Every commit
- Every pull request
- Before releases

Ensure all tests pass before submitting code.

