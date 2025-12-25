# PerturbLab Testing Guide

This guide describes how to run tests for PerturbLab.

## Quick Start

```bash
# Run all tests
make test

# Quick sanity check
make test-quick

# Test model registry
make test-models
```

## Available Test Commands

### 1. `make test`
Run all tests with standard verbosity.

```bash
make test
```

**Use case**: Regular test runs during development

**Output**: Summary with pass/fail for each test

---

### 2. `make test-fast`
Run tests with fail-fast mode (stops at first failure).

```bash
make test-fast
```

**Use case**: Quick validation when you want to fix failures one at a time

**Equivalent to**: `pytest tests/ -x -v`

---

### 3. `make test-imports`
Run only import and model registry tests.

```bash
make test-imports
```

**Use case**: Verify that all models and submodules import correctly

**Tests**: 
- Core package imports
- Model registry functionality
- Direct model imports
- Submodule imports
- Dependency checking

**Time**: ~2-3 seconds

---

### 4. `make test-verbose`
Run tests with maximum verbosity and show print statements.

```bash
make test-verbose
```

**Use case**: Debugging test failures, seeing detailed output

**Equivalent to**: `pytest tests/ -vv -s`

---

### 5. `make test-coverage`
Run tests and generate coverage report.

```bash
make test-coverage
```

**Use case**: Check which parts of code are tested

**Output**:
- Terminal coverage summary
- HTML report in `htmlcov/index.html`

**Dependencies**: Installs `pytest-cov` if needed

---

### 6. `make test-quick`
Quick sanity test that verifies PerturbLab can be imported.

```bash
make test-quick
```

**Use case**: Fast check after installation or code changes

**Output**: 
```
✓ PerturbLab v1.0.0 imported successfully
```

**Time**: ~1 second

---

### 7. `make test-models`
Test model registry and check registered models.

```bash
make test-models
```

**Use case**: Verify all models are properly registered

**Output**:
```
✓ Found 6 model registries: ['scFoundation', 'CellFM', 'scELMo', 'gears', 'scGPT', 'UCE']
```

**Time**: ~1 second

---

## Running Tests Directly with pytest

You can also run pytest directly for more control:

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_imports.py

# Run specific test class
uv run pytest tests/test_imports.py::TestCoreImports

# Run specific test
uv run pytest tests/test_imports.py::TestCoreImports::test_import_perturblab

# Run with specific marker
uv run pytest tests/ -m "not slow"

# Run and show local variables on failure
uv run pytest tests/ -l

# Run and enter debugger on failure
uv run pytest tests/ --pdb
```

## Test Structure

```
tests/
├── __init__.py
├── README.md
└── test_imports.py
    ├── TestCoreImports              # Core package imports
    ├── TestModelRegistry            # Model registry functionality
    ├── TestDirectModelImports       # Direct model imports
    ├── TestSubmoduleImports         # Submodule imports
    ├── TestUtilityImports           # Utility function imports
    └── TestModelRegistryEdgeCases   # Edge cases and error handling
```

## Test Coverage

Current test coverage includes:

- ✅ Core package imports (`perturblab`, types, MODELS)
- ✅ Model registry with lazy loading
- ✅ Hierarchical model organization
- ✅ Dependency checking and error handling
- ✅ Direct model imports with `__getattr__` lazy loading
- ✅ Submodule imports (methods, tools, metrics, data, engine)
- ✅ Registry methods (keys, values, items, list_keys, `__dir__`)

## Continuous Integration

Tests run automatically on:
- Every commit
- Every pull request
- Before releases

All tests must pass before merging code.

## Troubleshooting

### Tests fail with import errors

Check that all dependencies are installed:

```bash
uv pip install -e .
```

### Tests are slow

Use `test-fast` to stop at first failure:

```bash
make test-fast
```

### Need to debug a test

Use verbose mode with pytest directly:

```bash
uv run pytest tests/test_imports.py::TestCoreImports::test_import_perturblab -vv -s --pdb
```

### Coverage report not generating

Install pytest-cov:

```bash
uv pip install pytest-cov
```

## Writing New Tests

When adding new tests:

1. Use descriptive names: `test_feature_does_something`
2. Add docstrings explaining what is tested
3. Group related tests in test classes
4. Handle optional dependencies with `pytest.skip()`
5. Use fixtures for common setup
6. Keep tests fast and independent

Example:

```python
class TestMyFeature:
    """Tests for my new feature."""
    
    def test_feature_works(self):
        """Test that the feature works correctly."""
        from perturblab import my_feature
        
        result = my_feature()
        assert result is not None
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)

