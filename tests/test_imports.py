"""Test module for verifying PerturbLab imports and model registration.

This module tests:
1. Core package imports
2. Model registry lazy loading
3. Model class imports through registry
4. Direct model imports
5. Submodule imports (methods, tools, etc.)
"""

import pytest
import sys
from typing import List


class TestCoreImports:
    """Test core PerturbLab package imports."""

    def test_import_perturblab(self):
        """Test that the main package can be imported."""
        import perturblab
        assert perturblab is not None

    def test_import_version(self):
        """Test that version information is accessible."""
        from perturblab import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_import_models(self):
        """Test that MODELS registry can be imported."""
        from perturblab import MODELS
        assert MODELS is not None

    def test_import_types(self):
        """Test that core types can be imported."""
        try:
            from perturblab.types import (
                GeneVocab,
                CellData,
                PerturbationData,
                WeightedGraph,
                GeneGraph,
            )
            assert all([
                GeneVocab is not None,
                CellData is not None,
                PerturbationData is not None,
                WeightedGraph is not None,
                GeneGraph is not None,
            ])
        except ModuleNotFoundError as e:
            if 'anndata' in str(e):
                pytest.skip("anndata not installed")
            else:
                raise


class TestModelRegistry:
    """Test model registry functionality."""

    def test_registry_initialization(self):
        """Test that the model registry initializes correctly."""
        from perturblab import MODELS
        
        # Registry should be accessible
        assert MODELS is not None
        assert hasattr(MODELS, 'list_keys')
        assert hasattr(MODELS, 'get')

    def test_list_all_models(self):
        """Test that all models are registered and can be listed."""
        from perturblab import MODELS
        
        # Get all registered model keys (includes child registries)
        all_keys = MODELS.list_keys()
        
        # Should have models registered
        assert len(all_keys) > 0
        
        # Print for debugging
        print(f"\nRegistered models: {all_keys}")
        
        # Check some expected models (case-sensitive child registry names)
        expected_models = ['gears', 'scGPT', 'UCE', 'scFoundation']
        for model_name in expected_models:
            assert model_name in all_keys, f"Model '{model_name}' not found in registry"

    def test_model_lazy_loading(self):
        """Test that models are loaded lazily (not imported until accessed)."""
        from perturblab import MODELS
        
        # Before accessing, model modules should not be in sys.modules
        # (except if they were imported in previous tests)
        model_module = 'perturblab.models.gears._modeling.model'
        
        # Clear from sys.modules if present
        if model_module in sys.modules:
            del sys.modules[model_module]
        
        # List keys should not trigger full import
        MODELS.list_keys()
        
        # The actual model implementation should still not be imported
        # Note: __init__.py will be imported, but not the _modeling submodule
        assert model_module not in sys.modules

    def test_get_model_class(self):
        """Test that model classes can be retrieved through registry."""
        from perturblab import MODELS
        
        # Get GEARS model through hierarchical path
        GEARS = MODELS.get('gears.GEARSModel')
        assert GEARS is not None
        assert callable(GEARS)
        
        # Check that it's a class
        assert isinstance(GEARS, type)

    def test_hierarchical_access(self):
        """Test hierarchical model access through child registries."""
        from perturblab import MODELS
        
        # Access child registry
        gears_registry = MODELS.child('gears')
        assert gears_registry is not None
        
        # Should have GEARSModel class
        GEARS = MODELS.get('gears.GEARSModel')
        assert GEARS is not None
        assert callable(GEARS)

    def test_model_with_dependencies(self):
        """Test that models with missing dependencies raise appropriate errors."""
        from perturblab import MODELS
        from perturblab.utils import DependencyError
        
        # Try to load a model that might have missing dependencies
        # This test should either succeed (dependencies present) or raise DependencyError
        try:
            scgpt_model = MODELS.get('scGPT.scGPTModel')
            # If we get here, dependencies are satisfied
            assert scgpt_model is not None
        except (DependencyError, ImportError) as e:
            # This is expected if dependencies are missing
            assert 'requires' in str(e).lower() or 'missing' in str(e).lower() or 'no module' in str(e).lower()
            pytest.skip(f"Skipping due to missing dependencies: {e}")


class TestDirectModelImports:
    """Test direct imports of model modules."""

    def test_direct_import_gears(self):
        """Test that GEARS can be imported directly."""
        try:
            from perturblab.models.gears import GEARS
            assert GEARS is not None
            assert callable(GEARS)
        except ImportError as e:
            pytest.fail(f"Failed to import GEARS directly: {e}")

    def test_direct_import_gears_components(self):
        """Test that GEARS components can be imported."""
        from perturblab.models.gears import GEARSInput, GEARSOutput
        assert GEARSInput is not None
        assert GEARSOutput is not None

    def test_direct_import_scgpt(self):
        """Test that scGPT can be imported directly (if dependencies available)."""
        from perturblab.utils import DependencyError
        
        try:
            from perturblab.models.scgpt import scGPT
            assert scGPT is not None
        except (DependencyError, ImportError):
            pytest.skip("scGPT dependencies not available")

    def test_model_requirements_access(self):
        """Test that model requirements can be accessed."""
        from perturblab.models.gears import requirements, dependencies
        
        assert isinstance(requirements, list)
        assert isinstance(dependencies, list)
        
        # GEARS should have torch_geometric as a requirement
        assert any('torch' in req.lower() for req in requirements)


class TestSubmoduleImports:
    """Test imports of submodules (methods, tools, etc.)."""

    def test_import_methods_gears(self):
        """Test that GEARS methods can be imported."""
        try:
            from perturblab.methods import gears
            assert gears is not None
            
            # Test key functions
            assert hasattr(gears, 'build_go_similarity_graph')
            assert hasattr(gears, 'build_coexpression_graph')
            assert hasattr(gears, 'build_graphs')
            assert hasattr(gears, 'build_model')
            assert hasattr(gears, 'build_training_components')
        except ModuleNotFoundError as e:
            if 'natsort' in str(e) or 'anndata' in str(e):
                pytest.skip(f"Missing dependency: {e}")
            else:
                raise

    def test_import_tools(self):
        """Test that tools can be imported."""
        try:
            from perturblab import tools
            assert tools is not None
            
            # Test some key tools
            from perturblab.tools import (
                compute_gene_similarity_from_go,
                project_bipartite_graph,
            )
            assert compute_gene_similarity_from_go is not None
            assert project_bipartite_graph is not None
        except ModuleNotFoundError as e:
            if 'anndata' in str(e):
                pytest.skip(f"Missing dependency: {e}")
            else:
                raise

    def test_import_metrics(self):
        """Test that metrics can be imported."""
        try:
            from perturblab import metrics
            assert metrics is not None
            
            # Test some key metrics
            from perturblab.metrics import r2, pearson, mse
            assert all([r2 is not None, pearson is not None, mse is not None])
        except ModuleNotFoundError as e:
            if 'natsort' in str(e):
                pytest.skip(f"Missing dependency: {e}")
            else:
                raise

    def test_import_data(self):
        """Test that data module can be imported."""
        from perturblab import data
        assert data is not None
        
        # Test data loading function
        from perturblab.data import load_dataset
        assert load_dataset is not None
        assert callable(load_dataset)

    def test_import_engine(self):
        """Test that engine components can be imported."""
        from perturblab.engine import Trainer
        assert Trainer is not None
        
        try:
            from perturblab.engine import DistributedTrainer
            assert DistributedTrainer is not None
        except ImportError:
            # DistributedTrainer might have additional dependencies
            pass


class TestUtilityImports:
    """Test utility function imports."""

    def test_import_check_dependencies(self):
        """Test that dependency checking utility can be imported."""
        from perturblab.utils import check_dependencies
        assert check_dependencies is not None
        assert callable(check_dependencies)

    def test_check_dependencies_function(self):
        """Test that check_dependencies function works correctly."""
        from perturblab.utils import check_dependencies, DependencyError
        
        # Should not raise error for standard library
        check_dependencies(requirements=['sys', 'os'])
        
        # Should raise error for non-existent package
        with pytest.raises(DependencyError):
            check_dependencies(requirements=['this_package_definitely_does_not_exist_12345'])
        
        # Should not raise error for optional dependencies (just log)
        check_dependencies(dependencies=['this_optional_package_does_not_exist'])


class TestModelRegistryEdgeCases:
    """Test edge cases and error handling in model registry."""

    def test_get_nonexistent_model(self):
        """Test that getting a non-existent model returns None or raises KeyError."""
        from perturblab import MODELS
        
        # Non-existent model should return None (default behavior)
        result = MODELS.get('this_model_does_not_exist', default=None)
        assert result is None

    def test_registry_keys_method(self):
        """Test that registry keys() method works."""
        from perturblab import MODELS
        
        keys = list(MODELS.keys())
        # keys() returns models + child registries
        assert len(keys) > 0
        # Child registries should be present
        child_names = list(MODELS._child_registries.keys())
        for child in child_names:
            assert child in keys

    def test_registry_values_method(self):
        """Test that registry values() method works."""
        from perturblab import MODELS
        
        values = list(MODELS.values())
        # values() includes child registries
        assert len(values) > 0
        
        # All values should be classes or registries
        for value in values:
            assert value is not None
            # Should be either a ModelRegistry or a class
            from perturblab.core import ModelRegistry
            assert isinstance(value, (type, ModelRegistry))

    def test_registry_items_method(self):
        """Test that registry items() method works."""
        from perturblab import MODELS
        
        items = list(MODELS.items())
        # items() includes child registries
        assert len(items) > 0
        
        # Each item should be a (key, value) tuple
        for key, value in items:
            assert isinstance(key, str)
            assert value is not None
            # Value should be either a class or a ModelRegistry
            from perturblab.core import ModelRegistry
            assert isinstance(value, (type, ModelRegistry))

    def test_registry_dir_method(self):
        """Test that registry __dir__ method works for autocomplete."""
        from perturblab import MODELS
        
        dir_items = dir(MODELS)
        
        # Should contain child registry names
        assert 'gears' in dir_items
        assert 'scGPT' in dir_items
        
        # Should also contain registry methods
        assert 'get' in dir_items
        assert 'list_keys' in dir_items


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

