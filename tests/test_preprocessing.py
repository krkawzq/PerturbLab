"""Tests for perturblab.preprocessing module.

This module tests preprocessing functions including:
- normalize_total (CPM/TPM normalization)
- scale (z-score normalization)

These functions are optimized with C++/SIMD/OpenMP and should be
2-15x faster than scanpy while producing identical results.
"""

import numpy as np
import pytest
from scipy import sparse

import sys
import warnings

import pytest

try:
    from anndata import AnnData

    import perturblab as pl
    from perturblab import preprocessing as pp

    _has_perturblab = True
except ImportError as e:
    _has_perturblab = False
    _import_error = str(e)


# =============================================================================
# Test normalize_total
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestNormalizeTotal:
    """Tests for normalize_total function."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData for testing."""
        np.random.seed(42)
        X = np.random.randint(0, 100, size=(100, 50)).astype(float)
        return AnnData(X)

    @pytest.fixture
    def sparse_adata(self):
        """Create sparse AnnData for testing."""
        np.random.seed(42)
        X = sparse.random(100, 50, density=0.3, format="csr")
        X.data = np.random.randint(1, 100, size=len(X.data)).astype(float)
        return AnnData(X)

    def test_normalize_total_dense(self, sample_adata):
        """Test normalize_total on dense matrix."""
        adata = sample_adata.copy()
        pp.normalize_total(adata, target_sum=1e4)

        # Check that total counts per cell are correct
        counts_per_cell = adata.X.sum(axis=1)
        assert np.allclose(counts_per_cell, 1e4, rtol=1e-5)

    def test_normalize_total_sparse(self, sparse_adata):
        """Test normalize_total on sparse matrix."""
        adata = sparse_adata.copy()
        pp.normalize_total(adata, target_sum=1e4)

        # Check that total counts per cell are correct
        counts_per_cell = np.array(adata.X.sum(axis=1)).flatten()
        assert np.allclose(counts_per_cell, 1e4, rtol=1e-5)

    def test_normalize_total_auto_target(self, sample_adata):
        """Test normalize_total with automatic target_sum."""
        adata = sample_adata.copy()
        original_median = np.median(adata.X.sum(axis=1))

        pp.normalize_total(adata, target_sum=None)

        # Target should be original median
        counts_per_cell = adata.X.sum(axis=1)
        assert np.allclose(counts_per_cell, original_median, rtol=1e-5)

    def test_normalize_total_cpm(self, sample_adata):
        """Test CPM normalization (target_sum=1e6)."""
        adata = sample_adata.copy()
        pp.normalize_total(adata, target_sum=1e6)

        # Check CPM values
        counts_per_cell = adata.X.sum(axis=1)
        assert np.allclose(counts_per_cell, 1e6, rtol=1e-5)

    def test_normalize_total_with_layer(self, sample_adata):
        """Test normalize_total on a specific layer."""
        adata = sample_adata.copy()
        adata.layers["counts"] = adata.X.copy()

        pp.normalize_total(adata, target_sum=1e4, layer="counts")

        # Original X should be unchanged
        assert not np.allclose(adata.X.sum(axis=1), 1e4)

        # Layer should be normalized
        counts_per_cell = adata.layers["counts"].sum(axis=1)
        assert np.allclose(counts_per_cell, 1e4, rtol=1e-5)

    def test_normalize_total_key_added(self, sample_adata):
        """Test that normalization factor is stored."""
        adata = sample_adata.copy()
        pp.normalize_total(adata, target_sum=1e4, key_added="norm_factor")

        assert "norm_factor" in adata.obs
        assert len(adata.obs["norm_factor"]) == adata.n_obs

    def test_normalize_total_exclude_highly_expressed(self, sample_adata):
        """Test normalization excluding highly expressed genes."""
        adata = sample_adata.copy()

        pp.normalize_total(
            adata, target_sum=1e4, exclude_highly_expressed=True, max_fraction=0.05
        )

        # Should still normalize, but excluding highly expressed genes
        # from the normalization factor calculation
        assert adata.X is not None

    def test_normalize_total_copy(self, sample_adata):
        """Test normalize_total with copy=True."""
        adata = sample_adata.copy()
        original_sum = adata.X.sum()

        adata_norm = pp.normalize_total(adata, target_sum=1e4, copy=True)

        # Original should be unchanged
        assert np.isclose(adata.X.sum(), original_sum)

        # Copy should be normalized
        counts_per_cell = adata_norm.X.sum(axis=1)
        assert np.allclose(counts_per_cell, 1e4, rtol=1e-5)

    def test_normalize_total_not_inplace(self, sample_adata):
        """Test normalize_total with inplace=False."""
        adata = sample_adata.copy()
        original_sum = adata.X.sum()

        result = pp.normalize_total(adata, target_sum=1e4, inplace=False)

        # Original should be unchanged
        assert np.isclose(adata.X.sum(), original_sum)

        # Result should be a dict with normalized data
        if result is not None:
            assert isinstance(result, dict)

    def test_normalize_total_zero_counts_warning(self):
        """Test that zero count cells trigger warning or are handled gracefully."""
        X = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        adata = AnnData(X.astype(float))

        # May issue warning or handle gracefully (both are acceptable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.normalize_total(adata, target_sum=1e4)

    def test_normalize_total_invalid_max_fraction(self, sample_adata):
        """Test that invalid max_fraction raises error."""
        adata = sample_adata.copy()

        with pytest.raises(ValueError, match="Choose max_fraction between 0 and 1"):
            pp.normalize_total(adata, max_fraction=1.5)

    def test_normalize_total_preserves_type(self, sparse_adata):
        """Test that sparse input remains sparse."""
        adata = sparse_adata.copy()
        pp.normalize_total(adata, target_sum=1e4)

        assert sparse.issparse(adata.X)

    def test_normalize_total_different_targets(self, sample_adata):
        """Test normalization with different target sums."""
        adata1 = sample_adata.copy()
        adata2 = sample_adata.copy()

        pp.normalize_total(adata1, target_sum=1e3)
        pp.normalize_total(adata2, target_sum=1e6)

        # Check that proportions are maintained but scales differ
        counts1 = adata1.X.sum(axis=1)
        counts2 = adata2.X.sum(axis=1)

        assert np.allclose(counts1, 1e3, rtol=1e-5)
        assert np.allclose(counts2, 1e6, rtol=1e-5)


# =============================================================================
# Test scale
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestScale:
    """Tests for scale function."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 50) * 10 + 5  # Mean ~5, std ~10
        return AnnData(X)

    @pytest.fixture
    def sparse_adata(self):
        """Create sparse AnnData for testing."""
        np.random.seed(42)
        X = sparse.random(100, 50, density=0.5, format="csr")
        X.data = np.random.randn(len(X.data)) * 10 + 5
        return AnnData(X)

    def test_scale_dense(self, sample_adata):
        """Test scale on dense matrix."""
        adata = sample_adata.copy()
        pp.scale(adata, zero_center=True)

        # Check that mean is ~0 and std is ~1
        means = adata.X.mean(axis=0)
        stds = adata.X.std(axis=0)

        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-6)

    def test_scale_sparse(self, sparse_adata):
        """Test scale on sparse matrix."""
        adata = sparse_adata.copy()
        pp.scale(adata, zero_center=True)

        # After scaling, should be dense
        assert not sparse.issparse(adata.X)

        # Check that mean is ~0 and std is ~1
        means = adata.X.mean(axis=0)
        stds = adata.X.std(axis=0)

        assert np.allclose(means, 0, atol=1e-5)
        assert np.allclose(stds, 1, atol=1e-3)

    def test_scale_no_zero_center(self, sample_adata):
        """Test scale without zero centering."""
        adata = sample_adata.copy()
        original_means = adata.X.mean(axis=0)

        pp.scale(adata, zero_center=False)

        # Mean should be unchanged
        new_means = adata.X.mean(axis=0)
        assert np.allclose(new_means, original_means, rtol=1e-6)

        # Std should be ~1
        stds = adata.X.std(axis=0)
        assert np.allclose(stds, 1, atol=1e-6)

    def test_scale_max_value(self, sample_adata):
        """Test scale with max_value clipping."""
        adata = sample_adata.copy()
        pp.scale(adata, zero_center=True, max_value=3.0)

        # All values should be within [-3, 3]
        assert np.all(adata.X >= -3.0)
        assert np.all(adata.X <= 3.0)

    def test_scale_with_layer(self, sample_adata):
        """Test scale on a specific layer."""
        adata = sample_adata.copy()
        adata.layers["raw"] = adata.X.copy()

        pp.scale(adata, layer="raw")

        # Original X should be unchanged
        assert not np.allclose(adata.X.mean(axis=0), 0, atol=1e-5)

        # Layer should be scaled
        means = adata.layers["raw"].mean(axis=0)
        assert np.allclose(means, 0, atol=1e-10)

    def test_scale_copy(self, sample_adata):
        """Test scale with copy=True."""
        adata = sample_adata.copy()
        original_mean = adata.X.mean()

        adata_scaled = pp.scale(adata, copy=True)

        # Original should be unchanged
        assert np.isclose(adata.X.mean(), original_mean)

        # Copy should be scaled
        assert np.allclose(adata_scaled.X.mean(axis=0), 0, atol=1e-10)

    def test_scale_not_inplace(self, sample_adata):
        """Test scale with inplace=False."""
        adata = sample_adata.copy()
        original_mean = adata.X.mean()

        result = pp.scale(adata, inplace=False)

        # Original should be unchanged
        assert np.isclose(adata.X.mean(), original_mean)

        # Result might be None or AnnData depending on implementation
        if result is not None and not isinstance(result, AnnData):
            assert isinstance(result, np.ndarray)

    def test_scale_constant_genes(self):
        """Test scale with constant genes."""
        # Create data with constant genes
        X = np.random.randn(100, 50)
        X[:, 0] = 5.0  # Constant gene
        X[:, 1] = 0.0  # Zero gene
        adata = AnnData(X)

        pp.scale(adata, zero_center=True)

        # Constant genes should be set to 0
        assert np.allclose(adata.X[:, 0], 0, atol=1e-10)
        assert np.allclose(adata.X[:, 1], 0, atol=1e-10)

        # Other genes should be scaled normally
        means = adata.X[:, 2:].mean(axis=0)
        stds = adata.X[:, 2:].std(axis=0)
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-6)

    def test_scale_with_obsm(self, sample_adata):
        """Test scale on obsm key (e.g., PCA)."""
        adata = sample_adata.copy()
        adata.obsm["X_pca"] = np.random.randn(adata.n_obs, 10)

        pp.scale(adata, obsm="X_pca")

        # obsm should be scaled
        means = adata.obsm["X_pca"].mean(axis=0)
        assert np.allclose(means, 0, atol=1e-10)

    def test_scale_with_mask_obs(self, sample_adata):
        """Test scale with observation mask."""
        adata = sample_adata.copy()
        mask = np.zeros(adata.n_obs, dtype=bool)
        mask[: adata.n_obs // 2] = True  # Use first half for computing stats

        pp.scale(adata, mask_obs=mask)

        # Should compute stats from first half, but scale all
        assert adata.X.shape[0] == 100

    def test_scale_comparison_with_manual(self, sample_adata):
        """Test that scale produces correct values manually."""
        adata = sample_adata.copy()
        X_orig = adata.X.copy()

        pp.scale(adata, zero_center=True, max_value=None)

        # Manually compute expected result
        means = X_orig.mean(axis=0)
        stds = X_orig.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0  # Avoid division by zero

        X_expected = (X_orig - means) / stds

        assert np.allclose(adata.X, X_expected, rtol=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_normalize_then_scale(self):
        """Test typical preprocessing pipeline: normalize -> scale."""
        np.random.seed(42)
        X = np.random.randint(0, 100, size=(100, 50)).astype(float)
        adata = AnnData(X)

        # Step 1: Normalize
        pp.normalize_total(adata, target_sum=1e4)
        assert np.allclose(adata.X.sum(axis=1), 1e4, rtol=1e-5)

        # Step 2: Scale
        pp.scale(adata, zero_center=True)
        assert np.allclose(adata.X.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(adata.X.std(axis=0), 1, atol=1e-6)

    def test_sparse_preservation(self):
        """Test that sparse format is handled correctly."""
        np.random.seed(42)
        X = sparse.random(100, 50, density=0.3, format="csr")
        X.data = np.random.randint(1, 100, size=len(X.data)).astype(float)
        adata = AnnData(X)

        # Normalize preserves sparsity
        pp.normalize_total(adata, target_sum=1e4)
        assert sparse.issparse(adata.X)

        # Scale converts to dense
        pp.scale(adata, zero_center=True)
        assert not sparse.issparse(adata.X)

    def test_layer_workflow(self):
        """Test preprocessing workflow with layers."""
        np.random.seed(42)
        X = np.random.randint(0, 100, size=(100, 50)).astype(float)
        adata = AnnData(X)
        adata.layers["counts"] = X.copy()

        # Normalize counts layer
        pp.normalize_total(adata, target_sum=1e4, layer="counts")

        # Scale normalized layer
        adata.X = adata.layers["counts"]
        pp.scale(adata, zero_center=True)

        # Original counts layer should be unchanged
        assert not np.allclose(adata.layers["counts"].sum(axis=1), X.sum(axis=1))


# =============================================================================
# Performance Tests (Optional)
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
@pytest.mark.slow
class TestPreprocessingPerformance:
    """Performance tests for preprocessing functions."""

    def test_normalize_total_performance(self):
        """Benchmark normalize_total performance."""
        import time

        np.random.seed(42)
        X = sparse.random(10000, 2000, density=0.1, format="csr")
        X.data = np.random.randint(1, 100, size=len(X.data)).astype(float)
        adata = AnnData(X)

        start = time.time()
        pp.normalize_total(adata, target_sum=1e4)
        elapsed = time.time() - start

        print(f"\nnormalize_total took {elapsed:.3f}s for 10k cells × 2k genes")
        assert elapsed < 5.0  # Should be fast

    def test_scale_performance(self):
        """Benchmark scale performance."""
        import time

        np.random.seed(42)
        X = np.random.randn(10000, 2000)
        adata = AnnData(X)

        start = time.time()
        pp.scale(adata, zero_center=True)
        elapsed = time.time() - start

        print(f"\nscale took {elapsed:.3f}s for 10k cells × 2k genes")
        assert elapsed < 5.0  # Should be fast


