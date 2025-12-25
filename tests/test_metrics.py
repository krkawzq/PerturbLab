"""Tests for perturblab.metrics module.

This module tests all metric functions including:
- Expression metrics (R², Pearson, MSE, etc.)
- Distribution metrics (MMD, Wasserstein)
- Direction consistency
- DEG overlap
- Spatial autocorrelation
- Classification metrics
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

try:
    import perturblab as pl
    from perturblab import metrics

    _has_perturblab = True
except ImportError:
    _has_perturblab = False


# =============================================================================
# Test Expression Metrics
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestExpressionMetrics:
    """Tests for expression prediction metrics."""

    def test_r2_perfect_prediction(self):
        """Test R² for perfect prediction."""
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = metrics.r2(y_pred, y_true)
        assert score == 1.0

    def test_r2_poor_prediction(self):
        """Test R² for poor prediction."""
        y_pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = metrics.r2(y_pred, y_true)
        assert score < 0.5

    def test_pearson_perfect_correlation(self):
        """Test Pearson correlation for perfect correlation."""
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect linear
        score = metrics.pearson(y_pred, y_true)
        assert abs(score - 1.0) < 1e-6

    def test_pearson_no_correlation(self):
        """Test Pearson correlation for no correlation."""
        np.random.seed(42)
        y_pred = np.random.randn(100)
        y_true = np.random.randn(100)
        score = metrics.pearson(y_pred, y_true)
        assert abs(score) < 0.3  # Should be close to 0

    def test_mse_zero_error(self):
        """Test MSE for zero error."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        score = metrics.mse(y_pred, y_true)
        assert score == 0.0

    def test_mse_known_value(self):
        """Test MSE with known value."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([2.0, 3.0, 4.0])
        score = metrics.mse(y_pred, y_true)
        assert score == 1.0

    def test_rmse_relation_to_mse(self):
        """Test that RMSE is sqrt of MSE."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([2.0, 4.0, 6.0])
        mse_val = metrics.mse(y_pred, y_true)
        rmse_val = metrics.rmse(y_pred, y_true)
        assert abs(rmse_val - np.sqrt(mse_val)) < 1e-6

    def test_mae_zero_error(self):
        """Test MAE for zero error."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        score = metrics.mae(y_pred, y_true)
        assert score == 0.0

    def test_mae_known_value(self):
        """Test MAE with known value."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([2.0, 4.0, 5.0])
        score = metrics.mae(y_pred, y_true)
        assert score == (1.0 + 2.0 + 2.0) / 3.0

    def test_cosine_identical_vectors(self):
        """Test cosine similarity for identical vectors."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        score = metrics.cosine(y_pred, y_true)
        assert abs(score - 1.0) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors."""
        y_pred = np.array([1.0, 0.0])
        y_true = np.array([0.0, 1.0])
        score = metrics.cosine(y_pred, y_true)
        assert abs(score) < 1e-6

    def test_l2_zero_distance(self):
        """Test L2 distance for identical vectors."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        score = metrics.l2(y_pred, y_true)
        assert score == 0.0

    def test_l2_known_value(self):
        """Test L2 distance with known value."""
        y_pred = np.array([0.0, 0.0])
        y_true = np.array([3.0, 4.0])
        score = metrics.l2(y_pred, y_true)
        assert score == 5.0  # sqrt(3^2 + 4^2)

    def test_evaluate_perturbation_without_control(self):
        """Test evaluate_perturbation without control data."""
        np.random.seed(42)
        pred = np.random.rand(50, 100)
        true = np.random.rand(50, 100)

        result = metrics.evaluate_perturbation(pred, true, ctrl=None, include_delta=False)

        assert "R2" in result
        assert "Pearson" in result
        assert "MSE" in result
        assert "RMSE" in result
        assert "MAE" in result
        assert "Cosine" in result
        assert "L2" in result
        assert "R2_delta" not in result  # Should not be computed

    def test_evaluate_perturbation_with_control(self):
        """Test evaluate_perturbation with control data."""
        np.random.seed(42)
        pred = np.random.rand(50, 100)
        true = np.random.rand(50, 100)
        ctrl = np.random.rand(50, 100)

        result = metrics.evaluate_perturbation(pred, true, ctrl, include_delta=True)

        # Absolute metrics
        assert "R2" in result
        assert "Pearson" in result

        # Delta metrics
        assert "R2_delta" in result
        assert "Pearson_delta" in result


# =============================================================================
# Test Distribution Metrics
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestDistributionMetrics:
    """Tests for distribution comparison metrics."""

    def test_mmd_identical_distributions(self):
        """Test MMD for identical distributions."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        score = metrics.mmd(X, X, gamma=1.0)
        assert score < 1e-6  # Should be very close to 0

    def test_mmd_different_distributions(self):
        """Test MMD for different distributions."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 50) + 2.0  # Shifted distribution
        score = metrics.mmd(X, Y, gamma=1.0)
        assert score > 0.01  # Should be significantly > 0

    def test_mmd_per_gene(self):
        """Test MMD with per_gene=True."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 50)
        scores = metrics.mmd(X, Y, gamma=1.0, per_gene=True)
        assert scores.shape == (50,)
        assert all(scores >= 0)

    def test_mmd_sparse_input(self):
        """Test MMD with sparse input."""
        np.random.seed(42)
        X = sparse.random(100, 50, density=0.1, format="csr")
        Y = sparse.random(100, 50, density=0.1, format="csr")
        score = metrics.mmd(X.toarray(), Y.toarray(), gamma=1.0)
        assert score >= 0

    def test_wasserstein_identical_distributions(self):
        """Test Wasserstein distance for identical distributions."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        score = metrics.wasserstein_distance(X, X)
        assert score < 1e-10

    def test_wasserstein_different_distributions(self):
        """Test Wasserstein distance for different distributions."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 50) + 1.0
        score = metrics.wasserstein_distance(X, Y)
        assert score > 0.5

    def test_wasserstein_per_gene(self):
        """Test Wasserstein distance with per_gene=True."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 50)
        scores = metrics.wasserstein_distance(X, Y, per_gene=True)
        assert scores.shape == (50,)
        assert all(scores >= 0)

    def test_compute_distribution_metrics(self):
        """Test compute_distribution_metrics."""
        np.random.seed(42)
        pred = np.random.randn(100, 50)
        true = np.random.randn(100, 50)

        result = metrics.compute_distribution_metrics(pred, true, mmd_gamma=1.0)

        assert "MMD" in result
        assert "Wasserstein" in result
        assert result["MMD"] >= 0
        assert result["Wasserstein"] >= 0


# =============================================================================
# Test Direction Consistency
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestDirectionConsistency:
    """Tests for direction consistency metrics."""

    def test_direction_accuracy_perfect(self):
        """Test direction accuracy for perfect prediction."""
        np.random.seed(42)
        ctrl = np.random.rand(50, 100)
        delta = np.random.randn(100) * 2.0  # Same delta for both
        pred = ctrl + delta
        true = ctrl + delta

        acc = metrics.delta_direction_accuracy(pred, true, ctrl)
        assert acc == 1.0

    def test_direction_accuracy_random(self):
        """Test direction accuracy for random prediction."""
        np.random.seed(42)
        pred = np.random.randn(50, 100)
        true = np.random.randn(50, 100)
        ctrl = np.random.randn(50, 100)

        acc = metrics.delta_direction_accuracy(pred, true, ctrl)
        assert 0.0 <= acc <= 1.0
        # Random should be around 0.5, but allow variation
        assert 0.3 <= acc <= 0.7

    def test_direction_accuracy_per_gene(self):
        """Test direction accuracy with per_gene=True."""
        np.random.seed(42)
        pred = np.random.randn(50, 100)
        true = np.random.randn(50, 100)
        ctrl = np.random.randn(50, 100)

        result = metrics.delta_direction_accuracy(pred, true, ctrl, per_gene=True)
        assert result.shape == (100,)
        assert result.dtype == bool

    def test_compute_direction_metrics(self):
        """Test compute_direction_metrics."""
        np.random.seed(42)
        pred = np.random.randn(50, 100)
        true = np.random.randn(50, 100)
        ctrl = np.random.randn(50, 100)

        result = metrics.compute_direction_metrics(pred, true, ctrl)

        assert "delta_agreement_acc" in result
        assert "n_genes_up_pred" in result
        assert "n_genes_down_pred" in result
        assert "n_genes_up_true" in result
        assert "n_genes_down_true" in result
        assert "n_genes_agree" in result

        assert 0.0 <= result["delta_agreement_acc"] <= 1.0
        assert result["n_genes_up_pred"] + result["n_genes_down_pred"] == 100


# =============================================================================
# Test DEG Overlap Metrics
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestDEGOverlapMetrics:
    """Tests for DEG overlap metrics."""

    @pytest.fixture
    def sample_degs(self):
        """Create sample DEG DataFrames for testing."""
        pred_degs = pd.DataFrame(
            {
                "feature": [f"Gene{i}" for i in range(100)],
                "p_value": np.random.rand(100),
                "fdr": np.random.rand(100),
                "log2_fold_change": np.random.randn(100),
            }
        )
        pred_degs = pred_degs.sort_values("p_value")

        true_degs = pd.DataFrame(
            {
                "feature": [f"Gene{i}" for i in range(100)],
                "p_value": np.random.rand(100),
                "fdr": np.random.rand(100),
                "log2_fold_change": np.random.randn(100),
            }
        )
        true_degs = true_degs.sort_values("p_value")

        return pred_degs, true_degs

    def test_deg_overlap_topn(self, sample_degs):
        """Test DEG overlap for top-N genes."""
        pred_degs, true_degs = sample_degs
        overlap = metrics.deg_overlap_topn(pred_degs, true_degs, top_n=20)
        assert 0.0 <= overlap <= 1.0

    def test_deg_overlap_topn_perfect(self):
        """Test DEG overlap for perfect top-N overlap."""
        genes = [f"Gene{i}" for i in range(50)]
        pred_degs = pd.DataFrame({"feature": genes, "p_value": np.arange(50)})
        true_degs = pd.DataFrame({"feature": genes, "p_value": np.arange(50)})

        overlap = metrics.deg_overlap_topn(pred_degs, true_degs, top_n=20)
        assert overlap == 1.0

    def test_deg_overlap_topn_insufficient_genes(self):
        """Test DEG overlap when insufficient genes."""
        pred_degs = pd.DataFrame({"feature": ["Gene1", "Gene2"], "p_value": [0.01, 0.02]})
        true_degs = pd.DataFrame({"feature": ["Gene1", "Gene2"], "p_value": [0.01, 0.02]})

        overlap = metrics.deg_overlap_topn(pred_degs, true_degs, top_n=20)
        assert np.isnan(overlap)

    def test_deg_overlap_pvalue(self, sample_degs):
        """Test DEG overlap for p-value threshold."""
        pred_degs, true_degs = sample_degs
        overlap = metrics.deg_overlap_pvalue(pred_degs, true_degs, p_threshold=0.05)
        assert 0.0 <= overlap <= 1.0 or np.isnan(overlap)

    def test_deg_overlap_fdr(self, sample_degs):
        """Test DEG overlap for FDR threshold."""
        pred_degs, true_degs = sample_degs
        overlap = metrics.deg_overlap_fdr(pred_degs, true_degs, fdr_threshold=0.05)
        assert 0.0 <= overlap <= 1.0 or np.isnan(overlap)

    def test_compute_deg_overlap_metrics(self, sample_degs):
        """Test compute_deg_overlap_metrics."""
        pred_degs, true_degs = sample_degs

        result = metrics.compute_deg_overlap_metrics(
            pred_degs,
            true_degs,
            top_n_list=[20, 50],
            p_thresholds=[0.05],
            fdr_thresholds=[0.05],
        )

        assert "Top20_DEG_Overlap" in result
        assert "Top50_DEG_Overlap" in result
        # Key format is P<0{threshold with decimal removed}
        assert any("P<" in k and "DEG_Overlap" in k for k in result)
        assert any("FDR<" in k and "DEG_Overlap" in k for k in result)


# =============================================================================
# Test Spatial Metrics
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestSpatialMetrics:
    """Tests for spatial autocorrelation metrics."""

    @pytest.fixture
    def ring_graph(self):
        """Create a ring lattice graph for testing."""
        n = 100
        graph = sparse.lil_matrix((n, n))
        for i in range(n):
            graph[i, (i - 1) % n] = 1
            graph[i, (i + 1) % n] = 1
        return graph.tocsr()

    def test_morans_i_autocorrelated(self, ring_graph):
        """Test Moran's I for autocorrelated data."""
        n = 100
        # Create spatially autocorrelated pattern
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        moran = metrics.morans_i(ring_graph, x)
        assert moran > 0.5  # Should show positive autocorrelation

    def test_morans_i_random(self, ring_graph):
        """Test Moran's I for random data."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        moran = metrics.morans_i(ring_graph, x)
        assert abs(moran) < 0.3  # Should be close to 0

    def test_morans_i_matrix(self, ring_graph):
        """Test Moran's I for matrix input."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(50, n)  # 50 genes × 100 cells
        morans = metrics.morans_i(ring_graph, X)
        assert morans.shape == (50,)

    def test_morans_i_constant_values(self, ring_graph):
        """Test Moran's I for constant values."""
        n = 100
        x = np.ones(n)  # Constant
        # Constant values should raise ZeroDivisionError or return NaN
        try:
            moran = metrics.morans_i(ring_graph, x)
            assert np.isnan(moran)
        except ZeroDivisionError:
            # Expected for constant values (variance is zero)
            pass

    def test_gearys_c_autocorrelated(self, ring_graph):
        """Test Geary's C for autocorrelated data."""
        n = 100
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        geary = metrics.gearys_c(ring_graph, x)
        assert geary < 1.0  # Should show positive autocorrelation

    def test_gearys_c_random(self, ring_graph):
        """Test Geary's C for random data."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        geary = metrics.gearys_c(ring_graph, x)
        assert 0.5 < geary < 1.5  # Should be close to 1

    def test_gearys_c_matrix(self, ring_graph):
        """Test Geary's C for matrix input."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(50, n)
        gearys = metrics.gearys_c(ring_graph, X)
        assert gearys.shape == (50,)

    def test_compute_spatial_metrics(self, ring_graph):
        """Test compute_spatial_metrics."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)

        result = metrics.compute_spatial_metrics(ring_graph, x)

        assert "morans_i" in result
        assert "gearys_c" in result


# =============================================================================
# Test Classification Metrics
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestClassificationMetrics:
    """Tests for classification/clustering metrics."""

    def test_confusion_matrix_from_arrays(self):
        """Test confusion matrix from arrays."""
        orig = ["A", "A", "B", "B", "C", "C"]
        new = ["A", "B", "B", "B", "C", "C"]

        cm = metrics.confusion_matrix(orig, new, normalize=False)

        assert cm.shape[0] == 3  # 3 unique labels
        assert cm.loc["A", "A"] == 1
        assert cm.loc["A", "B"] == 1
        assert cm.loc["B", "B"] == 2

    def test_confusion_matrix_normalized(self):
        """Test confusion matrix with normalization."""
        orig = ["A", "A", "B", "B"]
        new = ["A", "B", "B", "B"]

        cm = metrics.confusion_matrix(orig, new, normalize=True)

        assert cm.loc["A", "A"] == 0.5
        assert cm.loc["A", "B"] == 0.5
        assert cm.loc["B", "B"] == 1.0

    def test_confusion_matrix_from_dataframe(self):
        """Test confusion matrix from DataFrame."""
        df = pd.DataFrame({"true": ["A", "A", "B", "B"], "pred": ["A", "B", "B", "B"]})

        cm = metrics.confusion_matrix("true", "pred", data=df)

        assert isinstance(cm, pd.DataFrame)
        assert cm.shape == (2, 2)

    def test_confusion_matrix_perfect_classification(self):
        """Test confusion matrix for perfect classification."""
        orig = ["A", "B", "C"]
        new = ["A", "B", "C"]

        cm = metrics.confusion_matrix(orig, new, normalize=True)

        # Diagonal should be all 1.0
        assert cm.loc["A", "A"] == 1.0
        assert cm.loc["B", "B"] == 1.0
        assert cm.loc["C", "C"] == 1.0


# =============================================================================
# Test Comprehensive Evaluation
# =============================================================================


@pytest.mark.skipif(not _has_perturblab, reason="requires perturblab")
class TestComprehensiveEvaluation:
    """Tests for comprehensive evaluation function."""

    def test_evaluate_prediction_without_degs(self):
        """Test evaluate_prediction without DEG data."""
        np.random.seed(42)
        pred = np.random.rand(50, 100)
        true = np.random.rand(50, 100)
        ctrl = np.random.rand(50, 100)

        result = metrics.evaluate_prediction(
            pred,
            true,
            ctrl,
            include_expression=True,
            include_distribution=True,
            include_direction=True,
            include_deg_overlap=False,
        )

        # Should have expression metrics
        assert "R2" in result
        assert "Pearson" in result

        # Should have distribution metrics
        assert "MMD" in result
        assert "Wasserstein" in result

        # Should have direction metrics
        assert "delta_agreement_acc" in result

    def test_evaluate_prediction_selective_metrics(self):
        """Test evaluate_prediction with selective metrics."""
        np.random.seed(42)
        pred = np.random.rand(50, 100)
        true = np.random.rand(50, 100)
        ctrl = np.random.rand(50, 100)

        result = metrics.evaluate_prediction(
            pred,
            true,
            ctrl,
            include_expression=True,
            include_distribution=False,
            include_direction=False,
            include_deg_overlap=False,
        )

        # Should have expression metrics only
        assert "R2" in result
        assert "MMD" not in result
        assert "delta_agreement_acc" not in result

    def test_evaluate_prediction_with_degs(self):
        """Test evaluate_prediction with DEG data."""
        np.random.seed(42)
        pred = np.random.rand(50, 100)
        true = np.random.rand(50, 100)
        ctrl = np.random.rand(50, 100)

        # Create DEG DataFrames
        pred_degs = pd.DataFrame(
            {
                "feature": [f"Gene{i}" for i in range(100)],
                "p_value": np.random.rand(100),
                "fdr": np.random.rand(100),
                "log2_fold_change": np.random.randn(100),
            }
        )

        true_degs = pd.DataFrame(
            {
                "feature": [f"Gene{i}" for i in range(100)],
                "p_value": np.random.rand(100),
                "fdr": np.random.rand(100),
                "log2_fold_change": np.random.randn(100),
            }
        )

        result = metrics.evaluate_prediction(
            pred,
            true,
            ctrl,
            pred_degs=pred_degs,
            true_degs=true_degs,
            include_deg_overlap=True,
            deg_top_n=[20],
        )

        # Should have DEG overlap metrics
        assert "Top20_DEG_Overlap" in result

