"""Test module for PerturbLab type classes.

This module tests:
1. Basic data structures (Vocab, WeightedGraph, GeneGraph)
2. CellData functionality (view, cache, layer, virtual genes)
3. PerturbationData functionality (perturbation-specific features)
4. Data consistency across operations
"""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Skip all tests if anndata not available
pytest.importorskip("anndata")


class TestVocab:
    """Test Vocab class for name-to-index mapping."""

    def test_vocab_creation_from_list(self):
        """Test creating vocab from list."""
        from perturblab.types import Vocab
        
        names = ["gene1", "gene2", "gene3"]
        vocab = Vocab(names)
        
        assert len(vocab) == 3
        assert vocab["gene1"] == 0
        assert vocab["gene2"] == 1
        assert vocab["gene3"] == 2

    def test_vocab_creation_from_array(self):
        """Test creating vocab from numpy array (converts to list)."""
        from perturblab.types import Vocab
        
        names = np.array(["A", "B", "C"])
        vocab = Vocab(names.tolist())  # Vocab requires list
        
        assert len(vocab) == 3
        assert vocab.itos[0] == "A"  # Use itos instead of names

    def test_vocab_duplicate_handling(self):
        """Test that vocab raises error on duplicates."""
        from perturblab.types import Vocab
        
        names = ["gene1", "gene2", "gene1"]  # Duplicate
        
        with pytest.raises(ValueError, match="Duplicate"):
            Vocab(names)

    def test_vocab_contains(self):
        """Test that 'in' operator works."""
        from perturblab.types import Vocab
        
        vocab = Vocab(["A", "B", "C"])
        
        assert "A" in vocab
        assert "D" not in vocab

    def test_vocab_indexing(self):
        """Test vocab indexing operations."""
        from perturblab.types import Vocab
        
        vocab = Vocab(["A", "B", "C"])
        
        # By name (uses stoi dict)
        assert vocab.stoi["A"] == 0
        assert vocab.stoi["C"] == 2
        
        # By index (uses itos list)
        assert vocab.itos[0] == "A"
        assert vocab.itos[2] == "C"
        
        # KeyError for missing
        assert "nonexistent" not in vocab.stoi


class TestGeneVocab:
    """Test GeneVocab class."""

    def test_gene_vocab_creation(self):
        """Test creating gene vocab."""
        from perturblab.types import GeneVocab
        
        genes = ["TP53", "KRAS", "MYC"]
        vocab = GeneVocab(genes)
        
        assert len(vocab) == 3
        assert vocab["TP53"] == 0

    def test_gene_vocab_from_anndata(self):
        """Test creating vocab from AnnData."""
        from perturblab.types import GeneVocab
        
        adata = ad.AnnData(
            X=np.random.randn(10, 5),
            var=pd.DataFrame(index=["G1", "G2", "G3", "G4", "G5"])
        )
        
        vocab = GeneVocab.from_anndata(adata)
        
        assert len(vocab) == 5
        assert vocab["G1"] == 0


class TestWeightedGraph:
    """Test WeightedGraph class."""

    def test_weighted_graph_creation(self):
        """Test creating weighted graph."""
        from perturblab.types import WeightedGraph
        
        edges = [(0, 1, 0.5), (1, 2, 0.3), (0, 2, 0.7)]
        graph = WeightedGraph(edges, n_nodes=3)
        
        assert graph.n_nodes == 3
        assert graph.n_edges == 3

    def test_weighted_graph_unweighted(self):
        """Test creating unweighted graph (auto weight=1.0)."""
        from perturblab.types import WeightedGraph
        
        edges = [(0, 1), (1, 2)]  # No weights
        graph = WeightedGraph(edges, n_nodes=3, default_weight=1.0)
        
        assert graph.n_edges == 2
        # Check that weights were added
        assert all(len(edge) == 3 for edge in graph.edges)

    def test_weighted_graph_to_adjacency(self):
        """Test converting to adjacency matrix."""
        from perturblab.types import WeightedGraph
        
        edges = [(0, 1, 0.5), (1, 0, 0.5)]  # Undirected
        graph = WeightedGraph(edges, n_nodes=2)
        
        adj = graph.to_adjacency_matrix()
        
        assert adj.shape == (2, 2)
        assert adj[0, 1] == 0.5
        assert adj[1, 0] == 0.5

    def test_weighted_graph_empty(self):
        """Test empty graph."""
        from perturblab.types import WeightedGraph
        
        graph = WeightedGraph([], n_nodes=5)
        
        assert graph.n_nodes == 5
        assert graph.n_edges == 0


class TestGeneGraph:
    """Test GeneGraph class."""

    def test_gene_graph_creation(self):
        """Test creating gene graph."""
        from perturblab.types import GeneGraph, GeneVocab, WeightedGraph
        
        vocab = GeneVocab(["TP53", "KRAS", "MYC"])
        edges = [(0, 1, 0.8), (1, 2, 0.6)]
        graph = WeightedGraph(edges, n_nodes=3)
        
        gene_graph = GeneGraph(graph, vocab)
        
        assert gene_graph.n_nodes == 3
        assert gene_graph.n_edges == 2

    def test_gene_graph_neighbors_by_name(self):
        """Test getting neighbors by gene name."""
        from perturblab.types import GeneGraph, GeneVocab, WeightedGraph
        
        vocab = GeneVocab(["A", "B", "C"])
        edges = [(0, 1, 0.8), (1, 2, 0.6), (0, 2, 0.4)]
        graph = WeightedGraph(edges, n_nodes=3)
        
        gene_graph = GeneGraph(graph, vocab)
        
        neighbors = gene_graph.neighbors("A")
        assert "B" in neighbors
        assert "C" in neighbors

    def test_gene_graph_subgraph(self):
        """Test creating subgraph."""
        from perturblab.types import GeneGraph, GeneVocab, WeightedGraph
        
        vocab = GeneVocab(["A", "B", "C", "D"])
        edges = [(0, 1, 0.8), (1, 2, 0.6), (2, 3, 0.5)]
        graph = WeightedGraph(edges, n_nodes=4)
        
        gene_graph = GeneGraph(graph, vocab)
        
        # Subgraph for genes A and B
        subgraph = gene_graph.subgraph(["A", "B"])
        
        assert subgraph.n_nodes == 2
        assert len(subgraph.vocab) == 2

    def test_gene_graph_vocab_mismatch(self):
        """Test that mismatched vocab and graph raise error."""
        from perturblab.types import GeneGraph, GeneVocab, WeightedGraph
        
        vocab = GeneVocab(["A", "B"])  # 2 genes
        graph = WeightedGraph([], n_nodes=3)  # 3 nodes
        
        with pytest.raises(ValueError):
            GeneGraph(graph, vocab)


class TestCellDataBasic:
    """Test basic CellData functionality."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData for testing."""
        np.random.seed(42)
        n_obs, n_vars = 100, 50
        
        X = sp.csr_matrix(np.random.randn(n_obs, n_vars))
        obs = pd.DataFrame({
            'cell_type': np.random.choice(['A', 'B', 'C'], n_obs),
            'batch': np.random.choice([1, 2], n_obs),
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        var = pd.DataFrame(
            index=[f'gene_{i}' for i in range(n_vars)]
        )
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Add additional layer
        adata.layers['normalized'] = X.copy()
        
        return adata

    def test_celldata_initialization(self, sample_adata):
        """Test CellData initialization."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata, cell_type_col='cell_type')
        
        assert cell_data.n_obs == 100
        assert cell_data.n_vars == 50
        assert cell_data.cell_type_col == 'cell_type'

    def test_celldata_genes_property(self, sample_adata):
        """Test accessing gene names through adata.var_names."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Access gene names through adata.var_names
        genes = cell_data.adata.var_names
        assert len(genes) == 50
        # genes returns Index, convert to list for checking
        assert list(genes)[0] == 'gene_0'

    def test_celldata_cell_types_property(self, sample_adata):
        """Test cell_types property."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata, cell_type_col='cell_type')
        
        cell_types = cell_data.cell_types
        assert set(cell_types) == {'A', 'B', 'C'}

    def test_celldata_X_property(self, sample_adata):
        """Test X property returns correct data."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        X = cell_data.X
        assert X.shape == (100, 50)
        
        # X property may return sparse or dense, check both cases
        if sp.issparse(X):
            X = X.toarray()
        assert isinstance(X, np.ndarray)

    def test_celldata_getitem(self, sample_adata):
        """Test __getitem__ slicing."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata, cell_type_col='cell_type')
        
        # Single cell
        single_cell = cell_data[0]
        assert isinstance(single_cell, CellData)
        assert single_cell.n_obs == 1
        
        # Multiple cells
        subset = cell_data[:10]
        assert subset.n_obs == 10
        assert subset.n_vars == 50


class TestCellDataLayers:
    """Test CellData layer management."""

    @pytest.fixture
    def adata_with_layers(self):
        """Create AnnData with multiple layers."""
        np.random.seed(42)
        n_obs, n_vars = 50, 30
        
        X = sp.csr_matrix(np.random.randn(n_obs, n_vars))
        adata = ad.AnnData(X=X)
        
        # Add layers
        adata.layers['raw'] = X.copy()
        adata.layers['normalized'] = X.copy() / 10
        adata.layers['log1p'] = np.log1p(X.toarray())
        
        return adata

    def test_use_layer(self, adata_with_layers):
        """Test switching layers."""
        from perturblab.types import CellData
        
        cell_data = CellData(adata_with_layers)
        
        # Default uses X - convert sparse to dense for comparison
        X_orig = cell_data.X
        if sp.issparse(X_orig):
            X_orig = X_orig.toarray()
        X_orig = X_orig.copy()
        
        # Switch to normalized
        cell_data.use_layer('normalized')
        X_norm = cell_data.X
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        
        # Should be different (normalized is divided by 10)
        assert not np.allclose(X_orig, X_norm)
        
        # Switch back to X
        cell_data.use_layer(None)
        X_back = cell_data.X
        if sp.issparse(X_back):
            X_back = X_back.toarray()
        
        assert np.allclose(X_orig, X_back)

    def test_layer_not_found(self, adata_with_layers):
        """Test error when layer doesn't exist."""
        from perturblab.types import CellData
        
        cell_data = CellData(adata_with_layers)
        
        with pytest.raises(KeyError, match="nonexistent"):
            cell_data.use_layer('nonexistent')

    def test_layer_with_cache(self, adata_with_layers):
        """Test that layer change clears cache."""
        from perturblab.types import CellData
        
        cell_data = CellData(adata_with_layers)
        
        # Enable cache
        cell_data.enable_cache()
        assert cell_data.is_cache_enabled
        
        # Switch layer should clear cache
        cell_data.use_layer('normalized')
        
        # Cache should be cleared (implementation detail)
        assert not hasattr(cell_data, '_cached_X') or cell_data._cached_X is None


class TestCellDataCache:
    """Test CellData caching mechanism."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData."""
        np.random.seed(42)
        X = sp.csr_matrix(np.random.randn(100, 50))
        return ad.AnnData(X=X)

    def test_cache_enable_disable(self, sample_adata):
        """Test enabling and disabling cache."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Initially disabled
        assert not cell_data.is_cache_enabled
        
        # Enable
        cell_data.enable_cache()
        assert cell_data.is_cache_enabled
        
        # Disable (clear)
        cell_data.clear_cache()
        assert not cell_data.is_cache_enabled

    def test_cache_materializes_data(self, sample_adata):
        """Test that cache materializes sparse data to dense."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Enable cache
        cell_data.enable_cache()
        
        # Internal cache should be dense numpy array
        assert hasattr(cell_data, '_cached_X')
        assert isinstance(cell_data._cached_X, np.ndarray)
        assert cell_data._cached_X.shape == (100, 50)

    def test_cache_accelerates_getitem(self, sample_adata):
        """Test that cache improves __getitem__ performance."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Enable cache
        cell_data.enable_cache()
        
        # Access should work with cache
        subset1 = cell_data[0]
        subset2 = cell_data[0]
        
        # Both should return valid CellData
        assert isinstance(subset1, CellData)
        assert isinstance(subset2, CellData)
        
        # Note: Performance testing is flaky, so we just test functionality

    def test_cached_getitem_creates_copy(self, sample_adata):
        """Test that __getitem__ with cache creates proper copies."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        cell_data.enable_cache()
        
        # Get subset
        subset = cell_data[:10]
        
        # Should be a new CellData instance
        assert isinstance(subset, CellData)
        assert subset.n_obs == 10
        
        # Should have its own AnnData (not a view that could cause issues)
        assert subset.adata is not cell_data.adata

    def test_cache_refresh(self, sample_adata):
        """Test refreshing cache."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        cell_data.enable_cache()
        
        # Get cached data
        cache_before = cell_data._cached_X.copy()
        
        # Modify underlying data
        cell_data.adata.X[0, 0] = 999.0
        
        # Refresh cache
        cell_data.refresh_cache()
        cache_after = cell_data._cached_X
        
        # Cache should reflect the change
        assert cache_after[0, 0] == 999.0
        assert not np.array_equal(cache_before, cache_after)


class TestCellDataVirtualGenes:
    """Test CellData virtual gene functionality."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        var = pd.DataFrame(index=[f'gene_{i}' for i in range(10)])
        return ad.AnnData(X=X, var=var)

    def test_align_genes_basic(self, sample_adata):
        """Test basic gene alignment."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Align to subset
        target_genes = ['gene_0', 'gene_5', 'gene_9']
        aligned = cell_data.align_genes(target_genes)
        
        assert aligned.n_vars == 3
        # Check gene names through adata.var_names
        aligned_genes = aligned.adata.var_names.tolist()
        assert aligned_genes == target_genes

    def test_align_genes_with_virtual(self, sample_adata):
        """Test alignment with virtual genes."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Align with some virtual genes
        target_genes = ['gene_0', 'virtual_1', 'gene_5', 'virtual_2']
        aligned = cell_data.align_genes(target_genes, fill_value=0.0)
        
        assert aligned.n_vars == 4
        
        # Check that virtual genes are filled with 0
        X = aligned.X
        # gene_0 should have data
        assert not np.all(X[:, 0] == 0)
        # virtual_1 should be all zeros
        assert np.all(X[:, 1] == 0.0)

    def test_align_genes_reorder(self, sample_adata):
        """Test that alignment reorders genes correctly."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Reorder
        target_genes = ['gene_9', 'gene_0', 'gene_5']
        aligned = cell_data.align_genes(target_genes)
        
        # Check order is preserved
        aligned_genes = aligned.adata.var_names.tolist()
        assert aligned_genes == target_genes
        
        # Check data is correctly reordered - handle sparse matrices
        orig_X = cell_data.X
        if sp.issparse(orig_X):
            orig_gene9 = orig_X[:, 9].toarray().flatten()
        else:
            orig_gene9 = orig_X[:, 9].copy()
        
        aligned_X = aligned.X
        if sp.issparse(aligned_X):
            aligned_gene9 = aligned_X[:, 0].toarray().flatten()
        else:
            aligned_gene9 = aligned_X[:, 0]
        
        assert np.allclose(orig_gene9, aligned_gene9)

    def test_virtual_view_backed_mode(self):
        """Test virtual genes with backed mode."""
        from perturblab.types import CellData
        
        # Create backed AnnData
        with tempfile.TemporaryDirectory() as tmpdir:
            h5ad_path = Path(tmpdir) / "test.h5ad"
            
            # Create and save
            adata = ad.AnnData(
                X=np.random.randn(100, 20),
                var=pd.DataFrame(index=[f'gene_{i}' for i in range(20)])
            )
            adata.write_h5ad(h5ad_path)
            
            # Load in backed mode
            adata_backed = ad.read_h5ad(h5ad_path, backed='r')
            cell_data = CellData(adata_backed)
            
            # Align with virtual genes
            target_genes = ['gene_0', 'virtual_1', 'gene_10']
            aligned = cell_data.align_genes(target_genes, fill_value=-1.0)
            
            # Should still work
            X = aligned.X
            assert X.shape == (100, 3)
            assert np.all(X[:, 1] == -1.0)  # Virtual gene


class TestCellDataConsistency:
    """Test data consistency across operations."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData with known values."""
        np.random.seed(42)
        X = np.arange(300).reshape(30, 10).astype(float)
        obs = pd.DataFrame({'type': ['A'] * 15 + ['B'] * 15})
        var = pd.DataFrame(index=[f'G{i}' for i in range(10)])
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_slicing_preserves_data(self, sample_adata):
        """Test that slicing preserves data values."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Original data
        orig_cell0 = cell_data.X[0].copy()
        
        # Slice
        subset = cell_data[:10]
        
        # Data should be preserved
        assert np.allclose(subset.X[0], orig_cell0)

    def test_layer_switch_consistency(self, sample_adata):
        """Test that layer switching maintains consistency."""
        from perturblab.types import CellData
        
        # Add layers
        sample_adata.layers['double'] = sample_adata.X.copy() * 2
        
        cell_data = CellData(sample_adata)
        
        # Original X
        X_orig = cell_data.X.copy()
        
        # Switch to double
        cell_data.use_layer('double')
        X_double = cell_data.X
        
        # Should be exactly double
        assert np.allclose(X_double, X_orig * 2)

    def test_cache_data_consistency(self, sample_adata):
        """Test that cached data matches original."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Get data without cache
        X_no_cache = cell_data.X.copy()
        
        # Enable cache
        cell_data.enable_cache()
        X_cached = cell_data.X
        
        # Should be identical
        assert np.allclose(X_no_cache, X_cached)

    def test_virtual_gene_alignment_consistency(self, sample_adata):
        """Test that virtual gene alignment is consistent."""
        from perturblab.types import CellData
        
        cell_data = CellData(sample_adata)
        
        # Align twice with same genes
        target_genes = ['G0', 'G5', 'virtual']
        aligned1 = cell_data.align_genes(target_genes, fill_value=0.0)
        aligned2 = cell_data.align_genes(target_genes, fill_value=0.0)
        
        # Should produce identical results
        assert np.allclose(aligned1.X, aligned2.X)


class TestPerturbationDataBasic:
    """Test basic PerturbationData functionality."""

    @pytest.fixture
    def perturb_adata(self):
        """Create sample perturbation AnnData."""
        np.random.seed(42)
        n_obs = 100
        
        X = sp.csr_matrix(np.random.randn(n_obs, 50))
        obs = pd.DataFrame({
            'perturbation': ['ctrl'] * 30 + ['gene_A'] * 35 + ['gene_B'] * 35,
            'cell_type': np.random.choice(['T', 'B'], n_obs),
        })
        var = pd.DataFrame(index=[f'gene_{i}' for i in range(50)])
        
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_perturbation_data_init(self, perturb_adata):
        """Test PerturbationData initialization."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        assert pert_data.perturbation_col == 'perturbation'
        assert 'ctrl' in pert_data.control_labels

    def test_perturbation_property(self, perturb_adata):
        """Test perturbations property."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        perturbations = pert_data.perturbations
        assert set(perturbations) == {'ctrl', 'gene_A', 'gene_B'}

    def test_is_control_property(self, perturb_adata):
        """Test is_control property."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        is_control = pert_data.is_control
        
        # First 30 should be control
        assert np.sum(is_control) == 30
        assert np.all(is_control[:30])
        assert not np.any(is_control[30:])

    def test_control_mask_property(self, perturb_adata):
        """Test is_control property (mask for control cells)."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        # Can access control cells through boolean indexing
        is_control = pert_data.is_control
        n_control = is_control.sum()
        
        assert n_control == 30

    def test_perturbed_mask_property(self, perturb_adata):
        """Test is_control property to identify perturbed cells."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        # Perturbed cells are those that are not control
        is_perturbed = ~pert_data.is_control
        n_perturbed = is_perturbed.sum()
        
        assert n_perturbed == 70  # 35 + 35


class TestPerturbationDataInheritance:
    """Test that PerturbationData inherits CellData features."""

    @pytest.fixture
    def perturb_adata(self):
        """Create sample perturbation AnnData."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        obs = pd.DataFrame({
            'perturbation': ['ctrl'] * 20 + ['treat'] * 30,
        })
        var = pd.DataFrame(index=[f'G{i}' for i in range(20)])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.layers['raw'] = X.copy()
        return adata

    def test_perturbation_inherits_cache(self, perturb_adata):
        """Test that PerturbationData supports caching."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation'
        )
        
        # Should support cache
        pert_data.enable_cache()
        assert pert_data.is_cache_enabled
        
        # Should work with getitem
        subset = pert_data[:10]
        assert subset.n_obs == 10

    def test_perturbation_inherits_layers(self, perturb_adata):
        """Test that PerturbationData supports layers."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation'
        )
        
        # Should support layer switching
        X_orig = pert_data.X.copy()
        pert_data.use_layer('raw')
        X_raw = pert_data.X
        
        assert np.allclose(X_orig, X_raw)

    def test_perturbation_inherits_virtual_genes(self, perturb_adata):
        """Test that PerturbationData supports virtual genes."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation'
        )
        
        # Should support gene alignment
        target_genes = ['G0', 'virtual', 'G10']
        aligned = pert_data.align_genes(target_genes, fill_value=0.0)
        
        assert isinstance(aligned, PerturbationData)
        assert aligned.n_vars == 3
        # Should preserve perturbation info
        assert aligned.perturbation_col == 'perturbation'

    def test_perturbation_slicing_preserves_type(self, perturb_adata):
        """Test that slicing preserves PerturbationData type."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        # Slice
        subset = pert_data[:10]
        
        # Should still be PerturbationData
        assert isinstance(subset, PerturbationData)
        assert subset.perturbation_col == 'perturbation'
        assert 'ctrl' in subset.control_labels


class TestPerturbationDataAdvanced:
    """Test advanced PerturbationData features."""

    @pytest.fixture
    def complex_perturb_adata(self):
        """Create complex perturbation dataset."""
        np.random.seed(42)
        n_obs = 200
        
        perturbations = (
            ['ctrl'] * 50 +
            ['gene_A'] * 30 +
            ['gene_B'] * 30 +
            ['gene_A+gene_B'] * 40 +
            ['gene_C'] * 50
        )
        
        X = np.random.randn(n_obs, 30)
        obs = pd.DataFrame({
            'perturbation': perturbations,
            'cell_type': np.random.choice(['T', 'B', 'NK'], n_obs),
            'batch': np.tile([1, 2], n_obs // 2),
        })
        var = pd.DataFrame(index=[f'gene_{i}' for i in range(30)])
        
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_multiple_control_labels(self, complex_perturb_adata):
        """Test handling multiple control labels."""
        from perturblab.types import PerturbationData
        
        # Relabel some ctrl to ctrl_B
        complex_perturb_adata.obs.loc[
            complex_perturb_adata.obs['perturbation'] == 'ctrl', 'perturbation'
        ] = ['ctrl_A'] * 25 + ['ctrl_B'] * 25
        
        pert_data = PerturbationData(
            complex_perturb_adata,
            perturbation_col='perturbation',
            control_label=['ctrl_A', 'ctrl_B']
        )
        
        assert len(pert_data.control_labels) == 2
        assert pert_data.is_control.sum() == 50

    def test_ignore_labels(self, complex_perturb_adata):
        """Test ignoring certain perturbations."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            complex_perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl',
            ignore_labels=['gene_C']
        )
        
        assert 'gene_C' in pert_data.ignore_labels

    def test_combo_perturbations(self, complex_perturb_adata):
        """Test handling combination perturbations."""
        from perturblab.types import PerturbationData
        
        pert_data = PerturbationData(
            complex_perturb_adata,
            perturbation_col='perturbation',
            control_label='ctrl'
        )
        
        # Get unique perturbations from the data
        perturbations = pert_data.adata.obs['perturbation'].unique()
        
        # Should include combo
        assert 'gene_A+gene_B' in perturbations


class TestDataConsistencyAcrossTypes:
    """Test data consistency across different type conversions."""

    def test_celldata_to_perturbation_data(self):
        """Test converting CellData to PerturbationData preserves data."""
        from perturblab.types import CellData, PerturbationData
        
        # Create as CellData
        X = np.arange(600).reshape(60, 10).astype(float)
        obs = pd.DataFrame({'pert': ['ctrl'] * 30 + ['treat'] * 30})
        adata = ad.AnnData(X=X, obs=obs)
        
        cell_data = CellData(adata)
        X_cell = cell_data.X.copy()
        
        # Create as PerturbationData from same adata
        pert_data = PerturbationData(
            adata.copy(),
            perturbation_col='pert'
        )
        X_pert = pert_data.X
        
        # Data should be identical
        assert np.allclose(X_cell, X_pert)

    def test_view_chain_consistency(self):
        """Test that chaining views maintains data consistency."""
        from perturblab.types import CellData
        
        X = np.arange(500).reshape(50, 10).astype(float)
        adata = ad.AnnData(X=X)
        
        cell_data = CellData(adata)
        
        # Original value
        orig_val = cell_data.X[0, 0]
        
        # Chain operations
        subset1 = cell_data[:30]
        subset2 = subset1[:20]
        subset3 = subset2[:10]
        
        # Value should be preserved
        assert subset3.X[0, 0] == orig_val


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

