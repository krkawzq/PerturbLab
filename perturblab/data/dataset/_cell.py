"""Cell dataset with lazy virtual gene view and backed mode support.

This module provides `CellDataset`, a wrapper around AnnData with enhanced features:
- Zero-copy gene alignment with virtual genes
- Backed mode support for large datasets
- Efficient data loading and transformation
"""

import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse
import torch
import warnings
from typing import Optional, Literal, Any, Union, List, Tuple, Callable, Dict

DuplicatedGenePolicy = Literal['error', 'first', 'last', 'remove']


class _VirtualViewBackedArray:
    """Lazy wrapper for backed AnnData with virtual genes.
    
    Provides array-like interface without loading entire matrix into memory.
    Data is loaded on-demand when sliced or accessed.
    """
    
    def __init__(
        self,
        base_X,
        gene_indices: np.ndarray,
        target_genes: pd.Index,
        virtual_genes: Dict[str, float],
        n_cells: int,
    ):
        self.base_X = base_X
        self.gene_indices = gene_indices
        self.target_genes = target_genes
        self.virtual_genes = virtual_genes
        self._shape = (n_cells, len(target_genes))
        self._dtype = base_X.dtype if hasattr(base_X, 'dtype') else np.float32
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the virtual array."""
        return self._shape
    
    @property
    def dtype(self):
        """Data type of the array."""
        return self._dtype
    
    def __getitem__(self, key):
        """Lazy slicing - loads only requested data from disk."""
        # Parse slice indices
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx, col_idx = key, slice(None)
        
        # Normalize indices
        row_idx = self._normalize_index(row_idx, 0)
        col_idx = self._normalize_index(col_idx, 1)
        
        # Determine target columns after slicing
        if isinstance(col_idx, slice):
            target_col_indices = np.arange(*col_idx.indices(self.shape[1]))
        elif isinstance(col_idx, (list, np.ndarray)):
            target_col_indices = np.asarray(col_idx)
        else:
            target_col_indices = np.array([col_idx])
        
        # Map to source columns (filter out virtual genes)
        source_cols = []
        target_to_source = {}  # target_idx -> position in source_cols
        
        for i, target_idx in enumerate(target_col_indices):
            source_idx = self.gene_indices[target_idx]
            if source_idx >= 0:  # Real gene
                target_to_source[i] = len(source_cols)
                source_cols.append(source_idx)
        
        # Load real genes from disk (only the columns we need)
        if len(source_cols) > 0:
            real_data = self.base_X[row_idx, source_cols]
            if scipy.sparse.issparse(real_data):
                real_data = real_data.toarray()
            else:
                real_data = np.asarray(real_data)
            
            # Ensure 2D
            if real_data.ndim == 1:
                real_data = real_data.reshape(-1, 1)
        else:
            # All virtual genes
            n_rows = self._get_slice_length(row_idx, 0)
            real_data = np.empty((n_rows, 0), dtype=self._dtype)
        
        # Build result with virtual genes filled
        n_rows = real_data.shape[0]
        n_cols = len(target_col_indices)
        result = np.zeros((n_rows, n_cols), dtype=self._dtype)
        
        for i, target_idx in enumerate(target_col_indices):
            source_idx = self.gene_indices[target_idx]
            if source_idx >= 0:  # Real gene
                result[:, i] = real_data[:, target_to_source[i]]
            else:  # Virtual gene
                gene_name = self.target_genes[target_idx]
                fill_val = self.virtual_genes[gene_name]
                result[:, i] = fill_val
        
        # Squeeze if single element
        if result.shape == (1, 1):
            return result[0, 0]
        elif result.shape[1] == 1:
            return result.ravel()
        
        return result
    
    def _normalize_index(self, idx, axis: int):
        """Normalize index to standard form."""
        max_len = self.shape[axis]
        
        if isinstance(idx, slice):
            return idx
        elif isinstance(idx, int):
            if idx < 0:
                idx += max_len
            return slice(idx, idx + 1)
        elif isinstance(idx, (list, np.ndarray)):
            return np.asarray(idx)
        else:
            return idx
    
    def _get_slice_length(self, idx, axis: int) -> int:
        """Get length of sliced dimension."""
        max_len = self.shape[axis]
        
        if isinstance(idx, slice):
            start, stop, step = idx.indices(max_len)
            return len(range(start, stop, step))
        elif isinstance(idx, (list, np.ndarray)):
            return len(idx)
        else:
            return 1
    
    def toarray(self):
        """Convert to dense array (loads entire matrix - use with caution!)."""
        return self[:, :]
    
    def __repr__(self) -> str:
        return f"<VirtualViewBackedArray {self.shape} (backed, {len(self.virtual_genes)} virtual genes)>"


class CellDataset:
    """Cell dataset with lazy virtual gene view and backed mode support.
    
    Wraps AnnData with enhanced features for single-cell analysis:
    - **Zero-Copy Gene Alignment**: Virtual genes filled on-demand
    - **Backed Mode**: Support for on-disk datasets
    - **Efficient Views**: No copy until data access
    - **PyTorch Integration**: Direct tensor conversion
    
    Args:
        adata: AnnData object
        cell_type_col: Column name for cell types in obs
        gene_name_col: Column name for gene names in var
        cell_id_col: Column name for cell IDs in obs
        duplicated_gene_policy: How to handle duplicate genes
            - 'error': Raise error if duplicates found
            - 'first': Keep first occurrence
            - 'last': Keep last occurrence
            - 'remove': Remove all duplicates
    
    Attributes:
        adata: Underlying AnnData object
        cell_type_col: Column name for cell types in obs
        gene_name_col: Column name for gene names in var
        cell_id_col: Column name for cell IDs in obs
    
    Example:
        >>> # Create from AnnData
        >>> dataset = CellDataset(
        ...     adata,
        ...     cell_type_col='cell_type',
        ...     gene_name_col='gene_name'
        ... )
        >>> 
        >>> # Align genes with virtual genes (zero-copy)
        >>> aligned = dataset.align_genes(['GENE1', 'NEW_GENE', 'GENE3'])
        >>> 
        >>> # Convert to tensor
        >>> tensor = aligned.to_tensor()
    """
    
    # Proxy attributes that delegate to underlying AnnData
    # Note: 'isbacked' and 'filename' have explicit @property definitions below
    _PROXY_ATTRIBUTES = frozenset({
        "layers", "raw", "obs", 
        "obsm", "varm", "obsp", "varp", "uns", 
        "n_obs", "obs_names",
        "to_df", "is_view"
    })

    def __init__(
        self,
        adata: ad.AnnData,
        cell_type_col: Optional[str] = None,
        gene_name_col: Optional[str] = None,
        cell_id_col: Optional[str] = None,
        duplicated_gene_policy: DuplicatedGenePolicy = 'error',
    ):
        """Initialize CellDataset from AnnData object.
        
        Args:
            adata: AnnData object
            cell_type_col: Column name for cell types in obs
            gene_name_col: Column name for gene names in var
            cell_id_col: Column name for cell IDs in obs
            duplicated_gene_policy: How to handle duplicate genes
        
        Example:
            >>> dataset = CellDataset(
            ...     adata,
            ...     cell_type_col='cell_type',
            ...     gene_name_col='gene_name',
            ...     duplicated_gene_policy='first'
            ... )
        """
        # Column names
        self.cell_type_col = cell_type_col
        self.gene_name_col = gene_name_col
        self.cell_id_col = cell_id_col
        
        # Configuration
        self._duplicated_policy = duplicated_gene_policy
        self._layer_name: Optional[str] = None
        
        # Label encoding
        self.type_to_idx: Optional[Dict[str, int]] = None
        
        # Virtual view metadata (internal)
        self._is_virtual_view = False
        self._target_genes: Optional[pd.Index] = None
        self._virtual_genes: Optional[Dict[str, float]] = None
        self._gene_indices: Optional[np.ndarray] = None
        
        # Process duplicates
        self.adata = self._process_duplicates(adata, duplicated_gene_policy)
        self._validate_columns()
        
        # Build label encoder
        if cell_type_col and cell_type_col in self.adata.obs:
            unique_types = sorted(self.adata.obs[cell_type_col].unique())
            self.type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    @classmethod
    def _initialize(cls) -> 'CellDataset':
        """Internal factory for creating uninitialized instances.
        
        Used internally for creating views and virtual views.
        External users should use __init__ directly.
        
        Returns:
            CellDataset: Uninitialized instance
        """
        # Bypass __init__ to create empty instance
        instance = cls.__new__(cls)
        
        # Initialize attributes
        instance.adata = None
        instance.cell_type_col = None
        instance.gene_name_col = None
        instance.cell_id_col = None
        instance._duplicated_policy = 'error'
        instance._layer_name = None
        instance.type_to_idx = None
        instance._is_virtual_view = False
        instance._target_genes = None
        instance._virtual_genes = None
        instance._gene_indices = None
        
        return instance
    
    # ====================================================================================
    # Core Functionality: Gene Alignment
    # ====================================================================================
    
    def align_genes(
        self, 
        target_genes: Union[List[str], pd.Index], 
        fill_value: float = 0.0
    ) -> 'CellDataset':
        """Align dataset to target gene list (lazy virtual view).
        
        **Zero-Copy Design**: 
        - Reorder/subset genes → AnnData view
        - Missing genes → Virtual view (filled on-demand)
        - Call `materialize()` for hard copy
        
        Args:
            target_genes: Target gene list (desired order)
            fill_value: Fill value for missing genes
        
        Returns:
            CellDataset: Aligned dataset (view or virtual view)
        
        Example:
            >>> # Reorder genes (view)
            >>> aligned = dataset.align_genes(['GENE1', 'GENE2', 'GENE3'])
            >>> 
            >>> # Add missing genes (virtual view)
            >>> aligned = dataset.align_genes(['GENE1', 'NEW_GENE', 'GENE3'])
            >>> 
            >>> # Materialize when needed
            >>> materialized = aligned.materialize()
        """
        target_genes = pd.Index(target_genes)
        
        # Materialize if already a virtual view
        if self._is_virtual_view:
            return self.materialize().align_genes(target_genes, fill_value)
        
        current_genes = self.gene_names

        # 1. Already aligned
        if current_genes.equals(target_genes):
            return self

        # 2. Check for missing genes
        missing_genes = target_genes.difference(current_genes)

        # 3. Build gene index mapping
        if self.gene_name_col:
            gene_to_idx = pd.Series(
                np.arange(len(self.adata.var_names)), 
                index=self.adata.var[self.gene_name_col]
            )
        else:
            gene_to_idx = pd.Series(
                np.arange(len(self.adata.var_names)), 
                index=self.adata.var_names
            )
        
        gene_indices = np.array([
            gene_to_idx.get(gene, -1) for gene in target_genes
        ])
        
        # 4. No missing genes → AnnData view
        if len(missing_genes) == 0:
            actual_indices = gene_indices  # All should be >= 0
            new_adata = self.adata[:, actual_indices]
            return self._create_from_existing(new_adata)
        
        # 5. Has missing genes → Virtual view (zero-copy!)
        return self._create_virtual_view(
            target_genes=target_genes,
            gene_indices=gene_indices,
            virtual_genes={gene: fill_value for gene in missing_genes},
        )
    
    # ====================================================================================
    # Core Functionality: Split & Sample
    # ====================================================================================
    
    def split(
        self, 
        test_size: float = 0.2, 
        stratify: Union[bool, str] = True, 
        random_state: int = 42,
        shuffle: bool = True,
        dry_run: bool = False,
        split_col: str = 'split'
    ) -> Union[Dict[str, 'CellDataset'], 'CellDataset']:
        """Split dataset into train and test sets.
        
        Args:
            test_size: Fraction for test set
            stratify: Whether to stratify split
                - True: Use cell_type_col
                - str: Use specified column
                - False: Random split
            random_state: Random seed
            shuffle: Whether to shuffle data
            dry_run: If True, only add split labels to obs without actual splitting
            split_col: Column name for split labels (used when dry_run=True)
        
        Returns:
            If dry_run=False: Dict with keys 'train' and 'test' containing datasets
            If dry_run=True: Self with split column added to obs
        
        Example:
            >>> # Actual split
            >>> splits = dataset.split(test_size=0.2, stratify=True)
            >>> train = splits['train']
            >>> test = splits['test']
            >>> 
            >>> # Dry run - only mark splits in obs
            >>> dataset = dataset.split(test_size=0.2, dry_run=True)
            >>> print(dataset.obs['split'].value_counts())
        """
        from perturblab.utils import get_logger
        from perturblab.tools.data_split import split_cells
        
        logger = get_logger()
        
        # Determine stratify labels
        stratify_labels = None
        if stratify:
            if isinstance(stratify, str):
                stratify_labels = self.adata.obs[stratify].values
            elif stratify is True and self.cell_type_col:
                stratify_labels = self.cell_types.values

        # Use decoupled split algorithm
        train_idx, test_idx = split_cells(
            n_cells=self.n_cells,
            test_size=test_size,
            stratify_labels=stratify_labels,
            random_state=random_state,
            shuffle=shuffle
        )

        # Dry run mode: only add split labels to obs
        if dry_run:
            if split_col in self.adata.obs.columns:
                logger.info(
                    f"Column '{split_col}' already exists in obs, overwriting with new split labels"
                )
            
            # Create split labels
            split_labels = np.empty(self.n_cells, dtype=object)
            split_labels[train_idx] = 'train'
            split_labels[test_idx] = 'test'
            
            # Add to obs
            self.adata.obs[split_col] = split_labels
            
            logger.info(
                f"Added split labels to obs['{split_col}']: "
                f"{len(train_idx)} train, {len(test_idx)} test"
            )
            
            return self
        
        # Normal mode: return split datasets
        return {
            'train': self[train_idx],
            'test': self[test_idx]
        }

    def sample(
        self, 
        n: Optional[int] = None, 
        frac: Optional[float] = None, 
        by_cell_type: bool = False,
        balance: bool = False,
        sampler: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        random_state: int = 42,
        replace: bool = False
    ) -> 'CellDataset':
        """Sample cells from dataset.
        
        Args:
            n: Number of cells to sample
            frac: Fraction of cells to sample
            by_cell_type: Sample within each cell type
            balance: Force equal number per cell type (requires n)
            sampler: Custom weight function
            random_state: Random seed
            replace: Sample with replacement
        
        Returns:
            CellDataset: Sampled dataset
        
        Example:
            >>> # Sample 1000 cells
            >>> sampled = dataset.sample(n=1000)
            >>> 
            >>> # Balanced sampling per cell type
            >>> balanced = dataset.sample(n=100, balance=True)
        """
        from perturblab.tools.data_split import (
            sample_cells_simple,
            sample_cells_weighted,
            sample_cells_by_group
        )
        
        # A. Custom weight sampling
        if sampler:
            weights = sampler(self.adata.obs)
            sel_idx = sample_cells_weighted(
                n_cells=self.n_cells,
                weights=weights,
                n=n,
                frac=frac,
                random_state=random_state,
                replace=replace
            )
            return self[sel_idx]
        
        # B. By cell type
        if by_cell_type or balance:
            if not self.cell_type_col:
                raise ValueError("Requires cell_type_col")
            
            # Build group indices dict
            groups = self.adata.obs.groupby(self.cell_type_col)
            group_indices = groups.indices
            
            sel_idx = sample_cells_by_group(
                group_indices=group_indices,
                n=n,
                frac=frac,
                balance=balance,
                random_state=random_state,
                replace=replace
            )
            return self[sel_idx]
        
        # C. Simple random
        sel_idx = sample_cells_simple(
            n_cells=self.n_cells,
            n=n,
            frac=frac,
            random_state=random_state,
            replace=replace
        )
        return self[sel_idx]
    
    # ====================================================================================
    # Core Functionality: Data Loading
    # ====================================================================================
    
    def to_tensor(
        self, 
        idx: Union[slice, np.ndarray, None] = None, 
        sparse: bool = False
    ) -> Union[torch.Tensor, torch.sparse.Tensor]:
        """Convert to PyTorch tensor.
        
        Args:
            idx: Optional row indices to slice
            sparse: If True, return sparse COO tensor
        
        Returns:
            torch.Tensor or torch.sparse.Tensor
        
        Note:
            Virtual genes are automatically filled.
            For backed data, loads requested data from disk.
        
        Example:
            >>> # Dense tensor
            >>> tensor = dataset.to_tensor()
            >>> 
            >>> # Sparse tensor
            >>> sparse_tensor = dataset.to_tensor(sparse=True)
        """
        target = self.X
        if idx is not None:
            target = target[idx]

        # 1. Dense tensor
        if not sparse:
            if scipy.sparse.issparse(target):
                data = target.toarray()
            elif hasattr(target, "shape") and hasattr(target, "__array__") and not isinstance(target, np.ndarray):
                data = np.array(target)
            else:
                data = np.array(target)
            return torch.from_numpy(data).float()
        
        # 2. Sparse tensor
        else:
            if not scipy.sparse.issparse(target):
                tmp = scipy.sparse.coo_matrix(target)
            else:
                tmp = target.tocoo()
            values = torch.from_numpy(tmp.data).float()
            indices = torch.from_numpy(np.vstack((tmp.row, tmp.col))).long()
            return torch.sparse_coo_tensor(indices, values, tmp.shape)
    
    def materialize(self) -> 'CellDataset':
        """Materialize virtual view into actual AnnData.
        
        For backed data, loads data into memory.
        Regular views returned as-is.
        
        Returns:
            CellDataset: Materialized dataset
        
        Example:
            >>> virtual = dataset.align_genes(['GENE1', 'NEW_GENE'])
            >>> concrete = virtual.materialize()
        """
        if not self._is_virtual_view:
            return self
        
        # Get base X
        if self._layer_name:
            base_X = self.adata.layers[self._layer_name]
        else:
            base_X = self.adata.X
        
        # Materialize virtual genes
        X_materialized = self._materialize_virtual_X(base_X)
        
        # Build var DataFrame
        var_df = pd.DataFrame(index=self._target_genes)
        if self.gene_name_col:
            var_df[self.gene_name_col] = self._target_genes.values
        
        # Copy obs and metadata
        obs_copy = self.adata.obs.copy()
        uns_copy = dict(self.adata.uns) if self.adata.uns else {}
        
        # Materialize obsm if backed
        obsm_copy = {}
        for key in self.adata.obsm.keys():
            obsm_data = self.adata.obsm[key]
            if hasattr(obsm_data, 'toarray'):
                obsm_copy[key] = obsm_data.toarray()
            else:
                obsm_copy[key] = np.array(obsm_data)
        
        # Create new AnnData
        new_adata = ad.AnnData(
            X=X_materialized,
            obs=obs_copy,
            var=var_df,
            uns=uns_copy,
            obsm=obsm_copy if obsm_copy else None,
        )
        
        instance = CellDataset(
            new_adata,
            cell_type_col=self.cell_type_col,
            gene_name_col=self.gene_name_col,
            cell_id_col=self.cell_id_col,
            duplicated_gene_policy=self._duplicated_policy,
        )
        instance._layer_name = self._layer_name
        instance.type_to_idx = self.type_to_idx
        
        return instance
    
    def copy(self) -> 'CellDataset':
        """Create hard copy (materializes virtual views).
        
        Returns:
            CellDataset: Fully materialized copy
        
        Example:
            >>> copied = dataset.copy()
        """
        if self._is_virtual_view:
            return self.materialize()
        
        instance = CellDataset(
            self.adata.copy(),
            cell_type_col=self.cell_type_col,
            gene_name_col=self.gene_name_col,
            cell_id_col=self.cell_id_col,
            duplicated_gene_policy=self._duplicated_policy,
        )
        instance._layer_name = self._layer_name
        instance.type_to_idx = self.type_to_idx.copy() if self.type_to_idx else None
        
        return instance
    
    # ====================================================================================
    # Properties
    # ====================================================================================
    
    @property
    def is_virtual_view(self) -> bool:
        """Check if this is a virtual view (has virtual genes)."""
        return self._is_virtual_view
    
    @property
    def isbacked(self) -> bool:
        """Check if underlying AnnData is backed (on-disk)."""
        return self.adata.isbacked if self.adata else False
    
    @property
    def filename(self):
        """Path to backing file (if backed)."""
        return self.adata.filename if self.isbacked else None
    
    def use_layer(self, layer_name: str) -> 'CellDataset':
        """Switch to a different layer for X access.
        
        Args:
            layer_name: Layer name in adata.layers
        
        Returns:
            Self for method chaining
        """
        self._layer_name = layer_name
        return self

    @property
    def X(self):
        """Get expression matrix (fills virtual genes on-demand).
        
        For backed AnnData, returns a lazy wrapper.
        """
        # Get base data
        if self._layer_name:
            base_X = self.adata.layers[self._layer_name]
        else:
            base_X = self.adata.X
        
        # Not a virtual view
        if not self._is_virtual_view:
            return base_X
        
        # Virtual view: lazy wrapper for backed, immediate for in-memory
        if self.isbacked:
            return _VirtualViewBackedArray(
                base_X=base_X,
                gene_indices=self._gene_indices,
                target_genes=self._target_genes,
                virtual_genes=self._virtual_genes,
                n_cells=self.adata.n_obs,
            )
        
        return self._materialize_virtual_X(base_X)

    @property
    def gene_names(self) -> pd.Index:
        """Get gene names (includes virtual genes if virtual view)."""
        if self._is_virtual_view:
            return self._target_genes
        
        if self.gene_name_col:
            return pd.Index(self.adata.var[self.gene_name_col])
        return self.adata.var_names
    
    @property
    def var(self) -> pd.DataFrame:
        """Get gene metadata (includes virtual genes if virtual view)."""
        if self._is_virtual_view:
            var_df = pd.DataFrame(index=self._target_genes)
            if self.gene_name_col:
                var_df[self.gene_name_col] = self._target_genes.values
            var_df['is_virtual'] = [idx < 0 for idx in self._gene_indices]
            return var_df
        
        return self.adata.var

    @property
    def cell_types(self):
        """Get cell type labels."""
        if self.cell_type_col:
            return self.adata.obs[self.cell_type_col]
        return None
    
    @property
    def idx_to_type(self) -> Dict[int, str]:
        """Reverse mapping from index to cell type."""
        if self.type_to_idx:
            return {v: k for k, v in self.type_to_idx.items()}
        return {}
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape (n_cells, n_genes)."""
        if self._is_virtual_view:
            return (self.adata.n_obs, len(self._target_genes))
        return self.adata.shape
    
    @property
    def n_genes(self) -> int:
        """Number of genes (includes virtual genes)."""
        if self._is_virtual_view:
            return len(self._target_genes)
        return self.adata.n_vars
    
    @property
    def n_vars(self) -> int:
        """Alias for n_genes (AnnData compatibility)."""
        return self.n_genes

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.adata.n_obs

    @property
    def var_names(self) -> pd.Index:
        """Alias for gene_names."""
        return self.gene_names
    
    # ====================================================================================
    # Private Methods
    # ====================================================================================
    
    def _create_from_existing(self, new_adata: ad.AnnData) -> 'CellDataset':
        """Create new dataset from existing adata (clears virtual view)."""
        instance = CellDataset(
            new_adata,
            cell_type_col=self.cell_type_col,
            gene_name_col=self.gene_name_col,
            cell_id_col=self.cell_id_col,
            duplicated_gene_policy=self._duplicated_policy,
        )
        instance._layer_name = self._layer_name
        instance.type_to_idx = self.type_to_idx
        return instance
    
    def _create_virtual_view(
        self,
        target_genes: pd.Index,
        gene_indices: np.ndarray,
        virtual_genes: Dict[str, float],
    ) -> 'CellDataset':
        """Create virtual view with virtual genes."""
        instance = CellDataset._initialize()
        instance.adata = self.adata
        instance.cell_type_col = self.cell_type_col
        instance.gene_name_col = self.gene_name_col
        instance.cell_id_col = self.cell_id_col
        instance._duplicated_policy = self._duplicated_policy
        instance._layer_name = self._layer_name
        instance.type_to_idx = self.type_to_idx
        
        # Virtual view metadata
        instance._is_virtual_view = True
        instance._target_genes = target_genes
        instance._virtual_genes = virtual_genes
        instance._gene_indices = gene_indices
        
        return instance
    
    def _process_duplicates(
        self, 
        adata: ad.AnnData, 
        policy: DuplicatedGenePolicy
    ) -> ad.AnnData:
        """Process duplicate gene names."""
        if self.gene_name_col:
            names = adata.var[self.gene_name_col]
        else:
            names = adata.var_names
        
        if names.is_unique:
            return adata
        
        if policy == 'error':
            raise ValueError("Duplicate genes found")
        
        keep_map = {'first': 'first', 'last': 'last', 'remove': False}
        return adata[:, ~names.duplicated(keep=keep_map[policy])]
    
    def _validate_columns(self):
        """Validate that specified columns exist."""
        if self.cell_type_col and self.cell_type_col not in self.adata.obs:
            raise KeyError(f"Missing column: {self.cell_type_col}")
    
    def _materialize_virtual_X(self, base_X):
        """Materialize virtual view into actual array."""
        n_cells = self.adata.n_obs
        n_target_genes = len(self._target_genes)
        
        is_sparse = scipy.sparse.issparse(base_X)
        dtype = base_X.dtype if hasattr(base_X, 'dtype') else np.float32
        
        if is_sparse:
            rows, cols, data = [], [], []
            
            for target_idx, source_idx in enumerate(self._gene_indices):
                if source_idx >= 0:
                    col_data = base_X[:, source_idx]
                    if scipy.sparse.issparse(col_data):
                        col_data = col_data.toarray().ravel()
                    else:
                        col_data = np.asarray(col_data).ravel()
                    
                    nonzero_rows = np.where(col_data != 0)[0]
                    rows.extend(nonzero_rows)
                    cols.extend([target_idx] * len(nonzero_rows))
                    data.extend(col_data[nonzero_rows])
                else:
                    gene_name = self._target_genes[target_idx]
                    fill_val = self._virtual_genes[gene_name]
                    if fill_val != 0:
                        rows.extend(range(n_cells))
                        cols.extend([target_idx] * n_cells)
                        data.extend([fill_val] * n_cells)
            
            result = scipy.sparse.coo_matrix(
                (data, (rows, cols)), 
                shape=(n_cells, n_target_genes),
                dtype=dtype
            ).tocsr()
        else:
            result = np.zeros((n_cells, n_target_genes), dtype=dtype)
            
            for target_idx, source_idx in enumerate(self._gene_indices):
                if source_idx >= 0:
                    result[:, target_idx] = np.asarray(base_X[:, source_idx]).ravel()
                else:
                    gene_name = self._target_genes[target_idx]
                    fill_val = self._virtual_genes[gene_name]
                    result[:, target_idx] = fill_val
        
        return result
    
    # ====================================================================================
    # Magic Methods
    # ====================================================================================
    
    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, index) -> 'CellDataset':
        """Slice cells (maintains virtual view structure)."""
        if self._is_virtual_view:
            return self._create_virtual_view(
                target_genes=self._target_genes,
                gene_indices=self._gene_indices,
                virtual_genes=self._virtual_genes,
            )._update_adata(self.adata[index])
        
        return self._create_from_existing(self.adata[index])
    
    def _update_adata(self, new_adata: ad.AnnData) -> 'CellDataset':
        """Update adata reference (for virtual view slicing)."""
        self.adata = new_adata
        return self

    def __getattr__(self, name):
        """Proxy attributes to underlying AnnData."""
        if name in self._PROXY_ATTRIBUTES:
            return getattr(self.adata, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        n_cells, n_genes = self.shape
        
        # Determine status
        if self.isbacked:
            status = f"backed at '{self.filename}'"
        elif self._is_virtual_view:
            status = "virtual view"
        elif self.adata.is_view:
            status = "view"
        else:
            status = "copy"
        
        parts = [
            f"CellDataset with n_obs × n_vars = {n_cells} × {n_genes}",
        ]
        
        if self._is_virtual_view:
            n_virtual = sum(1 for idx in self._gene_indices if idx < 0)
            parts.append(f"    {n_virtual} virtual genes")
        
        parts.append(f"    {status}")
        
        if self.cell_type_col:
            n_types = len(self.cell_types.unique())
            parts.append(f"    {n_types} cell types")
        
        return "\n".join(parts)
    
    # ====================================================================================
    # Utility: DataLoader Collate Function
    # ====================================================================================
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for torch DataLoader.
        
        Args:
            batch: List of samples (dict or tensor)
        
        Returns:
            Batched data as dict or stacked tensor
        
        Example:
            >>> from torch.utils.data import DataLoader
            >>> loader = DataLoader(
            ...     dataset,
            ...     batch_size=32,
            ...     collate_fn=CellDataset.collate_fn
            ... )
        """
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch)
        
        elem = batch[0]
        collated = {}
        for key in elem:
            if isinstance(elem[key], torch.Tensor):
                collated[key] = torch.stack([d[key] for d in batch])
            elif isinstance(elem[key], (int, float)):
                collated[key] = torch.tensor([d[key] for d in batch])
            else:
                collated[key] = [d[key] for d in batch]
        return collated
