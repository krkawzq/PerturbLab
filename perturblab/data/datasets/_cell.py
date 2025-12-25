"""Cell dataset wrapper for PyTorch integration.

Provides PyTorch-compatible dataset interface for CellData objects.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import torch

from perturblab.core import TorchDataset
from perturblab.types import CellData
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["CellDataset"]


class CellDataset(TorchDataset[CellData]):
    """PyTorch-compatible dataset wrapper for CellData.

    Wraps a CellData object to provide PyTorch DataLoader compatibility.
    The generic type is constrained to CellData, indicating this dataset
    holds and operates on CellData objects.

    Note:
        While the dataset holds CellData (generic type T=CellData),
        __getitem__ returns training samples as Dict[str, torch.Tensor] for
        PyTorch compatibility.

    Each item returned by __getitem__ is a dictionary with:
        - 'x': Gene expression tensor (shape: [n_genes])
        - 'cell_id': Cell identifier (if available)
        - 'cell_type': Cell type label (if available)
        - Additional metadata as needed

    Args:
        cell_data: CellData object to wrap.
        return_sparse: If True, returns sparse tensors. If False, converts to dense.
            Defaults to False.
        return_metadata: If True, includes cell metadata in returned dict.
            Defaults to True.
        device: Device for tensors ('cpu' or 'cuda'). Defaults to 'cpu'.

    Examples:
        >>> from perturblab.data.datasets import CellDataset
        >>> from perturblab.types import CellData
        >>> from torch.utils.data import DataLoader
        >>>
        >>> # Create CellData
        >>> cell_data = CellData(adata, cell_type_col='cell_type')
        >>>
        >>> # Wrap in CellDataset
        >>> dataset = CellDataset(cell_data)
        >>>
        >>> # Access underlying data
        >>> data = dataset.data  # Returns CellData
        >>>
        >>> # Use with DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> for batch in loader:
        ...     x = batch['x']  # Gene expression
        ...     cell_types = batch['cell_type']  # Cell type labels
        ...     # Training logic
    """

    def __init__(
        self,
        cell_data: CellData,
        *,
        return_sparse: bool = False,
        return_metadata: bool = True,
        device: str = "cpu",
    ):
        """Initialize CellDataset.

        Parameters
        ----------
        cell_data : CellData
            CellData object to wrap.
        return_sparse : bool, default=False
            Whether to return sparse tensors.
        return_metadata : bool, default=True
            Whether to include metadata in returned items.
        device : str, default='cpu'
            Device for tensors.
        """
        super().__init__()

        self._cell_data = cell_data
        self.return_sparse = return_sparse
        self.return_metadata = return_metadata
        self.device = device

    @property
    def data(self) -> CellData:
        """Get the underlying CellData object.

        Returns
        -------
        CellData
            The wrapped CellData object.

        Examples
        --------
        >>> dataset = CellDataset(cell_data)
        >>> cell_data = dataset.data  # Access underlying data
        >>> print(cell_data.n_genes)
        """
        return self._cell_data

    def __len__(self) -> int:
        """Return number of cells in the dataset.

        Returns
        -------
        int
            Number of cells.
        """
        return len(self._cell_data)

    def __getitem__(self, index: int | slice) -> dict[str, torch.Tensor | str | int | list]:
        """Get cell(s) by index.

        Parameters
        ----------
        index : int or slice
            Cell index or slice.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'x': Gene expression tensor
            - 'index': Original cell index
            - 'cell_id': Cell identifier (if return_metadata=True)
            - 'cell_type': Cell type (if return_metadata=True)
            - 'cell_type_idx': Encoded cell type index (if available)

        Examples
        --------
        >>> dataset = CellDataset(cell_data)
        >>>
        >>> # Get single cell
        >>> item = dataset[0]
        >>> print(item['x'].shape)  # [n_genes]
        >>>
        >>> # Get multiple cells
        >>> items = dataset[0:10]
        >>> print(items['x'].shape)  # [10, n_genes]
        """
        # Get expression data
        if isinstance(index, slice):
            # Slice returns 2D array
            expr_data = self._cell_data.X[index]
            batch_size = expr_data.shape[0]
            is_batch = True
        else:
            # Single index returns 1D array
            expr_data = self._cell_data.X[index]
            if expr_data.ndim == 1:
                expr_data = expr_data.reshape(1, -1)
            batch_size = 1
            is_batch = False

        # Convert to dense if needed
        if scipy.sparse.issparse(expr_data):
            if self.return_sparse:
                # Convert to sparse COO tensor
                coo = scipy.sparse.coo_matrix(expr_data)
                indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
                values = torch.FloatTensor(coo.data)
                shape = torch.Size(coo.shape)
                x_tensor = torch.sparse_coo_tensor(indices, values, shape, device=self.device)
            else:
                # Convert to dense
                x_tensor = torch.tensor(
                    expr_data.toarray(), dtype=torch.float32, device=self.device
                )
        else:
            x_tensor = torch.tensor(np.asarray(expr_data), dtype=torch.float32, device=self.device)

        # Squeeze if single item
        if not is_batch:
            x_tensor = x_tensor.squeeze(0)

        # Build result dict
        result = {
            "x": x_tensor,
            "index": index if isinstance(index, int) else list(range(*index.indices(len(self)))),
        }

        # Add metadata if requested
        if self.return_metadata:
            # Cell IDs
            if (
                self._cell_data.cell_id_col
                and self._cell_data.cell_id_col in self._cell_data.obs.columns
            ):
                cell_ids = self._cell_data.obs[self._cell_data.cell_id_col].iloc[index]
                result["cell_id"] = cell_ids.tolist() if is_batch else cell_ids

            # Cell types
            if (
                self._cell_data.cell_type_col is not None
                and self._cell_data.cell_type_col in self._cell_data.obs.columns
            ):
                cell_types = self._cell_data.obs[self._cell_data.cell_type_col].iloc[index]
                result["cell_type"] = cell_types.tolist() if is_batch else cell_types

                # Encoded cell type indices (requires both cell_type_col and type_to_idx)
                if (
                    self._cell_data.cell_type_col is not None
                    and self._cell_data.type_to_idx is not None
                ):
                    if is_batch:
                        type_indices = [
                            self._cell_data.type_to_idx.get(ct, -1) for ct in cell_types
                        ]
                        result["cell_type_idx"] = torch.tensor(
                            type_indices, dtype=torch.long, device=self.device
                        )
                    else:
                        type_idx = self._cell_data.type_to_idx.get(cell_types, -1)
                        result["cell_type_idx"] = torch.tensor(
                            type_idx, dtype=torch.long, device=self.device
                        )

        return result

    @property
    def gene_names(self) -> list[str]:
        """Get gene names from the dataset.

        Returns
        -------
        list of str
            List of gene names.
        """
        return self._cell_data.gene_names.tolist()

    @property
    def n_genes(self) -> int:
        """Get number of genes.

        Returns
        -------
        int
            Number of genes.
        """
        return self._cell_data.n_genes

    @property
    def n_cells(self) -> int:
        """Get number of cells.

        Returns
        -------
        int
            Number of cells.
        """
        return len(self._cell_data)

    @property
    def cell_types(self) -> list[str] | None:
        """Get unique cell types.

        Returns
        -------
        list of str or None
            List of unique cell types, or None if not available.
        """
        if (
            self._cell_data.cell_type_col is not None
            and self._cell_data.cell_type_col in self._cell_data.obs.columns
        ):
            return self._cell_data.obs[self._cell_data.cell_type_col].unique().tolist()
        return None

    def split(
        self,
        *,
        use_split: bool = True,
        split_col: str = "split",
        force_compute: bool = False,
        compute_only: bool = False,
        test_size: float = 0.2,
        stratify: bool | str = True,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> dict[str, CellDataset]:
        """Split dataset into train/test subsets.

        This method supports multiple modes:
        1. use_split=True: Use existing split column (if available)
        2. force_compute=True: Compute new split (overwrites existing)
        3. compute_only=True: Compute split but don't store in obs

        Returns lazy view subsets - no data copying until access.

        Parameters
        ----------
        use_split : bool, default=True
            Use existing split column if available.
            If False or column doesn't exist, computes new split.
        split_col : str, default='split'
            Column name for split labels in obs.
        force_compute : bool, default=False
            Force compute new split even if one exists.
            Overwrites existing split column.
        compute_only : bool, default=False
            Compute split but don't store in obs.
            Returns temporary subsets without modifying underlying data.
        test_size : float, default=0.2
            Fraction for test set (used if computing new split).
        stratify : bool or str, default=True
            Stratification strategy (used if computing new split):
            - True: Use cell_type_col
            - str: Use specified column
            - False: Random split
        random_state : int, default=42
            Random seed for reproducibility.
        shuffle : bool, default=True
            Whether to shuffle before splitting.

        Returns
        -------
        dict
            Dictionary with keys 'train' and 'test' containing CellDataset subsets.
            Each subset is a lazy view (zero-copy until accessed).

        Examples
        --------
        >>> from perturblab.data.datasets import CellDataset
        >>>
        >>> dataset = CellDataset(cell_data)
        >>>
        >>> # Use existing split (if available)
        >>> splits = dataset.split(use_split=True)
        >>> train_ds = splits['train']
        >>> test_ds = splits['test']
        >>>
        >>> # Force compute new split
        >>> splits = dataset.split(force_compute=True, test_size=0.3)
        >>>
        >>> # Compute temporary split (don't modify data)
        >>> splits = dataset.split(compute_only=True, test_size=0.1)
        """
        # Check if we should use existing split
        if use_split and not force_compute and not compute_only:
            if split_col in self._cell_data.obs.columns:
                # Use existing split column
                split_labels = self._cell_data.obs[split_col]

                # Get unique split names
                unique_splits = split_labels.unique()

                # Build subsets using boolean indexing
                subsets = {}
                for split_name in unique_splits:
                    # Get boolean mask
                    mask = (split_labels == split_name).values
                    indices = np.where(mask)[0]

                    # Create subset view (zero-copy)
                    subset_data = self._cell_data[indices]

                    # Wrap in CellDataset
                    subsets[split_name] = CellDataset(
                        subset_data,
                        return_sparse=self.return_sparse,
                        return_metadata=self.return_metadata,
                        device=self.device,
                    )

                return subsets

        # Need to compute new split
        if compute_only:
            # Compute split without modifying obs
            split_result = self._cell_data.split(
                test_size=test_size,
                stratify=stratify,
                random_state=random_state,
                shuffle=shuffle,
                dry_run=True,  # Add split column
                split_col=f"_temp_split_{random_state}",  # Temp column
            )

            # Get the temporary split column
            temp_col = f"_temp_split_{random_state}"
            split_labels = split_result.obs[temp_col]

            # Build subsets
            subsets = {}
            for split_name in split_labels.unique():
                mask = (split_labels == split_name).values
                indices = np.where(mask)[0]
                subset_data = split_result[indices]

                # Remove temporary split column from view
                subset_data.obs = subset_data.obs.drop(columns=[temp_col])

                subsets[split_name] = CellDataset(
                    subset_data,
                    return_sparse=self.return_sparse,
                    return_metadata=self.return_metadata,
                    device=self.device,
                )

            # Clean up temp column from original data
            if temp_col in self._cell_data.obs.columns:
                self._cell_data.obs.drop(columns=[temp_col], inplace=True)

            return subsets

        else:
            # Compute and store split (or force overwrite)
            if force_compute or split_col not in self._cell_data.obs.columns:
                # Compute split and store in obs
                self._cell_data.split(
                    test_size=test_size,
                    stratify=stratify,
                    random_state=random_state,
                    shuffle=shuffle,
                    dry_run=True,  # Only add labels, don't actually split
                    split_col=split_col,
                )

            # Now use the split column
            split_labels = self._cell_data.obs[split_col]

            subsets = {}
            for split_name in split_labels.unique():
                mask = (split_labels == split_name).values
                indices = np.where(mask)[0]
                subset_data = self._cell_data[indices]

                subsets[split_name] = CellDataset(
                    subset_data,
                    return_sparse=self.return_sparse,
                    return_metadata=self.return_metadata,
                    device=self.device,
                )

            return subsets

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CellDataset(n_cells={self.n_cells}, n_genes={self.n_genes}, "
            f"sparse={self.return_sparse}, device={self.device})"
        )
