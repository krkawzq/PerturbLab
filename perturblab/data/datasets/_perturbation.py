"""Perturbation dataset for PyTorch integration.

Provides PyTorch-compatible dataset interface for PerturbationData objects
with perturbation-specific splitting strategies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from perturblab.types import PerturbationData
from perturblab.utils import get_logger

from ._cell import CellDataset

logger = get_logger()

__all__ = ["PerturbationDataset"]


class PerturbationDataset(CellDataset):
    """PyTorch-compatible dataset for perturbation analysis.

    Extends CellDataset to work with PerturbationData, adding
    perturbation-specific splitting strategies for generalization testing.

    Generic type is constrained to PerturbationData, indicating this dataset
    specifically holds and operates on perturbation experiment data.

    Args:
        perturbation_data: PerturbationData object to wrap.
        return_sparse: If True, returns sparse tensors.
        return_metadata: If True, includes cell metadata in returned dict.
        device: Device for tensors ('cpu' or 'cuda').

    Examples:
        >>> from perturblab.data.datasets import PerturbationDataset
        >>> from perturblab.types import PerturbationData
        >>> from torch.utils.data import DataLoader
        >>>
        >>> # Create PerturbationData
        >>> pert_data = PerturbationData(
        ...     adata,
        ...     perturbation_col='condition',
        ...     control_label='ctrl'
        ... )
        >>>
        >>> # Wrap in PerturbationDataset
        >>> dataset = PerturbationDataset(pert_data)
        >>>
        >>> # Gene-level split for generalization testing
        >>> splits = dataset.split(
        ...     split_type='simulation',
        ...     test_size=0.2,
        ...     train_gene_fraction=0.7
        ... )
        >>>
        >>> train_ds = splits['train']
        >>> test_ds = splits['test']
    """

    def __init__(
        self,
        perturbation_data: PerturbationData,
        *,
        return_sparse: bool = False,
        return_metadata: bool = True,
        device: str = "cpu",
    ):
        """Initialize PerturbationDataset.

        Parameters
        ----------
        perturbation_data : PerturbationData
            PerturbationData object to wrap.
        return_sparse : bool, default=False
            Whether to return sparse tensors.
        return_metadata : bool, default=True
            Whether to include metadata in returned items.
        device : str, default='cpu'
            Device for tensors.
        """
        # Call parent with PerturbationData (which is a subclass of CellData)
        super().__init__(
            cell_data=perturbation_data,
            return_sparse=return_sparse,
            return_metadata=return_metadata,
            device=device,
        )

    @property
    def data(self) -> PerturbationData:
        """Get the underlying PerturbationData object.

        Returns
        -------
        PerturbationData
            The wrapped PerturbationData object.

        Examples
        --------
        >>> dataset = PerturbationDataset(pert_data)
        >>> pert_data = dataset.data  # Access underlying PerturbationData
        >>> print(pert_data.perturbations)
        """
        return self._cell_data  # type: ignore

    @property
    def perturbations(self) -> "pd.Series":  # type: ignore
        """Get perturbation labels.

        Returns
        -------
        pd.Series
            Series of perturbation labels.
        """
        return self.data.perturbations

    @property
    def unique_perturbations(self) -> List[str]:
        """Get unique perturbation labels.

        Returns
        -------
        list of str
            List of unique perturbations (excluding controls and ignored).
        """
        import numpy as np

        valid_mask = ~self.data.perturbations.isin(self.data.ignore_labels)
        all_perts = self.data.perturbations[valid_mask].unique()

        # Exclude controls
        pert_only = [p for p in all_perts if p not in self.data.control_labels]

        return sorted(pert_only)

    def split(
        self,
        *,
        split_type: str = "simple",
        use_split: bool = True,
        split_col: str = "split",
        force_compute: bool = False,
        compute_only: bool = False,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify: Union[bool, str] = False,
        random_state: int = 42,
        **kwargs,
    ) -> Dict[str, "PerturbationDataset"]:
        """Split dataset with perturbation-specific strategies.

        Supports multiple splitting strategies:
        - 'simple': Random split of all perturbations
        - 'simulation': Gene-level split (test on unseen genes)
        - 'combo_seen0/1/2': Combo perturbations with 0/1/2 seen genes
        - 'no_test': Only train/val split

        Parameters
        ----------
        split_type : str, default='simple'
            Splitting strategy:
            - 'simple': Random perturbation split
            - 'simulation': Gene-level split for generalization
            - 'simulation_single': Gene-level for single perturbations
            - 'combo_seen0': Test combos where both genes are unseen
            - 'combo_seen1': Test combos where one gene is unseen
            - 'combo_seen2': Test combos where both genes are seen
            - 'no_test': Only train/val, no test set
        use_split : bool, default=True
            Use existing split column if available (ignored if force_compute=True).
        split_col : str, default='split'
            Column name for split labels.
        force_compute : bool, default=False
            Force compute new split even if one exists.
        compute_only : bool, default=False
            Compute split but don't store in obs (temporary split).
        test_size : float, default=0.2
            Fraction for test set.
        val_size : float, optional
            Fraction for validation set. If None, no val set.
        stratify : bool or str, default=False
            Stratification for simple split only.
        random_state : int, default=42
            Random seed for reproducibility.
        **kwargs
            Additional arguments for specific split strategies:
            - train_gene_fraction: For gene-level splits (default: 0.7)
            - test_perts: Explicit list of test perturbations
            - test_genes: Explicit list of test genes

        Returns
        -------
        dict
            Dictionary with keys 'train', 'val' (optional), 'test' containing
            PerturbationDataset subsets (zero-copy views).

        Examples
        --------
        >>> from perturblab.data.datasets import PerturbationDataset
        >>>
        >>> dataset = PerturbationDataset(pert_data)
        >>>
        >>> # Simple random split (use existing if available)
        >>> splits = dataset.split(split_type='simple', use_split=True)
        >>> train_ds = splits['train']
        >>> test_ds = splits['test']
        >>>
        >>> # Gene-level split for generalization testing
        >>> splits = dataset.split(
        ...     split_type='simulation',
        ...     test_size=0.2,
        ...     val_size=0.1,
        ...     train_gene_fraction=0.7,
        ...     force_compute=True
        ... )
        >>>
        >>> # Combo perturbation split (test on unseen gene combos)
        >>> splits = dataset.split(
        ...     split_type='combo_seen0',  # Both genes unseen
        ...     test_size=0.15
        ... )
        >>>
        >>> # Temporary split (don't modify data)
        >>> splits = dataset.split(
        ...     split_type='simulation',
        ...     compute_only=True,
        ...     test_size=0.2
        ... )
        """
        import numpy as np

        # Check if we should use existing split
        if use_split and not force_compute and not compute_only:
            if split_col in self.data.obs.columns:
                logger.info(f"Using existing split column '{split_col}'")

                # Use existing split (delegate to parent)
                # This returns Dict[str, CellDataset]
                cell_splits = super().split(
                    use_split=True,
                    split_col=split_col,
                    force_compute=False,
                    compute_only=False,
                )

                # Convert CellDataset to PerturbationDataset
                subsets = {}
                for split_name, cell_ds in cell_splits.items():
                    # Wrap the CellData view in PerturbationDataset
                    subsets[split_name] = PerturbationDataset(
                        cell_ds.data,  # This is a CellData view, cast to PerturbationData
                        return_sparse=self.return_sparse,
                        return_metadata=self.return_metadata,
                        device=self.device,
                    )

                return subsets

        # Need to compute new split using PerturbationData's method
        if compute_only:
            # Compute temporary split
            temp_col = f"_temp_split_{random_state}"

            # Call PerturbationData.split with dry_run=True
            self.data.split(
                split_type=split_type,
                test_size=test_size,
                val_size=val_size,
                stratify=stratify,
                seed=random_state,
                dry_run=True,  # Only add labels
                split_col=temp_col,
                **kwargs,
            )

            # Build temporary subsets
            split_labels = self.data.obs[temp_col]
            subsets = {}

            for split_name in split_labels.unique():
                mask = (split_labels == split_name).values
                indices = np.where(mask)[0]
                subset_data = self.data[indices]

                # Remove temp column from view
                subset_data.obs = subset_data.obs.drop(columns=[temp_col])

                subsets[split_name] = PerturbationDataset(
                    subset_data,
                    return_sparse=self.return_sparse,
                    return_metadata=self.return_metadata,
                    device=self.device,
                )

            # Clean up temp column
            if temp_col in self.data.obs.columns:
                self.data.obs.drop(columns=[temp_col], inplace=True)

            return subsets

        else:
            # Compute and store split
            if force_compute or split_col not in self.data.obs.columns:
                logger.info(f"Computing new split with strategy '{split_type}'")

                # Call PerturbationData.split
                self.data.split(
                    split_type=split_type,
                    test_size=test_size,
                    val_size=val_size,
                    stratify=stratify,
                    seed=random_state,
                    dry_run=True,  # Only add labels
                    split_col=split_col,
                    **kwargs,
                )

            # Build subsets from split column
            split_labels = self.data.obs[split_col]
            subsets = {}

            for split_name in split_labels.unique():
                mask = (split_labels == split_name).values
                indices = np.where(mask)[0]
                subset_data = self.data[indices]

                subsets[split_name] = PerturbationDataset(
                    subset_data,
                    return_sparse=self.return_sparse,
                    return_metadata=self.return_metadata,
                    device=self.device,
                )

            return subsets

    def __repr__(self) -> str:
        """String representation."""
        n_perts = len(self.unique_perturbations)
        return (
            f"PerturbationDataset(n_cells={self.n_cells}, n_genes={self.n_genes}, "
            f"n_perturbations={n_perts}, sparse={self.return_sparse}, device={self.device})"
        )
