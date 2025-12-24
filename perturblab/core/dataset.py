"""Base dataset classes for PerturbLab.

Provides abstract base classes for dataset implementations:
- Dataset: Most abstract dataset interface (no enforced methods)
- TorchDataset: PyTorch-compatible dataset (requires __len__ and __getitem__)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from torch.utils.data import Dataset as _BaseTorchDataset

T = TypeVar("T")

class Dataset(ABC, Generic[T]):
    """Abstract base class for all datasets.
    
    Generic type T represents the underlying data type (e.g., CellData, AnnData).
    
    Subclasses must implement:
    - data property: Returns the underlying data object of type T
    """
    
    @property
    @abstractmethod
    def data(self) -> T:
        """Get the underlying data object.
        
        Returns
        -------
        T
            The underlying data object.
        """
        pass

class TorchDataset(Dataset, _BaseTorchDataset, ABC, Generic[T]):
    """PyTorch-compatible dataset (with torch.utils.data.Dataset).
    
    This version includes torch.utils.data.Dataset as a parent class,
    ensuring full compatibility with PyTorch's ecosystem including
    DataLoader, Sampler, and all other torch.utils.data utilities.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        """Get item by integer index."""
        pass
