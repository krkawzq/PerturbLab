"""Base interfaces for Model Input/Output data structures.

This module provides the ModelIO base class, which endows dataclasses with
PyTorch-specific capabilities like device movement, gradient detachment,
and dictionary-like access patterns.
"""

from __future__ import annotations

import copy
from abc import ABC
from dataclasses import asdict, fields, replace
from typing import Any, Dict, Iterator, Optional, Type, TypeVar, Union

import torch

__all__ = ["ModelIO"]

T = TypeVar("T", bound="ModelIO")

class ModelIO(ABC):
    """Base class for all Model Input and Output data structures.

    Inheriting from this class automatically provides tensor management
    utilities (to, detach, cpu, cuda) and dict-like compatibility.

    Usage
    -----
    @dataclass
    class MyInput(ModelIO):
        x: torch.Tensor
        mask: torch.Tensor

    batch = MyInput(x=..., mask=...)
    batch = batch.to('cuda')  # Moves all tensors to GPU
    """

    def to(self: T, device: Union[str, torch.device], non_blocking: bool = False) -> T:
        """Move all Tensor fields to the specified device.

        Parameters
        ----------
        device : str or torch.device
            The target device (e.g., 'cuda:0', 'cpu').
        non_blocking : bool, default=False
            If True and the source is in pinned memory, the copy will be asynchronous.

        Returns
        -------
        T
            A new instance with tensors moved to the target device.
        """
        updates = {}
        for field in fields(self):
            val = getattr(self, field.name)

            if isinstance(val, torch.Tensor):
                updates[field.name] = val.to(device, non_blocking=non_blocking)
            elif hasattr(val, "to"):  # Handle nested ModelIO or nn.Module
                updates[field.name] = val.to(device, non_blocking=non_blocking)
            elif isinstance(val, (list, tuple)):
                # Handle lists of tensors
                converted = []
                for v in val:
                    if isinstance(v, torch.Tensor):
                        converted.append(v.to(device, non_blocking=non_blocking))
                    elif hasattr(v, "to"):
                        converted.append(v.to(device, non_blocking=non_blocking))
                    else:
                        converted.append(v)
                updates[field.name] = type(val)(converted)
            elif isinstance(val, dict):
                # Handle dicts of tensors
                converted_dict = {}
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        converted_dict[k] = v.to(device, non_blocking=non_blocking)
                    elif hasattr(v, "to"):
                        converted_dict[k] = v.to(device, non_blocking=non_blocking)
                    else:
                        converted_dict[k] = v
                updates[field.name] = converted_dict

        # Use replace to create a shallow copy with updated fields
        return replace(self, **updates)

    def cpu(self: T) -> T:
        """Move all tensors to CPU."""
        return self.to("cpu")

    def cuda(self: T, device: Optional[Union[int, str, torch.device]] = None) -> T:
        """Move all tensors to CUDA device.

        Parameters
        ----------
        device : int, str, or torch.device, optional
            Target CUDA device. If None, uses default CUDA device.

        Returns
        -------
        T
            A new instance with tensors moved to CUDA.
        """
        if device is None:
            return self.to("cuda")
        return self.to(f"cuda:{device}" if isinstance(device, int) else device)

    def detach(self: T) -> T:
        """Detach all Tensor fields from the computation graph.

        Useful when passing outputs to metrics or storing history.

        Returns
        -------
        T
            A new instance with detached tensors.
        """
        updates = {}
        for field in fields(self):
            val = getattr(self, field.name)

            if isinstance(val, torch.Tensor):
                updates[field.name] = val.detach()
            elif hasattr(val, "detach"):
                updates[field.name] = val.detach()
            elif isinstance(val, (list, tuple)):
                # Handle lists of tensors
                detached = []
                for v in val:
                    if isinstance(v, torch.Tensor):
                        detached.append(v.detach())
                    elif hasattr(v, "detach"):
                        detached.append(v.detach())
                    else:
                        detached.append(v)
                updates[field.name] = type(val)(detached)
            elif isinstance(val, dict):
                # Handle dicts of tensors
                detached_dict = {}
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        detached_dict[k] = v.detach()
                    elif hasattr(v, "detach"):
                        detached_dict[k] = v.detach()
                    else:
                        detached_dict[k] = v
                updates[field.name] = detached_dict

        return replace(self, **updates)

    def clone(self: T) -> T:
        """Create a deep copy of the object."""
        return copy.deepcopy(self)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access: output['key']."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __contains__(self, key: str) -> bool:
        """Check if key exists: 'key' in output."""
        return hasattr(self, key)

    def keys(self) -> Iterator[str]:
        """Iterate over field names."""
        return (f.name for f in fields(self))

    def values(self) -> Iterator[Any]:
        """Iterate over field values."""
        return (getattr(self, f.name) for f in fields(self))

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over (key, value) pairs."""
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instance from dictionary, filtering unknown keys.

        Useful when loading data from a flexible config or dataloader
        that might have extra fields.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
