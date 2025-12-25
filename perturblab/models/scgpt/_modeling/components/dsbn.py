"""Domain-Specific Batch Normalization for scGPT.

This module implements Domain-Specific Batch Normalization (DSBN), which maintains
separate batch normalization statistics for different domains/batches in the data.

DSBN is particularly useful for the following scenarios:
    1. Batch effect correction in single-cell data.
    2. Domain adaptation tasks.
    3. Multi-dataset integration.

References:
    Chang et al. (2019). "Domain-Specific Batch Normalization for Unsupervised Domain Adaptation." CVPR 2019.
    Original implementation: https://github.com/wgchang/DSBN

Copyright:
    Modified from https://github.com/wgchang/DSBN/blob/master/model/dsbn.py
    Adapted for scGPT and PerturbLab.
"""

import torch
from torch import nn

__all__ = [
    "DomainSpecificBatchNorm1d",
    "DomainSpecificBatchNorm2d",
]


class _DomainSpecificBatchNorm(nn.Module):
    """Base class for Domain-Specific Batch Normalization.

    Maintains a list of batch normalization layers, one for each domain, which allows
    domain-specific normalization statistics while sharing model parameters elsewhere.

    Attributes:
        num_domains (int): Number of domains.
        bns (nn.ModuleList): List of batch normalization layers (one per domain).
        cur_domain (int or None): Currently active domain index.
    """

    _version = 2

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """Initializes the Domain-Specific Batch Normalization base module.

        Args:
            num_features (int): Number of features (channels) in the input.
            num_domains (int): Number of domains/batches to handle.
            eps (float, optional): Value added to denominator for numerical stability. Defaults to 1e-5.
            momentum (float, optional): Value used for running mean and variance computation. Defaults to 0.1.
            affine (bool, optional): If True, has learnable affine parameters. Defaults to True.
            track_running_stats (bool, optional): Tracks running mean and variance if True. Defaults to True.
        """
        super(_DomainSpecificBatchNorm, self).__init__()
        self._cur_domain = None
        self.num_domains = num_domains
        self.bns = nn.ModuleList(
            [
                self.bn_handle(num_features, eps, momentum, affine, track_running_stats)
                for _ in range(num_domains)
            ]
        )

    @property
    def bn_handle(self) -> nn.Module:
        """Returns the batch normalization class to use (e.g., 1d, 2d)."""
        raise NotImplementedError

    @property
    def cur_domain(self) -> int | None:
        """int or None: Currently active domain index."""
        return self._cur_domain

    @cur_domain.setter
    def cur_domain(self, domain_label: int):
        """Sets the currently active domain index.

        Args:
            domain_label (int): The domain label to set as current.
        """
        self._cur_domain = domain_label

    def reset_running_stats(self):
        """Resets running statistics (mean and variance) for all domains."""
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        """Resets the learnable parameters for all domains."""
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input: torch.Tensor):
        """Checks if input tensor has the correct dimensionality.

        Args:
            input (torch.Tensor): The tensor to check.

        Raises:
            NotImplementedError: Needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, domain_label: int) -> torch.Tensor:
        """Executes the forward pass through domain-specific batch normalization.

        Args:
            x (torch.Tensor): Input tensor.
            domain_label (int): Domain index to use (from 0 to num_domains-1).

        Returns:
            torch.Tensor: Normalized tensor using the specified domain's statistics.

        Raises:
            ValueError: If domain_label is out of range.
        """
        self._check_input_dim(x)
        if domain_label >= self.num_domains:
            raise ValueError(
                f"Domain label {domain_label} exceeds the number of domains {self.num_domains}"
            )
        bn = self.bns[domain_label]
        self.cur_domain = domain_label
        return bn(x)


class DomainSpecificBatchNorm1d(_DomainSpecificBatchNorm):
    """Domain-Specific Batch Normalization for 1D inputs.

    This module applies domain-specific batch normalization over 2D or 3D inputs
    shaped (batch_size, num_features) or (batch_size, num_features, length).

    Example:
        dsbn = DomainSpecificBatchNorm1d(num_features=128, num_domains=3)
        x = torch.randn(32, 128)
        out = dsbn(x, domain_label=0)  # Use domain 0's statistics
        print(out.shape)  # torch.Size([32, 128])
    """

    @property
    def bn_handle(self) -> nn.Module:
        """Returns nn.BatchNorm1d class."""
        return nn.BatchNorm1d

    def _check_input_dim(self, input: torch.Tensor):
        """Checks that input tensor is 2D or 3D.

        Args:
            input (torch.Tensor): The tensor to check.

        Raises:
            ValueError: If tensor is not 2D or 3D.
        """
        if input.dim() > 3:
            raise ValueError(f"expected at most 3D input (got {input.dim()}D input)")


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    """Domain-Specific Batch Normalization for 2D inputs.

    This module applies domain-specific batch normalization over 4D inputs
    shaped (batch_size, num_features, height, width).

    Example:
        dsbn = DomainSpecificBatchNorm2d(num_features=64, num_domains=3)
        x = torch.randn(32, 64, 28, 28)
        out = dsbn(x, domain_label=1)  # Use domain 1's statistics
        print(out.shape)  # torch.Size([32, 64, 28, 28])
    """

    @property
    def bn_handle(self) -> nn.Module:
        """Returns nn.BatchNorm2d class."""
        return nn.BatchNorm2d

    def _check_input_dim(self, input: torch.Tensor):
        """Checks that input tensor is 4D.

        Args:
            input (torch.Tensor): The tensor to check.

        Raises:
            ValueError: If tensor is not 4D.
        """
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
