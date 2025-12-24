"""Gradient Reversal Layer for Domain Adaptation.

Implements the Gradient Reversal Layer (GRL), which acts as the identity function during
the forward pass, but reverses and scales gradients by a factor lambda during the backward pass.
This is commonly used in domain-adversarial training and domain adaptation tasks.

References:
    Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR, 2016.
"""

import torch
from torch.autograd import Function

__all__ = ["grad_reverse", "GradReverse"]


class GradReverse(Function):
    """Gradient Reversal Layer (GRL).

    During forward, returns the input unchanged.
    During backward, multiplies the gradient by -lambda.

    Typical usage: learn domain-invariant features for domain adaptation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        """Forward pass for Gradient Reversal Layer.

        Args:
            ctx: Context object used to stash information for backward computation.
            x (torch.Tensor): Input tensor.
            lambd (float): Gradient reversal coefficient.

        Returns:
            torch.Tensor: Output tensor (same as input).
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass for Gradient Reversal Layer.

        Args:
            ctx: Context object with saved lambda from forward pass.
            grad_output (torch.Tensor): Gradient output from the subsequent layer.

        Returns:
            Tuple[torch.Tensor, None]: Reversed and scaled gradient with respect to input,
            and None for lambda (no gradient needed).
        """
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Applies the Gradient Reversal Layer to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        lambd (float, optional): Gradient reversal coefficient, controlling strength of
            adversarial signal. Default is 1.0.

    Returns:
        torch.Tensor: Output tensor (same as input in forward, but reversed gradients in backward).

    Example:
        >>> import torch
        >>> x = torch.randn(32, 128, requires_grad=True)
        >>> feat_main_task = x
        >>> feat_domain_clf = grad_reverse(x, lambd=1.0)
        >>> # Backprop: feat_domain_clf gradients reversed by lambda

    Note:
        The lambda parameter is often scheduled from 0 to 1 as training progresses to let
        the model first learn the primary task, then enforce domain invariance.
    """
    return GradReverse.apply(x, lambd)
