"""GEARS loss functions.

This module provides loss computation for GEARS models, including:
- Standard MSE loss
- Direction regularization loss
- Uncertainty-aware loss (for epistemic uncertainty quantification)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from perturblab.models.gears import GEARSOutput

__all__ = [
    "build_loss",
    "GEARSLoss",
    "GEARSUncertaintyLoss",
]


class GEARSLoss(nn.Module):
    """GEARS loss module with direction regularization.

    This loss combines MSE loss with optional direction consistency regularization
    to encourage predictions to have the correct sign of change relative to control.

    Args:
        direction_lambda: Weight for direction loss. Defaults to 1e-3.
        reduction: Loss reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
        ctrl_expression: Optional control expression to register as buffer.
            If None, must be provided dynamically in forward. Defaults to None.

    Examples:
        >>> from perturblab.methods.gears import build_loss
        >>>
        >>> # Build loss with static config
        >>> loss_fn = build_loss(
        ...     loss_type='standard',
        ...     direction_lambda=1e-3
        ... )
        >>>
        >>> # Use in training
        >>> loss = loss_fn(
        ...     outputs=model_outputs,
        ...     labels=batch_dict['labels'],
        ...     metadata=batch_dict['metadata'],
        ...     ctrl_expression=ctrl_expr  # Dynamic parameter
        ... )
    """

    def __init__(
        self,
        direction_lambda: float = 1e-3,
        reduction: str = "mean",
        ctrl_expression: torch.Tensor | None = None,
    ):
        super().__init__()
        self.default_direction_lambda = direction_lambda
        self.reduction = reduction

        # Register control expression as buffer if provided
        if ctrl_expression is not None:
            self.register_buffer("ctrl_expression", ctrl_expression)
        else:
            self.ctrl_expression = None

    def forward(
        self,
        outputs: GEARSOutput,
        labels: dict,
        metadata: dict,
        ctrl_expression: torch.Tensor | None = None,
        direction_lambda: float | None = None,
        **loss_kwargs,
    ) -> torch.Tensor:
        """Computes GEARS loss.

        Args:
            outputs: Model outputs (GEARSOutput with predictions field).
            labels: Dictionary from batch_dict['labels'] with 'predictions' key.
            metadata: Dictionary from batch_dict['metadata'] with 'perturbations' key.
            ctrl_expression: Control expression for direction loss. If None, uses
                registered buffer or skips direction loss.
            direction_lambda: Dynamic override for direction loss weight. If None,
                uses default value from initialization.
            **loss_kwargs: Additional dynamic parameters (e.g., epoch, step).

        Returns:
            Scalar loss tensor.

        Examples:
            >>> # Static parameters only
            >>> loss = loss_fn(outputs, labels, metadata)
            >>>
            >>> # Dynamic direction_lambda
            >>> loss = loss_fn(
            ...     outputs, labels, metadata,
            ...     direction_lambda=0.01  # Override
            ... )
            >>>
            >>> # With epoch-based scheduling
            >>> current_lambda = base_lambda * (1 + epoch * 0.1)
            >>> loss = loss_fn(
            ...     outputs, labels, metadata,
            ...     direction_lambda=current_lambda
            ... )
        """
        pred = outputs.predictions
        target = labels["predictions"]

        # Main MSE loss
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)

        # Direction loss (optional)
        direction_lambda = (
            direction_lambda if direction_lambda is not None else self.default_direction_lambda
        )

        if direction_lambda > 0:
            # Use provided ctrl_expression or registered buffer
            ctrl_expr = ctrl_expression if ctrl_expression is not None else self.ctrl_expression

            if ctrl_expr is not None:
                # Compute deltas
                pred_delta = pred - ctrl_expr
                target_delta = target - ctrl_expr

                # Direction consistency loss
                direction_loss = F.mse_loss(
                    torch.sign(pred_delta), torch.sign(target_delta), reduction=self.reduction
                )

                mse_loss = mse_loss + direction_lambda * direction_loss

        return mse_loss


class GEARSUncertaintyLoss(nn.Module):
    """GEARS loss with uncertainty quantification.

    Extends GEARSLoss to support epistemic uncertainty estimation via
    log-variance prediction.

    Args:
        uncertainty_reg: Weight for uncertainty regularization. Defaults to 1.0.
        direction_lambda: Weight for direction loss. Defaults to 1e-3.
        reduction: Loss reduction method. Defaults to 'mean'.
        ctrl_expression: Optional control expression buffer. Defaults to None.

    Examples:
        >>> loss_fn = build_loss(
        ...     loss_type='uncertainty',
        ...     uncertainty_reg=1.0,
        ...     direction_lambda=1e-3
        ... )
        >>>
        >>> # Forward with uncertainty
        >>> loss = loss_fn(
        ...     outputs=model_outputs,  # Must have log_variance field
        ...     labels=batch_dict['labels'],
        ...     metadata=batch_dict['metadata'],
        ...     uncertainty_reg=0.5  # Dynamic override
        ... )
    """

    def __init__(
        self,
        uncertainty_reg: float = 1.0,
        direction_lambda: float = 1e-3,
        reduction: str = "mean",
        ctrl_expression: torch.Tensor | None = None,
    ):
        super().__init__()
        self.default_uncertainty_reg = uncertainty_reg
        self.default_direction_lambda = direction_lambda
        self.reduction = reduction

        if ctrl_expression is not None:
            self.register_buffer("ctrl_expression", ctrl_expression)
        else:
            self.ctrl_expression = None

    def forward(
        self,
        outputs: GEARSOutput,
        labels: dict,
        metadata: dict,
        ctrl_expression: torch.Tensor | None = None,
        uncertainty_reg: float | None = None,
        direction_lambda: float | None = None,
        **loss_kwargs,
    ) -> torch.Tensor:
        """Computes GEARS uncertainty loss.

        Args:
            outputs: Model outputs with predictions and log_variance fields.
            labels: Dictionary with 'predictions' key.
            metadata: Dictionary with 'perturbations' key.
            ctrl_expression: Control expression for direction loss.
            uncertainty_reg: Dynamic override for uncertainty weight.
            direction_lambda: Dynamic override for direction weight.
            **loss_kwargs: Additional dynamic parameters.

        Returns:
            Scalar loss tensor combining prediction loss and uncertainty regularization.

        Notes:
            The uncertainty loss uses the formulation:
            L = (1/2) * exp(-log_var) * MSE + (1/2) * log_var

            This encourages the model to predict higher variance for harder samples.
        """
        pred = outputs.predictions
        log_var = outputs.log_variance
        target = labels["predictions"]

        if log_var is None:
            raise ValueError("Model outputs do not contain log_variance. Use GEARSLoss instead.")

        # Dynamic parameter overrides
        uncertainty_reg = (
            uncertainty_reg if uncertainty_reg is not None else self.default_uncertainty_reg
        )
        direction_lambda = (
            direction_lambda if direction_lambda is not None else self.default_direction_lambda
        )

        # Uncertainty-aware MSE loss
        # L = 0.5 * exp(-log_var) * (pred - target)^2 + 0.5 * log_var
        precision = torch.exp(-log_var)
        mse_term = F.mse_loss(pred, target, reduction="none")
        uncertainty_loss = 0.5 * precision * mse_term + 0.5 * log_var

        if self.reduction == "mean":
            uncertainty_loss = uncertainty_loss.mean()
        elif self.reduction == "sum":
            uncertainty_loss = uncertainty_loss.sum()

        total_loss = uncertainty_loss * uncertainty_reg

        # Direction loss (optional)
        if direction_lambda > 0:
            ctrl_expr = ctrl_expression if ctrl_expression is not None else self.ctrl_expression

            if ctrl_expr is not None:
                pred_delta = pred - ctrl_expr
                target_delta = target - ctrl_expr

                direction_loss = F.mse_loss(
                    torch.sign(pred_delta), torch.sign(target_delta), reduction=self.reduction
                )

                total_loss = total_loss + direction_lambda * direction_loss

        return total_loss


def build_loss(loss_type: str = "standard", **config) -> nn.Module:
    """Builds a GEARS loss module.

    This factory function creates the appropriate loss module based on the
    specified type and configuration. The returned module can be used in
    training loops and supports both static configuration and dynamic parameters.

    Args:
        loss_type: Type of loss to build ('standard', 'uncertainty').
            Defaults to 'standard'.
        **config: Loss configuration parameters:
            - direction_lambda: Weight for direction loss (float, default: 1e-3)
            - uncertainty_reg: Weight for uncertainty regularization (float, default: 1.0)
            - reduction: Loss reduction method ('mean', 'sum', 'none', default: 'mean')
            - ctrl_expression: Control expression tensor (optional)

    Returns:
        nn.Module that computes loss with signature:
            forward(outputs, labels, metadata, **loss_kwargs) -> Tensor

    Examples:
        >>> from perturblab.methods.gears import build_loss
        >>>
        >>> # Standard GEARS loss
        >>> loss_fn = build_loss(
        ...     loss_type='standard',
        ...     direction_lambda=1e-3
        ... )
        >>>
        >>> # Uncertainty-aware loss
        >>> loss_fn = build_loss(
        ...     loss_type='uncertainty',
        ...     uncertainty_reg=1.0,
        ...     direction_lambda=1e-3
        ... )
        >>>
        >>> # Use in training with static parameters
        >>> loss = loss_fn(outputs, batch_dict['labels'], batch_dict['metadata'])
        >>>
        >>> # Use with dynamic parameters
        >>> current_lambda = base_lambda * (1.0 + epoch * 0.1)
        >>> loss = loss_fn(
        ...     outputs,
        ...     batch_dict['labels'],
        ...     batch_dict['metadata'],
        ...     direction_lambda=current_lambda,  # Override
        ...     ctrl_expression=ctrl_expr
        ... )
        >>>
        >>> # Register as part of model
        >>> model = MODELS.build("GEARS.default", config)
        >>> loss_fn = build_loss('standard', direction_lambda=config.direction_lambda)

    Notes:
        The loss module is an nn.Module, which means:
        - It can be moved to GPU with .to(device)
        - It can have learnable parameters (e.g., adaptive loss weights)
        - It can be saved/loaded with model checkpoints
        - It integrates seamlessly with PyTorch training loops

    See Also:
        GEARSLoss: Standard loss implementation
        GEARSUncertaintyLoss: Uncertainty-aware loss implementation
    """
    loss_modules = {
        "standard": GEARSLoss,
        "uncertainty": GEARSUncertaintyLoss,
    }

    if loss_type not in loss_modules:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. " f"Available: {list(loss_modules.keys())}"
        )

    LossClass = loss_modules[loss_type]

    # Filter config to only include parameters for this loss class
    import inspect

    sig = inspect.signature(LossClass.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    return LossClass(**filtered_config)
