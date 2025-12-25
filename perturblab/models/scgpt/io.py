"""Input/Output schema definitions for scGPT models.

Inherits from core.model_io.ModelIO to provide unified device management
and dictionary-like access patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from perturblab.core.model_io import ModelIO

__all__ = ["scGPTInput", "scGPTOutput"]


@dataclass(kw_only=True)
class scGPTInput(ModelIO):
    """Container for scGPT model inputs and task control flags.

    This unified input schema supports all scGPT variants and task modes.

    Data Attributes
    ---------------
    src : Tensor
        Gene token indices. Shape: (batch_size, seq_len)
    values : Tensor
        Gene expression values. Shape: (batch_size, seq_len)
    src_key_padding_mask : Tensor
        Mask indicating padding positions (True/1 for padding). Shape: (batch_size, seq_len)
    batch_labels : Optional[Tensor], default=None
        Batch/Domain labels. Shape: (batch_size,)
    mod_types : Optional[Tensor], default=None
        Modality type indices (MultiOmic only). Shape: (batch_size, seq_len)
    pert_flags : Optional[Tensor], default=None
        Perturbation flags (Generator only). Shape: (batch_size, seq_len)

    Task Control Flags
    ------------------
    CLS : bool, default=False
        Whether to compute Classification task output.
    CCE : bool, default=False
        Whether to compute Contrastive Cell Embedding loss.
    MVC : bool, default=False
        Whether to compute Masked Value Prediction task.
    ECS : bool, default=False
        Whether to compute Elastic Cell Similarity loss.
    do_sample : bool, default=False
        Whether to sample from the output distribution (for MVC/MLM) instead of taking the mean.
    
    Note
    ----
    All fields must be specified as keyword arguments due to dataclass inheritance constraints.
    Example: scGPTInput(src=..., values=..., src_key_padding_mask=...)
    """

    # Data Tensors (required fields)
    src: Tensor
    values: Tensor
    src_key_padding_mask: Tensor
    
    # Optional Data Tensors
    batch_labels: Optional[Tensor] = None
    mod_types: Optional[Tensor] = None
    pert_flags: Optional[Tensor] = None

    # Task Flags
    CLS: bool = False
    CCE: bool = False
    MVC: bool = False
    ECS: bool = False
    do_sample: bool = False


@dataclass(kw_only=True)
class scGPTOutput(ModelIO):
    """Container for scGPT model outputs.

    Aggregates all possible outputs. Fields are None if the corresponding
    task flag in input was False.

    Attributes
    ----------
    mlm_output : Optional[Tensor]
        Masked Language Modeling predictions. Shape: (batch_size, seq_len)
    cell_emb : Optional[Tensor]
        Cell-level embedding. Shape: (batch_size, d_model)
    cls_output : Optional[Tensor]
        Classification logits. Shape: (batch_size, n_cls)
    mvc_output : Optional[Tensor]
        Masked Value Prediction outputs. Shape: (batch_size, seq_len)
    loss_cce : Optional[Tensor]
        Contrastive Cell Embedding loss (scalar).
    loss_ecs : Optional[Tensor]
        Elastic Cell Similarity loss (scalar).
    dab_output : Optional[Tensor]
        Domain Adversarial discriminator output.
    mlm_zero_probs : Optional[Tensor]
        Zero-inflation probability for MLM.
    mvc_zero_probs : Optional[Tensor]
        Zero-inflation probability for MVC.
    
    Note
    ----
    All fields must be specified as keyword arguments due to dataclass inheritance constraints.
    Example: scGPTOutput(mlm_output=..., cell_emb=...)
    """

    # Primary Outputs
    mlm_output: Optional[Tensor] = None
    cell_emb: Optional[Tensor] = None

    # Task-Specific Outputs
    cls_output: Optional[Tensor] = None
    mvc_output: Optional[Tensor] = None

    # Losses
    loss_cce: Optional[Tensor] = None
    loss_ecs: Optional[Tensor] = None

    # Advanced / Internal Outputs
    dab_output: Optional[Tensor] = None
    mlm_zero_probs: Optional[Tensor] = None
    mvc_zero_probs: Optional[Tensor] = None
