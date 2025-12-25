"""Configuration definition for scGPT model variants.

This configuration handles hyperparameters for standard scGPT, multi-omics,
and generative tasks.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from perturblab.core.config import Config

__all__ = ["scGPTConfig"]


@dataclass(kw_only=True)
class scGPTConfig(Config):
    """Configuration for scGPT models.

    This unified config supports all model variants:
    - TransformerModel (Standard)
    - MultiOmicTransformerModel
    - TransformerGenerator

    Attributes
    ----------
    ntoken : int
        Size of the gene vocabulary.
    d_model : int
        Dimension of the model embeddings and hidden states.
    nhead : int
        Number of attention heads.
    d_hid : int
        Dimension of the feedforward network hidden layer.
    nlayers : int
        Number of transformer encoder layers.
    nlayers_cls : int, default=3
        Number of layers in the classification decoder.
    n_cls : int, default=1
        Number of output classes for classification decoder.
    vocab_size : int, optional
        Alias for ntoken (for compatibility).
    dropout : float, default=0.5
        Dropout probability.
    pad_token : str, default="<pad>"
        Token string used for padding.
    pad_value : int, default=0
        Value used for padding in input values.

    # Model Architecture Flags
    do_mvc : bool, default=False
        Enable Masked Value Prediction (MVC) objective.
    do_dab : bool, default=False
        Enable Domain Adversarial Batch (DAB) training.
    use_batch_labels : bool, default=False
        Use batch labels for domain adaptation.
    num_batch_labels : int, default=0
        Number of unique batch/domain labels.
    domain_spec_batchnorm : Union[bool, str], default=False
        Use Domain-Specific BatchNorm. Options: False, True, "dsbn", "batchnorm", "do_affine".

    # Input Embedding Settings
    input_emb_style : Literal["category", "continuous", "scaling"], default="continuous"
        Style of input value embedding.
    n_input_bins : int, default=51
        Number of bins for categorical input embedding.
    cell_emb_style : Literal["cls", "avg-pool", "w-pool"], default="cls"
        Method to extract cell embedding from transformer output.

    # Training / Loss Settings
    mvc_decoder_style : str, default="inner-product"
        Architecture style for MVC decoder.
    ecs_threshold : float, default=0.3
        Elastic Cell Similarity threshold.
    explicit_zero_prob : bool, default=False
        Model explicit probability of zero expression (Zero-Inflated modeling).
    pre_norm : bool, default=False
        Use Pre-Norm architecture (vs Post-Norm).

    # Multi-omics Settings
    use_mod : bool, default=False
        Enable multi-omics modality embeddings.
    ntokens_mod : int, optional
        Number of modality tokens (required if use_mod=True).

    # Acceleration
    use_fast_transformer : bool, default=False
        Use optimized transformer implementation (FlashAttention/Linear).
    fast_transformer_backend : Literal["linear", "flash"], default="flash"
        Backend for fast transformer.
    """

    # Core Architecture
    ntoken: int
    d_model: int
    nhead: int
    d_hid: int
    nlayers: int

    # Defaults
    nlayers_cls: int = 3
    n_cls: int = 1
    dropout: float = 0.5
    pad_token: str = "<pad>"
    pad_value: int = 0

    # Feature Flags
    do_mvc: bool = False
    do_dab: bool = False
    use_batch_labels: bool = False
    num_batch_labels: int = 0
    domain_spec_batchnorm: bool | str = False

    # Embeddings
    input_emb_style: Literal["category", "continuous", "scaling"] = "continuous"
    n_input_bins: int = 51
    cell_emb_style: Literal["cls", "avg-pool", "w-pool"] = "cls"

    # Loss & Decoder
    mvc_decoder_style: str = "inner-product"
    ecs_threshold: float = 0.3
    explicit_zero_prob: bool = False
    pre_norm: bool = False

    # Multi-omics
    use_mod: bool = False
    ntokens_mod: Optional[int] = None

    # Acceleration
    use_fast_transformer: bool = False
    fast_transformer_backend: Literal["linear", "flash"] = "flash"

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.use_mod and self.ntokens_mod is None:
            raise ValueError("ntokens_mod must be specified when use_mod=True")

        if self.input_emb_style == "category" and self.n_input_bins <= 0:
            raise ValueError("n_input_bins must be > 0 when input_emb_style='category'")

        if self.use_batch_labels and self.num_batch_labels <= 0:
            raise ValueError("num_batch_labels must be > 0 when use_batch_labels=True")
