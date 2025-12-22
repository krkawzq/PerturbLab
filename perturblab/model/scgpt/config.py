import logging
from typing import Any, Literal

from ..configuration import ModelConfig

logger = logging.getLogger(__name__)

class scGPTConfig(ModelConfig):
    """Configuration class for scGPT models."""

    def __init__(
        self,
        *,
        scgpt_mode: Literal['singlecell', 'multiomic', 'perturbation'] = 'perturbation',
        use_default_gene_vocab: bool = True,
        specials: list[str] | None = None,
        special_first: bool = True,
        default_token: str | None = "<pad>",
        ntoken: int | None = None,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 12,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: int | None = None,
        domain_spec_batchnorm: bool | str = False,
        input_emb_style: str = "continuous",
        n_input_bins: int | None = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        pert_pad_id: int = 2,
        decoder_activation: str | None = None,
        decoder_adaptive_bias: bool = False,
        use_mod: bool = False,
        ntokens_mod: int | None = None,
        vocab_mod: Any | None = None,
        max_seq_len: int | None = None,
        batch_label_key: str = 'batch',
        dab_weight: float = 1.0,
        **kwargs,
    ):
        """Initialize scGPTConfig with model hyperparameters."""
        super().__init__(**kwargs)
        
        # Batch set attributes from local variables
        self._set_all(locals())

    @classmethod
    def _get_key_mapping(cls) -> dict[str, str]:
        """Get mapping for legacy key names to current attribute names."""
        return {
            "d_model": "embsize",
            "nhead": "nheads",
            "nlayers_cls": "n_layers_cls",
            "do_mvc": "MVC",
            "do_dab": "DAB",
            "n_input_bins": "n_bins",
            "use_fast_transformer": "fast_transformer",
        }
