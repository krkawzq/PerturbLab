import json
import os
from typing import Any, Literal

from ..configuration import ModelConfig


class scGPTConfig(ModelConfig):
    def __init__(
        self,
        *,
        scgpt_mode: Literal['singlecell', 'multiomic', 'perturbation'] = 'perturbation',
        
        use_default_gene_vocab: bool = True,
        specials: list[str] | None = None,
        special_first: bool = True,
        default_token: str | None = "<pad>",
        
        ntoken: int = None,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
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
        max_seq_len: int | None = None,  # None = no length limit, use actual data length
        batch_label_key: str = 'batch',
        dab_weight: float = 1.0,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.use_default_gene_vocab = use_default_gene_vocab
        self.specials = specials
        self.special_first = special_first
        self.default_token = default_token
        
        self.ntoken = ntoken
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nlayers_cls = nlayers_cls
        self.n_cls = n_cls
        self.dropout = dropout
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.use_batch_labels = use_batch_labels
        self.num_batch_labels = num_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.mvc_decoder_style = mvc_decoder_style
        self.ecs_threshold = ecs_threshold
        self.explicit_zero_prob = explicit_zero_prob
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend
        self.pre_norm = pre_norm
        self.pert_pad_id = pert_pad_id
        self.decoder_activation = decoder_activation
        self.decoder_adaptive_bias = decoder_adaptive_bias
        self.use_mod = use_mod
        self.ntokens_mod = ntokens_mod
        self.vocab_mod = vocab_mod
        self.max_seq_len = max_seq_len
        self.batch_label_key = batch_label_key
        self.dab_weight = dab_weight
    
    def save(self, path: str):
        """
        Save config to a file (args.json format, following scGPT convention).
        
        Args:
            path: File path where config will be saved (e.g., "model/args.json")
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Convert to scGPT args.json format (only essential model parameters)
        args_dict = {
            "pad_token": self.pad_token,
            "input_emb_style": self.input_emb_style,
            "n_bins": self.n_input_bins,
            "nlayers": self.nlayers,
            "nheads": self.nhead,
            "embsize": self.d_model,
            "d_hid": self.d_hid,
            "dropout": self.dropout,
            "n_layers_cls": self.nlayers_cls,
            "pad_value": self.pad_value,
            "fast_transformer": self.use_fast_transformer,
            "MVC": self.do_mvc,
            "DAB": self.do_dab,
        }
        
        with open(path, 'w') as f:
            json.dump(args_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'scGPTConfig':
        """
        Load config from a file (args.json format, following scGPT convention).
        Only loads model architecture parameters, ignoring training configurations.
        
        Note: ntoken will be None if not in args.json. The model should set it
        from vocab.json after loading the config.
        
        Args:
            path: Path to config file (e.g., "model/args.json")
            
        Returns:
            scGPTConfig instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
        
        with open(path, 'r') as f:
            args = json.load(f)
        
        # ntoken is not typically in args.json, will be set by model from vocab
        ntoken = args.get('ntoken', None)
        
        # Only extract model architecture parameters from args.json
        # Training configs (epochs, lr, batch_size, etc.) are ignored
        return cls(
            model_series='scgpt',
            model_name='pretrained',
            # Core architecture parameters
            ntoken=ntoken,
            d_model=args.get('embsize', 512),
            nhead=args.get('nheads', 8),
            d_hid=args.get('d_hid', 512),
            nlayers=args.get('nlayers', 12),
            nlayers_cls=args.get('n_layers_cls', 3),
            n_cls=args.get('n_cls', 1),
            dropout=args.get('dropout', 0.2),
            # Token parameters
            pad_token=args.get('pad_token', '<pad>'),
            pad_value=args.get('pad_value', -2),
            # Task flags (using MVC flag if present)
            do_mvc=args.get('MVC', False),
            do_dab=args.get('DAB', False),
            # Input/embedding parameters
            input_emb_style=args.get('input_emb_style', 'continuous'),
            n_input_bins=args.get('n_bins'),  # Can be None
            # Other architecture configs (with safe defaults)
            cell_emb_style=args.get('cell_emb_style', 'cls'),
            mvc_decoder_style=args.get('mvc_decoder_style', 'inner product'),
            ecs_threshold=args.get('ecs_threshold', 0.3),
            explicit_zero_prob=args.get('explicit_zero_prob', False),
            use_fast_transformer=args.get('fast_transformer', False),
            fast_transformer_backend=args.get('fast_transformer_backend', 'flash'),
            pre_norm=args.get('pre_norm', False),
            pert_pad_id=args.get('pert_pad_id', 2),
            # Batch normalization
            use_batch_labels=args.get('use_batch_labels', False),
            num_batch_labels=args.get('num_batch_labels'),  # Can be None
            domain_spec_batchnorm=args.get('domain_spec_batchnorm', args.get('DSBN', False)),
            # Advanced parameters (usually None for pretrained models)
            decoder_activation=args.get('decoder_activation'),
            decoder_adaptive_bias=args.get('decoder_adaptive_bias', False),
            use_mod=args.get('use_mod', False),
            ntokens_mod=args.get('ntokens_mod'),  # Can be None
            vocab_mod=args.get('vocab_mod'),  # Can be None
        )
