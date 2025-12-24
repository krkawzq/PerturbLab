"""GeneCompass configuration.

GeneCompass is a knowledge-informed, cross-species foundation model based on the BERT architecture.
It incorporates five kinds of biological prior knowledge to enhance single-cell analysis.

Copyright: GeneCompass Authors, adapted for PerturbLab.
"""

from dataclasses import dataclass
from typing import Optional

from perturblab.core.config import ModelConfig

__all__ = ["GeneCompassConfig", "dependencies"]

# Required dependencies for GeneCompass.
dependencies = [
    "transformers",
]


@dataclass
class GeneCompassConfig(ModelConfig):
    """Configuration for the GeneCompass model.

    This configuration defines the parameters for the GeneCompass model, which extends
    BERT by integrating biological prior knowledge for improved single-cell RNA-seq analysis
    and cross-species transfer learning.

    Attributes:
        vocab_size (int): Vocabulary size (number of genes), typically 20,000-30,000. Defaults to 30000.
        hidden_size (int): Hidden dimensionality of transformer layers. Defaults to 768.
        num_hidden_layers (int): Number of transformer encoder layers. Range: 6-24. Defaults to 6.
        num_attention_heads (int): Number of attention heads per layer. Must divide hidden_size. Defaults to 12.
        intermediate_size (int): Inner dimension of feed-forward networks. Defaults to 3072.
        max_position_embeddings (int): Maximum sequence length supported. Defaults to 5000.

        hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.1.
        attention_probs_dropout_prob (float): Dropout probability for attention weights. Defaults to 0.1.
        layer_norm_eps (float): Epsilon for layer normalization layers. Defaults to 1e-12.
        initializer_range (float): Standard deviation for weight initialization. Defaults to 0.02.

        type_vocab_size (int): Vocabulary size for token type (segment) embeddings. Defaults to 2.
        position_embedding_type (str): Type of position embeddings ("absolute", "relative_key", "relative_key_query"). Defaults to "absolute".

        use_values (bool): Whether to predict expression values (regression). Defaults to True.
        use_cls_token (bool): Whether to prepend a CLS token for pooled sequence representation. Defaults to False.
        warmup_steps (int): Number of steps for learning rate warmup. Defaults to 10000.
        emb_warmup_steps (int): Number of steps for embedding warmup (biological knowledge integration). Defaults to 2000.

        mlm_weight (float): Weight for masked language modeling loss (gene ID prediction). Defaults to 0.2.
        value_weight (float): Weight for gene expression value prediction loss. Defaults to 0.8.

        pad_token_id (int): Token ID for padding. Defaults to 0.
        mask_token_id (int): Token ID for masking. Defaults to 1.

    Example:
        config = GeneCompassConfig(
            vocab_size=25000,
            hidden_size=768,
            num_hidden_layers=12,
            use_values=True,
            use_cls_token=True
        )

    Notes:
        - hidden_size must be divisible by num_attention_heads.
        - For cross-species transfer, provide appropriate species embeddings in the knowledge input.
        - Embedding warmup can stabilize training with external knowledge.

    References:
        GeneCompass: Decoding Universal Gene Expression Signatures Across
        Species and Sequencing Platforms. bioRxiv 2023.
    """

    # BERT architecture parameters
    vocab_size: int = 30000
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 5000

    # Regularization parameters
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    # Embedding parameters
    type_vocab_size: int = 2
    position_embedding_type: str = "absolute"

    # Knowledge integration settings
    use_values: bool = True
    use_cls_token: bool = False
    warmup_steps: int = 10000
    emb_warmup_steps: int = 2000

    # Loss configuration
    mlm_weight: float = 0.2
    value_weight: float = 0.8

    # Special token IDs
    pad_token_id: int = 0
    mask_token_id: int = 1

    def __post_init__(self):
        """Validates configuration values.

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_attention_heads`.
        Warns:
            UserWarning: If loss weights do not sum to 1.0, or if embedding warmup
                steps exceed learning rate warmup steps.
        """
        super().__post_init__()

        # Validate that hidden_size is divisible by num_attention_heads.
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads}). "
                f"Got hidden_size // num_attention_heads = "
                f"{self.hidden_size / self.num_attention_heads:.2f}"
            )

        # Validate that loss weights sum to 1.0.
        total_weight = self.mlm_weight + self.value_weight
        if abs(total_weight - 1.0) > 1e-6:
            import warnings

            warnings.warn(
                f"Loss weights (mlm_weight={self.mlm_weight}, "
                f"value_weight={self.value_weight}) sum to {total_weight}, "
                f"not 1.0. This may affect training dynamics.",
                UserWarning,
            )

        # Check that emb_warmup_steps does not exceed warmup_steps.
        if self.emb_warmup_steps > self.warmup_steps:
            import warnings

            warnings.warn(
                f"emb_warmup_steps ({self.emb_warmup_steps}) > warmup_steps "
                f"({self.warmup_steps}). Embedding warmup will extend beyond "
                f"learning rate warmup.",
                UserWarning,
            )
