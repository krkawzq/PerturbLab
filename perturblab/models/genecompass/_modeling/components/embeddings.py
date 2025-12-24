"""scBERT Embedding Layer Implementation.

This module implements the complex embedding layer for scBERT, which integrates
multiple sources of prior knowledge (Promoter, Co-expression, Gene Family, PECA-GRN)
along with continuous gene expression values.

Original source: https://github.com/TencentAILabHealthcare/scBERT
Adapted for PerturbLab with strict weight compatibility.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["QuickGELU", "ContinuousValueEncoder", "PriorEmbedding", "KnowledgeBertEmbeddings"]


class QuickGELU(nn.Module):
    """Fast approximation of GELU activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class ContinuousValueEncoder(nn.Module):
    """Encode real number values to a vector using neural nets projection.

    Architecture:
        Input(1) -> Linear -> ReLU -> Linear -> LayerNorm -> Dropout
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 255):
        super().__init__()
        self.max_value = max_value
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # Expand last dimension: [B, L] -> [B, L, 1]
        x = x.unsqueeze(-1)
        x = x.float()
        # Clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PriorEmbedding(nn.Module):
    """Simple linear projection for prior knowledge features."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear1(x)


class KnowledgeBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type and prior knowledge embeddings.

    This layer fuses:
    1. Gene ID embeddings (word_embeddings)
    2. Continuous expression values (optional)
    3. Biological prior knowledge (Promoter, Co-expression, etc.)
    4. Positional embeddings (standard BERT absolute position)
    """

    def __init__(self, config: Any, knowledges: Dict[str, Any]):
        super().__init__()

        # 1. Base Embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # 2. Config Flags
        # Use config.to_dict() safely if available, else getattr
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__
        self.use_values = cfg_dict.get("use_values", False)
        self.use_promoter = cfg_dict.get("use_promoter", False)
        self.use_co_exp = cfg_dict.get("use_co_exp", False)
        self.use_gene_family = cfg_dict.get("use_gene_family", False)
        self.use_peca_grn = cfg_dict.get("use_peca_grn", False)
        self.value_dim = 1

        # 3. Prior Knowledge Embeddings
        num_hidden = 0
        token_num = 0  # Will be set if any knowledge is used

        # Helper to register knowledge buffers safely
        # Note: We must use the exact attribute names 'promoter_embeddings', etc.

        if self.use_promoter:
            if knowledges.get("promoter") is None:
                raise ValueError("promoter knowledge missing")
            self.register_buffer("promoter_knowledge", knowledges["promoter"])
            self.promoter_embeddings = PriorEmbedding(
                knowledges["promoter"].size(1), config.hidden_size
            )
            token_num = knowledges["promoter"].size(0)
            num_hidden += 1

        if self.use_co_exp:
            if knowledges.get("co_exp") is None:
                raise ValueError("co_exp knowledge missing")
            self.register_buffer("co_exp_knowledge", knowledges["co_exp"])
            self.co_exp_embeddings = PriorEmbedding(
                knowledges["co_exp"].size(1), config.hidden_size
            )
            token_num = knowledges["co_exp"].size(0)
            num_hidden += 1

        if self.use_gene_family:
            if knowledges.get("gene_family") is None:
                raise ValueError("gene_family knowledge missing")
            self.register_buffer("gene_family_knowledge", knowledges["gene_family"])
            self.gene_family_embeddings = PriorEmbedding(
                knowledges["gene_family"].size(1), config.hidden_size
            )
            token_num = knowledges["gene_family"].size(0)
            num_hidden += 1

        if self.use_peca_grn:
            if knowledges.get("peca_grn") is None:
                raise ValueError("peca_grn knowledge missing")
            self.register_buffer("peca_grn_knowledge", knowledges["peca_grn"])
            self.peca_grn_embeddings = PriorEmbedding(
                knowledges["peca_grn"].size(1), config.hidden_size
            )
            token_num = knowledges["peca_grn"].size(0)
            num_hidden += 1

        # Handle Homologous Gene Mapping (Human <-> Mouse)
        if num_hidden > 0:
            self.homologous_gene_human2mouse = knowledges.get("homologous_gene_human2mouse", {})
            # Construct mapping index tensor
            homologous_index = torch.arange(token_num)
            if self.homologous_gene_human2mouse:
                keys = list(self.homologous_gene_human2mouse.keys())
                vals = list(self.homologous_gene_human2mouse.values())
                homologous_index[keys] = torch.as_tensor(vals, dtype=torch.long)
            self.register_buffer("homologous_index", homologous_index)

        # 4. Concatenation Projection Layer
        # Projects concatenated embeddings back to hidden_size
        # Input dim: Hidden(word) + [Hidden(prior) * num_prior] + [Value(1) if used]

        total_input_dim = config.hidden_size * (1 + num_hidden)
        if self.use_values:
            total_input_dim += self.value_dim

        self.concat_embeddings = nn.Sequential(
            OrderedDict(
                [
                    ("cat_fc", nn.Linear(total_input_dim, config.hidden_size)),
                    ("cat_ln", nn.LayerNorm(config.hidden_size)),
                    ("cat_gelu", QuickGELU()),
                    ("cat_proj", nn.Linear(config.hidden_size, config.hidden_size)),
                ]
            )
        )

        # 5. Standard BERT Embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Note: Name 'LayerNorm' (PascalCase) is preserved from original BERT/TensorFlow port
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position IDs Buffer
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        values: Optional[torch.FloatTensor] = None,
        species: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        emb_warmup_alpha: float = 1.0,
    ) -> Tensor:
        """Forward pass for embedding generation.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            values: Continuous expression values for genes.
            species: Species indicator (for homologous gene mapping).
            token_type_ids: Segment token indices.
            position_ids: Indices of positions of each input sequence token.
            inputs_embeds: Optionally provided embeddings (bypasses input_ids).
            past_key_values_length: Length of past key values (for generation).
            emb_warmup_alpha: Scaling factor for prior knowledge embeddings.

        Returns:
            Tensor: Final embeddings [batch_size, seq_len, hidden_size].
        """
        # 1. Determine Input Shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 2. Position IDs
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # 3. Token Type IDs
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        # 4. Word Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 5. Concatenate Values (if enabled)
        if self.use_values and values is not None:
            # [B, L] -> [B, L, 1]
            values_unsqueezed = values.unsqueeze(-1).float()
            inputs_embeds = torch.cat([inputs_embeds, values_unsqueezed], dim=2)

        # 6. Prepare Gene Indices for Prior Knowledge (Species Handling)
        input_ids_shifted = input_ids.detach().clone() if input_ids is not None else None

        # Note: Homologous mapping logic only runs if prior knowledge is used AND species is provided
        if (
            species is not None
            and hasattr(self, "homologous_index")
            and input_ids_shifted is not None
        ):
            # species shape usually [B, 1], mask where species == 1 (e.g. mouse)
            is_target_species = species.squeeze(1) == 1
            if is_target_species.any():
                # Apply mapping: index -> homologous_index[index]
                input_ids_shifted[is_target_species] = self.homologous_index[
                    input_ids_shifted[is_target_species]
                ]

        # 7. Concatenate Prior Knowledge (if enabled)
        # Note: Prior embeddings are scaled by emb_warmup_alpha

        if self.use_promoter:
            promoter_inputs = self.promoter_knowledge[input_ids_shifted]
            promoter_embeds = self.promoter_embeddings(promoter_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * promoter_embeds), dim=2)

        if self.use_co_exp:
            co_exp_inputs = self.co_exp_knowledge[input_ids_shifted]
            co_exp_embeds = self.co_exp_embeddings(co_exp_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * co_exp_embeds), dim=2)

        if self.use_gene_family:
            gene_family_inputs = self.gene_family_knowledge[input_ids_shifted]
            gene_family_embeds = self.gene_family_embeddings(gene_family_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * gene_family_embeds), dim=2)

        if self.use_peca_grn:
            peca_grn_inputs = self.peca_grn_knowledge[input_ids_shifted]
            peca_grn_embeds = self.peca_grn_embeddings(peca_grn_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * peca_grn_embeds), dim=2)

        # 8. Project Fusion to Hidden Size
        inputs_embeds = self.concat_embeddings(inputs_embeds)

        # 9. Add Positional & Token Type Embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 10. Norm & Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
