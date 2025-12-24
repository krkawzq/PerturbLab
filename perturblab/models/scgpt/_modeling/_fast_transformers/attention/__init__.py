#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implementations of different types of attention mechanisms."""


from .aft_attention import AFTFullAttention, AFTSimpleAttention
from .attention_layer import AttentionLayer
from .causal_linear_attention import CausalLinearAttention
from .clustered_attention import ClusteredAttention
from .conditional_full_attention import ConditionalFullAttention
from .exact_topk_attention import ExactTopKAttention
from .full_attention import FullAttention
from .improved_clustered_attention import ImprovedClusteredAttention
from .improved_clustered_causal_attention import ImprovedClusteredCausalAttention
from .linear_attention import LinearAttention
from .local_attention import LocalAttention
from .reformer_attention import ReformerAttention
