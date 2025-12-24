#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implementations of feature maps to be used with linear attention and causal
linear attention."""


from .base import ActivationFunctionFeatureMap, elu_feature_map
from .fourier_features import (
    Favor,
    GeneralizedRandomFeatures,
    RandomFourierFeatures,
    SmoothedRandomFourierFeatures,
)
