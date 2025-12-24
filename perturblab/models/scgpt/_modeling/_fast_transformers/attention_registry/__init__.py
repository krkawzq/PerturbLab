#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Allow for the dynamic registration of new attention implementations.

This module provides a Registry implementation that other modules can use to
register attention implementations for the builders.
"""

from .registry import AttentionRegistry, RecurrentAttentionRegistry, RecurrentCrossAttentionRegistry
from .spec import Bool, Callable, Choice, EventDispatcherInstance, Float, Int, Optional, Spec
