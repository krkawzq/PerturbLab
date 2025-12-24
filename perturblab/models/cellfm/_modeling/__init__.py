"""CellFM model implementation (in progress).

This module will contain the CellFM model implementation once migration is complete.

Current Status: Infrastructure ready, model implementation pending.
"""

from perturblab.utils import create_lazy_loader

from perturblab.models.cellfm.config import requirements, dependencies

# Define lazy loading module map
_LAZY_MODULES = {
    "CellFMModel": ".model",
}

# Create lazy loader
__getattr__, __dir__ = create_lazy_loader(
    requirements=requirements,
    dependencies=dependencies,
    lazy_modules=_LAZY_MODULES,
    package_name=__package__,
    install_hint="pip install perturblab[cellfm]",
)

__all__ = list(_LAZY_MODULES.keys())
