"""scGPT model implementation with lazy loading and dependency checking.

This module uses lazy loading to defer imports until actual usage, allowing
users without scGPT dependencies to use other parts of PerturbLab.
"""

from perturblab.utils import create_lazy_loader

# Import dependencies from config
try:
    from ..config import dependencies
except (ImportError, AttributeError):
    dependencies = []

# Define lazy loading module map (unified models)
_LAZY_MODULES = {
    # Unified model implementations (from model.py)
    "scGPTBaseModel": ".model",
    "scGPTModel": ".model",
    "scGPTMultiOmicModel": ".model",
    "scGPTPerturbationModel": ".model",
}

# Create lazy loader with dependency checking
__getattr__, __dir__ = create_lazy_loader(
    dependencies=dependencies,
    lazy_modules=_LAZY_MODULES,
    package_name=__package__,
    install_hint="pip install perturblab[scgpt]",
)

__all__ = list(_LAZY_MODULES.keys())
