"""GEARS model implementation with lazy loading and dependency checking.

This module uses lazy loading to defer imports until actual usage, allowing
users without torch_geometric to use other parts of PerturbLab.
"""

from perturblab.utils import create_lazy_loader

dependencies = ["torch_geometric"]

# Define lazy loading module map
_LAZY_MODULES = {
    "MLP": ".model",
    "GEARSModel": ".model",
}

# Create lazy loader with dependency checking
__getattr__, __dir__ = create_lazy_loader(
    dependencies=dependencies,
    lazy_modules=_LAZY_MODULES,
    package_name=__package__,
    install_hint="pip install perturblab[gears]",
)

__all__ = list(_LAZY_MODULES.keys())
