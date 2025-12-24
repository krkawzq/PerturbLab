"""scFoundation model implementation with lazy loading and dependency checking.

This module uses lazy loading to defer imports until actual usage.
"""

from perturblab.utils import create_lazy_loader

# Import dependencies from config
try:
    from ..config import dependencies
except (ImportError, AttributeError):
    dependencies = []

# Define lazy loading module map
_LAZY_MODULES = {
    'scFoundationModel': '.model',
}

# Create lazy loader with dependency checking
__getattr__, __dir__ = create_lazy_loader(
    dependencies=dependencies,
    lazy_modules=_LAZY_MODULES,
    package_name=__package__,
    install_hint="pip install perturblab[scfoundation]"
)

__all__ = list(_LAZY_MODULES.keys())

