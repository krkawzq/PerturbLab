"""GeneCompass model implementation with lazy loading."""

from perturblab.utils import create_lazy_loader

from perturblab.models.genecompass.config import requirements, dependencies

# Define lazy loading module map
_LAZY_MODULES = {
    "GeneCompassModel": "model",
}

# Create lazy loader
__getattr__, __dir__ = create_lazy_loader(
    requirements=requirements,
    dependencies=dependencies,
    lazy_modules=_LAZY_MODULES,
    package_name=__package__,
    install_hint="pip install perturblab[genecompass]",
)

__all__ = list(_LAZY_MODULES.keys())
