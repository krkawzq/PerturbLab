# 惰性导入：避免循环导入
def __getattr__(name):
    """惰性导入 scFoundation 模型类"""
    if name == 'scFoundationConfig':
        from .config import scFoundationConfig
        return scFoundationConfig
    elif name == 'scFoundationModel':
from .model import scFoundationModel

        return scFoundationModel
    elif name == 'scFoundationPerturbationModel':
from .perturbation_model import scFoundationPerturbationModel

        return scFoundationPerturbationModel
    elif name == 'load_gene_list':
        from .config import load_gene_list
        return load_gene_list
    elif name == 'load_training_config':
        from .config import load_training_config
        return load_training_config
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'scFoundationConfig',
    'scFoundationModel',
    'scFoundationPerturbationModel',
    'load_gene_list',
    'load_training_config',
]
