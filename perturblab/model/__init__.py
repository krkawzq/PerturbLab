from .base import PerturbationModel
from .configuration import ModelConfig


# 惰性导入：避免循环导入和减少启动时间
def __getattr__(name):
    """惰性导入模型类"""
    if name == 'scGPTConfig':
        from .scgpt import scGPTConfig
        return scGPTConfig
    elif name == 'scGPTModel':
        from .scgpt import scGPTModel
        return scGPTModel
    elif name == 'scGPTPerturbationModel':
        from .scgpt import scGPTPerturbationModel
        return scGPTPerturbationModel
    elif name == 'scFoundationConfig':
        from .scfoundation import scFoundationConfig
        return scFoundationConfig
    elif name == 'scFoundationModel':
        from .scfoundation import scFoundationModel
        return scFoundationModel
    elif name == 'scFoundationPerturbationModel':
        from .scfoundation import scFoundationPerturbationModel
        return scFoundationPerturbationModel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'PerturbationModel',
    'ModelConfig',
    'scGPTConfig',
    'scGPTModel',
    'scGPTPerturbationModel',
    'scFoundationConfig',
    'scFoundationModel',
    'scFoundationPerturbationModel',
]
