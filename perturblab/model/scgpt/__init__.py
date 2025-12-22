def __getattr__(name):
    """惰性导入 scGPT 模型类"""
    if name == 'scGPTConfig':
        from .config import scGPTConfig
        return scGPTConfig
    elif name == 'scGPTModel':
        from .model import scGPTModel
        return scGPTModel
    elif name == 'scGPTPerturbationModel':
        from .model import scGPTPerturbationModel
        return scGPTPerturbationModel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['scGPTConfig', 'scGPTModel', 'scGPTPerturbationModel']
