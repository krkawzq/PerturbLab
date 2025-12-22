import inspect
import json
import os
from typing import Any

from typing_extensions import Self


class ModelConfig:
    def __init__(
        self,
        *,
        model_series: str = None,
        model_name: str = None,
        model_type: str = None,
        **kwargs,
    ):
        self.model_series = model_series
        self.model_name = model_name
        self.model_type = model_type
        # Generate name from model_series and model_name, or use provided name
        self.name = kwargs.get('name') or (f"{model_series}-{model_name}" if model_series and model_name else None)
        
    def _set_all(self, local: dict, kwargs: dict = {}):
        kwargs.update(local)
        for k, v in kwargs.items():
            if k not in ('self', '__class__', 'kwargs'):
                setattr(self, k, v)

    @classmethod
    def _get_key_mapping(cls) -> dict[str, str]:
        """Return mapping from class attribute names to JSON key names"""
        return {}

    def to_dict(self) -> dict[str, Any]:
        mapping = self._get_key_mapping()
        raw_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
        # Apply name mapping if exists
        return {mapping.get(k, k): v for k, v in raw_dict.items()}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Self:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
            
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        mapping = cls._get_key_mapping()
        # Reverse mapping: JSON key -> class attribute
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        sig = inspect.signature(cls.__init__)
        init_kwargs = {}
        
        for name, param in sig.parameters.items():
            if name in ["self", "kwargs"]:
                continue
            
            # Try mapped JSON key first, then original name
            json_key = mapping.get(name, name)
            
            if json_key in config_dict:
                init_kwargs[name] = config_dict[json_key]
            elif param.default is not inspect.Parameter.empty:
                # Use default value from __init__ if not in JSON
                init_kwargs[name] = param.default
            else:
                init_kwargs[name] = None
                
        return cls(**init_kwargs)
