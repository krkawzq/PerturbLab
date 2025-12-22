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
        self.name = kwargs.get('name', None) or model_series + '-' + model_name
        self.model_type = model_type
        self.model_series = model_series
        self.model_name = model_name
    
    
    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
        
