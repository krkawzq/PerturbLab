from ..configuration import ModelConfig
import logging

logger = logging.getLogger(__name__)

class CellFMConfig(ModelConfig):
    def __init__(self,
                 model_series: str = 'cellfm',
                 model_name: str = 'default',
                 **kwargs):
        super().__init__(model_series=model_series, model_name=model_name, **kwargs)

    def to_dict(self):
        return super().to_dict()
