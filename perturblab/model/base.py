import os
from abc import ABC, abstractmethod

from .configuration import ModelConfig


class PerturbationModel(ABC):
    """Base class for all perturbation prediction models.
    
    This class does NOT inherit from nn.Module to allow more flexibility.
    Each subclass should implement its own from_pretrained logic, which may
    use HuggingFace, direct URLs, or other sources.
    """
    
    def __init__(self, config: ModelConfig, **kwargs):
        self.config = config
        self.model = None

    @abstractmethod
    def save(self, save_directory: str):
        """Saves model weights and config to a directory.
        
        Args:
            save_directory: Directory path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        **kwargs
    ) -> 'PerturbationModel':
        """Loads a pretrained model."""
        pass
    
    @abstractmethod
    def to(self, device: str):
        """Moves the model to a device."""
        pass
    
    @abstractmethod
    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        pass
    
    @abstractmethod
    def eval(self):
        """Sets the model to evaluation mode."""
        pass

    def predict_perturbation(self, *args, **kwargs):
        """Predicts perturbation effects.
        
        Returns:
            Dictionary with 'pred' key containing predictions.
        """
        raise NotImplementedError("This model does not support perturbation prediction.")
    
    def predict_embeddings(self, *args, **kwargs):
        """Predicts cell or gene embeddings.
        
        Returns:
            Dictionary with 'cell' or 'gene' keys containing embeddings.
        """
        raise NotImplementedError("This model does not support embedding prediction.")
    
    def get_dataloader(self, *args, **kwargs):
        """Creates DataLoader for the dataset.
        
        Returns:
            DataLoader or dictionary of DataLoaders.
        """
        raise NotImplementedError("This model does not support dataloader creation.")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model.
        
        Returns:
            Dictionary containing model outputs with 'cell' key for embeddings.
        """
        raise NotImplementedError("This model does not support forward pass.")
    
    def compute_loss(self, *args, **kwargs):
        """Computes loss for the given batch.
        
        Returns:
            Dictionary with 'loss' key containing the computed loss.
        """
        raise NotImplementedError("This model does not support loss computation.")
    
    def train_model(self, *args, **kwargs):
        """Trains the model on the provided dataset."""
        raise NotImplementedError("This model does not support training.")

    @abstractmethod
    @classmethod
    def load(cls, model_path: str):
        """Loads a model from a path."""
        pass
    
    @abstractmethod
    def save(self, model_path: str):
        """Saves a model to a path."""
        pass

    @classmethod
    def pretrained_models(cls):
        return cls._pretrained_models

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, use_cache: bool = True, **kwargs):
        """Loads a pretrained model.
        
        Args:
            model_name_or_path: Local path or HF repo ID (e.g. 'perturblab/scgpt')
            use_cache: Whether to use local HF cache
        """
        import os
        from huggingface_hub import snapshot_download

        if os.path.exists(model_name_or_path):
            return cls.load(model_name_or_path)
        
        short_name = model_name_or_path
        prefix = 'perturblab/'
        if short_name.startswith(prefix):
            short_name = short_name[len(prefix):]

        if short_name in cls.pretrained_models():
            
            print(f"Downloading {model_name_or_path} from HuggingFace...")
            model_path = snapshot_download(
                repo_id=model_name_or_path, 
                force_download=not use_cache
            )
            return cls.load(model_path)
        
        raise ValueError(
            f"Model '{model_name_or_path}' not found in local path or pretrained registry. "
            f"Available models: {cls.pretrained_models()}"
        )
