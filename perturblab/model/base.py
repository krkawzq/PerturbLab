import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .configuration import ModelConfig


class PerturbationModel(ABC):
    """
    Base class for all perturbation prediction models.
    This class does NOT inherit from nn.Module to allow more flexibility.
    
    Each subclass should implement its own from_pretrained logic,
    which may use HuggingFace, direct URLs, or other sources.
    """
    
    def __init__(self, config: ModelConfig, **kwargs):
        self.config = config
        self.model = None  # Should be set by subclass to the actual nn.Module
    
    @abstractmethod
    def train(self, mode: bool = True):
        """Set the model to training mode."""
        pass
    
    @abstractmethod
    def eval(self):
        """Set the model to evaluation mode."""
        pass
        
    # Model persistence methods
    @abstractmethod
    def save(self, save_directory: str):
        """
        Save model weights and config to a directory.
        
        Args:
            save_directory: Directory path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        **kwargs
    ) -> 'PerturbationModel':
        pass
    
    # Prediction methods - optional to implement
    def predict_embeddings(
        self, 
        dataset, 
        batch_size: int = 32,
        embedding_type: Literal["cell", "gene"] = "cell",
        **kwargs
    ) -> np.ndarray:
        """
        Unified embedding prediction method.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size for inference
            embedding_type: Type of embedding to predict ("cell" or "gene")
            **kwargs: Additional arguments for specific models
            
        Returns:
            np.ndarray: Embeddings of shape (n_samples, embedding_dim) for cells
                       or (n_genes, embedding_dim) for genes
        """
        raise NotImplementedError("This model is not able to predict embeddings.")
    
    def predict_perturbation(self, *args, **kwargs):
        """Predict perturbation effects."""
        raise NotImplementedError("This model is not able to predict perturbation effects.")
    
    def train_model(self, *args, **kwargs):
        """Train the model."""
        raise NotImplementedError("This model is not able to train.")
