import torch
import torch.nn as nn
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Dict, ClassVar
from pathlib import Path

from .configuration import ModelConfig

def download_from_huggingface(
    model_name_or_path: str,
    organization: str = "perturblab",
    **hf_kwargs
) -> str:
    """
    Helper function to download models from HuggingFace.
    This is a standalone utility function that can be used by any model class.
    
    Args:
        model_name_or_path: Can be:
            - Model name: "scgpt-human" (will be resolved to "{organization}/scgpt-human")
            - Full HuggingFace repo ID: "perturblab/scgpt-human"
            - URL or other identifier
        organization: HuggingFace organization name (default: "perturblab")
        **hf_kwargs: Additional arguments for huggingface_hub.snapshot_download
            - revision: str, specific branch/tag/commit
            - token: str, HuggingFace API token for private repos
            - cache_dir: str, custom cache directory
            - force_download: bool, force re-download
            - resume_download: bool, resume interrupted downloads
    
    Returns:
        Local path to the downloaded model directory
        
    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails
        
    Example:
        >>> path = download_from_huggingface("scgpt-human")
        >>> path = download_from_huggingface("perturblab/scgpt-brain", revision="v1.0")
    """
    try:
        from huggingface_hub import snapshot_download
        
        # If it's just a model name (e.g., "scgpt-human"), prepend organization
        if "/" not in model_name_or_path:
            repo_id = f"{organization}/{model_name_or_path}"
        else:
            repo_id = model_name_or_path
        
        # Download from HuggingFace (uses cache automatically)
        model_path = snapshot_download(
            repo_id=repo_id,
            **hf_kwargs
        )
        
        return model_path
        
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from HuggingFace. "
            "Install it with: pip install huggingface_hub"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to download model '{model_name_or_path}' from HuggingFace.\n"
            f"Error: {str(e)}"
        )


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
