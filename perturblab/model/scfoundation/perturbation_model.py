import logging
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ...data import PerturbationData
from ..gears.source import GEARS, PertData
from ..gears.source.inference import (compute_metrics, deeper_analysis,
                                      evaluate, non_dropout_analysis)
from ..gears.source.utils import print_sys
from .config import scFoundationConfig
from .model import scFoundationModel

logger = logging.getLogger(__name__)


class scFoundationPerturbationModel(scFoundationModel):
    """
    scFoundation model for perturbation prediction using GEARS framework.
    
    This model extends scFoundationModel to perform gene perturbation prediction.
    It uses the GEARS (Graph-Enhanced Gene Activation Modeling) framework with
    scFoundation as the base encoder.
    
    Architecture:
        scFoundation (encoder) → GEARS (GNN + decoder) → Perturbation prediction
    
    Features:
        - Graph Neural Networks for gene relationships
        - Perturbation-specific embeddings
        - Gene-specific decoders
        - Support for single and combinatorial perturbations
    
    Example:
        ```python
        from perturblab.data import PerturbationData
        from perturblab.model.scfoundation import scFoundationPerturbationModel
        
        # Create model
        model = scFoundationPerturbationModel.from_pretrained(
            'scfoundation-cell',
            device='cuda'
        )
        
        # Prepare data (must be in GEARS format)
        pert_data = PerturbationData(adata, perturb_col='condition')
        pert_data.set_gears_format(fallback_cell_type='unknown')
        pert_data.split_data(split_type='simulation', seed=1)
        pert_data.compute_de_genes(n_top_genes=20)
        
        # Initialize perturbation model
        model.init_perturbation(
            pert_data,
            hidden_size=64,
            num_go_gnn_layers=1,
            num_gene_gnn_layers=1
        )
        
        # Train
        model.train_perturbation(
            epochs=20,
            lr=1e-3,
            result_dir='./results'
        )
        
        # Predict
        predictions = model.predict_perturbation(test_adata, perturbations=['SAMD1'])
        ```
    """
    
    def __init__(
        self,
        config: scFoundationConfig,
        device: str = 'cuda',
        gene_list: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize scFoundation perturbation model.
        
        Args:
            config (scFoundationConfig): Model configuration
            device (str): Device to use. Default: 'cuda'
            gene_list (Optional[List[str]]): List of gene names
            **kwargs: Additional arguments
        """
        super().__init__(config, device, gene_list, **kwargs)
        
        # GEARS components (initialized later)
        self.gears_model = None
        self.pert_data = None
        self.best_model = None
        
        logger.info("Initialized scFoundation perturbation model")
    
    def init_perturbation(
        self,
        dataset: PerturbationData,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        num_similar_genes_go_graph: int = 20,
        num_similar_genes_co_express_graph: int = 20,
        coexpress_threshold: float = 0.4,
        uncertainty: bool = False,
        uncertainty_reg: float = 1,
        direction_lambda: float = 1e-1,
        no_perturb: bool = False,
        cell_fitness_pred: bool = False,
        weight_bias_track: bool = False,
        proj_name: str = 'GEARS',
        exp_name: str = 'scFoundation-GEARS',
        **kwargs
    ):
        """
        Initialize GEARS perturbation prediction components.
        
        This method sets up the GEARS framework on top of scFoundation encoder.
        The dataset must be in GEARS format (use dataset.set_gears_format() first).
        
        Args:
            dataset (PerturbationData): Training data in GEARS format
            hidden_size (int): Hidden dimension for GEARS. Default: 64
            num_go_gnn_layers (int): Number of GO GNN layers. Default: 1
            num_gene_gnn_layers (int): Number of gene GNN layers. Default: 1
            decoder_hidden_size (int): Decoder hidden dimension. Default: 16
            num_similar_genes_go_graph (int): K for GO graph. Default: 20
            num_similar_genes_co_express_graph (int): K for co-expression graph. Default: 20
            coexpress_threshold (float): Co-expression threshold. Default: 0.4
            uncertainty (bool): Enable uncertainty mode. Default: False
            uncertainty_reg (float): Uncertainty regularization. Default: 1
            direction_lambda (float): Direction loss weight. Default: 1e-1
            no_perturb (bool): Baseline mode (no perturbation). Default: False
            cell_fitness_pred (bool): Enable cell fitness prediction. Default: False
            weight_bias_track (bool): Track with wandb. Default: False
            proj_name (str): Wandb project name. Default: 'GEARS'
            exp_name (str): Wandb experiment name. Default: 'scFoundation-GEARS'
            **kwargs: Additional GEARS arguments
        
        Raises:
            ValueError: If dataset is not in GEARS format
        """
        # Verify GEARS format
        if not dataset.gears_format:
            raise ValueError(
                "Dataset must be in GEARS format. "
                "Please call dataset.set_gears_format() first."
            )
        
        # Check required fields
        required_fields = ['condition', 'split', 'ctrl_indices']
        missing = []
        if 'condition' not in dataset.adata.obs:
            missing.append('condition (use set_gears_format)')
        if 'split' not in dataset.adata.obs:
            missing.append('split (use split_data)')
        if 'ctrl_indices' not in dataset.adata.obsm:
            missing.append('ctrl_indices (use pair_cells)')
        
        if missing:
            raise ValueError(
                f"Dataset missing required fields: {', '.join(missing)}"
            )
        
        logger.info("Converting PerturbationData to GEARS PertData format...")
        
        # Convert to GEARS PertData
        # We'll use a temporary directory for GEARS internal data
        temp_data_path = os.path.join(os.getcwd(), '.gears_temp')
        os.makedirs(temp_data_path, exist_ok=True)
        
        self.pert_data = dataset.to_gears(
            data_path=temp_data_path,
            check_de_genes=True
        )
        
        logger.info(f"Initializing GEARS with hidden_size={hidden_size}")
        
        # Initialize GEARS model
        self.gears_model = GEARS(
            self.pert_data,
            device=self.device,
            weight_bias_track=weight_bias_track,
            proj_name=proj_name,
            exp_name=exp_name
        )
        
        # Get scFoundation encoder path for GEARS
        # GEARS will load scFoundation weights from this path
        encoder_config = self.config.to_model_config_dict()
        
        # Initialize GEARS model with scFoundation encoder
        self.gears_model.model_initialize(
            hidden_size=hidden_size,
            num_go_gnn_layers=num_go_gnn_layers,
            num_gene_gnn_layers=num_gene_gnn_layers,
            decoder_hidden_size=decoder_hidden_size,
            num_similar_genes_go_graph=num_similar_genes_go_graph,
            num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
            coexpress_threshold=coexpress_threshold,
            uncertainty=uncertainty,
            uncertainty_reg=uncertainty_reg,
            direction_lambda=direction_lambda,
            no_perturb=no_perturb,
            cell_fitness_pred=cell_fitness_pred,
            # Pass scFoundation model info to GEARS
            model_type='scfoundation',
            load_path=None,  # We'll directly set the encoder
            **kwargs
        )
        
        # Replace GEARS's encoder with our scFoundation model
        # This is more direct than loading from path
        self.gears_model.model.singlecell_model = self.model
        self.gears_model.model.pretrained = True
        
        logger.info("✓ GEARS perturbation model initialized successfully")
        logger.info(f"  - Base encoder: scFoundation ({self.config.model_series}-{self.config.model_name})")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - GO GNN layers: {num_go_gnn_layers}")
        logger.info(f"  - Gene GNN layers: {num_gene_gnn_layers}")
        logger.info(f"  - Number of genes: {self.gears_model.num_genes}")
        logger.info(f"  - Number of perturbations: {self.gears_model.num_perts}")
    
    def train_model(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        result_dir: str = './results',
        finetune_method: Optional[str] = None
    ):
        """
        Train perturbation prediction model.
        
        Args:
            epochs (int): Number of training epochs. Default: 20
            lr (float): Learning rate. Default: 1e-3
            weight_decay (float): Weight decay. Default: 5e-4
            result_dir (str): Directory to save results. Default: './results'
            finetune_method (str): Fine-tuning strategy for scFoundation encoder:
                - None: Train all parameters
                - 'frozen': Freeze scFoundation encoder
                - 'finetune_lr_1': Fine-tune encoder with 0.1x learning rate
                Default: None
        
        Returns:
            Dict: Training results and metrics
        """
        if self.gears_model is None:
            raise ValueError(
                "GEARS model not initialized. "
                "Please call init_perturbation() first."
            )
        
        os.makedirs(result_dir, exist_ok=True)
        
        # Set fine-tuning method in config
        if finetune_method is not None:
            self.gears_model.config['finetune_method'] = finetune_method
        
        logger.info("="*60)
        logger.info("Starting perturbation prediction training")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Result directory: {result_dir}")
        if finetune_method:
            logger.info(f"Fine-tuning method: {finetune_method}")
        logger.info("="*60)
        
        # Train using GEARS
        self.gears_model.train(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            result_dir=result_dir
        )
        
        # Store best model
        self.best_model = deepcopy(self.gears_model.best_model)
        
        logger.info("✓ Training completed")
        
        return {
            'best_model': self.best_model,
            'config': self.gears_model.config
        }
    
    def predict_perturbation(
        self,
        dataset: PerturbationData,
        perturbations: Optional[List[str]] = None,
        return_anndata: bool = False
    ):
        """
        Predict gene expression changes for given perturbations.
        
        Args:
            dataset (PerturbationData): Test data
            perturbations (Optional[List[str]]): List of perturbations to predict.
                If None, predict for all perturbations in dataset.
            return_anndata (bool): Return results as AnnData. Default: False
        
        Returns:
            Dict or AnnData: Predictions and metrics
        """
        if self.best_model is None:
            if self.gears_model is None:
                raise ValueError(
                    "Model not initialized. "
                    "Please call init_perturbation() and train_perturbation() first."
                )
            self.best_model = self.gears_model.model
        
        logger.info("Predicting perturbation effects...")
        
        # Use GEARS predict method
        # This requires the data to be in PyG format
        test_loader = self.pert_data.dataloader.get('test_loader')
        
        if test_loader is None:
            raise ValueError("No test data available")
        
        # Evaluate on test set
        test_res = evaluate(
            test_loader,
            self.best_model,
            self.gears_model.config.get('uncertainty', False),
            self.device
        )
        
        test_metrics, test_pert_res = compute_metrics(test_res)
        
        logger.info("✓ Prediction completed")
        logger.info(f"  - Test MSE: {test_metrics['mse']:.4f}")
        logger.info(f"  - Test Top 20 DE MSE: {test_metrics['mse_de']:.4f}")
        logger.info(f"  - Test Pearson: {test_metrics['pearson']:.4f}")
        
        results = {
            'predictions': test_res,
            'metrics': test_metrics,
            'per_perturbation_metrics': test_pert_res
        }
        
        if return_anndata:
            # Convert results back to AnnData format
            # This would require additional implementation
            logger.warning("return_anndata=True not yet implemented, returning dict")
        
        return results
    
    def save_perturbation(self, save_directory: str):
        """
        Save perturbation model and GEARS configuration.
        
        Args:
            save_directory (str): Directory to save model
        """
        if self.gears_model is None:
            raise ValueError("No GEARS model to save")
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base scFoundation model
        super().save(save_directory)
        
        # Save GEARS-specific components
        self.gears_model.save_model(save_directory)
        
        logger.info(f"✓ Perturbation model saved to {save_directory}")
    
    def load_perturbation(self, load_directory: str):
        """
        Load perturbation model and GEARS configuration.
        
        Args:
            load_directory (str): Directory to load model from
        """
        if not os.path.exists(load_directory):
            raise FileNotFoundError(f"Directory not found: {load_directory}")
        
        # Load GEARS model
        if self.gears_model is None:
            raise ValueError(
                "GEARS model not initialized. "
                "Please call init_perturbation() first, then load weights."
            )
        
        self.gears_model.load_pretrained(load_directory)
        self.best_model = self.gears_model.model
        
        logger.info(f"✓ Perturbation model loaded from {load_directory}")

