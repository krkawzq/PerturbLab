import json
import logging
import os
import pickle
import time
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm

from ...data import PerturbationData
from ...utils import download_from_huggingface
from ..base import PerturbationModel
from .config import scELMoConfig

logger = logging.getLogger(__name__)

class scELMoModel(PerturbationModel):
    """
    scELMo: Embeddings from Language Models for Single-cell Data Analysis.
    
    scELMo uses language model embeddings (e.g., GPT-3.5) to extract features from
    single-cell data. It loads pre-computed gene embeddings and aggregates them
    to compute cell embeddings.
    
    Features:
    - Load pre-computed gene embeddings from pickle files
    - Compute cell embeddings using weighted average or average aggregation
    - Support for multiple embedding sources
    - Support for from_pretrained to load pre-computed embeddings
    
    Args:
        config (scELMoConfig): Model configuration
        gene_embeddings (Optional[Dict]): Pre-loaded gene embeddings dictionary.
            If None, embeddings can be loaded using load_gene_embeddings() or from_pretrained().
    
    Example:
        ```python
        from perturblab.model.scelmo import scELMoModel, scELMoConfig
        
        # Load from pretrained embeddings
        model = scELMoModel.from_pretrained(
            'gpt-3.5-gene-embeddings',
            aggregation_mode='wa'
        )
        
        # Compute cell embeddings
        embeddings = model.predict_embeddings(
            adata,
            aggregation_mode='wa'
        )
        ```
    """
    
    def __init__(
        self,
        config: scELMoConfig,
        gene_embeddings: Optional[Dict[str, np.ndarray]] = None,
        gene_embeddings_path: Optional[str] = None,
    ):
        """
        Initialize scELMo model.
        
        Args:
            config: scELMoConfig configuration object
            gene_embeddings: Pre-loaded gene embeddings dictionary (optional)
            gene_embeddings_path: Path to pickle file containing gene embeddings (optional)
                If both gene_embeddings and gene_embeddings_path are provided,
                gene_embeddings takes precedence.
        """
        super().__init__(config)
        
        self.config = config
        self.gene_embeddings: Optional[Dict[str, np.ndarray]] = None
        
        # Load gene embeddings if provided
        if gene_embeddings is not None:
            self.gene_embeddings = gene_embeddings
            logger.info(f"Loaded {len(gene_embeddings)} gene embeddings")
        elif gene_embeddings_path is not None:
            self.load_gene_embeddings(gene_embeddings_path)
        
        logger.info("Initialized scELMo model")
    
    def load_gene_embeddings(self, path: str):
        """
        Load gene embeddings from a pickle file.
        
        Supports two formats:
        1. Dictionary format: {gene_name: embedding_array}
        2. Saved format: {'embeddings': {gene_name: embedding_array}, 'gene_list': [gene_names]}
        
        Args:
            path: Path to pickle file containing gene embeddings
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gene embeddings file not found: {path}")
        
        logger.info(f"Loading gene embeddings from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both formats: dict directly or dict with 'embeddings' key
        if isinstance(data, dict):
            if 'embeddings' in data:
                # New format with gene_list
                self.gene_embeddings = data['embeddings']
                gene_list = data.get('gene_list', list(self.gene_embeddings.keys()))
                logger.info(f"Loaded {len(self.gene_embeddings)} gene embeddings (gene_list: {len(gene_list)} genes)")
            else:
                # Old format: direct dictionary
                self.gene_embeddings = data
                logger.info(f"Loaded {len(self.gene_embeddings)} gene embeddings")
        else:
            raise ValueError(f"Unexpected format in pickle file: {type(data)}")
        
        # Check embedding dimension
        if self.gene_embeddings:
            sample_emb = next(iter(self.gene_embeddings.values()))
            if isinstance(sample_emb, np.ndarray):
                embed_dim = sample_emb.shape[-1] if sample_emb.ndim > 0 else len(sample_emb)
            else:
                embed_dim = len(sample_emb)
            
            if embed_dim != self.config.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: config says {self.config.embedding_dim}, "
                    f"but loaded embeddings have dimension {embed_dim}. "
                    f"Updating config..."
                )
                self.config.embedding_dim = embed_dim
    
    def annotate_gene_embeddings(
        self,
        gene_list: List[str],
        api_key: str = 'sk-',
        chat_model: str = 'gpt-3.5-turbo-1106',
        embedding_model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        delay_sec: float = 1.0,
        save_path: Optional[str] = None,
        save_descriptions: Optional[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Annotate genes and compute embeddings using language model API.
        
        This method follows the scELMo workflow:
        1. For each gene, generate a description using ChatCompletion API
        2. Get embedding of the description using Embedding API
        3. Store embeddings in a dictionary
        
        Args:
            gene_list: List of gene names to annotate
            api_key: OpenAI API key (required for API calls)
            chat_model: Model for generating gene descriptions. Default: 'gpt-3.5-turbo-1106'
            embedding_model: Model for generating embeddings. If None, uses config.api_model
            prompt_template: Custom prompt template. Should contain {gene} placeholder.
                Default: "Please summarize the major function of gene: {gene}. "
                        "Use academic language in one paragraph and include pathway information."
            delay_sec: Delay between API calls (seconds) to avoid rate limits. Default: 1.0
            save_path: Optional path to save gene embeddings pickle file
            save_descriptions: Optional path to save gene descriptions pickle file
            max_retries: Maximum number of retries for failed API calls. Default: 3
            **kwargs: Additional arguments passed to API calls
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping gene names to their embeddings
        
        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        
        Example:
            ```python
            model = scELMoModel(config)
            
            # Annotate genes and get embeddings
            gene_list = ['TP53', 'BRCA1', 'EGFR']
            embeddings = model.annotate_gene_embeddings(
                gene_list=gene_list,
                api_key='your-api-key',
                delay_sec=1.0,
                save_path='./gene_embeddings.pkl'
            )
            
            # Use the embeddings
            model.gene_embeddings = embeddings
            ```
        """
        
        import openai
        
        if not api_key:
            raise ValueError("api_key is required for annotate_gene_embeddings")
        
        # Set API key
        openai.api_key = api_key
        
        # Determine embedding model
        if embedding_model is None:
            embedding_model = self.config.api_model
        
        # Default prompt template
        if prompt_template is None:
            prompt_template = (
                "Please summarize the major function of gene: {gene}. "
                "Use academic language in one paragraph and include pathway information."
            )
        
        # Initialize storage
        gene_name_to_description = {}
        gene_name_to_embedding = {}
        
        logger.info(f"Annotating {len(gene_list)} genes using {chat_model}...")
        
        # Process each gene
        for gene in tqdm(gene_list, desc="Annotating genes"):
            if gene in gene_name_to_embedding:
                logger.debug(f"Gene {gene} already processed, skipping")
                continue
            
            retries = 0
            success = False
            
            while retries < max_retries and not success:
                try:
                    # Step 1: Generate gene description using ChatCompletion
                    prompt = prompt_template.format(gene=gene)
                    completion = openai.ChatCompletion.create(
                        model=chat_model,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    
                    description = completion.choices[0].message.content
                    gene_name_to_description[gene] = description
                    
                    # Step 2: Get embedding of the description
                    embedding = self._get_embedding(description, embedding_model)
                    gene_name_to_embedding[gene] = embedding
                    
                    success = True
                    logger.debug(f"Successfully processed gene {gene}")
                    
                    # Delay to avoid rate limits
                    if delay_sec > 0:
                        time.sleep(delay_sec)
                    
                except (
                    openai.APIError,
                    openai.error.APIError,
                    openai.error.APIConnectionError,
                    openai.error.RateLimitError,
                    openai.error.ServiceUnavailableError,
                    openai.error.Timeout
                ) as e:
                    retries += 1
                    if retries < max_retries:
                        wait_time = delay_sec * (2 ** retries)  # Exponential backoff
                        logger.warning(
                            f"API error for gene {gene}: {e}. "
                            f"Retrying ({retries}/{max_retries}) after {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to process gene {gene} after {max_retries} retries: {e}")
                        # Continue with next gene instead of raising
                        break
        
        logger.info(f"Successfully annotated {len(gene_name_to_embedding)} out of {len(gene_list)} genes")
        
        # Update embedding dimension if needed
        if gene_name_to_embedding:
            sample_emb = next(iter(gene_name_to_embedding.values()))
            embed_dim = len(sample_emb) if isinstance(sample_emb, np.ndarray) else len(sample_emb)
            if embed_dim != self.config.embedding_dim:
                logger.info(f"Updating embedding_dim from {self.config.embedding_dim} to {embed_dim}")
                self.config.embedding_dim = embed_dim
        
        # Save embeddings if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(gene_name_to_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved gene embeddings to {save_path}")
        
        # Save descriptions if requested
        if save_descriptions:
            os.makedirs(os.path.dirname(save_descriptions) if os.path.dirname(save_descriptions) else '.', exist_ok=True)
            with open(save_descriptions, 'wb') as f:
                pickle.dump(gene_name_to_description, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved gene descriptions to {save_descriptions}")
        
        # Update model's gene embeddings
        if self.gene_embeddings is None:
            self.gene_embeddings = {}
        self.gene_embeddings.update(gene_name_to_embedding)
        
        return gene_name_to_embedding
    
    def _get_embedding(self, text: str, model: str) -> np.ndarray:
        """
        Get embedding for a text using OpenAI Embedding API.
        
        Args:
            text: Text to embed
            model: Embedding model name
        
        Returns:
            np.ndarray: Embedding vector
        """
        # Clean text (remove newlines)
        text = text.replace("\n", " ")
        
        # Get embedding
        import openai
        response = openai.Embedding.create(input=[text], model=model)
        embedding = np.array(response['data'][0]['embedding'])
        
        return embedding
    
    def predict_embeddings(
        self,
        adata: Union[AnnData, PerturbationData],
        aggregation_mode: Optional[str] = None,
        gene_list: Optional[list] = None,
        return_numpy: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Compute cell embeddings from gene expression data.
        
        This method aggregates gene embeddings based on gene expression values
        to compute cell-level embeddings.
        
        Args:
            adata: AnnData or PerturbationData object
            aggregation_mode: Aggregation mode ('wa' for weighted average or 'aa' for average).
                If None, uses config.aggregation_mode
            gene_list: Optional list of gene names to use. If None, uses adata.var.index
            return_numpy: Whether to return numpy array. Default: True
        
        Returns:
            np.ndarray: Cell embeddings of shape (n_cells, embedding_dim)
        
        Raises:
            ValueError: If gene embeddings are not loaded
        """
        if self.gene_embeddings is None:
            raise ValueError(
                "Gene embeddings not loaded. "
                "Please load embeddings using load_gene_embeddings() or from_pretrained()."
            )
        
        # Get AnnData object
        if isinstance(adata, PerturbationData):
            adata = adata.adata
        
        # Determine aggregation mode
        if aggregation_mode is None:
            aggregation_mode = self.config.aggregation_mode
        
        if aggregation_mode not in ['wa', 'aa']:
            raise ValueError(f"aggregation_mode must be 'wa' or 'aa', got {aggregation_mode}")
        
        # Get gene list
        if gene_list is None:
            gene_list = list(adata.var.index)
        
        # Get expression matrix
        X = adata.X
        if issparse(X):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)
        
        # Build lookup matrix for gene embeddings
        n_genes = len(gene_list)
        embed_dim = self.config.embedding_dim
        lookup_embed = np.zeros(shape=(n_genes, embed_dim), dtype=np.float32)
        
        count_missing = 0
        for i, gene in enumerate(gene_list):
            if gene in self.gene_embeddings:
                emb = self.gene_embeddings[gene]
                # Handle different embedding formats
                if isinstance(emb, np.ndarray):
                    lookup_embed[i, :] = emb.flatten()[:embed_dim]
                else:
                    # Convert to numpy if needed
                    lookup_embed[i, :] = np.array(emb).flatten()[:embed_dim]
            else:
                count_missing += 1
        
        if count_missing > 0:
            logger.warning(
                f"Unable to match {count_missing} out of {n_genes} genes in the embeddings. "
                f"Missing genes will be represented as zero vectors."
            )
        
        # Compute cell embeddings based on aggregation mode
        if aggregation_mode == 'wa':
            # Weighted average: normalize by total expression per cell
            # adata.X / sum(adata.X, axis=1) @ lookup_embed
            cell_totals = np.sum(X, axis=1, keepdims=True)
            # Avoid division by zero
            cell_totals = np.where(cell_totals == 0, 1.0, cell_totals)
            normalized_X = X / cell_totals
            cell_embeddings = normalized_X @ lookup_embed
        else:  # 'aa'
            # Average: (adata.X @ lookup_embed) / n_genes
            cell_embeddings = (X @ lookup_embed) / n_genes
        
        logger.info(
            f"Computed cell embeddings: shape {cell_embeddings.shape}, "
            f"mode: {aggregation_mode}"
        )
        
        return cell_embeddings
    
    def train(self, mode: bool = True):
        """scELMo is a feature extractor, no training needed."""
        return self
    
    def eval(self):
        """scELMo is a feature extractor, always in eval mode."""
        return self
    
    def save(self, save_directory: str):
        """
        Save scELMo model configuration and gene embeddings.
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, 'config.json')
        self.config.save(config_path)
        
        # Save gene embeddings if available
        if self.gene_embeddings is not None:
            embeddings_path = os.path.join(save_directory, 'gene_embeddings.pkl')
            
            # Extract gene list from embeddings
            gene_list = list(self.gene_embeddings.keys())
            
            # Save embeddings with gene list
            save_data = {
                'embeddings': self.gene_embeddings,
                'gene_list': gene_list
            }
            
            with open(embeddings_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved gene embeddings ({len(gene_list)} genes) to {embeddings_path}")
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        aggregation_mode: str = 'wa',
        gene_embeddings_path: Optional[str] = None,
        embedding_dim: int = 1536,
        **kwargs
    ) -> 'scELMoModel':
        """
        Load scELMo model from pretrained embeddings.
        
        This method can handle multiple input formats:
        1. Path to a pickle file (gene_embeddings_path) - loads directly
        2. Path to a directory containing config.json and gene_embeddings.pkl
        3. Model name for HuggingFace download
        
        Args:
            model_name_or_path: Model name, path to directory, or path to pickle file.
                If it's a file path (ends with .pkl), it will be treated as gene_embeddings_path.
            aggregation_mode: Aggregation mode ('wa' or 'aa'). Default: 'wa'
            gene_embeddings_path: Optional path to gene embeddings pickle file.
                If provided, will override embeddings from model directory.
            embedding_dim: Embedding dimension. Will be auto-detected from embeddings if available.
            **kwargs: Additional arguments
        
        Returns:
            scELMoModel: Loaded model instance
        
        Example:
            ```python
            # Load directly from pickle file
            model = scELMoModel.from_pretrained('./embeddings/gpt35_genes.pkl')
            
            # Load from local directory
            model = scELMoModel.from_pretrained('./models/scelmo-gpt35')
            
            # Load with custom embeddings path
            model = scELMoModel.from_pretrained(
                './models/scelmo-gpt35',
                gene_embeddings_path='./embeddings/gpt35_genes.pkl'
            )
            ```
        """
        # Check if model_name_or_path is a pickle file
        if model_name_or_path.endswith('.pkl') or model_name_or_path.endswith('.pickle'):
            if os.path.exists(model_name_or_path):
                # Direct pickle file path
                logger.info(f"Loading model directly from pickle file: {model_name_or_path}")
                config = scELMoConfig(
                    aggregation_mode=aggregation_mode,
                    embedding_dim=embedding_dim,
                    **kwargs
                )
                model = cls(config=config)
                model.load_gene_embeddings(model_name_or_path)
                logger.info("✓ scELMo model loaded from pickle file")
                return model
            else:
                raise FileNotFoundError(f"Gene embeddings file not found: {model_name_or_path}")
        
        # Resolve model path (directory)
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            model_path = model_name_or_path
            logger.info(f"Loading model from local path: {model_path}")
        else:
            # Try built-in models or HuggingFace
            weights_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'weights'
            )
            model_path = os.path.join(weights_dir, model_name_or_path)
            
            if not os.path.isdir(model_path):
                # Try HuggingFace download
                try:
                    logger.info(f"Attempting to download '{model_name_or_path}' from HuggingFace...")
                    model_path = download_from_huggingface(
                        model_name_or_path,
                        organization="perturblab",
                        **kwargs
                    )
                    logger.info(f"✓ Model cached at: {model_path}")
                except Exception as e:
                    raise ValueError(
                        f"Model not found: {model_name_or_path}\n"
                        f"Tried local: {model_path}\n"
                        f"Tried HuggingFace: perturblab/{model_name_or_path}\n"
                        f"Error: {e}"
                    )
        
        # Load config
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            config = scELMoConfig.load(config_path)
        else:
            # Create default config if not found
            logger.warning(f"Config file not found at {config_path}, using default config")
            config = scELMoConfig(
                aggregation_mode=aggregation_mode,
                embedding_dim=embedding_dim,
                **kwargs
            )
        
        # Override aggregation_mode if provided
        if aggregation_mode is not None:
            config.aggregation_mode = aggregation_mode
        
        # Load gene embeddings
        gene_embeddings = None
        
        # Priority: 1) gene_embeddings_path parameter, 2) model directory
        if gene_embeddings_path is not None:
            if os.path.exists(gene_embeddings_path):
                logger.info(f"Loading gene embeddings from provided path: {gene_embeddings_path}")
                # Use load_gene_embeddings to handle both formats
                temp_model = cls(config=config)
                temp_model.load_gene_embeddings(gene_embeddings_path)
                gene_embeddings = temp_model.gene_embeddings
            else:
                logger.warning(f"Provided gene_embeddings_path not found: {gene_embeddings_path}")
        
        if gene_embeddings is None:
            embeddings_path = os.path.join(model_path, 'gene_embeddings.pkl')
            if os.path.exists(embeddings_path):
                logger.info(f"Loading gene embeddings from model directory: {embeddings_path}")
                # Use load_gene_embeddings to handle both formats
                temp_model = cls(config=config)
                temp_model.load_gene_embeddings(embeddings_path)
                gene_embeddings = temp_model.gene_embeddings
        
        # Create model instance
        model = cls(config=config, gene_embeddings=gene_embeddings)
        
        logger.info("✓ scELMo model loaded successfully")
        return model

