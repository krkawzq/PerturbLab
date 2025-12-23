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
    """scELMo: Embeddings from Language Models for Single-cell Data Analysis.
    
    scELMo uses language model embeddings (e.g., GPT-3.5) to extract features from
    single-cell data. It loads pre-computed gene embeddings and aggregates them
    to compute cell embeddings.
    
    Features:
    - Load pre-computed gene embeddings from pickle files
    - Compute cell embeddings using weighted average or average aggregation
    - Support for multiple embedding sources
    """
    
    def __init__(
        self,
        config: scELMoConfig,
        gene_embeddings: Optional[Dict[str, np.ndarray]] = None,
        gene_embeddings_path: Optional[str] = None,
    ):
        """Initializes scELMo model.
        
        Args:
            config: scELMo configuration object.
            gene_embeddings: Optional pre-loaded gene embeddings dictionary.
            gene_embeddings_path: Optional path to pickle file containing gene embeddings.
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
    
    def to(self, device: str):
        return self
    
    def train(self, mode: bool = True):
        return self
    
    def eval(self):
        return self
    
    def load_gene_embeddings(self, path: str):
        """Loads gene embeddings from a pickle file.
        
        Args:
            path: Path to pickle file containing gene embeddings.
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
        """Annotates genes and computes embeddings using language model API.
        
        Args:
            gene_list: List of gene names to annotate.
            api_key: OpenAI API key.
            chat_model: Model for generating gene descriptions.
            embedding_model: Model for generating embeddings.
            prompt_template: Custom prompt template.
            delay_sec: Delay between API calls in seconds.
            save_path: Optional path to save gene embeddings.
            save_descriptions: Optional path to save gene descriptions.
            max_retries: Maximum number of retries for failed API calls.
            **kwargs: Additional arguments passed to API calls.
        
        Returns:
            Dictionary mapping gene names to their embeddings.
        """
        
        import openai
        
        if not api_key:
            raise ValueError("api_key is required for annotate_gene_embeddings")
        
        # Detect OpenAI SDK version and setup
        try:
            from openai import __version__ as openai_version
            major_version = int(openai_version.split('.')[0])
            use_new_api = major_version >= 1
        except:
            # Fallback: try to detect by checking if OpenAI class exists
            use_new_api = hasattr(openai, 'OpenAI')
        
        if use_new_api:
            # New API (>= 1.0.0)
            client = openai.OpenAI(api_key=api_key)
            logger.info(f"Using OpenAI SDK v{openai_version} (new API)")
        else:
            # Old API (< 1.0.0)
            openai.api_key = api_key
            logger.info(f"Using OpenAI SDK v{openai_version} (legacy API)")
        
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
                    
                    if use_new_api:
                        # New API (>= 1.0.0)
                        completion = client.chat.completions.create(
                            model=chat_model,
                            messages=[{"role": "user", "content": prompt}],
                            **kwargs
                        )
                        description = completion.choices[0].message.content
                    else:
                        # Old API (< 1.0.0)
                        completion = openai.ChatCompletion.create(
                            model=chat_model,
                            messages=[{"role": "user", "content": prompt}],
                            **kwargs
                        )
                        description = completion.choices[0].message.content
                    
                    gene_name_to_description[gene] = description
                    
                    # Step 2: Get embedding of the description
                    embedding = self._get_embedding(
                        description, embedding_model, 
                        client if use_new_api else None
                    )
                    gene_name_to_embedding[gene] = embedding
                    
                    success = True
                    logger.debug(f"Successfully processed gene {gene}")
                    
                    # Delay to avoid rate limits
                    if delay_sec > 0:
                        time.sleep(delay_sec)
                    
                except Exception as e:
                    # Handle both old and new API errors
                    is_retryable = False
                    if use_new_api:
                        # New API exceptions
                        is_retryable = any([
                            'APIError' in type(e).__name__,
                            'APIConnectionError' in type(e).__name__,
                            'RateLimitError' in type(e).__name__,
                            'APITimeoutError' in type(e).__name__,
                        ])
                    else:
                        # Old API exceptions
                        is_retryable = any([
                            hasattr(openai, 'error') and isinstance(e, (
                                getattr(openai.error, 'APIError', type(None)),
                                getattr(openai.error, 'APIConnectionError', type(None)),
                                getattr(openai.error, 'RateLimitError', type(None)),
                                getattr(openai.error, 'ServiceUnavailableError', type(None)),
                                getattr(openai.error, 'Timeout', type(None)),
                            ))
                        ])
                    
                    if is_retryable:
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
                            break
                    else:
                        # Non-retryable error
                        logger.error(f"Non-retryable error for gene {gene}: {e}")
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
    
    def _get_embedding(
        self, 
        text: str, 
        model: str, 
        client=None
    ) -> np.ndarray:
        """Gets embedding for text using OpenAI Embedding API.
        
        Args:
            text: Input text to embed.
            model: Embedding model name.
            client: OpenAI client instance (for new API >= 1.0.0).
        
        Returns:
            Embedding vector as numpy array.
        """
        # Clean text (remove newlines)
        text = text.replace("\n", " ")
        
        # Get embedding
        if client is not None:
            # New API (>= 1.0.0)
            response = client.embeddings.create(input=[text], model=model)
            embedding = np.array(response.data[0].embedding)
        else:
            # Old API (< 1.0.0)
            import openai
            response = openai.Embedding.create(input=[text], model=model)
            embedding = np.array(response['data'][0]['embedding'])
        
        return embedding
    
    def _get_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        client=None,
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """Gets embeddings for multiple texts in batches using OpenAI Embedding API.
        
        Args:
            texts: List of input texts to embed.
            model: Embedding model name.
            client: OpenAI client instance (for new API >= 1.0.0).
            batch_size: Maximum number of texts per API call.
        
        Returns:
            List of embedding vectors as numpy arrays.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Clean texts (remove newlines)
            batch = [text.replace("\n", " ") for text in batch]
            
            # Get embeddings
            if client is not None:
                # New API (>= 1.0.0)
                response = client.embeddings.create(input=batch, model=model)
                embeddings = [np.array(item.embedding) for item in response.data]
            else:
                # Old API (< 1.0.0)
                import openai
                response = openai.Embedding.create(input=batch, model=model)
                embeddings = [np.array(item['embedding']) for item in response['data']]
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def predict_embeddings(
        self,
        dataset: Union[AnnData, PerturbationData],
        aggregation_mode: Optional[str] = None,
        gene_list: Optional[list] = None,
        split: Optional[str] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Computes cell embeddings from gene expression data.
        
        Args:
            dataset: AnnData or PerturbationData object.
            aggregation_mode: Aggregation mode ('wa' or 'aa').
            gene_list: Optional list of gene names to use.
            split: Specific split to predict on ('train', 'val', 'test'), or None for all splits.
        
        Returns:
            Dictionary with 'cell' key containing cell embeddings.
            If split is None and dataset has splits, returns nested dict.
        """
        if self.gene_embeddings is None:
            raise ValueError(
                "Gene embeddings not loaded. "
                "Please load embeddings using load_gene_embeddings() or from_pretrained()."
            )
        
        # Get AnnData object
        if isinstance(dataset, PerturbationData):
            adata = dataset.adata
        else:
            adata = dataset
        
        # Handle split logic
        has_split = "split" in adata.obs
        
        if split is not None:
            if has_split:
                if split not in adata.obs["split"].values:
                    raise ValueError(f"Split '{split}' not found in dataset")
                adata = adata[adata.obs["split"] == split]
            return self._predict_embeddings_single(adata, aggregation_mode, gene_list)
        elif has_split:
            split_names = adata.obs["split"].unique()
            result = {}
            for split_name in split_names:
                subset = adata[adata.obs["split"] == split_name]
                emb_result = self._predict_embeddings_single(subset, aggregation_mode, gene_list)
                result[str(split_name)] = emb_result
            return result
        else:
            return {'train': self._predict_embeddings_single(adata, aggregation_mode, gene_list)}
    
    def _predict_embeddings_single(
        self,
        adata: AnnData,
        aggregation_mode: Optional[str] = None,
        gene_list: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes embeddings for a single AnnData subset."""
        if aggregation_mode is None:
            aggregation_mode = self.config.aggregation_mode
        
        if aggregation_mode not in ['wa', 'aa']:
            raise ValueError(f"aggregation_mode must be 'wa' or 'aa', got {aggregation_mode}")
        
        if gene_list is None:
            gene_list = list(adata.var.index)
        
        n_genes = len(gene_list)
        embed_dim = self.config.embedding_dim
        lookup_embed = np.zeros(shape=(n_genes, embed_dim), dtype=np.float32)
        
        # Build gene embedding lookup matrix
        count_missing = 0
        for i, gene in enumerate(gene_list):
            emb = self.gene_embeddings.get(gene)
            if emb is not None:
                # Compatible with both list and ndarray
                emb_val = emb if isinstance(emb, np.ndarray) else np.array(emb)
                lookup_embed[i, :] = emb_val.flatten()[:embed_dim]
            else:
                count_missing += 1
        
        # Check missing gene ratio and warn if too high
        missing_ratio = count_missing / n_genes
        if count_missing > 0:
            logger.warning(
                f"Missing {count_missing}/{n_genes} ({missing_ratio:.1%}) genes in embeddings. "
                f"They will be zero-padded."
            )
            if missing_ratio > 0.5:
                logger.error(
                    f"Over 50% of genes are missing embeddings! "
                    f"This may indicate gene ID mismatch (e.g., Symbol vs Ensembl ID). "
                    f"Results may be unreliable."
                )
        
        # Keep X sparse to avoid memory explosion
        X = adata.X
        
        # Ensure X is numeric type
        if hasattr(X, 'dtype') and X.dtype == object:
            X = X.astype(np.float32)
        
        # Matrix multiplication: (N_cells, N_genes) @ (N_genes, Embed_dim) -> (N_cells, Embed_dim)
        # scipy.sparse supports @ operator directly
        raw_embeddings = X @ lookup_embed
        
        if aggregation_mode == 'wa':
            # Weighted average: (X @ E) / cell_totals
            # For sparse matrix, sum(axis=1) returns np.matrix or np.array
            cell_totals = np.array(X.sum(axis=1)).flatten()
            
            # Avoid division by zero
            cell_totals = np.where(cell_totals == 0, 1.0, cell_totals)
            
            # Broadcast division: (N, Dim) / (N, 1)
            cell_embeddings = raw_embeddings / cell_totals[:, None]
        else:
            # Simple average (aa)
            cell_embeddings = raw_embeddings / n_genes
        
        # Ensure output is float32
        cell_embeddings = cell_embeddings.astype(np.float32)
        
        logger.info(
            f"Computed cell embeddings: shape {cell_embeddings.shape}, "
            f"mode: {aggregation_mode}, missing genes: {missing_ratio:.1%}"
        )
        
        return {'cell': cell_embeddings}
    
    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        return self
    
    def eval(self):
        """Sets the model to evaluation mode."""
        return self
    
    def save(self, model_path: str):
        """Saves scELMo model configuration and gene embeddings.
        
        Args:
            model_path: Directory to save the model.
        """
        os.makedirs(model_path, exist_ok=True)
        
        # 1. Save config
        config_path = os.path.join(model_path, 'config.json')
        self.config.save(config_path)
        
        # 2. Save gene embeddings if available
        if self.gene_embeddings is not None:
            embeddings_path = os.path.join(model_path, 'gene_embeddings.pkl')
            
            # Standardize save format: wrapper dict with metadata
            save_data = {
                'embeddings': self.gene_embeddings,
                'gene_list': list(self.gene_embeddings.keys())
            }
            
            with open(embeddings_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved gene embeddings ({len(save_data['gene_list'])} genes) to {embeddings_path}")
        
        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(
        cls,
        model_path: str,
        aggregation_mode: Optional[str] = None,
        embedding_dim: Optional[int] = None,  # Added missing argument
        gene_embeddings_path: Optional[str] = None,
        **kwargs
    ) -> 'scELMoModel':
        """Loads scELMo model from a saved directory or pickle file.
        
        Args:
            model_path: Path to the saved model directory (containing config.json) 
                       OR direct path to a .pkl file.
            aggregation_mode: Optional override for aggregation mode.
            embedding_dim: Optional override for embedding dimension.
            gene_embeddings_path: Optional explicit path to embeddings file.
            **kwargs: Additional config overrides.
        
        Returns:
            Loaded scELMoModel instance.
        """
        # 1. Determine paths for config and embeddings
        if os.path.isfile(model_path) and (model_path.endswith('.pkl') or model_path.endswith('.pickle')):
            # Case A: model_path is directly the embeddings file
            logger.info(f"Loading directly from pickle file: {model_path}")
            target_config_path = None
            target_embeddings_path = model_path
        else:
            # Case B: model_path is a directory (Standard)
            target_config_path = os.path.join(model_path, 'config.json')
            target_embeddings_path = os.path.join(model_path, 'gene_embeddings.pkl')

        # Allow explicit override of embeddings path
        if gene_embeddings_path:
            target_embeddings_path = gene_embeddings_path

        # 2. Load Configuration
        if target_config_path and os.path.exists(target_config_path):
            config = scELMoConfig.load(target_config_path)
            logger.info(f"Loaded config from {target_config_path}")
        else:
            logger.warning(f"Config file not found at {target_config_path if target_config_path else 'None'}. Using default config.")
            config = scELMoConfig()

        # 3. Apply Overrides (Priority: kwargs > args > loaded config)
        if aggregation_mode is not None:
            config.aggregation_mode = aggregation_mode
        if embedding_dim is not None:
            config.embedding_dim = embedding_dim
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"Override config.{key} = {value}")

        # 4. Load Gene Embeddings
        gene_embeddings = None
        if os.path.exists(target_embeddings_path):
            logger.info(f"Loading gene embeddings from: {target_embeddings_path}")
            try:
                with open(target_embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Handle both 'wrapper dict' format (from save method) and 'raw dict' format
                if isinstance(data, dict):
                    if 'embeddings' in data and 'gene_list' in data:
                        gene_embeddings = data['embeddings']
                    else:
                        gene_embeddings = data
                else:
                    logger.warning(f"Unexpected data type in embeddings file: {type(data)}")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
        else:
            logger.warning(f"Embeddings file not found at: {target_embeddings_path}")

        # 5. Instantiate and return
        model = cls(config=config, gene_embeddings=gene_embeddings)
        logger.info("✓ scELMo model loaded successfully")
        return model

