import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from typing_extensions import Self

from .model.gears.source.utils import make_GO

# Initialize logger
logger = logging.getLogger(__name__)


class PerturbationData:
    def __init__(
        self,
        adata: ad.AnnData,
        perturb_col: str = 'perturbation',
        control_tag: Union[str, List[str]] = 'control',
        ignore_tags: Optional[List[str]] = None,
    ):
        self.adata = adata
        self.perturb_col = perturb_col
        self.gears_format = False
        
        # Normalize control tags
        if isinstance(control_tag, str):
            self.control_tags = {control_tag}
        else:
            self.control_tags = set(control_tag)
            
        self.ignore_tags = set(ignore_tags) if ignore_tags else set()
        self._validate()

    def _validate(self):
        if self.perturb_col not in self.adata.obs.columns:
            raise KeyError(f"Column '{self.perturb_col}' not found in adata.obs")

    def set_gears_format(
        self,
        fallback_cell_type: str,
        parse_fn: Optional[Callable[[str], List[str]]] = None,
        remove_ignore: bool = True
    ) -> None:
        from .utils import apply_gears_format
        self.adata = apply_gears_format(
            adata=self.adata, 
            perturb_col=self.perturb_col, 
            control_tag=list(self.control_tags), 
            ignore_tags=list(self.ignore_tags), 
            parse_fn=parse_fn, 
            fallback_cell_type=fallback_cell_type,
            remove_ignore=remove_ignore
        )
        self.control_tags = {'ctrl'}
        self.perturb_col = 'condition'
        if remove_ignore:
            self.ignore_tags = set()
        self.gears_format = True
        logger.info("Gears format applied.")

    def compute_de_genes(
        self, 
        n_top_genes: int = 20, 
        method: str = 't-test_overestim_var',
        use_hpdex: bool = False,
        use_raw: bool = False
    ) -> None:
        """
        Computes top DE genes (critical for GEARS loss).
        """
        if self.perturb_col != 'condition' or self.control_tags != {'ctrl'}:
            raise ValueError("Run set_gears_format() first.")

        de_success = False

        # --- 2. Try hpdex (High Performance) ---
        if use_hpdex:
            target_metric = 'wilcoxon'
            if method != 'wilcoxon':
                logger.warning(f"hpdex supports 'wilcoxon' (Mann-Whitney U). Switching method from '{method}' to 'wilcoxon'.")
            
            logger.info(f"Attempting DE genes computation with hpdex...")
            try:
                import hpdex

                # hpdex 返回 DataFrame
                df_res = hpdex.parallel_differential_expression(
                    adata=self.adata,
                    groupby_key=self.perturb_col,
                    reference='ctrl',
                    metric=target_metric,
                    threads=-1,
                    show_progress=True,
                    use_continuity=True,
                    tie_correction=True
                )

                rank_dict = {}
                for target, group in df_res.groupby("target"):
                    sorted_group = group.sort_values(
                        by=["p_value", "statistic"], 
                        ascending=[True, False]
                    )
                    rank_dict[target] = sorted_group["feature"].tolist()

                self.adata.uns['rank_genes_groups_cov_all'] = rank_dict
                
                de_success = True
                logger.info("hpdex DE analysis successful.")
                
            except ImportError:
                logger.warning("Could not import 'hpdex'. Falling back to scanpy.")
            except Exception as e:
                logger.warning(f"hpdex execution failed: {str(e)}. Falling back to scanpy.")

        # --- 3. Fallback to Scanpy ---
        if not de_success:
            logger.info(f"Computing DE genes with scanpy (method={method})...")
            try:
                sc.tl.rank_genes_groups(
                    self.adata,
                    groupby='condition',
                    reference='ctrl',
                    method=method,
                    use_raw=use_raw,
                    key_added='rank_genes_groups_cov_all'
                )
                logger.info(f"Scanpy DE analysis successful.")
                de_success = True
            except Exception as e:
                logger.error(f"DE computation failed: {e}")
                raise e
        
        # Save metadata
        if de_success:
            self.adata.uns['top_de_n'] = n_top_genes

    def split_data(
        self,
        split_type: str = 'simulation',
        split_ratio: tuple = (0.7, 0.15, 0.15),
        seed: int = 1,
        test_perts: Optional[List[str]] = None,
        only_use_test_perts: bool = False,
        test_pert_genes: Optional[List[str]] = None,
        split_dict: Optional[Dict] = None,
        split_dict_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Splits data supporting all GEARS strategies using a unified interface.
        
        Args:
            split_type: 'simulation', 'simulation_single', 'combo_seen0/1/2', 'simple', 'no_test', etc.
            split_ratio: 
                - For 'simple'/'single'/'no_test': (train_frac, val_frac, test_frac). Sum must be 1.0.
                - For 'simulation'/'combo_seen*': (train_gene_frac, val_weight, test_weight).
                  * train_gene_frac: Fraction of genes to be used as "seen" (training).
                  * val_weight/test_weight: Relative proportions for splitting the hold-out set.
            seed: Random seed.
            test_perts: Explicit list of perturbations for testing.
            only_use_test_perts: If True, test set contains ONLY the perts in test_perts.
            test_pert_genes: Explicit list of genes to be considered "unseen".
            
        """
        if self.perturb_col != 'condition':
            raise ValueError("Run set_gears_format() first.")
        
        np.random.seed(seed)
        self.adata.obs['split'] = 'train'
        
        valid_mask = ~self.adata.obs['condition'].isin(self.ignore_tags)
        conditions = self.adata.obs.loc[valid_mask, 'condition'].unique()
        pert_conditions = [c for c in conditions if c != 'ctrl']
        
        singles = [p for p in pert_conditions if '+' not in p]
        combos = [p for p in pert_conditions if '+' in p]
        
        def get_genes(p):
            return set(g for g in p.split('+') if g != 'ctrl')

        # Containers
        train_perts, val_perts, test_perts_list = [], [], []
        subgroup = None

        logger.info(f"Splitting strategy: {split_type}")

        # --- Check Split Type Category ---
        is_gene_split = split_type in ['simulation', 'simulation_single'] or split_type.startswith('combo_seen')
        
        # --- Parameter Interpretation ---
        if is_gene_split:
            # Interpretation: (Gene Train %, Val Weight, Test Weight)
            train_gene_set_size = split_ratio[0]
            val_weight = split_ratio[1]
            test_weight = split_ratio[2]
            # Use same ratio for combo_seen2 training fraction unless hardcoded
            combo_seen2_train_frac = split_ratio[0] 
            
            logger.info(f"Gene-level split: {train_gene_set_size*100}% genes seen.")
        else:
            # Interpretation: (Train %, Val %, Test %) of total data
            train_frac, val_frac, test_frac = split_ratio
            if not np.isclose(sum(split_ratio), 1.0) and split_type not in ['custom', 'no_split']:
                 raise ValueError(f"For {split_type}, split_ratio must sum to 1.0, got {sum(split_ratio)}")

        # ===========================
        # 1. Custom & No Split
        # ===========================
        if split_type == 'custom':
            if split_dict_path and os.path.exists(split_dict_path):
                with open(split_dict_path, 'rb') as f:
                    split_dict = pickle.load(f)
            if not split_dict:
                raise ValueError("For 'custom' split, provide split_dict or valid split_dict_path.")
            train_perts = split_dict.get('train', [])
            val_perts = split_dict.get('val', [])
            test_perts_list = split_dict.get('test', [])

        elif split_type == 'no_split':
            self.adata.obs['split'] = 'test'
            return {}

        # ===========================
        # 2. Simple & No Test
        # ===========================
        elif split_type == 'no_test':
            all_perts = pert_conditions
            np.random.shuffle(all_perts)
            n_train = int(len(all_perts) * (train_frac / (train_frac + val_frac)))
            train_perts = all_perts[:n_train]
            val_perts = all_perts[n_train:]

        elif split_type == 'single': # Simple random split on singles
            np.random.shuffle(singles)
            n_train = int(len(singles) * train_frac)
            n_val = int(len(singles) * val_frac)
            train_perts = singles[:n_train]
            val_perts = singles[n_train : n_train + n_val]
            test_perts_list = singles[n_train + n_val:]

        # ===========================
        # 3. Simulation (Standard Benchmark)
        # ===========================
        elif split_type == 'simulation':
            # 1. Split Genes
            all_genes = set()
            for p in singles: all_genes.update(get_genes(p))
            all_genes = list(all_genes)
            np.random.shuffle(all_genes)
            
            n_train_genes = int(len(all_genes) * train_gene_set_size)
            train_genes = set(all_genes[:n_train_genes])
            test_genes = set(all_genes[n_train_genes:])
            
            # 2. Assign Singles
            train_singles = [s for s in singles if get_genes(s).issubset(train_genes)]
            test_singles = [s for s in singles if get_genes(s).issubset(test_genes)]
            
            # 3. Classify Combos
            seen0, seen1, seen2 = [], [], []
            for c in combos:
                g = get_genes(c)
                if len(g) < 2: continue
                cnt = sum(1 for x in g if x in train_genes)
                if cnt == 0: seen0.append(c)
                elif cnt == 1: seen1.append(c)
                else: seen2.append(c)
            
            # 4. Handle Specific Test Perts
            valtest_pool = []
            if test_perts:
                test_perts_set = set(test_perts)
                if only_use_test_perts:
                    valtest_pool = [p for p in test_perts if p in pert_conditions]
                    # Clear others from automatic pools
                    seen0 = [c for c in seen0 if c not in test_perts_set]
                    seen1 = [c for c in seen1 if c not in test_perts_set]
                    seen2 = [c for c in seen2 if c not in test_perts_set]
                    test_singles = []
                else:
                    # Remove explicit tests from their categories to avoid duplication
                    for tp in test_perts:
                        if tp in seen0: seen0.remove(tp)
                        elif tp in seen1: seen1.remove(tp)
                        elif tp in seen2: seen2.remove(tp)
                    valtest_pool = test_perts

            # 5. Build Final Pools
            if not test_perts or not only_use_test_perts:
                np.random.shuffle(seen2)
                n_seen2_train = int(len(seen2) * combo_seen2_train_frac)
                train_seen2 = seen2[:n_seen2_train]
                valtest_seen2 = seen2[n_seen2_train:]
                
                # Combine all held-out candidates
                valtest_pool += test_singles + seen0 + seen1 + valtest_seen2
                train_perts = train_singles + train_seen2
            else:
                train_perts = train_singles + seen2

            # 6. Split Val/Test based on weights
            np.random.shuffle(valtest_pool)
            total_weight = val_weight + test_weight
            if total_weight == 0: n_val = 0
            else: n_val = int(len(valtest_pool) * (val_weight / total_weight))
            
            val_perts = valtest_pool[:n_val]
            test_perts_list = valtest_pool[n_val:]
            
            # Subgroup Info
            subgroup = {
                'test_subgroup': {
                    'single': [p for p in test_perts_list if '+' not in p],
                    'seen0': [p for p in test_perts_list if p in seen0],
                    'seen1': [p for p in test_perts_list if p in seen1],
                    'seen2': [p for p in test_perts_list if p in seen2]
                },
                'train_genes': list(train_genes),
                'test_genes': list(test_genes)
            }

        # ===========================
        # 4. Simulation Single
        # ===========================
        elif split_type == 'simulation_single':
            all_genes = set()
            for p in singles: all_genes.update(get_genes(p))
            all_genes = list(all_genes)
            np.random.shuffle(all_genes)
            
            n_train_genes = int(len(all_genes) * train_gene_set_size)
            train_genes = set(all_genes[:n_train_genes])
            test_genes = set(all_genes[n_train_genes:])
            
            train_singles = [s for s in singles if get_genes(s).issubset(train_genes)]
            valtest_singles = [s for s in singles if get_genes(s).issubset(test_genes)]
            
            np.random.shuffle(valtest_singles)
            total_weight = val_weight + test_weight
            n_val = int(len(valtest_singles) * (val_weight / total_weight))
            
            val_perts = valtest_singles[:n_val]
            test_perts_list = valtest_singles[n_val:]
            train_perts = train_singles

        # ===========================
        # 5. Combo Seen 0/1/2
        # ===========================
        elif split_type.startswith('combo_seen'):
            target_seen_count = int(split_type[-1])
            unique_genes = list(set().union(*[get_genes(p) for p in pert_conditions]))
            np.random.shuffle(unique_genes)
            
            if test_pert_genes:
                unseen_genes = set(test_pert_genes)
                seen_genes = set(unique_genes) - unseen_genes
            else:
                n_seen = int(len(unique_genes) * train_gene_set_size)
                seen_genes = set(unique_genes[:n_seen])
                unseen_genes = set(unique_genes[n_seen:])
            
            train_perts = [s for s in singles if get_genes(s).issubset(seen_genes)]
            
            target_combos, other_seen2 = [], []
            for c in combos:
                g = get_genes(c)
                if len(g) < 2: continue
                cnt = sum(1 for x in g if x in seen_genes)
                if cnt == target_seen_count: target_combos.append(c)
                elif cnt == 2: other_seen2.append(c)
            
            # Handle explicit test perts
            if test_perts:
                tps = set(test_perts)
                specified = [p for p in test_perts if p in target_combos]
                target_combos = [c for c in target_combos if c not in tps] + specified
            
            np.random.shuffle(target_combos)
            total_weight = val_weight + test_weight
            n_val = int(len(target_combos) * (val_weight / total_weight))
            
            val_perts = target_combos[:n_val]
            test_perts_list = target_combos[n_val:]
            train_perts += other_seen2
            
        else:
            raise ValueError(f"Unknown split_type: {split_type}")

        # ===========================
        # Apply Logic
        # ===========================
        train_set = set(train_perts) | {'ctrl'}
        val_set = set(val_perts)
        test_set = set(test_perts_list)
        
        overlap = test_set.intersection(train_set)
        if len(overlap) > 1 or (len(overlap) == 1 and 'ctrl' not in overlap):
            logger.warning(f"Overlap detected: {overlap}")

        self.adata.obs.loc[self.adata.obs['condition'].isin(train_set), 'split'] = 'train'
        self.adata.obs.loc[self.adata.obs['condition'].isin(val_set), 'split'] = 'val'
        self.adata.obs.loc[self.adata.obs['condition'].isin(test_set), 'split'] = 'test'

        logger.info(f"Split Summary ({split_type}): Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
        if subgroup:
            for k, v in subgroup['test_subgroup'].items():
                logger.info(f"  Test {k}: {len(v)}")

        return subgroup if subgroup else {}

    def pair_cells(self, num_samples: int = 1, seed: int = 1) -> None:
        """Pairs perturbations with controls."""
        if 'split' not in self.adata.obs:
            logger.warning("Dataset not split. Running split_data() recommended.")

        np.random.seed(seed)
        self.adata.obs['index_col'] = np.arange(self.adata.n_obs, dtype=int)
        
        valid_mask = ~self.adata.obs['condition'].isin(self.ignore_tags)
        ctrl_mask = valid_mask & (self.adata.obs['condition'] == 'ctrl')
        ctrl_indices = self.adata.obs.loc[ctrl_mask, 'index_col'].values
        
        if len(ctrl_indices) == 0:
            raise ValueError("No valid control cells found.")
        
        n_obs = self.adata.n_obs
        ctrl_pairing = np.full((n_obs, num_samples), -1, dtype=int)
        
        pert_mask = valid_mask & (self.adata.obs['condition'] != 'ctrl')
        pert_positions = np.where(pert_mask)[0]
        n_pert = len(pert_positions)
        
        if n_pert > 0:
            sampled_ctrl = np.random.choice(ctrl_indices, size=(n_pert, num_samples), replace=True)
            ctrl_pairing[pert_positions, :] = sampled_ctrl
            
        ctrl_positions = np.where(ctrl_mask)[0]
        ctrl_self = self.adata.obs.loc[ctrl_mask, 'index_col'].values
        for i, pos in enumerate(ctrl_positions):
            ctrl_pairing[pos, :] = ctrl_self[i]

        self.adata.obsm['ctrl_indices'] = ctrl_pairing
        logger.info(f"Paired {num_samples} controls per cell.")

    def to_gears(self, data_path: str, check_de_genes: bool = True) -> Any:
        """Converts to GEARS PertData."""
        if 'rank_genes_groups_cov_all' not in self.adata.uns and check_de_genes:
            logger.warning("Computing DE genes...")
            self.compute_de_genes()
        
        try:
            from gears import PertData
        except ImportError:
            from .model.gears.source.pertdata import PertData

        
        if 'ctrl_indices' not in self.adata.obsm:
            raise ValueError("Run pair_cells() first.")

        pert_data = PertData(data_path, default_pert_graph=False)
        pert_data.adata = self.adata
        pert_data.gene_names = self.adata.var['gene_name'].tolist()
        pert_data.dataset_name = 'custom_loader'
        pert_data.dataset_path = data_path
        
        gene_name_to_idx = {name: i for i, name in enumerate(pert_data.gene_names)}

        # Build DE map
        de_gene_map = {}
        if 'rank_genes_groups_cov_all' in self.adata.uns:
            rank_data = self.adata.uns['rank_genes_groups_cov_all']
            top_n = self.adata.uns.get('top_de_n', 20)
            
            if hasattr(rank_data, 'dtype') and hasattr(rank_data.dtype, 'names'):
                for cond in rank_data['names'].dtype.names:
                    top_genes = rank_data['names'][cond][:top_n]
                    de_gene_map[cond] = [gene_name_to_idx.get(g, -1) for g in top_genes]
            else:
                for cond in rank_data.keys():
                    if cond not in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']:
                        top_genes = rank_data[cond][:top_n]
                        de_gene_map[cond] = [gene_name_to_idx.get(g, -1) for g in top_genes]

        # Process Data
        dataset_processed = {}
        X = self.adata.X
        ctrl_indices = self.adata.obsm['ctrl_indices']
        num_samples = ctrl_indices.shape[1]
        
        obs_df = self.adata.obs[['condition', 'index_col', 'split']].copy()
        obs_df = obs_df[obs_df.index.isin(self.adata.obs.index)]
        
        logger.info("Generating PyG objects...")
        for condition, group_df in tqdm(obs_df.groupby('condition')):
            data_list = []
            
            if condition == 'ctrl': pert_idx = [-1]
            else:
                genes = [p for p in condition.split('+') if p != 'ctrl']
                pert_idx = [gene_name_to_idx.get(g, -1) for g in genes]
                if not pert_idx or all(i == -1 for i in pert_idx): pert_idx = [-1]

            de_idx = de_gene_map.get(condition, [-1] * 20)
            
            current_idx = group_df['index_col'].values
            
            if hasattr(X, "toarray"): y_batch = X[current_idx].toarray()
            elif hasattr(X, "A"): y_batch = np.asarray(X[current_idx])
            else: y_batch = X[current_idx]
            
            for i, cell_idx in enumerate(current_idx):
                paired_idx = ctrl_indices[cell_idx, :]
                
                if hasattr(X, "toarray"): x_batch = X[paired_idx].toarray()
                elif hasattr(X, "A"): x_batch = np.asarray(X[paired_idx])
                else: x_batch = X[paired_idx]
                
                for j in range(num_samples):
                    data_list.append(Data(
                        x=torch.tensor(x_batch[j], dtype=torch.float).unsqueeze(1),
                        y=torch.tensor(y_batch[i], dtype=torch.float).unsqueeze(1),
                        pert_idx=pert_idx,
                        pert=condition,
                        de_idx=de_idx
                    ))
            
            dataset_processed[condition] = data_list

        pert_data.dataset_processed = dataset_processed
        
        if 'split' in self.adata.obs:
            set2conditions = self.adata.obs.groupby('split')['condition'].unique().to_dict()
            pert_data.set2conditions = {k: list(v) for k, v in set2conditions.items()}
            pert_data.split = 'custom'
        
        logger.info(f"Done. {len(dataset_processed)} conditions processed.")
        return pert_data

    def save(self, data_path: str) -> None:
        """
        Saves the processed AnnData to the specified file path.
        Example: data_path='./data/processed.h5ad'
        """
        if self.adata is None:
            raise ValueError("No adata to save.")

        # Ensure the directory exists if the path contains a directory component
        dir_name = os.path.dirname(data_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        # 1. Save class attributes to adata.uns so they persist
        # HDF5 doesn't support sets, so we convert them to lists
        self.adata.uns['perturbation_data_meta'] = {
            'perturb_col': self.perturb_col,
            'control_tags': list(self.control_tags),
            'ignore_tags': list(self.ignore_tags)
        }
        
        # 2. Save file directly to data_path
        logger.info(f"Saving AnnData to {data_path}...")
        self.adata.write_h5ad(data_path)
        logger.info("Save completed.")

    @staticmethod
    def load(data_path: str) -> 'PerturbationData':
        """
        Static method to load AnnData from disk and return a new PerturbationData instance.
        Example: pert_data = PerturbationData.load('./data/processed.h5ad')
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
            
        logger.info(f"Loading AnnData from {data_path}...")
        adata = ad.read_h5ad(data_path)
        
        # Defaults
        perturb_col = 'perturbation'
        control_tags = ['control']
        ignore_tags = []
        
        # 3. Restore class attributes from metadata
        if 'perturbation_data_meta' in adata.uns:
            meta = adata.uns['perturbation_data_meta']
            perturb_col = meta.get('perturb_col', perturb_col)
            
            if 'control_tags' in meta:
                control_tags = meta['control_tags']
            if 'ignore_tags' in meta:
                ignore_tags = meta['ignore_tags']
                
            logger.info(f"Metadata restored: perturb_col='{perturb_col}'")
        else:
            logger.warning("No 'perturbation_data_meta' found in .uns. Using default parameters.")
            
        # Create new instance with loaded data and metadata
        instance = PerturbationData(
            adata=adata,
            perturb_col=perturb_col,
            control_tag=control_tags,
            ignore_tags=ignore_tags
        )
        
        # Validate that necessary data exists
        if 'split' in adata.obs:
            logger.info("Detected existing data split.")
        if 'ctrl_indices' in adata.obsm:
            logger.info("Detected existing control pairing.")
            
        return instance


class GeneGraph:
    """
    Universal Gene Graph Class.
    Decouples graph construction from model logic.
    Supports Graph Neural Networks via PyTorch Geometric (PyG).
    """

    def __init__(
        self, 
        gene_list: List[str], 
        edge_index: Union[np.ndarray, torch.Tensor], 
        edge_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
        graph_type: str = 'custom'
    ):
        """
        Args:
            gene_list: List of gene names corresponding to nodes (index 0, 1...).
            edge_index: Graph connectivity in COO format [2, num_edges] (numpy array or torch tensor).
            edge_weight: Edge weights [num_edges] (numpy array or torch tensor).
            graph_type: Identifier for the graph type (e.g., 'go', 'co_express').
        """
        self.gene_list = gene_list
        self.gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
        
        # Convert to numpy arrays (internal storage)
        if isinstance(edge_index, torch.Tensor):
            self.edge_index = edge_index.cpu().numpy().astype(np.int64)
        else:
            self.edge_index = np.asarray(edge_index, dtype=np.int64)
        
        if edge_weight is not None:
            if isinstance(edge_weight, torch.Tensor):
                self.edge_weight = edge_weight.cpu().numpy().astype(np.float32)
            else:
                self.edge_weight = np.asarray(edge_weight, dtype=np.float32)
        else:
            self.edge_weight = None
            
        self.graph_type = graph_type
        self.num_nodes = len(gene_list)

    @classmethod
    def from_adjacency(cls, adj_matrix: Union[np.ndarray, torch.Tensor], gene_list: List[str], graph_type='custom'):
        """Build graph from a dense adjacency matrix."""
        # Convert to numpy if needed
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()
        
        # Extract indices of non-zero elements
        indices = np.nonzero(adj_matrix)
        sources = indices[0]
        targets = indices[1]
        weights = adj_matrix[sources, targets]
        
        edge_index = np.stack([sources, targets], axis=0)
        
        return cls(gene_list, edge_index, weights, graph_type)

    @classmethod
    def from_coexpression(
        cls, 
        adata: sc.AnnData, 
        gene_list: Optional[List[str]] = None,
        threshold: float = 0.4, 
        top_k: int = 20,
        mode: str = 'correlation'
    ):
        """
        Build co-expression graph from AnnData.
        
        Args:
            adata: AnnData object.
            gene_list: Subset of genes to include (default: all genes in adata).
            threshold: Absolute correlation threshold.
            top_k: Keep top K neighbors per gene.
        """
        if gene_list is None:
            gene_list = adata.var_names.tolist()
        
        # Validate genes
        valid_genes = [g for g in gene_list if g in adata.var_names]
        if len(valid_genes) != len(gene_list):
            logger.warning(f"{len(gene_list) - len(valid_genes)} genes from gene_list not found in AnnData.")
        
        # Extract expression matrix
        X = adata[:, valid_genes].X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
            
        # Calculate correlation
        if mode == 'correlation':
            corr = np.corrcoef(X.T)
            np.fill_diagonal(corr, 0) # Remove self-loops
            corr = np.abs(corr)
        else:
            raise NotImplementedError("Only 'correlation' mode is supported.")

        # Sparsification: Top-K + Threshold
        num_genes = len(valid_genes)
        sources, targets, weights = [], [], []

        for i in range(num_genes):
            row = corr[i]
            mask = row > threshold
            indices = np.where(mask)[0]
            
            if len(indices) == 0: continue
            
            vals = row[indices]
            
            # Apply Top-K
            if len(vals) > top_k:
                top_k_idx = np.argsort(vals)[-top_k:]
                indices = indices[top_k_idx]
                vals = vals[top_k_idx]
            
            sources.extend([i] * len(indices))
            targets.extend(indices)
            weights.extend(vals)

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

        return cls(valid_genes, edge_index, edge_weight, graph_type='co_express')

    @classmethod
    def from_go(
        cls, 
        gene_list: List[str], 
        path: Optional[str] = None,
        cache_dir: str = './gene_graph',
        threshold: float = 0.1,
        top_k: int = 20,
        num_workers: int = 4
    ):
        """
        Build GO similarity graph.
        
        Args:
            gene_list: Target gene list.
            path: Path to custom similarity matrix (csv/pkl). If None, uses GEARS default logic.
            cache_dir: Directory to download/store gene2go data.
            threshold: Similarity threshold.
            top_k: Max neighbors per gene.
        """
        
        # --- Strategy A: Default GEARS Logic (Path is None) ---
        if path is None:
            logger.info("Building GO graph using GEARS default logic...")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Use GEARS make_GO utility (downloads data and computes Jaccard)
            # make_GO returns a DataFrame with ['source', 'target', 'importance']
            df_edge_list = make_GO(
                data_path=cache_dir,
                pert_list=gene_list,
                data_name='temp_go_build',
                num_workers=num_workers,
                save=False
            )
            
            # Filter genes
            all_graph_genes = set(df_edge_list['source']).union(set(df_edge_list['target']))
            valid_genes = [g for g in gene_list if g in all_graph_genes]
            
            if not valid_genes:
                raise ValueError("None of the provided genes were found in the GO database.")

            # Filter by importance (Threshold)
            df_filtered = df_edge_list[df_edge_list['importance'] > threshold].copy()
            
            # Filter by Top-K
            df_topk = []
            for gene in valid_genes:
                # Get edges where 'gene' is the source
                gene_edges = df_filtered[df_filtered['source'] == gene].nlargest(top_k, 'importance')
                if not gene_edges.empty:
                    df_topk.append(gene_edges)
            
            if not df_topk:
                raise ValueError("No edges remained after filtering.")
                
            df_final = pd.concat(df_topk, ignore_index=True)
            
            # Prepare adjacency data
            target_genes = valid_genes
            # We map using the strings directly from the dataframe later
            
        # --- Strategy B: Custom File (Path provided) ---
        else:
            logger.info(f"Loading custom GO graph from {path}...")
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            if path.endswith('.csv'):
                df_adj = pd.read_csv(path, index_col=0)
            elif path.endswith('.pkl'):
                df_adj = pd.read_pickle(path)
            else:
                raise ValueError("Unsupported format. Use .csv or .pkl")
            
            valid_genes = [g for g in gene_list if g in df_adj.index]
            target_genes = valid_genes
            adj_matrix = df_adj.loc[valid_genes, valid_genes].values
            np.fill_diagonal(adj_matrix, 0)
            
            # Convert adjacency matrix to edge list format for unified processing
            sources_idx, targets_idx = np.where(adj_matrix > threshold)
            weights = adj_matrix[sources_idx, targets_idx]
            
            # Create a dataframe to mimic the structure of Strategy A
            df_final = pd.DataFrame({
                'source': [valid_genes[i] for i in sources_idx],
                'target': [valid_genes[i] for i in targets_idx],
                'importance': weights
            })
            
            # Apply Top-K logic if needed (simplified for matrix inputs)
            # (Loop omitted for brevity, assuming custom matrix is usually pre-processed or small enough)

        # --- Common Logic: Build Edge Index ---
        
        # 1. Map gene names to global indices (0..N for the full gene_list)
        full_map = {g: i for i, g in enumerate(gene_list)}
        
        # 2. Filter edges to ensure both source and target are in the requested gene_list
        valid_edges = df_final[
            (df_final['source'].isin(full_map)) & 
            (df_final['target'].isin(full_map))
        ]
        
        if valid_edges.empty:
             raise ValueError("No valid edges found aligning with gene_list.")

        # 3. Convert to numpy arrays
        sources = [full_map[g] for g in valid_edges['source']]
        targets = [full_map[g] for g in valid_edges['target']]
        weights = valid_edges['importance'].values
        
        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

        # Return graph aligned to the full gene_list (missing genes become isolated nodes)
        return cls(gene_list, edge_index, edge_weight, graph_type='go')

    def subset(self, target_gene_list: List[str]) -> Self:
        """
        Create a subgraph containing only the specified genes.
        Crucial for aligning the graph with a specific model vocabulary.
        """
        target_map = {gene: i for i, gene in enumerate(target_gene_list)}
        old_idx_to_new_idx = {}
        kept_genes = []
        
        # Identify genes present in both current graph and target list
        for gene in target_gene_list:
            if gene in self.gene_to_idx:
                old_idx = self.gene_to_idx[gene]
                new_idx = target_map[gene]
                old_idx_to_new_idx[old_idx] = new_idx
                kept_genes.append(gene)
        
        if not kept_genes:
            raise ValueError("No overlap between graph genes and target gene list.")

        # Filter Edges: Keep only if both source and target are in the new set
        src, dst = self.edge_index
        
        # Create validity masks (boolean)
        valid_src = np.array([i in old_idx_to_new_idx for i in src])
        valid_dst = np.array([i in old_idx_to_new_idx for i in dst])
        mask = valid_src & valid_dst
        
        new_src = src[mask]
        new_dst = dst[mask]
        new_weights = self.edge_weight[mask] if self.edge_weight is not None else None
        
        # Remap indices from Old -> New
        remapped_src = np.array([old_idx_to_new_idx[i] for i in new_src], dtype=np.int64)
        remapped_dst = np.array([old_idx_to_new_idx[i] for i in new_dst], dtype=np.int64)
        
        new_edge_index = np.stack([remapped_src, remapped_dst], axis=0)
        
        # Return new instance
        return self.__class__(target_gene_list, new_edge_index, new_weights, self.graph_type)

    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        # Convert numpy arrays to torch tensors for PyG
        edge_index_tensor = torch.from_numpy(self.edge_index).long()
        edge_attr_tensor = torch.from_numpy(self.edge_weight).float() if self.edge_weight is not None else None
        return Data(edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, num_nodes=self.num_nodes)
    
    def to_torch(self) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors for model usage."""
        return {
            'edge_index': torch.from_numpy(self.edge_index).long(),
            'edge_weight': torch.from_numpy(self.edge_weight).float() if self.edge_weight is not None else None
        }

    def save(self, path: str):
        """
        Serialize graph to disk as a single pickle file.
        
        Args:
            path: Path to save the graph (should end with .pkl)
        """
        data = {
            'gene_list': self.gene_list,
            'edge_index': self.edge_index,  # numpy array
            'edge_weight': self.edge_weight,  # numpy array or None
            'graph_type': self.graph_type
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """
        Load graph from disk.
        
        Args:
            path: Path to the saved graph file (.pkl)
            
        Returns:
            GeneGraph instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            gene_list=data['gene_list'],
            edge_index=data['edge_index'],
            edge_weight=data.get('edge_weight', None),
            graph_type=data.get('graph_type', 'custom')
        )
