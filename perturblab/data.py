import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger("[PerturbationData]")

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
            try:
                from .model.gears.source.pertdata import PertData
            except ImportError:
                class PertData:
                    def __init__(self, *args, **kwargs): pass
        
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
