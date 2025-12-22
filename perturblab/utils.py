import warnings
from typing import Callable, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse


def apply_gears_format(
    adata: ad.AnnData, 
    perturb_col: str = 'perturbation',
    control_tag: Union[str, List[str]] = 'control',
    ignore_tags: Optional[List[str]] = None,
    parse_fn: Optional[Callable[[str], List[str]]] = None,
    fallback_cell_type: str = 'K562',
    remove_ignore: bool = True
) -> ad.AnnData:
    """
    Standardizes an AnnData object into the format required by the GEARS model.
    
    Optimized with fully vectorized label parsing and dictionary mapping.
    """
    
    # 1. Validation & Setup
    if perturb_col not in adata.obs:
        raise KeyError(f"Column '{perturb_col}' not found in adata.obs")

    if isinstance(control_tag, str):
        control_tags = {control_tag}
    else:
        control_tags = set(control_tag)
        
    ignore_tags_set = set(ignore_tags) if ignore_tags else set()

    # 2. Filter Valid Cells (Vectorized Filter)
    # Use the Series directly to avoid copying the whole adata if not needed initially
    obs_series = adata.obs[perturb_col].astype(str)
    
    if ignore_tags_set and remove_ignore:
        valid_mask = ~obs_series.isin(ignore_tags_set)
        if not valid_mask.any():
            raise ValueError("No cells left after filtering ignore_tags.")
        
        adata_gears = adata[valid_mask].copy()
        # Re-slice the series for the subset
        obs_series = adata_gears.obs[perturb_col].astype(str)
    else:
        adata_gears = adata.copy()

    # 3. Vectorized Label Parsing (The Core Optimization)
    # Strategy: Compute logic on Unique labels (K) -> Map back to All cells (N)
    unique_series = pd.Series(obs_series.unique())

    # 3.1 Identify special categories
    is_ctrl = unique_series.isin(control_tags)
    is_ignore = unique_series.isin(ignore_tags_set)

    # 3.2 Parse gene strings
    # Note: Custom parse_fn still requires apply (Python loop), but only runs K times.
    if parse_fn is None:
        def default_parser(pert_str: str) -> List[str]:
            parts = {p.strip() for p in pert_str.split('+')}
            parts.discard("")
            return sorted(list(parts))
        parse_fn = default_parser
    
    # Result is a Series of lists: [['A'], ['A', 'B'], ...]
    parsed_lists = unique_series.apply(parse_fn)
    
    # 3.3 Compute attributes for logic
    counts = parsed_lists.map(len)
    
    # Prepare string components (Vectorized)
    # Case: Single Gene -> "Gene" (extract first element safely)
    first_gene = parsed_lists.str[0]
    # Case: Multi Gene -> "GeneA+GeneB" (join list)
    joined_genes = parsed_lists.str.join('+') 
    
    # 3.4 Construct Condition Strings using np.select (Vectorized if/else)
    # Logic priority (top to bottom):
    conditions = [
        is_ctrl,                         # 1. Is explicitly a control tag?
        is_ignore,                       # 2. Is in ignore list (and remove_ignore=False)?
        counts == 0,                     # 3. Parsed to empty (treat as ctrl)
        counts == 1,                     # 4. Single perturbation
        counts >= 2                      # 5. Multi perturbation
    ]
    
    choices = [
        'ctrl',                          # -> 'ctrl'
        unique_series,                   # -> Keep original raw label
        'ctrl',                          # -> 'ctrl'
        first_gene + '+ctrl',            # -> 'Gene+ctrl'
        joined_genes                     # -> 'GeneA+GeneB'
    ]
    
    # Generate the mapped values
    # default='ctrl' covers any edge cases
    final_labels = np.select(conditions, choices, default='ctrl')
    
    # 4. Apply Map & Optimize Memory
    # map is efficient; astype('category') saves massive memory for repeated strings
    label_map = dict(zip(unique_series, final_labels))
    adata_gears.obs['condition'] = obs_series.map(label_map).astype('category')

    # 5. Metadata Standardization
    if 'cell_type' not in adata_gears.obs.columns:
        adata_gears.obs['cell_type'] = fallback_cell_type
    
    if 'gene_name' not in adata_gears.var.columns:
        adata_gears.var['gene_name'] = adata_gears.var_names

    # 6. Sparse Matrix Optimization
    if not sparse.isspmatrix_csr(adata_gears.X):
        adata_gears.X = sparse.csr_matrix(adata_gears.X)

    # 7. Warnings (computed from vectorized stats)
    # Check if we have both singles (count==1) and multis (count>=2) in the *active* dataset
    # (Excluding controls and ignores)
    active_mask = ~(is_ctrl | is_ignore | (counts == 0))
    active_counts = counts[active_mask]
    
    if not active_counts.empty:
        has_single = (active_counts == 1).any()
        has_multi = (active_counts >= 2).any()
        
        if has_single and has_multi:
            warnings.warn(
                "Mixed Perturbation Format Detected: Dataset contains both single and "
                "combinatorial (2+) perturbations. GEARS typically expects specific splits.",
                UserWarning
            )

    return adata_gears
