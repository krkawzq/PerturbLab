"""GEARS-specific perturbation string processing utilities.

This module provides parsing and filtering helpers for perturbation condition strings
(e.g., 'TP53', 'KRAS+MYC', 'ctrl'). It is separated from graph construction and model training code.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Set, Union, overload

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import anndata as ad

from perturblab.types import PerturbationData

if TYPE_CHECKING:
    from perturblab.types import GeneVocab

__all__ = [
    "filter_perturbations_by_genes",
    "extract_genes_from_perturbations",
    "format_gears",
    "build_collate_fn",
]


def filter_perturbations_by_genes(
    perturbations: Iterable[str],
    allowed_genes: Set[str] | List[str],
) -> List[str]:
    """
    Filter perturbation strings, retaining only those where all constituent genes are in `allowed_genes`.

    **Note on Input Format**:
    This function expects perturbation strings in **GEARS format**, where multi-gene
    perturbations are joined by ``+`` (e.g., ``'GeneA+GeneB'``). Single genes should
    be raw strings (e.g., ``'GeneC'``) or GEARS-style single perturbations (e.g., ``'GeneC+ctrl'``).

    Args:
        perturbations: Iterable of GEARS-formatted perturbation strings (e.g., ['TP53+ctrl', 'KRAS+MYC']).
        allowed_genes: Set or list of valid gene symbols to keep.

    Returns:
        List[str]: A list of perturbation strings where every gene component exists in `allowed_genes`.
        Control perturbations ('ctrl', 'control') are always preserved.

    Example:
        >>> # Inputs must be '+' delimited
        >>> perts = ['TP53+ctrl', 'KRAS+MYC', 'TP53+UNKNOWN']
        >>> valid_genes = {'TP53', 'KRAS', 'MYC'}
        >>> filter_perturbations_by_genes(perts, valid_genes)
        ['TP53+ctrl', 'KRAS+MYC']
    """
    if not isinstance(allowed_genes, set):
        allowed_genes = set(allowed_genes)

    filtered = []
    for pert in perturbations:
        # Always retain control conditions
        if pert.lower() in {"ctrl", "control"}:
            filtered.append(pert)
            continue
        
        # GEARS format assumption: genes are split by '+'
        genes = pert.split("+")
        
        # Filter out 'ctrl' from the gene check (e.g. for 'TP53+ctrl')
        genes_to_check = [g for g in genes if g.lower() not in {"ctrl", "control"}]
        
        if all(gene in allowed_genes for gene in genes_to_check):
            filtered.append(pert)
    return filtered


def extract_genes_from_perturbations(
    perturbations: Iterable[str],
    exclude_control: bool = True,
) -> List[str]:
    """
    Extract all unique gene names from a list of GEARS-formatted perturbation strings.

    **Note on Input Format**:
    This function expects perturbation strings in **GEARS format**, where multi-gene
    perturbations are joined by ``+`` (e.g., ``'GeneA+GeneB'``). It splits these strings
    to extract individual gene components.

    Args:
        perturbations: Iterable of GEARS-formatted strings (e.g. 'GeneA+GeneB', 'GeneC+ctrl').
        exclude_control: If True, ignores 'ctrl' and 'control' tokens during extraction.

    Returns:
        List[str]: A sorted list of unique gene symbols found across all perturbations.

    Example:
        >>> # Extracts 'KRAS', 'MYC', 'TP53' from combinatorics
        >>> extract_genes_from_perturbations(['TP53+ctrl', 'KRAS+MYC'])
        ['KRAS', 'MYC', 'TP53']
    """
    all_genes = set()
    for pert in perturbations:
        # Split by '+' as per GEARS format
        genes = pert.split("+")
        for gene in genes:
            if exclude_control and gene.lower() in {"ctrl", "control"}:
                continue
            all_genes.add(gene)
    return sorted(list(all_genes))


@overload
def format_gears(
    data: ad.AnnData,
    perturb_col: str = 'perturbation',
    control_tag: Union[str, List[str]] = 'control',
    ignore_tags: Optional[List[str]] = None,
    parse_fn: Optional[Callable[[str], List[str]]] = None,
    fallback_cell_type: str = 'K562',
    remove_ignore: bool = True,
    inplace: bool = False
) -> Optional[ad.AnnData]:
    ...

@overload
def format_gears(
    data: PerturbationData,
    perturb_col: str = 'perturbation',
    control_tag: Union[str, List[str]] = 'control',
    ignore_tags: Optional[List[str]] = None,
    parse_fn: Optional[Callable[[str], List[str]]] = None,
    fallback_cell_type: str = 'K562',
    remove_ignore: bool = True,
    inplace: bool = False
) -> Optional[PerturbationData]:
    ...

def format_gears(
    data: Union[ad.AnnData, PerturbationData],
    perturb_col: str = 'perturbation',
    control_tag: Union[str, List[str]] = 'control',
    ignore_tags: Optional[List[str]] = None,
    parse_fn: Optional[Callable[[str], List[str]]] = None,
    fallback_cell_type: str = 'K562',
    remove_ignore: bool = True,
    inplace: bool = False
) -> Optional[Union[ad.AnnData, PerturbationData]]:
    """Standardizes data into the GEARS model required format.

    This function supports both AnnData and PerturbationData inputs and returns
    the same type. It creates standardized columns and metadata required by GEARS:
        - Creates 'condition' column for perturbation names
        - Stores sorted unique perturbations in adata.uns['pert_list']
        - Handles single and multi-gene perturbations
        - Filters control and ignored labels

    Args:
        data: AnnData object or PerturbationData instance.
        perturb_col: Column in obs holding perturbation labels.
        control_tag: Tag(s) representing control condition.
        ignore_tags: List of tags to ignore and optionally remove.
        parse_fn: Optional custom string parser for perturbation labels.
            Default parser splits by '+' and sorts genes.
        fallback_cell_type: Default value if cell_type column is missing.
        remove_ignore: If True, removes cells matching ignore_tags.
        inplace: If True, modifies data in place and returns None.
            If False (default), returns a new copy.

    Returns:
        Standardized data in the same type as input, or None if inplace=True.

    Examples:
        >>> # Return new copy (default)
        >>> adata_formatted = format_gears(
        ...     adata,
        ...     perturb_col='perturbation',
        ...     control_tag='ctrl'
        ... )
        
        >>> # Modify in place
        >>> format_gears(
        ...     adata,
        ...     perturb_col='perturbation',
        ...     control_tag='ctrl',
        ...     inplace=True
        ... )
        
        >>> # With PerturbationData
        >>> from perturblab.types import PerturbationData
        >>> pert_data = PerturbationData(adata, perturbation_col='condition')
        >>> pert_data_formatted = format_gears(
        ...     pert_data,
        ...     perturb_col='condition',
        ...     control_tag='ctrl'
        ... )
    """
    # Extract AnnData from PerturbationData if needed
    is_perturbation_data = isinstance(data, PerturbationData)
    if is_perturbation_data:
        adata = data.adata
        # Use PerturbationData's perturbation_col if not explicitly provided
        if perturb_col == 'perturbation' and data.perturbation_col:
            perturb_col = data.perturbation_col
        # Use PerturbationData's control labels if not explicitly provided
        if control_tag == 'control' and data.control_labels:
            control_tag = list(data.control_labels)
    else:
        adata = data
    # --- 1. Validation and Setup ---
    if perturb_col not in adata.obs:
        raise KeyError(f"Column '{perturb_col}' not found in adata.obs")

    if isinstance(control_tag, str):
        control_tags = {control_tag}
    else:
        control_tags = set(control_tag)
    ignore_tags_set = set(ignore_tags) if ignore_tags else set()

    # --- 2. Filter out ignored tags if requested ---
    obs_series = adata.obs[perturb_col].astype(str)
    
    if ignore_tags_set and remove_ignore:
        valid_mask = ~obs_series.isin(ignore_tags_set)
        if not valid_mask.any():
            raise ValueError("No cells left after filtering ignore_tags.")
        if inplace:
            # Inplace subset
            if is_perturbation_data:
                # For PerturbationData, update the underlying adata
                adata_gears = adata[valid_mask]
                data.adata = adata_gears
            else:
                # For AnnData, use internal inplace subsetting
                adata._inplace_subset_obs(valid_mask.values)
                adata_gears = adata
        else:
            adata_gears = adata[valid_mask].copy()
        
        # Refresh obs_series after filtering
        obs_series = adata_gears.obs[perturb_col].astype(str)
    else:
        if inplace:
            adata_gears = adata
        else:
            adata_gears = adata.copy()

    # --- 3. Vectorized label parsing on unique labels ---
    unique_series = pd.Series(obs_series.unique())

    # Determine mask for control and ignored
    is_ctrl = unique_series.isin(control_tags)
    is_ignore = unique_series.isin(ignore_tags_set)

    if parse_fn is None:
        def default_parser(pert_str: str) -> List[str]:
            parts = {p.strip() for p in pert_str.split('+')}
            parts.discard("")
            return sorted(list(parts))
        parse_fn = default_parser

    parsed_lists = unique_series.apply(parse_fn)
    counts = parsed_lists.map(len)
    # Defensive: avoid errors if list is empty
    first_gene = parsed_lists.map(lambda x: x[0] if len(x) > 0 else "")
    joined_genes = parsed_lists.str.join('+')

    # Build condition assignment
    # Each row is: (1) control? (2) ignore? (3) empty? (4) single? (5) multi?
    conditions = [
        is_ctrl,                      # 1. Is a control tag
        is_ignore,                    # 2. Is in ignore list
        counts == 0,                  # 3. Empty parsed genes == treat as control
        counts == 1,                  # 4. Single-gene pert
        counts >= 2                   # 5. Multi-gene pert
    ]
    choices = [
        'ctrl',                       # For controls
        unique_series,                # For ignored, keep raw label
        'ctrl',                       # Empty = treat as control
        first_gene + '+ctrl',         # For single-pert, add '+ctrl' for GEARS compatibility
        joined_genes                  # For multi-perturbation, join with '+'
    ]
    final_labels = np.select(conditions, choices, default='ctrl')

    # --- 4. Apply mapping to every cell and store in obs['condition'] ---
    label_map = dict(zip(unique_series, final_labels))
    adata_gears.obs['condition'] = obs_series.map(label_map).astype('category')

    # --- 5. Store the non-control unique perturbation list ---
    unique_conditions = adata_gears.obs['condition'].unique().tolist()
    if 'ctrl' in unique_conditions:
        unique_conditions.remove('ctrl')
    adata_gears.uns['pert_list'] = sorted(unique_conditions)

    # --- 6. (Meta)data standardization ---
    if 'cell_type' not in adata_gears.obs.columns:
        adata_gears.obs['cell_type'] = fallback_cell_type
    if 'gene_name' not in adata_gears.var.columns:
        adata_gears.var['gene_name'] = adata_gears.var_names

    # --- 7. Ensure dense matrix storage is compressed sparse row ---
    if not sparse.isspmatrix_csr(adata_gears.X):
        adata_gears.X = sparse.csr_matrix(adata_gears.X)

    # --- 8. User warning for format mixture ---
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
    
    # Return in the same type as input
    if inplace:
        if is_perturbation_data:
            # Update PerturbationData's internal state
            data.adata = adata_gears
            data.perturbation_col = 'condition'
            if isinstance(control_tag, str):
                data.control_labels = {control_tag}
            else:
                data.control_labels = set(control_tag)
        return None
    else:
        if is_perturbation_data:
            return PerturbationData(
                adata_gears,
                perturbation_col='condition',
                control_label=control_tag,
                ignore_labels=list(ignore_tags) if ignore_tags else None,
                cell_type_col=data.cell_type_col,
                gene_name_col=data.gene_name_col,
                cell_id_col=data.cell_id_col,
                duplicated_gene_policy=data._duplicated_policy,
            )
        else:
            return adata_gears


def build_collate_fn(vocab: Optional[GeneVocab] = None):
    """Builds a collate function for GEARS DataLoader.

    The collate function combines individual samples into a standardized batch
    dictionary with 'inputs' and 'labels' fields, enabling unified engine processing.

    Args:
        vocab: Optional GeneVocab for gene name to index mapping. If None,
            assumes samples already contain proper gene indices.

    Returns:
        Collate function that returns batch_dict with 'inputs' and 'labels'.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> from perturblab.methods.gears import build_collate_fn
        >>> from perturblab.types import GeneVocab
        >>>
        >>> # Create collate function
        >>> vocab = GeneVocab(['TP53', 'KRAS', 'MYC'])
        >>> collate_fn = build_collate_fn(vocab)
        >>>
        >>> # Use in DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>>
        >>> # Iterate over batches
        >>> for batch_dict in loader:
        ...     # batch_dict has 'inputs' and 'labels'
        ...     inputs = batch_dict['inputs']
        ...     labels = batch_dict['labels']
        ...     
        ...     # Use with model
        ...     from perturblab.models.gears import GEARSInput
        ...     model_inputs = GEARSInput(**inputs)
        ...     outputs = model(model_inputs)

    Notes:
        Expected sample format (dict from Dataset.__getitem__):
            - 'x': Gene expression, shape (n_genes,) - baseline/control expression
            - 'y': Target expression, shape (n_genes,) - perturbed expression
            - 'pert': Perturbation label (str)
            - 'pert_genes': List of perturbed gene names (str) or indices (int)
            - 'de_idx': Optional DE gene indices (list[int])

        Output batch_dict format:
            {
                'inputs': {
                    'gene_expression': Tensor,       # (batch*n_genes,)
                    'pert_idx': list[list[int]],     # per-sample perturbation indices
                    'graph_batch_indices': Tensor,   # (batch*n_genes,) batch assignment
                },
                'labels': {
                    'predictions': Tensor,           # (batch, n_genes) target expression
                    'de_indices': list[list[int]],   # per-sample DE gene indices
                },
                'metadata': {
                    'perturbations': list[str],      # perturbation labels
                    'sample_indices': list[int],     # original sample indices (if available)
                }
            }
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    def collate_fn(samples: List[Dict]) -> Dict:
        """Collates samples into standardized batch dictionary.

        Args:
            samples: List of sample dictionaries from Dataset.__getitem__.

        Returns:
            Batch dictionary with 'inputs', 'labels', and 'metadata' fields.
        """
        all_x = []
        all_y = []
        all_pert_idx = []
        all_perts = []
        all_de_idx = []
        all_sample_idx = []

        for idx, sample in enumerate(samples):
            # Extract fields
            x = sample['x']
            y = sample['y']
            pert = sample['pert']
            pert_genes = sample.get('pert_genes', [])
            de_idx = sample.get('de_idx', [])
            sample_idx = sample.get('sample_idx', idx)

            # Convert to tensors
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)

            # Ensure 1D
            x = x.flatten()
            y = y.flatten()

            # Convert gene names to indices if vocab provided
            if vocab is not None and pert_genes and len(pert_genes) > 0:
                if isinstance(pert_genes[0], str):
                    pert_idx_list = [vocab.stoi[g] for g in pert_genes if g in vocab.stoi]
                else:
                    pert_idx_list = pert_genes
            else:
                pert_idx_list = pert_genes if pert_genes else [-1]

            # Collect
            all_x.append(x)
            all_y.append(y)
            all_pert_idx.append(pert_idx_list)
            all_perts.append(pert)
            all_de_idx.append(de_idx if de_idx else [-1])
            all_sample_idx.append(sample_idx)

        # Stack expressions
        gene_expression = torch.cat(all_x)  # (batch_size * n_genes,)
        target_expression = torch.stack(all_y)  # (batch_size, n_genes)

        # Create batch indices: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
        n_genes = all_x[0].shape[0]
        batch_indices = torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.long),
            n_genes
        )

        # Construct standardized batch dictionary
        batch_dict = {
            'inputs': {
                'gene_expression': gene_expression,
                'pert_idx': all_pert_idx,
                'graph_batch_indices': batch_indices,
            },
            'labels': {
                'predictions': target_expression,
                'de_indices': all_de_idx,
            },
            'metadata': {
                'perturbations': all_perts,
                'sample_indices': all_sample_idx,
            }
        }

        return batch_dict

    return collate_fn
