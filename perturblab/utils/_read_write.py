"""Read and write functions for PerturbLab data structures.

This module provides I/O functions for loading data into PerturbLab's data structures,
including CellData and Gene Ontology data.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, overload

import anndata as ad
import pandas as pd

if TYPE_CHECKING:
    from perturblab.types import CellData
    from perturblab.types.math import DAG

from .logging import get_logger

logger = get_logger()


# ==============================================================================
# Helper Functions for Auto-Detection
# ==============================================================================

def _guess_cell_type_col(obs: pd.DataFrame) -> Optional[str]:
    """Guess the column name for cell types in obs.
    
    Args:
        obs: The obs DataFrame from AnnData.
        
    Returns:
        Guessed column name or None if not found.
    """
    # Common patterns for cell type columns (case-insensitive)
    patterns = [
        "cell_type",
        "celltype",
        "cell_types",
        "celltypes",
        "cell type",
        "type",
        "cluster",
        "clusters",
        "cell_annotation",
        "annotation",
        "cell_label",
        "label",
    ]
    
    # First try exact case-insensitive match
    obs_lower = {col.lower(): col for col in obs.columns}
    for pattern in patterns:
        if pattern in obs_lower:
            col = obs_lower[pattern]
            logger.info(f"Auto-detected cell_type_col: '{col}'")
            return col
    
    # Try partial match
    for pattern in patterns:
        for col in obs.columns:
            if pattern in col.lower():
                logger.info(f"Auto-detected cell_type_col: '{col}' (partial match)")
                return col
    
    return None


def _guess_gene_name_col(var: pd.DataFrame) -> Optional[str]:
    """Guess the column name for gene names in var.
    
    Args:
        var: The var DataFrame from AnnData.
        
    Returns:
        Guessed column name or None if not found.
    """
    # Common patterns for gene name columns (case-insensitive)
    patterns = [
        "gene_name",
        "gene_names",
        "genename",
        "gene",
        "genes",
        "gene_symbol",
        "symbol",
        "symbols",
        "gene_id",
        "gene_ids",
        "geneid",
        "feature_name",
        "feature",
    ]
    
    # First try exact case-insensitive match
    var_lower = {col.lower(): col for col in var.columns}
    for pattern in patterns:
        if pattern in var_lower:
            col = var_lower[pattern]
            logger.info(f"Auto-detected gene_name_col: '{col}'")
            return col
    
    # Try partial match
    for pattern in patterns:
        for col in var.columns:
            if pattern in col.lower():
                logger.info(f"Auto-detected gene_name_col: '{col}' (partial match)")
                return col
    
    return None


def _guess_cell_id_col(obs: pd.DataFrame) -> Optional[str]:
    """Guess the column name for cell IDs in obs.
    
    Args:
        obs: The obs DataFrame from AnnData.
        
    Returns:
        Guessed column name or None if not found.
    """
    # Common patterns for cell ID columns (case-insensitive)
    patterns = [
        "cell_id",
        "cellid",
        "cell_ids",
        "cell_name",
        "cellname",
        "barcode",
        "barcodes",
    ]
    
    # First try exact case-insensitive match
    obs_lower = {col.lower(): col for col in obs.columns}
    for pattern in patterns:
        if pattern in obs_lower:
            col = obs_lower[pattern]
            logger.info(f"Auto-detected cell_id_col: '{col}'")
            return col
    
    # Try partial match
    for pattern in patterns:
        for col in obs.columns:
            if pattern in col.lower():
                logger.info(f"Auto-detected cell_id_col: '{col}' (partial match)")
                return col
    
    return None


# Overload signatures for type hints
@overload
def read_h5ad(
    filename: Union[str, Path],
    backed: Optional[Literal["r", "r+"]] = None,
    cell_type_col: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    cell_id_col: Optional[str] = None,
    duplicated_gene_policy: Literal["error", "first", "last", "remove"] = "error",
    auto_guess: bool = True,
    return_type: Literal["celldata"] = "celldata",
    **kwargs,
) -> "CellData":
    ...


@overload
def read_h5ad(
    filename: Union[str, Path],
    backed: Optional[Literal["r", "r+"]] = None,
    cell_type_col: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    cell_id_col: Optional[str] = None,
    duplicated_gene_policy: Literal["error", "first", "last", "remove"] = "error",
    auto_guess: bool = True,
    return_type: Literal["anndata"] = ...,
    **kwargs,
) -> ad.AnnData:
    ...


def read_h5ad(
    filename: Union[str, Path],
    backed: Optional[Literal["r", "r+"]] = None,
    cell_type_col: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    cell_id_col: Optional[str] = None,
    duplicated_gene_policy: Literal["error", "first", "last", "remove"] = "error",
    auto_guess: bool = True,
    return_type: Literal["celldata", "anndata"] = "celldata",
    **kwargs,
) -> Union["CellData", ad.AnnData]:
    """Read h5ad file and return CellData or AnnData object.

    This function wraps AnnData's read_h5ad with enhanced features:
    - Auto-detection of cell type and gene name columns
    - Flexible return type (CellData or AnnData)
    - Backed mode for large datasets
    - Duplicate gene handling

    Args:
        filename: Path to the h5ad file.
        backed: Load file in backed mode for large datasets:
            - None: Load entirely into memory (default, recommended for < 10GB)
            - 'r': Read-only backed mode (data stays on disk, read-only access)
            - 'r+': Read-write backed mode (data stays on disk, can be modified)
        cell_type_col: Column name in obs for cell type labels.
            If None and auto_guess=True, will attempt to auto-detect.
        gene_name_col: Column name in var for gene names.
            If None, uses var_names. If auto_guess=True, will attempt to auto-detect.
        cell_id_col: Column name in obs for cell IDs.
            If None, uses obs_names. If auto_guess=True, will attempt to auto-detect.
        duplicated_gene_policy: How to handle duplicate gene names (only for CellData):
            - 'error': Raise error if duplicates found (default)
            - 'first': Keep first occurrence
            - 'last': Keep last occurrence
            - 'remove': Remove all duplicate genes
        auto_guess: Whether to automatically guess column names if not provided.
            Defaults to True.
        return_type: Type of object to return:
            - 'celldata': Return CellData object with enhanced features (default)
            - 'anndata': Return raw AnnData object
        **kwargs: Additional arguments passed to anndata.read_h5ad().

    Returns:
        CellData object (if return_type='celldata') or AnnData object (if return_type='anndata').

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If duplicate genes found and policy is 'error' (CellData only).

    Examples:
        >>> # Basic usage with auto-detection
        >>> data = read_h5ad('dataset.h5ad')
        >>> print(f"Loaded {data.n_cells} cells × {data.n_genes} genes")
        >>>
        >>> # Specify column names explicitly
        >>> data = read_h5ad(
        ...     'dataset.h5ad',
        ...     cell_type_col='cell_type',
        ...     gene_name_col='gene_symbol'
        ... )
        >>>
        >>> # Load in backed mode for large datasets (> 10GB)
        >>> large_data = read_h5ad('large_dataset.h5ad', backed='r')
        >>> print(f"Backed: {large_data.isbacked}")
        >>> print(f"File: {large_data.filename}")
        >>>
        >>> # Return raw AnnData instead of CellData
        >>> adata = read_h5ad('dataset.h5ad', return_type='anndata')
        >>> print(type(adata))  # <class 'anndata.AnnData'>
        >>>
        >>> # Handle duplicate genes
        >>> data = read_h5ad(
        ...     'dataset.h5ad',
        ...     duplicated_gene_policy='first',
        ...     cell_type_col='cell_type'
        ... )
        >>>
        >>> # Disable auto-guessing
        >>> data = read_h5ad(
        ...     'dataset.h5ad',
        ...     auto_guess=False,
        ...     cell_type_col='my_cell_types'
        ... )

    Notes:
        **Backed Mode**:
        - Backed mode keeps data on disk and loads chunks on-demand.
        - Use backed='r' for read-only access (safe, prevents accidental modification).
        - Use backed='r+' if you need to modify the data on disk.
        - Backed mode is ideal for datasets > 10GB that don't fit in memory.
        - CellData supports backed mode seamlessly with lazy loading.

        **Auto-Detection**:
        - When auto_guess=True (default), the function attempts to detect:
            - Cell type column: looks for 'cell_type', 'celltype', 'cluster', etc.
            - Gene name column: looks for 'gene_name', 'gene_symbol', 'symbol', etc.
            - Cell ID column: looks for 'cell_id', 'barcode', etc.
        - Auto-detection uses case-insensitive pattern matching.
        - Explicitly provided column names override auto-detection.

        **CellData Features**:
        - Zero-copy gene alignment with virtual genes
        - Efficient data loading and transformation
        - PyTorch DataLoader integration
        - Splitting and sampling utilities
        - Use data.materialize() to convert backed data to in-memory

        **Performance Tips**:
        - For large datasets: use backed='r' + enable_cache() for training
        - For small datasets: load into memory (backed=None)
        - For multiple operations: materialize() backed views to avoid repeated disk I/O
    """
    filename = Path(filename)
    
    # Check file exists
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    logger.info(f"Reading h5ad file: {filename}")
    
    # Read AnnData
    adata = ad.read_h5ad(filename, backed=backed, **kwargs)
    
    # Log dataset info
    mode_info = f" (backed mode: '{backed}')" if backed else ""
    logger.info(
        f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes{mode_info}"
    )
    
    # Auto-detect columns if requested
    if auto_guess:
        if cell_type_col is None:
            detected = _guess_cell_type_col(adata.obs)
            if detected:
                cell_type_col = detected
            else:
                logger.debug("Could not auto-detect cell_type_col")
        
        if gene_name_col is None:
            detected = _guess_gene_name_col(adata.var)
            if detected:
                gene_name_col = detected
            else:
                logger.debug("Could not auto-detect gene_name_col, will use var_names")
        
        if cell_id_col is None:
            detected = _guess_cell_id_col(adata.obs)
            if detected:
                cell_id_col = detected
            else:
                logger.debug("Could not auto-detect cell_id_col, will use obs_names")
    
    # Return raw AnnData if requested
    if return_type == "anndata":
        logger.info("Returning raw AnnData object")
        return adata
    
    # Import CellData here to avoid circular import
    from perturblab.types import CellData
    
    # Create CellData wrapper
    cell_data = CellData(
        adata,
        cell_type_col=cell_type_col,
        gene_name_col=gene_name_col,
        cell_id_col=cell_id_col,
        duplicated_gene_policy=duplicated_gene_policy,
    )
    
    # Log detected configuration
    config_parts = []
    if cell_type_col:
        config_parts.append(f"cell_type_col='{cell_type_col}'")
    if gene_name_col:
        config_parts.append(f"gene_name_col='{gene_name_col}'")
    if cell_id_col:
        config_parts.append(f"cell_id_col='{cell_id_col}'")
    
    if config_parts:
        logger.info(f"CellData configuration: {', '.join(config_parts)}")
    
    return cell_data


def read_obo(
    obo_path: Union[str, Path],
    load_obsolete: bool = False,
) -> Tuple[List[Dict], "DAG"]:
    """Parse OBO file and return GO term information and hierarchy.

    This is a standalone parser that extracts Gene Ontology term metadata
    and their hierarchical relationships (is_a) from an OBO format file.

    Args:
        obo_path: Path to the OBO file (e.g., go-basic.obo).
        load_obsolete: Whether to include obsolete GO terms. Defaults to False.

    Returns:
        A tuple containing:
            - List[Dict]: List of GO term dictionaries, each with keys:
                - 'id': GO term ID (e.g., 'GO:0006915')
                - 'name': Human-readable term name
                - 'namespace': One of 'biological_process', 'molecular_function',
                  or 'cellular_component'
                - 'definition': Detailed definition of the term
                - 'is_obsolete': Boolean indicating if term is obsolete
                - 'alt_ids': Set of alternative IDs for this term
            - DAG: Directed acyclic graph of GO term relationships,
              with parent -> child edges representing 'is_a' relationships.

    Raises:
        FileNotFoundError: If the OBO file doesn't exist.

    Examples:
        >>> # Load GO ontology
        >>> terms, dag = read_obo('go-basic.obo')
        >>> print(f"Loaded {len(terms)} GO terms")
        >>> print(f"DAG has {dag.n_edges} edges")
        >>>
        >>> # Access term information
        >>> term_dict = {t['id']: t for t in terms}
        >>> print(term_dict['GO:0006915']['name'])
        >>> # Output: 'apoptotic process'
        >>>
        >>> # Query hierarchy
        >>> parents = dag.get_parents('GO:0006915')
        >>> children = dag.get_children('GO:0006915')
        >>>
        >>> # Include obsolete terms
        >>> all_terms, dag = read_obo('go-basic.obo', load_obsolete=True)

    Notes:
        - The parser extracts only 'is_a' relationships. Other relationship
          types (part_of, regulates, etc.) are not included.
        - Download OBO files from: http://current.geneontology.org/ontology/
        - The DAG structure allows efficient querying of term hierarchies.
    """
    obo_path = Path(obo_path)
    
    if not obo_path.exists():
        raise FileNotFoundError(
            f"OBO file not found: {obo_path}\n"
            "Download from: http://current.geneontology.org/ontology/go-basic.obo"
        )

    logger.info(f"Parsing OBO file: {obo_path}")

    terms = []
    edges = []
    format_version = None
    data_version = None
    default_namespace = "default"

    with open(obo_path, "r") as f:
        in_header = True
        current_term = None

        for line in f:
            line = line.strip()

            # Parse header metadata
            if in_header:
                if line.startswith("format-version:"):
                    format_version = line.split(":", 1)[1].strip()
                elif line.startswith("data-version:"):
                    data_version = line.split(":", 1)[1].strip()
                elif line.startswith("default-namespace:"):
                    default_namespace = line.split(":", 1)[1].strip()
                elif line == "[Term]":
                    in_header = False
                    current_term = {
                        "id": "",
                        "name": "",
                        "namespace": default_namespace,
                        "definition": "",
                        "is_obsolete": False,
                        "alt_ids": set(),
                        "_parent_ids": set(),  # Temporary storage for parsing
                    }
                continue

            # Start of new term block
            if line == "[Term]":
                if current_term is not None:
                    # Save previous term if not obsolete (or if loading obsolete)
                    if not current_term["is_obsolete"] or load_obsolete:
                        terms.append(current_term)
                        # Add edges from this term to its parents
                        for parent_id in current_term["_parent_ids"]:
                            edges.append((parent_id, current_term["id"]))

                # Initialize new term
                current_term = {
                    "id": "",
                    "name": "",
                    "namespace": default_namespace,
                    "definition": "",
                    "is_obsolete": False,
                    "alt_ids": set(),
                    "_parent_ids": set(),
                }

            # Skip typedef sections (relationship definitions)
            elif line == "[Typedef]":
                if current_term is not None:
                    if not current_term["is_obsolete"] or load_obsolete:
                        terms.append(current_term)
                        for parent_id in current_term["_parent_ids"]:
                            edges.append((parent_id, current_term["id"]))
                    current_term = None

            # Parse term field values
            elif current_term is not None and ":" in line:
                key, _, value = line.partition(":")
                value = value.strip()

                if key == "id":
                    current_term["id"] = value
                elif key == "name":
                    current_term["name"] = value
                elif key == "namespace":
                    current_term["namespace"] = value
                elif key == "def":
                    # Definition is enclosed in quotes
                    if value.startswith('"'):
                        end_quote = value.find('"', 1)
                        if end_quote != -1:
                            current_term["definition"] = value[1:end_quote]
                elif key == "is_a":
                    # Format: "GO:0008150 ! biological_process"
                    parent_id = value.split()[0]
                    current_term["_parent_ids"].add(parent_id)
                elif key == "alt_id":
                    current_term["alt_ids"].add(value)
                elif key == "is_obsolete" and value == "true":
                    current_term["is_obsolete"] = True

        # Add last term
        if current_term is not None:
            if not current_term["is_obsolete"] or load_obsolete:
                terms.append(current_term)
                for parent_id in current_term["_parent_ids"]:
                    edges.append((parent_id, current_term["id"]))

    # Clean up temporary fields
    for term in terms:
        del term["_parent_ids"]

    # Import DAG here to avoid circular import
    from perturblab.types.math import DAG

    # Build DAG from edges
    dag = DAG(edges, validate=False) if edges else DAG([], validate=False)

    logger.info(
        f"Parsed {len(terms)} GO terms, {len(edges)} edges "
        f"(format: {format_version}, version: {data_version})"
    )

    return terms, dag

