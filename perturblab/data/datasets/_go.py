"""Gene Ontology (GO) dataset for gene annotation and functional analysis.

Provides a structured representation of gene-GO term relationships using
BipartiteGraph for efficient querying and analysis. Uses DAG for managing
GO ontology hierarchy.
"""

import pickle
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple

from perturblab.core import Dataset
from perturblab.types import GeneVocab, Vocab
from perturblab.types.math import BipartiteGraph, DAG
from perturblab.utils import get_logger, read_obo
from perturblab.data.resources import load_dataset

logger = get_logger()

__all__ = ["GODataset", "load_go_from_gears"]


class GODataset(Dataset[Tuple[BipartiteGraph, Optional[DAG]]]):
    """Gene Ontology dataset with bipartite graph structure.
    
    Inherits from `Dataset[Tuple[BipartiteGraph, Optional[DAG]]]`, providing
    a standardized interface for gene-GO term relationships. The underlying data
    (accessed via `.data`) is a tuple of (BipartiteGraph, DAG | None) containing:
    - BipartiteGraph: Gene-GO term connections
    - DAG | None: GO ontology hierarchy (if loaded)
    
    Stores gene-GO term relationships as a bipartite graph, with genes on one
    side and GO terms on the other. Supports efficient querying of gene annotations
    and GO term memberships.
    
    Factory Methods:
        - `from_gene2go()`: Create from gene-to-GO mapping
        - `from_go2gene()`: Create from GO-to-gene mapping
        - `from_ontology()`: Create from OBO ontology file
    
    Core Properties:
        data: Tuple[BipartiteGraph, DAG | None] - The underlying data tuple
        gene_vocab: GeneVocab - Gene name to index mapping
        go_vocab: Vocab - GO term ID to index mapping
        ontology_terms: Dict | None - GO term metadata (if ontology loaded)
        ontology_dag: DAG | None - GO term hierarchy (if ontology loaded)
    
    Convenience Properties:
        n_genes: Number of genes
        n_go_terms: Number of GO terms
        n_annotations: Number of gene-GO annotations
        genes: List of all gene names
        go_term_ids: List of all GO term IDs
    
    Example:
        >>> # Create from gene2go dictionary
        >>> gene2go = {'TP53': ['GO:0006915', 'GO:0006974'], 'BRCA1': ['GO:0006281']}
        >>> dataset = GODataset.from_gene2go(gene2go)
        >>> 
        >>> # Access underlying data (tuple of BipartiteGraph and DAG)
        >>> bipartite_graph, dag = dataset.data
        >>> print(f"Graph has {bipartite_graph.n_edges} edges")
        >>> print(f"DAG: {dag}")  # None if not loaded
        >>> 
        >>> # Query GO terms for a gene
        >>> go_terms = dataset.go_terms('TP53')
        >>> 
        >>> # Export back to dictionary
        >>> gene2go = dataset.to_gene2go()
        >>> go2gene = dataset.to_go2gene()
        >>> 
        >>> # Load with ontology (includes DAG)
        >>> dataset = GODataset.from_ontology()
        >>> bipartite_graph, dag = dataset.data
        >>> term_info = dataset.query('GO:0006915')
        >>> print(term_info['name'])  # 'apoptotic process'
    """
    
    def __init__(self):
        """Private constructor. Use factory methods to create instances."""
        # Internal data structures (private)
        self._gene_vocab: Optional[GeneVocab] = None
        self._go_vocab: Optional[Vocab] = None
        self._gene_to_go: Optional[BipartiteGraph] = None
        self._go_to_gene_inverse: Optional[Dict[int, List[int]]] = None
        
        # Public attributes
        self.ontology_terms: Optional[Dict[str, Dict]] = None
        self.ontology_dag: Optional[DAG] = None
    
    @property
    def data(self) -> Tuple[BipartiteGraph, Optional[DAG]]:
        """Get the underlying data (bipartite graph + ontology DAG).
        
        Returns:
            Tuple[BipartiteGraph, DAG | None]: A tuple containing:
                - BipartiteGraph: Gene-GO term bipartite graph
                - DAG | None: GO ontology hierarchy (if loaded)
        
        Raises:
            ValueError: If dataset not initialized.
        
        Example:
            >>> dataset = GODataset.from_ontology()
            >>> bipartite_graph, dag = dataset.data
            >>> print(f"Graph has {bipartite_graph.n_edges} edges")
            >>> print(f"DAG has {dag.n_edges if dag else 0} edges")
        """
        if self._gene_to_go is None:
            raise ValueError("GODataset not initialized. Use factory methods to create.")
        return (self._gene_to_go, self.ontology_dag)
    
    @classmethod
    def from_gene2go(
        cls,
        gene2go: Dict[str, List[str] | Set[str]],
    ) -> 'GODataset':
        """Create GODataset from gene-to-GO mapping.
        
        Args:
            gene2go: Dictionary mapping gene names to GO term IDs.
                Format: {'gene1': ['GO:001', 'GO:002'], ...}
        
        Returns:
            GODataset: Initialized dataset.
        
        Example:
            >>> gene2go = {
            ...     'TP53': ['GO:0006915', 'GO:0006974'],
            ...     'BRCA1': ['GO:0006281', 'GO:0006974'],
            ... }
            >>> dataset = GODataset.from_gene2go(gene2go)
            >>> print(dataset.go_terms('TP53'))
        """
        instance = cls()
        instance._build_from_gene2go(gene2go)
        
        logger.info(
            f"GODataset created from gene2go: {len(instance._gene_vocab)} genes, "
            f"{len(instance._go_vocab)} GO terms, {instance._gene_to_go.n_edges} annotations"
        )
        
        return instance
    
    @classmethod
    def from_go2gene(
        cls,
        go2gene: Dict[str, List[str] | Set[str]],
    ) -> 'GODataset':
        """Create GODataset from GO-to-gene mapping.
        
        Args:
            go2gene: Dictionary mapping GO term IDs to gene names.
                Format: {'GO:001': ['gene1', 'gene2'], ...}
        
        Returns:
            GODataset: Initialized dataset.
        
        Example:
            >>> go2gene = {
            ...     'GO:0006915': ['TP53', 'BAX', 'BCL2'],
            ...     'GO:0006974': ['TP53', 'BRCA1'],
            ... }
            >>> dataset = GODataset.from_go2gene(go2gene)
            >>> print(dataset.related_genes('GO:0006915'))
        """
        instance = cls()
        instance._build_from_go2gene(go2gene)
        
        logger.info(
            f"GODataset created from go2gene: {len(instance._gene_vocab)} genes, "
            f"{len(instance._go_vocab)} GO terms, {instance._gene_to_go.n_edges} annotations"
        )
        
        return instance
    
    @classmethod
    def from_ontology(
        cls,
        obo_path: Optional[str | Path] = None,
        download: bool = True,
        load_obsolete: bool = False,
    ) -> 'GODataset':
        """Create GODataset from GO ontology OBO file.
        
        This creates a dataset containing ALL GO terms from the ontology,
        with their hierarchical relationships. No gene annotations are included
        unless you merge with another dataset.
        
        Args:
            obo_path: Path to OBO file. Auto-downloads if None.
            download: Whether to download if file not found.
            load_obsolete: Whether to include obsolete terms.
        
        Returns:
            GODataset: Dataset with ontology structure but no gene annotations.
        
        Example:
            >>> # Auto-download and load
            >>> dataset = GODataset.from_ontology()
            >>> 
            >>> # Query ontology
            >>> term_info = dataset.query('GO:0006915')
            >>> print(term_info['name'])  # 'apoptotic process'
        """
        instance = cls()
        
        # Determine OBO file path
        if obo_path is None:
            if download:
                # Use new resource system to download GO basic
                logger.info("Downloading GO basic ontology...")
                obo_path = load_dataset('go/go_basic')
            else:
                raise ValueError("obo_path is None and download=False")
        else:
            # Use provided file path
            obo_path = Path(obo_path)
            if not obo_path.exists():
                raise FileNotFoundError(f"OBO file not found: {obo_path}")
        
        # Parse OBO file
        terms_list, dag = read_obo(obo_path, load_obsolete=load_obsolete)
        
        # Build term dictionary
        # Main ID -> term mapping (for vocab construction)
        main_term_ids = sorted([term['id'] for term in terms_list])
        
        # Build full ontology_terms dictionary (including alt_ids)
        ontology_terms = {}
        for term in terms_list:
            ontology_terms[term['id']] = term
            # Map alt_ids to the same term
            for alt_id in term['alt_ids']:
                ontology_terms[alt_id] = term
        
        # Store ontology
        instance.ontology_terms = ontology_terms
        instance.ontology_dag = dag
        
        # Create empty gene-GO mapping structures (ontology only, no annotations)
        # IMPORTANT: Use only main IDs for vocab to ensure index consistency
        instance._gene_vocab = GeneVocab([], default_token='<unk>', default_index=-1)
        instance._go_vocab = Vocab(main_term_ids, default_token='<unk>', default_index=-1)
        instance._gene_to_go = BipartiteGraph(
            edges=[],
            shape=(0, len(instance._go_vocab)),
        )
        
        logger.info(
            f"GODataset created from ontology: {len(terms_list)} GO terms, "
            f"{dag.n_edges} hierarchical edges"
        )
        
        return instance
    
    def _build_from_gene2go(self, gene2go: Dict[str, List[str] | Set[str]]):
        """Build internal structures from gene2go dictionary.
        
        Args:
            gene2go: Dictionary mapping genes to GO terms.
        """
        # Extract all genes and GO terms
        all_genes = sorted(gene2go.keys())
        all_go_terms = set()
        for go_terms in gene2go.values():
            all_go_terms.update(go_terms)
        all_go_terms = sorted(all_go_terms)
        
        # Create vocabularies
        self._gene_vocab = GeneVocab(all_genes, default_token='<unk>', default_index=-1)
        self._go_vocab = Vocab(all_go_terms, default_token='<unk>', default_index=-1)
        
        # Build edge list
        edges = []
        for gene, go_terms in gene2go.items():
            gene_idx = self._gene_vocab[gene]
            for go_term in go_terms:
                go_idx = self._go_vocab[go_term]
                edges.append((gene_idx, go_idx))
        
        # Create bipartite graph
        self._gene_to_go = BipartiteGraph(
            edges=edges,
            shape=(len(self._gene_vocab), len(self._go_vocab)),
        )
    
    def _build_from_go2gene(self, go2gene: Dict[str, List[str] | Set[str]]):
        """Build internal structures from go2gene dictionary.
        
        Args:
            go2gene: Dictionary mapping GO terms to genes.
        """
        # Extract all GO terms and genes
        all_go_terms = sorted(go2gene.keys())
        all_genes = set()
        for genes in go2gene.values():
            all_genes.update(genes)
        all_genes = sorted(all_genes)
        
        # Create vocabularies
        self._gene_vocab = GeneVocab(all_genes, default_token='<unk>', default_index=-1)
        self._go_vocab = Vocab(all_go_terms, default_token='<unk>', default_index=-1)
        
        # Build edge list
        edges = []
        for go_term, genes in go2gene.items():
            go_idx = self._go_vocab[go_term]
            for gene in genes:
                gene_idx = self._gene_vocab[gene]
                edges.append((gene_idx, go_idx))
        
        # Create bipartite graph
        self._gene_to_go = BipartiteGraph(
            edges=edges,
            shape=(len(self._gene_vocab), len(self._go_vocab)),
        )
    
    def go_terms(self, gene: str) -> List[str]:
        """Get GO terms annotated to a gene.
        
        Args:
            gene: Gene name or symbol.
        
        Returns:
            List[str]: List of GO term IDs. Returns empty list if gene not found.
        
        Example:
            >>> go_terms = dataset.go_terms('TP53')
            >>> print(go_terms[:5])
        """
        if self._gene_vocab is None:
            return []
        
        # Get gene index
        gene_idx = self._gene_vocab[gene]
        if gene_idx == self._gene_vocab.default_index:
            return []
        
        # Get GO term indices
        go_indices = self._gene_to_go.get_neighbors(gene_idx, check_bounds=False)
        
        # Convert indices to GO term IDs
        return [self._go_vocab[int(idx)] for idx in go_indices]
    
    def related_genes(self, go_term: str) -> List[str]:
        """Get genes annotated with a specific GO term.
        
        Args:
            go_term: GO term ID (e.g., 'GO:0006915').
        
        Returns:
            List[str]: List of gene names. Returns empty list if GO term not found.
        
        Example:
            >>> genes = dataset.related_genes('GO:0006915')
            >>> print(f"Found {len(genes)} genes")
        """
        if self._go_vocab is None:
            return []
        
        # Get GO term index
        go_idx = self._go_vocab[go_term]
        if go_idx == self._go_vocab.default_index:
            return []
        
        # Build inverse mapping if not cached
        if self._go_to_gene_inverse is None:
            self._go_to_gene_inverse = self._gene_to_go.get_inverse_mapping()
        
        # Get gene indices
        gene_indices = self._go_to_gene_inverse.get(go_idx, [])
        
        # Convert indices to gene names
        return [self._gene_vocab[idx] for idx in gene_indices]
    
    def query(self, go_term: str) -> Optional[Dict]:
        """Query GO term information from ontology.
        
        Returns GO term metadata including name, namespace, definition,
        and hierarchical information. Requires ontology to be loaded.
        
        Args:
            go_term: GO term ID (e.g., 'GO:0006915').
        
        Returns:
            Dict | None: Dictionary with GO term information, or None if not found.
                Keys include:
                - 'id': GO term ID
                - 'name': Human-readable name
                - 'namespace': biological_process, molecular_function, or cellular_component
                - 'definition': Detailed description
                - 'is_obsolete': Whether the term is obsolete
                - 'alt_ids': Set of alternative IDs
                - 'level': Shortest distance from root (if DAG loaded)
                - 'depth': Longest distance from root (if DAG loaded)
                - 'parents': Set of parent GO IDs (if DAG loaded)
                - 'children': Set of child GO IDs (if DAG loaded)
        
        Example:
            >>> dataset = GODataset.from_ontology()
            >>> info = dataset.query('GO:0006915')
            >>> print(info['name'])  # 'apoptotic process'
            >>> print(info['namespace'])  # 'biological_process'
            >>> print(info['level'])  # e.g., 4
        """
        if self.ontology_terms is None:
            logger.warning("Ontology not loaded. Cannot query GO term info.")
            return None
        
        if go_term not in self.ontology_terms:
            return None
        
        # Get term info
        term_info = self.ontology_terms[go_term].copy()
        
        # Add hierarchical info if DAG is available
        if self.ontology_dag is not None and go_term in self.ontology_dag:
            term_info['level'] = self.ontology_dag.get_level(go_term)
            term_info['depth'] = self.ontology_dag.get_depth(go_term)
            term_info['parents'] = self.ontology_dag.get_parents(go_term)
            term_info['children'] = self.ontology_dag.get_children(go_term)
        
        return term_info
    
    def to_gene2go(self) -> Dict[str, List[str]]:
        """Export dataset as gene-to-GO mapping dictionary.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping gene names to GO term IDs.
        
        Example:
            >>> dataset = GODataset.from_gene2go(gene2go)
            >>> exported = dataset.to_gene2go()
            >>> print(exported['TP53'])  # ['GO:0006915', 'GO:0006974', ...]
        """
        if self._gene_vocab is None:
            return {}
        
        gene2go = {}
        for gene in self._gene_vocab.itos:
            go_terms = self.go_terms(gene)
            if go_terms:  # Only include genes with annotations
                gene2go[gene] = go_terms
        
        return gene2go
    
    def to_go2gene(self) -> Dict[str, List[str]]:
        """Export dataset as GO-to-gene mapping dictionary.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping GO term IDs to gene names.
        
        Example:
            >>> dataset = GODataset.from_go2gene(go2gene)
            >>> exported = dataset.to_go2gene()
            >>> print(exported['GO:0006915'])  # ['TP53', 'BAX', 'BCL2', ...]
        """
        if self._go_vocab is None:
            return {}
        
        go2gene = {}
        for go_term in self._go_vocab.itos:
            genes = self.related_genes(go_term)
            if genes:  # Only include GO terms with annotations
                go2gene[go_term] = genes
        
        return go2gene
    
    @property
    def gene_vocab(self) -> GeneVocab:
        """Get the gene vocabulary.
        
        Returns:
            GeneVocab: Vocabulary mapping gene names to indices.
        
        Raises:
            ValueError: If dataset not initialized.
        """
        if self._gene_vocab is None:
            raise ValueError("GODataset not initialized.")
        return self._gene_vocab
    
    @property
    def go_vocab(self) -> Vocab:
        """Get the GO term vocabulary.
        
        Returns:
            Vocab: Vocabulary mapping GO term IDs to indices.
        
        Raises:
            ValueError: If dataset not initialized.
        """
        if self._go_vocab is None:
            raise ValueError("GODataset not initialized.")
        return self._go_vocab
    
    @property
    def n_genes(self) -> int:
        """Number of genes in the dataset."""
        return len(self._gene_vocab) if self._gene_vocab else 0
    
    @property
    def n_go_terms(self) -> int:
        """Number of GO terms in the dataset."""
        return len(self._go_vocab) if self._go_vocab else 0
    
    @property
    def n_annotations(self) -> int:
        """Number of gene-GO annotations."""
        return self._gene_to_go.n_edges if self._gene_to_go else 0
    
    @property
    def genes(self) -> List[str]:
        """List of all genes in the dataset."""
        return list(self._gene_vocab.itos) if self._gene_vocab else []
    
    @property
    def go_term_ids(self) -> List[str]:
        """List of all GO term IDs in the dataset."""
        return list(self._go_vocab.itos) if self._go_vocab else []
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GODataset(genes={self.n_genes}, "
            f"go_terms={self.n_go_terms}, "
            f"annotations={self.n_annotations})"
        )
    
    def __len__(self) -> int:
        """Return number of genes."""
        return self.n_genes


def load_go_from_gears(gears_provided_file_path: Optional[str] = None) -> GODataset:
    """Load GO dataset from GEARS gene2go pickle file.
    
    Args:
        gears_provided_file_path: Path to local gene2go pickle file.
            If None, downloads from default source with caching.
    
    Returns:
        GODataset: Loaded GO dataset.
    
    Example:
        >>> # Auto-download from GEARS (cached)
        >>> dataset = load_go_from_gears()
        >>> 
        >>> # Load from local file
        >>> dataset = load_go_from_gears('/path/to/gene2go.pkl')
        >>> 
        >>> # Query and export
        >>> go_terms = dataset.go_terms('TP53')
        >>> gene2go = dataset.to_gene2go()
    """
    logger.info("Loading GO dataset from GEARS source...")
    
    # Determine source
    if gears_provided_file_path is None:
        # Use new resource system
        logger.info("Downloading gene2go from GEARS...")
        pkl_path = load_dataset('go/gene2go_gears')
        
        with open(pkl_path, 'rb') as f:
            gene2go_dict: Dict[str, List[str]] = pickle.load(f)
    else:
        # Load from provided file path
        pkl_path = Path(gears_provided_file_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"File not found: {gears_provided_file_path}")
        
        logger.info(f"Loading pickle file from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            gene2go_dict: Dict[str, List[str]] = pickle.load(f)
    
    logger.info(f"Loaded {len(gene2go_dict)} genes from GEARS")
    
    # Count GO terms and annotations
    all_go_terms = set()
    n_annotations = 0
    for go_terms in gene2go_dict.values():
        all_go_terms.update(go_terms)
        n_annotations += len(go_terms)
    
    logger.info(
        f"Found {len(gene2go_dict)} genes, "
        f"{len(all_go_terms)} GO terms, "
        f"{n_annotations} annotations"
    )
    
    # Create dataset using factory method
    return GODataset.from_gene2go(gene2go_dict)

