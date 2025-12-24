from perturblab.data.datatype import GeneMap, GeneVocab, Vocab
from perturblab.data.datatype.math import BipartiteGraph
from typing import Literal, Dict
import pandas as pd
from perturblab.utils import get_logger
logger = get_logger()

def build_human_gene_map_from_sources(
    source: Literal['hgnc', 'ensembl', 'both'] = 'hgnc',
) -> GeneMap:
    """Build GeneMap for human genes from HGNC and/or Ensembl data.
    
    This function is specifically designed for human (Homo sapiens) genes and supports
    multiple data sources including HGNC (official nomenclature) and Ensembl.
    
    Args:
        source: Data source to use:
            - 'hgnc': HGNC only (official human gene nomenclature, recommended)
            - 'ensembl': Ensembl only (human genes from Ensembl GTF)
            - 'both': Merge HGNC + Ensembl (most comprehensive)
    
    Returns:
        GeneMap: Constructed human gene mapping system.
    
    Important Notes:
        - HGNC provides curated, official human gene symbols and is recommended
        - When using 'ensembl' or 'both', some Ensembl gene names (e.g., SNORA70, Y_RNA)
          correspond to multiple Ensembl IDs (gene copies on different locations)
        - To maintain the injective constraint (symbol -> ID), only the first ID
          for each gene name is kept
        - In 'both' mode, HGNC symbols take priority; Ensembl adds additional genes
          not present in HGNC
    
    Example:
        >>> # Recommended: Use HGNC for official nomenclature
        >>> gene_map = build_human_gene_map_from_sources(source='hgnc')
        >>> gene_map.standardize('TP53')  # 'TP53'
        >>> gene_map.id('TP53')  # 'ENSG00000141510'
        >>> 
        >>> # Most comprehensive: Merge HGNC + Ensembl
        >>> gene_map = build_human_gene_map_from_sources(source='both')
    """
    return _build_gene_map_impl(source=source, species='homo_sapiens', allow_hgnc=True)


def build_gene_map_from_sources(
    species: str,
) -> GeneMap:
    """Build GeneMap from Ensembl data for any species.
    
    This function uses Ensembl GTF data as the source for gene information.
    It is designed for general species support where HGNC is not available
    (HGNC is human-specific).
    
    Args:
        species: Species name in Ensembl format (e.g., 'mus_musculus', 'rattus_norvegicus').
            Use Ensembl.SPECIES_* constants for convenience.
    
    Returns:
        GeneMap: Constructed gene mapping system for the specified species.
    
    Important Notes:
        - This function only supports Ensembl as the data source
        - Some Ensembl gene names may correspond to multiple gene IDs (e.g., gene families,
          paralogs, or genes on different chromosomes)
        - To maintain the injective constraint (symbol -> ID), only the first encountered
          gene ID for each gene name is kept, and subsequent IDs with the same name are skipped
        - For human genes, consider using build_human_gene_map_from_sources() instead,
          which supports HGNC's curated official nomenclature
    
    Example:
        >>> from perturblab.data.downloader import Ensembl
        >>> 
        >>> # Build for mouse
        >>> mouse_map = build_gene_map_from_sources(species=Ensembl.SPECIES_MOUSE)
        >>> 
        >>> # Build for zebrafish
        >>> fish_map = build_gene_map_from_sources(species=Ensembl.SPECIES_ZEBRAFISH)
    """
    return _build_gene_map_impl(source='ensembl', species=species, allow_hgnc=False)


def _build_gene_map_impl(
    source: Literal['hgnc', 'ensembl', 'both'],
    species: str,
    allow_hgnc: bool,
) -> GeneMap:
    """Internal implementation for building GeneMap from various data sources.
    
    This is the core implementation that handles data extraction, deduplication,
    and construction of the GeneMap with proper constraint validation.
    
    Args:
        source: Data source(s) to use ('hgnc', 'ensembl', or 'both').
        species: Species name (e.g., 'homo_sapiens', 'mus_musculus').
        allow_hgnc: Whether HGNC source is allowed (only True for human).
    
    Returns:
        GeneMap: Constructed gene mapping system.
    
    Implementation Details:
        Data Processing Strategy:
        1. Phase 1: Register all symbols and their primary IDs
           - For HGNC: Each approved symbol gets one Ensembl ID
           - For Ensembl: Each unique gene name gets one gene ID (first encountered)
           - Symbols are added as their own primary aliases to ensure surjectivity
        
        2. Phase 2: Add additional aliases
           - HGNC: alias_symbols and previous_symbols
           - Ensembl: Alternative gene names from GTF (in 'both' mode)
           - Conflicts resolved by keeping first mapping (symbols have priority)
        
        3. Final validation: Ensure all symbols have at least one alias
        
        Handling Multiple IDs per Gene Name (Ensembl):
        - Some gene names (e.g., SNORA70, Y_RNA, Metazoa_SRP) have multiple IDs
        - These represent gene copies, paralogs, or loci on different chromosomes
        - To maintain injective constraint (symbol -> ID), only the first ID is kept
        - This is necessary but means some genomic features are excluded
        
        Alias Conflicts:
        - If an alias name matches an existing symbol, the symbol takes priority
        - Example: CAPS2 is both a symbol and an alias for CADPS2; CAPS2 symbol wins
    """
    # Import downloaders
    from perturblab.data.downloader import download_hgnc, download_ensembl, HGNC, Ensembl
    import re
    
    if source in ('hgnc', 'both') and not allow_hgnc:
        raise ValueError("HGNC is only available for human. Use build_human_gene_map_from_sources()")
    
    logger.info(f"Building GeneMap (source={source}, species={species})")
    
    all_symbols = []
    all_ids = []
    symbol_to_id_edges = []
    
    # Mappings for construction
    symbol_to_idx = {}
    id_to_idx = {}
    
    # *** FIX: 添加 id 到 symbol_idx 的反向映射，用于 both 模式下的合并 ***
    id_to_symbol_idx = {} 
    
    # alias_data stores: {alias_string: symbol_idx}
    alias_data = {}
    
    # Metadata: {gene_id: {column: value}}
    meta_data = {}
    
    # 1. Process HGNC data (human only)
    if source in ('hgnc', 'both'):
        logger.info("Downloading HGNC data...")
        df_hgnc = download_hgnc(columns=[
            HGNC.COL_SYMBOL,
            HGNC.COL_NAME,
            HGNC.COL_LOCUS_TYPE,
            HGNC.COL_LOCUS_GROUP,
            HGNC.COL_CHROMOSOME,
            HGNC.COL_GENE_GROUP_NAME,
            HGNC.COL_ALIASES,
            HGNC.COL_PREV_SYMBOLS,
            HGNC.COL_ENSEMBL_ID,
        ])
        
        logger.info(f"Processing {len(df_hgnc)} HGNC entries...")
        
        # Phase 1: Register all symbols and IDs, and add symbol -> ID mappings
        # Each symbol gets itself as primary alias (ensures surjectivity)
        for _, row in df_hgnc.iterrows():
            symbol = row['approved_symbol']
            ensembl_id = row['ensembl_gene_id']
            
            if not symbol or not ensembl_id:
                continue
            
            # Register symbol
            if symbol not in symbol_to_idx:
                symbol_to_idx[symbol] = len(all_symbols)
                all_symbols.append(symbol)
            symbol_idx = symbol_to_idx[symbol]
            
            # Register ID
            if ensembl_id not in id_to_idx:
                id_to_idx[ensembl_id] = len(all_ids)
                all_ids.append(ensembl_id)
            id_idx = id_to_idx[ensembl_id]
            
            # Add symbol -> ID mapping
            symbol_to_id_edges.append((symbol_idx, id_idx))
            id_to_symbol_idx[id_idx] = symbol_idx
            
            # Collect metadata for this gene ID
            meta_data[ensembl_id] = {
                'symbol': symbol,
                'name': row.get('approved_name', ''),
                'locus_type': row.get('locus_type', ''),
                'locus_group': row.get('locus_group', ''),
                'chromosome': row.get('location', ''),
                'gene_group': row.get('gene_group_name', ''),
                'source': 'hgnc',
            }
            
            # Phase 1: Add symbol as its own primary alias (priority)
            # This ensures every symbol has at least one alias (itself)
            alias_data[symbol] = symbol_idx
        
        # Phase 2: Add additional aliases (may conflict with symbols, but symbols take priority)
        for _, row in df_hgnc.iterrows():
            symbol = row['approved_symbol']
            ensembl_id = row['ensembl_gene_id']
            
            if not symbol or not ensembl_id:
                continue
            
            symbol_idx = symbol_to_idx[symbol]
            
            # Collect additional aliases (excluding the symbol itself, already added)
            additional_aliases = []
            if row['alias_symbols']:
                additional_aliases.extend([a.strip() for a in row['alias_symbols'].split('|') if a.strip()])
            if row['previous_symbols']:
                additional_aliases.extend([a.strip() for a in row['previous_symbols'].split('|') if a.strip()])
            
            # Add additional aliases (skip if already occupied by a symbol)
            for alias in additional_aliases:
                if alias and alias not in alias_data:
                    alias_data[alias] = symbol_idx
    
    # 2. Process Ensembl data
    if source in ('ensembl', 'both'):
        logger.info(f"Downloading Ensembl GTF for {species}...")
        gtf_path = download_ensembl(
            species=species,
            gtf_type=Ensembl.GTF_CHR,
            decompress=True
        )
        
        logger.info(f"Parsing Ensembl GTF from {gtf_path}...")
        
        ensembl_genes = {} # gene_id -> {gene_name, gene_biotype, chromosome}
        
        with open(gtf_path, 'r') as f:
            for line in f:
                if line.startswith('#') or '\tgene\t' not in line:
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                chrom = parts[0]
                attrs = parts[8]
                gene_id_match = re.search(r'gene_id "([^"]+)"', attrs)
                gene_name_match = re.search(r'gene_name "([^"]+)"', attrs)
                gene_biotype_match = re.search(r'gene_biotype "([^"]+)"', attrs)
                
                if not gene_id_match:
                    continue
                
                gene_id = gene_id_match.group(1)
                gene_name = gene_name_match.group(1) if gene_name_match else gene_id
                gene_biotype = gene_biotype_match.group(1) if gene_biotype_match else ''
                
                # Strip version from Ensembl ID (ENSG00000.1 -> ENSG00000)
                gene_id_base = gene_id.split('.')[0]
                
                if gene_id_base not in ensembl_genes:
                    ensembl_genes[gene_id_base] = {
                        'gene_name': gene_name,
                        'gene_biotype': gene_biotype,
                        'chromosome': chrom,
                    }
        
        logger.info(f"Extracted {len(ensembl_genes)} genes from Ensembl")
        
        # Phase 1: Register symbols/IDs for new genes (not in HGNC)
        # Note: Some gene_names may have multiple gene_ids (e.g. SNORA70).
        # We only keep the first gene_id for each gene_name to maintain injective constraint.
        for gene_id, gene_info in ensembl_genes.items():
            gene_name = gene_info['gene_name']
            gene_biotype = gene_info['gene_biotype']
            chrom = gene_info['chromosome']
            
            # Check if this ID already exists (from HGNC)
            if gene_id in id_to_idx:
                # ID already registered from HGNC, will handle Ensembl names in Phase 2
                # Update metadata with Ensembl-specific info if not from HGNC
                if gene_id in meta_data and 'gene_biotype' not in meta_data[gene_id]:
                    meta_data[gene_id]['gene_biotype'] = gene_biotype
                continue
            
            # Check if this gene_name already has a symbol -> ID mapping
            if gene_name in symbol_to_idx:
                # This gene_name was already processed (duplicate gene_name with different ID)
                # Skip to maintain injective symbol -> ID constraint
                continue
            
            # New gene_name: register symbol and ID
            symbol_to_idx[gene_name] = len(all_symbols)
            all_symbols.append(gene_name)
            symbol_idx = symbol_to_idx[gene_name]
            
            id_to_idx[gene_id] = len(all_ids)
            all_ids.append(gene_id)
            id_idx = id_to_idx[gene_id]
            
            symbol_to_id_edges.append((symbol_idx, id_idx))
            id_to_symbol_idx[id_idx] = symbol_idx
            
            # Collect metadata for Ensembl-only genes
            meta_data[gene_id] = {
                'symbol': gene_name,
                'name': '',
                'gene_biotype': gene_biotype,
                'chromosome': chrom,
                'source': 'ensembl',
            }
            
            # Add symbol as its own primary alias
            alias_data[gene_name] = symbol_idx
        
        # Phase 2: Add Ensembl gene names as additional aliases (for genes from HGNC)
        for gene_id, gene_info in ensembl_genes.items():
            gene_name = gene_info['gene_name']
            
            if gene_id in id_to_idx and gene_id in id_to_symbol_idx:
                # This ID is from HGNC, add Ensembl name as an additional alias
                hgnc_symbol_idx = id_to_symbol_idx[gene_id]
                
                # Update HGNC metadata with Ensembl biotype info
                if gene_id in meta_data:
                    meta_data[gene_id]['gene_biotype'] = gene_info['gene_biotype']
                
                # Add Ensembl name as alias only if not already a symbol
                if gene_name and gene_name not in alias_data:
                    alias_data[gene_name] = hgnc_symbol_idx

    # Ensure every symbol has at least one alias (surjective constraint)
    symbols_with_aliases = set(alias_data.values())
    for symbol_idx, symbol in enumerate(all_symbols):
        if symbol_idx not in symbols_with_aliases:
            if symbol not in alias_data:
                alias_data[symbol] = symbol_idx
    
    # Build vocabularies
    all_aliases = list(alias_data.keys())
    alias_to_symbol_edges = [(i, alias_data[alias]) for i, alias in enumerate(all_aliases)]
    
    logger.info(
        f"Building vocabularies: "
        f"{len(all_aliases)} aliases, "
        f"{len(all_symbols)} symbols, "
        f"{len(all_ids)} IDs"
    )
    
    alias_vocab = GeneVocab(all_aliases, default_token='<unk>', default_index=-1)
    symbol_vocab = GeneVocab(all_symbols, default_token='<unk>', default_index=-1)
    id_vocab = Vocab(all_ids, default_token='<unk>', default_index=-1)
    
    # Build mapping graphs
    alias_to_symbol = BipartiteGraph(
        alias_to_symbol_edges,
        shape=(len(all_aliases), len(all_symbols))
    )
    symbol_to_id = BipartiteGraph(
        symbol_to_id_edges,
        shape=(len(all_symbols), len(all_ids))
    )
    
    # Build metadata DataFrame
    meta = None
    if meta_data:
        meta = pd.DataFrame.from_dict(meta_data, orient='index')
        meta.index.name = 'gene_id'
        logger.info(f"Built metadata table with {len(meta)} entries and columns: {list(meta.columns)}")
    
    # Create GeneMap
    gene_map = GeneMap(
        alias_vocab=alias_vocab,
        symbol_vocab=symbol_vocab,
        id_vocab=id_vocab,
        alias_to_symbol=alias_to_symbol,
        symbol_to_id=symbol_to_id,
        build_fast_mapping=True,
        meta=meta,
    )
    
    logger.info(f"GeneMap built successfully: {gene_map}")
    stats = gene_map.get_coverage_stats()
    logger.info(f"Coverage: {stats}")
    
    return gene_map


# *** FIX: 修改全局缓存结构，使其支持按 source/key 缓存 ***
# Global caches (lazy initialization)
_human_gene_map_cache: Dict[str, GeneMap] = {}
_species_gene_map_cache: Dict[str, GeneMap] = {}


def get_default_human_gene_map(
    source: Literal['hgnc', 'ensembl', 'both'] = 'hgnc',
    rebuild: bool = False,
) -> GeneMap:
    """Get or create the default human gene mapping system.
    
    Singleton pattern with lazy initialization, cached by source.
    """
    global _human_gene_map_cache
    
    if source not in _human_gene_map_cache or rebuild:
        logger.info(f"Building default human GeneMap (source={source})")
        _human_gene_map_cache[source] = build_human_gene_map_from_sources(source=source)
    
    return _human_gene_map_cache[source]


def get_default_gene_map(
    species: str,
    rebuild: bool = False,
) -> GeneMap:
    """Get or create the default gene mapping system for any species.
    
    Singleton pattern with per-species caching.
    Note: Currently only supports 'ensembl' source for non-human species.
    """
    global _species_gene_map_cache
    
    # Cache key: species
    if species not in _species_gene_map_cache or rebuild:
        logger.info(f"Building default GeneMap for {species}")
        _species_gene_map_cache[species] = build_gene_map_from_sources(species=species)
    
    return _species_gene_map_cache[species]
