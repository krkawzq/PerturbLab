#!/usr/bin/env python3
"""
Test CellFM gene mapping functionality.
"""

import numpy as np
from anndata import AnnData

def test_gene_mapping():
    """Test gene mapping with sample data."""
    print("="*80)
    print("Testing CellFM Gene Mapping")
    print("="*80)
    
    from perturblab.model.cellfm import CellFMGeneMapper
    
    # Initialize mapper
    print("\n[1/4] Initializing gene mapper...")
    mapper = CellFMGeneMapper()
    print(f"✓ Loaded {mapper.n_genes:,} genes in vocabulary")
    
    # Test gene list mapping
    print("\n[2/4] Testing gene list mapping...")
    test_genes = [
        'A1BG',      # Should map directly
        'A2M',       # Should map directly
        'INVALID',   # Should fail
        'TP53',      # Should map directly
    ]
    
    mapped, failed = mapper.map_gene_list(test_genes, verbose=True)
    print(f"  Input genes: {test_genes}")
    print(f"  Mapped: {mapped}")
    print(f"  Failed: {failed}")
    
    # Test with AnnData
    print("\n[3/4] Testing AnnData preparation...")
    
    # Create sample AnnData
    n_cells = 100
    n_genes = 50
    
    # Use some real gene names
    gene_names = mapper.available_genes[:n_genes]
    
    # Create random expression data
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    
    adata = AnnData(X=X)
    adata.var_names = gene_names
    
    print(f"  Created sample AnnData: {adata.shape}")
    
    # Prepare with mapping
    adata_prepared = mapper.prepare_adata_with_mapping(
        adata,
        max_genes=30,
        min_cells=1,
        inplace=False,
    )
    
    print(f"  After preparation: {adata_prepared.shape}")
    print(f"  Sample genes: {list(adata_prepared.var_names[:5])}")
    
    # Test gene info lookup
    print("\n[4/4] Testing gene info lookup...")
    gene = 'A1BG'
    info = mapper.get_gene_info(gene)
    if info is not None:
        print(f"  Gene: {gene}")
        print(f"  Biotype: {info.get('biotype', 'N/A')}")
        print(f"  Feature: {info.get('feature', 'N/A')}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    
    return True

if __name__ == '__main__':
    try:
        test_gene_mapping()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

