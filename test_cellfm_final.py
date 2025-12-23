#!/usr/bin/env python3
"""
Test CellFM model loading for both 80M and 800M models.
"""

import sys

def test_cellfm_models():
    """Test loading both CellFM models."""
    print("="*80)
    print("Testing CellFM Model Loading")
    print("="*80)
    
    from perturblab.model.cellfm import CellFMModel
    
    test_cases = [
        ("cellfm-80m", "80M", 27855, 1536, 2, 48),
        ("cellfm-800m", "800M", 24071, 1536, 40, 48),
    ]
    
    results = []
    
    for model_name, expected_name, expected_genes, expected_dims, expected_layers, expected_heads in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")
        
        try:
            model = CellFMModel.from_pretrained(
                model_name,
                device='cpu',
                load_weights=True,
            )
            

            
            # Count parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            
            print(f"✓ Success!")
            print(f"  Model name: {model.config.model_name}")
            print(f"  Genes: {model.n_genes:,}")
            print(f"  Hidden dim: {model.config.enc_dims}")
            print(f"  Layers: {model.config.enc_nlayers}")
            print(f"  Attention heads: {model.config.enc_num_heads}")
            print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
            print(f"  Weights loaded: {model.model_loaded}")
            
            results.append((model_name, True, None))
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((model_name, False, str(e)))
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    print(f"Passed: {success_count}/{total_count}")
    
    for model_name, success, error in results:
        status = "✓" if success else "✗"
        print(f"  {status} {model_name}")
        if error:
            print(f"    Error: {error[:100]}...")
    
    return success_count == total_count

if __name__ == '__main__':
    try:
        success = test_cellfm_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

