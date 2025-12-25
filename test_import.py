#!/usr/bin/env python
"""Test scGPT import after fixing dataclass field order."""
print("Testing scGPT import...")

try:
    # Test 1: Import scGPT IO classes
    print("\n1. Importing scGPT IO classes...")
    from perturblab.models.scgpt import scGPTInput, scGPTOutput
    print("   ✓ Successfully imported scGPTInput and scGPTOutput")
    
    # Test 2: Import scGPT module
    print("\n2. Importing scGPT module...")
    import perturblab.models.scgpt as scgpt
    print(f"   ✓ Successfully imported scgpt module")
    print(f"   - SCGPT_REGISTRY: {scgpt.SCGPT_REGISTRY}")
    print(f"   - SCGPT_COMPONENTS: {scgpt.SCGPT_COMPONENTS}")
    
    # Test 3: Check if components are registered
    print("\n3. Checking component registration...")
    try:
        component_keys = scgpt.SCGPT_COMPONENTS.list_keys()
        print(f"   ✓ Registered components: {component_keys}")
    except Exception as e:
        print(f"   ✗ Failed to list components: {e}")
    
    # Test 4: Try to build GeneEncoder using Model()
    print("\n4. Testing Model() function...")
    from perturblab.models import Model
    try:
        encoder = Model("scGPT/components/GeneEncoder")(vocab_size=5000, dim=512)
        print(f"   ✓ Successfully created GeneEncoder: {type(encoder).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create GeneEncoder: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ All tests completed!")
    
except Exception as e:
    print(f"\n❌ Error during import: {e}")
    import traceback
    traceback.print_exc()

