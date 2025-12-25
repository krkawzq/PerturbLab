"""Test exact scGPTInput definition."""
from dataclasses import dataclass, fields
from typing import Optional
from torch import Tensor
import torch
from perturblab.core.model_io import ModelIO

print("Testing exact scGPTInput definition...")

# Exact copy from scgpt/io.py
@dataclass
class scGPTInputTest(ModelIO):
    # Data Tensors (required fields first)
    src: Tensor
    values: Tensor
    src_key_padding_mask: Tensor
    
    # Optional Data Tensors
    batch_labels: Optional[Tensor] = None
    mod_types: Optional[Tensor] = None
    pert_flags: Optional[Tensor] = None

    # Task Flags
    CLS: bool = False
    CCE: bool = False
    MVC: bool = False
    ECS: bool = False
    do_sample: bool = False

try:
    obj = scGPTInputTest(
        src=torch.tensor([1]),
        values=torch.tensor([2]),
        src_key_padding_mask=torch.tensor([3])
    )
    print("✓ scGPTInputTest works!")
    
    all_fields = fields(scGPTInputTest)
    print(f"\nTotal fields: {len(all_fields)}")
    for i, f in enumerate(all_fields):
        has_default = f.default != f.default_factory
        print(f"  {i+1}. {f.name}: has_default={has_default}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Now try to import actual scGPTInput...")
try:
    from perturblab.models.scgpt.io import scGPTInput
    print("✓ Successfully imported scGPTInput!")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()

