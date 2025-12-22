#!/usr/bin/env python3
"""
Convert UCE model files to PerturbLab standard format.

This script converts the original UCE model files into the PerturbLab format:
- model.pt: Model state dict
- tokens.pt: Token embeddings
- config.json: Model configuration
- species_chrom.csv, species_offsets.pkl: Auxiliary files
- README.md: Model information
"""

import json
import os
import shutil
from pathlib import Path

import torch

from perturblab.model.uce.config import UCEModelConfig


def convert_uce_model(
    source_dir: str,
    target_dir: str,
    model_name: str,
    nlayers: int,
    model_file: str = "4layer_model.torch"
):
    """
    Convert UCE model to PerturbLab format.
    
    Args:
        source_dir: Directory containing downloaded UCE files
        target_dir: Target directory for converted model
        model_name: Name of the model (e.g., '4layer', '33layer')
        nlayers: Number of transformer layers
        model_file: Name of the model weight file
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Converting UCE {model_name} model...")
    print(f"{'='*70}")
    
    # 1. Load and save model weights
    print(f"1. Processing model weights: {model_file}")
    model_path = source_dir / model_file
    if model_path.exists():
        # Load original checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state_dict (UCE saves the model directly or in a dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's the state_dict directly
            state_dict = checkpoint
        
        # Save as standard model.pt
        torch.save(state_dict, target_dir / 'model.pt')
        print(f"   ✓ Saved model.pt ({len(state_dict)} parameters)")
    else:
        print(f"   ⚠ Model file not found: {model_path}")
    
    # 2. Copy token embeddings
    print("2. Processing token embeddings: all_tokens.torch")
    tokens_path = source_dir / 'all_tokens.torch'
    if tokens_path.exists():
        # Load and save as tokens.pt
        tokens = torch.load(tokens_path, map_location='cpu')
        torch.save(tokens, target_dir / 'tokens.pt')
        print(f"   ✓ Saved tokens.pt (shape: {tokens.shape if hasattr(tokens, 'shape') else 'N/A'})")
    else:
        print(f"   ⚠ Token file not found: {tokens_path}")
    
    # 3. Copy auxiliary files
    print("3. Copying auxiliary files...")
    aux_files = [
        ('species_chrom.csv', 'species_chrom.csv'),
        ('species_offsets.pkl', 'species_offsets.pkl'),
        ('new_species_protein_embeddings.csv', 'new_species_protein_embeddings.csv'),
    ]
    
    for src_name, dst_name in aux_files:
        src_path = source_dir / src_name
        if src_path.exists():
            shutil.copy(src_path, target_dir / dst_name)
            print(f"   ✓ Copied {dst_name}")
        else:
            print(f"   ⚠ File not found: {src_name}")
    
    # 4. Copy protein embeddings directory
    print("4. Copying protein embeddings directory...")
    protein_emb_src = source_dir / 'protein_embeddings'
    protein_emb_dst = target_dir / 'protein_embeddings'
    
    if protein_emb_src.exists():
        if protein_emb_dst.exists():
            shutil.rmtree(protein_emb_dst)
        shutil.copytree(protein_emb_src, protein_emb_dst)
        
        # Count files
        pt_files = list(protein_emb_dst.glob('*.pt'))
        print(f"   ✓ Copied protein_embeddings/ ({len(pt_files)} species)")
    else:
        print(f"   ⚠ Protein embeddings directory not found")
    
    # 5. Create config.json
    print("5. Creating config.json...")
    config = UCEModelConfig(
        model_name=model_name,
        nlayers=nlayers,
        d_model=1280,
        nhead=20,
        d_hid=5120,
        output_dim=1280,
        token_dim=5120,
        dropout=0.05,
        species='human',
        # Set paths relative to model directory
        spec_chrom_csv_path='species_chrom.csv',
        token_file='tokens.pt',
        protein_embeddings_dir='protein_embeddings/',
        offset_pkl_path='species_offsets.pkl',
    )
    
    config_dict = {
        'model_series': config.model_series,
        'model_name': config.model_name,
        'model_type': config.model_type,
        'nlayers': config.nlayers,
        'd_model': config.d_model,
        'nhead': config.nhead,
        'd_hid': config.d_hid,
        'output_dim': config.output_dim,
        'token_dim': config.token_dim,
        'dropout': config.dropout,
        'pad_length': config.pad_length,
        'sample_size': config.sample_size,
        'pad_token_idx': config.pad_token_idx,
        'chrom_token_left_idx': config.chrom_token_left_idx,
        'chrom_token_right_idx': config.chrom_token_right_idx,
        'cls_token_idx': config.cls_token_idx,
        'CHROM_TOKEN_OFFSET': config.CHROM_TOKEN_OFFSET,
        'species': config.species,
        'embedding_model': config.embedding_model,
        'spec_chrom_csv_path': config.spec_chrom_csv_path,
        'token_file': config.token_file,
        'protein_embeddings_dir': config.protein_embeddings_dir,
        'offset_pkl_path': config.offset_pkl_path,
        'batch_first': config.batch_first,
    }
    
    with open(target_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"   ✓ Saved config.json")
    
    # 6. Create README.md
    print("6. Creating README.md...")
    readme_content = f"""# UCE {model_name.upper()} Model

## Model Information

- **Model**: Universal Cell Embeddings (UCE)
- **Variant**: {nlayers}-layer Transformer
- **Source**: https://github.com/snap-stanford/UCE
- **Paper**: [Universal Cell Embeddings: A Foundation Model for Cell Biology](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)

## Architecture

- **Layers**: {nlayers}
- **Model Dimension**: 1280
- **Attention Heads**: 20
- **Hidden Dimension**: 5120
- **Output Dimension**: 1280
- **Token Dimension**: 5120 (ESM2 protein embeddings)

## Usage

```python
from perturblab.model.uce import UCEModel

# Load pretrained model
model = UCEModel.from_pretrained('perturblab/uce-{model_name}')

# Or load from local path
model = UCEModel.from_pretrained('./weights/uce-{model_name}')

# Generate embeddings
result = model.predict_embeddings(
    data=adata,  # or PerturbationData
    species='human',
    batch_size=25
)

cell_embeddings = result['cell_embeddings']  # (n_cells, 1280)
gene_embeddings = result['gene_embeddings']  # (n_cells, seq_len, 1280)
```

## Files

- `model.pt`: Model state dict
- `tokens.pt`: Token embeddings (ESM2-650M + chromosome tokens)
- `config.json`: Model configuration
- `species_chrom.csv`: Gene to chromosome mapping
- `species_offsets.pkl`: Species offsets in token file
- `protein_embeddings/`: Protein embeddings for each species
- `README.md`: This file

## Supported Species

- Human (Homo sapiens)
- Mouse (Mus musculus)
- Frog (Xenopus tropicalis)
- Zebrafish (Danio rerio)
- Mouse Lemur (Microcebus murinus)
- Pig (Sus scrofa)
- Macaca fascicularis
- Macaca mulatta
- And more...

## Citation

```bibtex
@article{{rosen2023universal,
  title={{Universal Cell Embeddings: A Foundation Model for Cell Biology}},
  author={{Rosen, Yanay and Roohani, Yusuf and Agrawal, Ayush and Samotorcan, Leon and Consortium, Tabula Sapiens and Quake, Stephen R and Leskovec, Jure}},
  journal={{bioRxiv}},
  pages={{2023--11}},
  year={{2023}},
  publisher={{Cold Spring Harbor Laboratory}}
}}
```

## License

MIT License (see original repository for details)
"""
    
    with open(target_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    print(f"   ✓ Saved README.md")
    
    print(f"\n{'='*70}")
    print(f"✓ Conversion complete: {target_dir}")
    print(f"{'='*70}\n")


def main():
    """Main conversion script"""
    source_dir = './tmp/uce_download'
    
    # Convert 4-layer model
    convert_uce_model(
        source_dir=source_dir,
        target_dir='./weights/uce-4layer',
        model_name='4layer',
        nlayers=4,
        model_file='4layer_model.torch'
    )
    
    # Note: 33-layer model needs to be downloaded separately
    # Uncomment below if you have downloaded it
    # convert_uce_model(
    #     source_dir=source_dir,
    #     target_dir='./weights/uce-33layer',
    #     model_name='33layer',
    #     nlayers=33,
    #     model_file='33layer_model.torch'
    # )
    
    print("\n" + "="*70)
    print("All conversions complete!")
    print("="*70)
    print("\nConverted models:")
    print("  - weights/uce-4layer/")
    # print("  - weights/uce-33layer/")
    print("\nYou can now use:")
    print("  model = UCEModel.from_pretrained('weights/uce-4layer')")
    print("="*70)


if __name__ == '__main__':
    main()

