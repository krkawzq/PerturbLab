#!/usr/bin/env python3
"""
Download and convert UCE model files to PerturbLab format.
"""

import json
import os
import shutil
import tarfile
from pathlib import Path

import requests
import torch
from tqdm import tqdm

from perturblab.model.uce.config import UCEModelConfig


def download_file(url, save_path):
    """Download file with progress bar"""
    if os.path.exists(save_path):
        print(f"✓ File already exists: {save_path}")
        return True
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"\nDownloading {os.path.basename(save_path)}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(save_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
        
        print(f"✓ Downloaded: {save_path}")
        
        # Extract if tar.gz
        if save_path.endswith('.tar.gz'):
            print(f"Extracting {save_path}...")
            with tarfile.open(save_path) as tar:
                tar.extractall(path=os.path.dirname(save_path))
            print(f"✓ Extracted to {os.path.dirname(save_path)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download {save_path}: {e}")
        return False


def download_uce_files(target_dir='./tmp/uce_download'):
    """Download all UCE model files"""
    print("=" * 70)
    print("Downloading UCE model files...")
    print("=" * 70)
    
    downloads = [
        # 1. 4-layer model
        ("https://figshare.com/ndownloader/files/42706576", 
         f"{target_dir}/4layer_model.torch"),
        
        # 2. Token embeddings
        ("https://figshare.com/ndownloader/files/42706966", 
         f"{target_dir}/all_tokens.torch"),
        
        # 3. Species chromosome info
        ("https://figshare.com/ndownloader/files/42706558", 
         f"{target_dir}/species_chrom.csv"),
        
        # 4. Species offsets
        ("https://figshare.com/ndownloader/files/42706555", 
         f"{target_dir}/species_offsets.pkl"),
        
        # 5. Protein embeddings (tar.gz)
        ("https://figshare.com/ndownloader/files/42715213", 
         f"{target_dir}/protein_embeddings.tar.gz"),
        
        # 6. New species config
        ("https://figshare.com/ndownloader/files/42706585", 
         f"{target_dir}/new_species_protein_embeddings.csv"),
    ]
    
    success_count = 0
    for url, path in downloads:
        if download_file(url, path):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Downloaded {success_count}/{len(downloads)} files successfully!")
    print("=" * 70)
    
    return success_count == len(downloads)


def convert_uce_model(
    source_dir: str,
    target_dir: str,
    model_name: str,
    nlayers: int,
    model_file: str = "4layer_model.torch"
):
    """Convert UCE model to PerturbLab format"""
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
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        torch.save(state_dict, target_dir / 'model.pt')
        print(f"   ✓ Saved model.pt ({len(state_dict)} parameters)")
    else:
        print(f"   ✗ Model file not found: {model_path}")
        return False
    
    # 2. Copy token embeddings
    print("2. Processing token embeddings: all_tokens.torch")
    tokens_path = source_dir / 'all_tokens.torch'
    if tokens_path.exists():
        tokens = torch.load(tokens_path, map_location='cpu')
        torch.save(tokens, target_dir / 'tokens.pt')
        shape_info = tokens.shape if hasattr(tokens, 'shape') else 'N/A'
        print(f"   ✓ Saved tokens.pt (shape: {shape_info})")
    else:
        print(f"   ✗ Token file not found: {tokens_path}")
        return False
    
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
    
    # 4. Copy protein embeddings directory
    print("4. Copying protein embeddings directory...")
    protein_emb_src = source_dir / 'protein_embeddings'
    protein_emb_dst = target_dir / 'protein_embeddings'
    
    if protein_emb_src.exists():
        if protein_emb_dst.exists():
            shutil.rmtree(protein_emb_dst)
        shutil.copytree(protein_emb_src, protein_emb_dst)
        pt_files = list(protein_emb_dst.glob('*.pt'))
        print(f"   ✓ Copied protein_embeddings/ ({len(pt_files)} species)")
    else:
        print(f"   ✗ Protein embeddings directory not found")
        return False
    
    # 5. Create config.json
    print("5. Creating config.json...")
    config = UCEModelConfig(
        model_name=model_name,
        nlayers=nlayers,
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
    
    return True


def main():
    """Main script"""
    print("\n" + "="*70)
    print("UCE Model Download and Conversion Script")
    print("="*70 + "\n")
    
    # Step 1: Download files
    download_dir = './tmp/uce_download'
    if not download_uce_files(download_dir):
        print("\n✗ Download failed. Please check your internet connection and try again.")
        return
    
    # Step 2: Convert 4-layer model
    if convert_uce_model(
        source_dir=download_dir,
        target_dir='./weights/uce-4layer',
        model_name='4layer',
        nlayers=4,
        model_file='4layer_model.torch'
    ):
        print("\n" + "="*70)
        print("✓ All conversions complete!")
        print("="*70)
        print("\nConverted models:")
        print("  - weights/uce-4layer/")
        print("\nYou can now use:")
        print("  from perturblab.model.uce import UCEModel")
        print("  model = UCEModel.from_pretrained('weights/uce-4layer')")
        print("="*70 + "\n")
    else:
        print("\n✗ Conversion failed. Please check the error messages above.")


if __name__ == '__main__':
    main()

