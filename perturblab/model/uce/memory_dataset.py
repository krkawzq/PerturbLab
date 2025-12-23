"""
Memory-optimized Dataset for UCE model.

This module provides an in-memory alternative to the disk-based MultiDatasetSentences
to avoid IO bottlenecks during inference.
"""

import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse import issparse

warnings.filterwarnings("ignore")


class InMemoryMultiDatasetSentences(data.Dataset):
    """In-memory version of MultiDatasetSentences that avoids disk IO.
    
    This class stores count matrices in memory instead of using np.memmap,
    providing significant speedup for repeated inference at the cost of higher
    memory usage.
    
    Suitable for:
        - Small to medium datasets (< 1M cells)
        - Repeated inference on the same dataset
        - When disk IO is a bottleneck
        
    Not suitable for:
        - Very large datasets (> 1M cells) with limited RAM
        - One-time inference where memory is constrained
    """
    
    def __init__(
        self,
        sorted_dataset_names: List[str],
        shapes_dict: Dict[str, Tuple[int, int]],
        args: Any,
        dataset_to_protein_embeddings_path: Dict[str, np.ndarray],
        datasets_to_chroms_path: Dict[str, np.ndarray],
        datasets_to_starts_path: Dict[str, np.ndarray],
        counts_dict: Dict[str, np.ndarray],
    ) -> None:
        """Initialize in-memory dataset.
        
        Args:
            sorted_dataset_names: List of dataset names.
            shapes_dict: Dict mapping dataset names to (n_cells, n_genes) tuples.
            args: Arguments object with UCE configuration.
            dataset_to_protein_embeddings_path: Dict mapping dataset names to protein embedding indices.
            datasets_to_chroms_path: Dict mapping dataset names to chromosome arrays.
            datasets_to_starts_path: Dict mapping dataset names to genomic start position arrays.
            counts_dict: Dict mapping dataset names to count matrices (in memory).
        """
        super(InMemoryMultiDatasetSentences, self).__init__()
        
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.args = args
        self.counts_dict = counts_dict  # Store counts in memory
        
        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes
            self.total_num_cells += num_cells
        
        self.datasets = sorted_dataset_names
        
        # Store metadata dictionaries
        self.dataset_to_protein_embeddings = dataset_to_protein_embeddings_path
        self.dataset_to_chroms = datasets_to_chroms_path
        self.dataset_to_starts = datasets_to_starts_path
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """Get a single cell's tokenized sequence.
        
        Args:
            idx: Cell index across all datasets.
            
        Returns:
            Tuple of (batch_sentences, mask, idx, seq_len, cell_sentences)
        """
        if not isinstance(idx, int):
            raise NotImplementedError("Only integer indexing is supported")
        
        original_idx = idx
        
        # Find which dataset this index belongs to
        for dataset in sorted(self.datasets):
            if idx < self.num_cells[dataset]:
                # Get counts from memory instead of disk
                counts = self.counts_dict[dataset][idx]
                
                # Convert to tensor
                if issparse(counts):
                    counts = counts.toarray().flatten()
                counts = torch.tensor(counts, dtype=torch.float32).unsqueeze(0)
                
                # Compute weights
                weights = torch.log1p(counts)
                weights = weights / torch.sum(weights)
                
                # Sample cell sentences (same logic as original)
                batch_sentences, mask, seq_len, cell_sentences = sample_cell_sentences(
                    counts,
                    weights,
                    dataset,
                    self.args,
                    dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
                    dataset_to_chroms=self.dataset_to_chroms,
                    dataset_to_starts=self.dataset_to_starts,
                )
                
                return batch_sentences, mask, original_idx, seq_len, cell_sentences
            else:
                idx -= self.num_cells[dataset]
        
        raise IndexError(f"Index {original_idx} out of range for dataset with {self.total_num_cells} cells")
    
    def __len__(self) -> int:
        return self.total_num_cells
    
    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


def sample_cell_sentences(
    counts: torch.Tensor,
    batch_weights: torch.Tensor,
    dataset: str,
    args: Any,
    dataset_to_protein_embeddings: Dict[str, np.ndarray],
    dataset_to_chroms: Dict[str, np.ndarray],
    dataset_to_starts: Dict[str, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Sample and construct cell sentences with chromosome ordering.
    
    This function replicates the logic from UCE source (eval_data.py:115-183).
    
    Args:
        counts: Count vector for the cell.
        batch_weights: Sampling weights (log-normalized counts).
        dataset: Dataset name.
        args: Arguments object with configuration.
        dataset_to_protein_embeddings: Mapping to protein embedding indices.
        dataset_to_chroms: Mapping to chromosome arrays.
        dataset_to_starts: Mapping to genomic start positions.
        
    Returns:
        Tuple of (batch_sentences, mask, longest_seq_len, cell_sentences)
    """
    dataset_idxs = dataset_to_protein_embeddings[dataset]
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))
    mask = torch.zeros((counts.shape[0], args.pad_length))
    chroms = dataset_to_chroms[dataset]
    starts = dataset_to_starts[dataset]
    
    longest_seq_len = 0
    
    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # Re-normalize
        
        # Sample genes weighted by expression
        choice_idx = np.random.choice(
            np.arange(len(weights)),
            size=args.sample_size,
            p=weights,
            replace=True,
        )
        
        choosen_chrom = chroms[choice_idx]
        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]
        
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]
        
        ordered_choice_idx = np.full((args.pad_length), args.cls_token_idx)
        i = 1  # Start after CLS token
        
        # Shuffle chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms)
        
        # Build sequence with chromosome tokens
        for chrom in uq_chroms:
            # Open chromosome token
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET
            i += 1
            
            # Sort genes by start position within chromosome
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(choosen_starts[loc])
            
            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            
            # Close chromosome token
            ordered_choice_idx[i] = args.chrom_token_right_idx
            i += 1
        
        longest_seq_len = max(longest_seq_len, i)
        remainder_len = args.pad_length - i
        
        # Create mask
        cell_mask = torch.concat((
            torch.ones(i),
            torch.zeros(remainder_len)
        ))
        mask[c, :] = cell_mask
        
        # Pad remaining positions
        ordered_choice_idx[i:] = args.pad_token_idx
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
    
    cell_sentences_pe = cell_sentences.long()
    
    return cell_sentences_pe, mask, longest_seq_len, cell_sentences

