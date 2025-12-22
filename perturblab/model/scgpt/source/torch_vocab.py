# modified from torchtext.vocab (https://github.com/pytorch/text)
"""
Pure Python implementation of Vocab, replacing torchtext.vocab.Vocab.
This implementation is compatible with scGPT and does not require torchtext.
"""

from typing import Dict, List, Optional


class VocabPybind:
    """
    Pure Python Vocab implementation that replaces torchtext's C++ VocabPybind.
    Provides bidirectional mapping between tokens and indices.
    """
    
    def __init__(self, tokens: List[str], default_index: Optional[int] = None):
        """
        Args:
            tokens: List of tokens in order
            default_index: Default index for OOV tokens (optional)
        """
        self.itos_: List[str] = tokens  # index to string
        self.stoi_: Dict[str, int] = {token: idx for idx, token in enumerate(tokens)}  # string to index
        self.default_index_: Optional[int] = default_index
    
    def __len__(self) -> int:
        return len(self.itos_)
    
    def __contains__(self, token: str) -> bool:
        return token in self.stoi_
    
    def __getitem__(self, token: str) -> int:
        """Get index for a token."""
        if token in self.stoi_:
            return self.stoi_[token]
        if self.default_index_ is not None:
            return self.default_index_
        raise KeyError(f"Token '{token}' not found in vocabulary and no default index set")
    
    def lookup_token(self, index: int) -> str:
        """Get token for an index."""
        if 0 <= index < len(self.itos_):
            return self.itos_[index]
        raise IndexError(f"Index {index} out of range [0, {len(self.itos_)})")
    
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        """Get tokens for a list of indices."""
        return [self.lookup_token(idx) for idx in indices]
    
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """Get indices for a list of tokens."""
        return [self[token] for token in tokens]
    
    def set_default_index(self, index: Optional[int]) -> None:
        """Set default index for OOV tokens."""
        self.default_index_ = index
    
    def get_default_index(self) -> Optional[int]:
        """Get default index."""
        return self.default_index_
    
    def insert_token(self, token: str, index: int) -> None:
        """Insert a token at a specific index."""
        if token in self.stoi_:
            raise RuntimeError(f"Token '{token}' already exists in vocabulary")
        if index < 0 or index > len(self.itos_):
            raise RuntimeError(f"Index {index} out of range [0, {len(self.itos_)}]")
        
        # Insert token at index
        self.itos_.insert(index, token)
        # Rebuild stoi
        self.stoi_ = {t: i for i, t in enumerate(self.itos_)}
    
    def append_token(self, token: str) -> None:
        """Append a token to the end."""
        if token in self.stoi_:
            raise RuntimeError(f"Token '{token}' already exists in vocabulary")
        self.itos_.append(token)
        self.stoi_[token] = len(self.itos_) - 1
    
    def get_stoi(self) -> Dict[str, int]:
        """Get string-to-index mapping."""
        return self.stoi_.copy()
    
    def get_itos(self) -> List[str]:
        """Get index-to-string mapping."""
        return self.itos_.copy()


class Vocab:
    """
    Vocab wrapper class compatible with torchtext.vocab.Vocab API.
    This is a pure Python implementation that doesn't require torchtext.
    """
    
    def __init__(self, vocab: VocabPybind) -> None:
        """
        Args:
            vocab: VocabPybind object
        """
        self.vocab = vocab
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __contains__(self, token: str) -> bool:
        return token in self.vocab
    
    def __getitem__(self, token: str) -> int:
        return self.vocab[token]
    
    def forward(self, tokens: List[str]) -> List[int]:
        """Calls the `lookup_indices` method (for compatibility)."""
        return self.vocab.lookup_indices(tokens)
    
    def set_default_index(self, index: Optional[int]) -> None:
        self.vocab.set_default_index(index)
    
    def get_default_index(self) -> Optional[int]:
        return self.vocab.get_default_index()
    
    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)
    
    def append_token(self, token: str) -> None:
        self.vocab.append_token(token)
    
    def lookup_token(self, index: int) -> str:
        return self.vocab.lookup_token(index)
    
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return self.vocab.lookup_tokens(indices)
    
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)
    
    def get_stoi(self) -> Dict[str, int]:
        return self.vocab.get_stoi()
    
    def get_itos(self) -> List[str]:
        return self.vocab.get_itos()


def vocab(
    ordered_dict: Dict, 
    min_freq: int = 1, 
    specials: Optional[List[str]] = None, 
    special_first: bool = True
) -> Vocab:
    """
    Factory method for creating a vocab object which maps tokens to indices.
    
    This is a pure Python implementation that replaces torchtext.vocab.vocab.
    The ordering in ordered_dict is preserved when building the vocab.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their corresponding occurrence frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.
        special_first: Indicates whether to insert symbols at the beginning or at the end.

    Returns:
        Vocab: A `Vocab` object

    Examples:
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> print(v1['a'])  # prints 1
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> unk_token = '<unk>'
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])
        >>> v2.set_default_index(0)
        >>> print(v2['<unk>'])  # prints 0
        >>> print(v2['out of vocab'])  # prints 0 (default index)
    """
    specials = specials or []
    
    # Remove special tokens from ordered_dict if they exist
    for token in specials:
        ordered_dict.pop(token, None)

    # Filter tokens by minimum frequency
    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    # Add special tokens at the beginning or end
    if special_first:
        tokens[0:0] = specials  # Insert at beginning
    else:
        tokens.extend(specials)  # Append at end

    return Vocab(VocabPybind(tokens, None))
