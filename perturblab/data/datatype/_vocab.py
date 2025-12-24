"""Basic vocabulary class for token-index mapping."""

from perturblab.kernels.mapping import lookup_indices, lookup_tokens
from typing import Iterable
import numpy as np
import json


class Vocab:
    """Vocabulary for bidirectional token-index mapping.
    
    Maintains three core data structures:
    - itos: List mapping index to token (ordered)
    - stoi: Dict mapping token to index (fast lookup)
    - tokens: Set of all tokens (fast membership check)
    
    Args:
        tokens: List of unique string tokens.
        default_token: Token to return when index is out of bounds.
        default_index: Index to return when token is not found.
    """
    
    def __init__(
        self,
        tokens: list[str],
        default_token: str = '<unk>',
        default_index: int = -1
    ):
        self._check_tokens(tokens)
        self.itos = tokens  # Index to string (ordered)
        self.stoi = {token: i for i, token in enumerate(tokens)}  # String to index
        self.tokens = set(self.itos)  # Token set for fast membership check
        self.default_token = default_token
        self.default_index = default_index
    
    @staticmethod
    def _check_tokens(tokens: list[str]) -> None:
        """Validate token list."""
        if not isinstance(tokens, list):
            raise ValueError("tokens must be a list of strings")
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("tokens must be a list of strings")
        if len(tokens) != len(set(tokens)):
            raise ValueError("tokens must be unique")
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)
    
    def lookup_tokens(
        self,
        indices: list[int] | np.ndarray,
        fallback: str | None = None
    ) -> list[str]:
        """Map indices to tokens.
        
        Args:
            indices: Integer indices to query.
            fallback: Token to return for out-of-bounds indices.
                If None, uses self.default_token.
        
        Returns:
            list[str]: Tokens corresponding to the indices.
        """
        if fallback is None:
            fallback = self.default_token
        return lookup_tokens(indices, self.itos, fallback)

    def lookup_indices(
        self,
        tokens: Iterable[str],
        fallback: int | None = None
    ) -> list[int]:
        """Map tokens to indices.
        
        Args:
            tokens: String tokens to query.
            fallback: Index to return for unknown tokens.
                If None, uses self.default_index.
        
        Returns:
            list[int]: List of indices.
        """
        if fallback is None:
            fallback = self.default_index
        return list(lookup_indices(self.stoi, tokens, fallback))
    
    def __getitem__(
        self,
        query: str | int | list[str] | list[int] | np.ndarray
    ) -> int | str | list[int] | list[str]:
        """Support vectorized token-index lookup.
        
        Args:
            query: String token, integer index, or array of tokens/indices.
        
        Returns:
            - int: If query is str
            - str: If query is int
            - list[int]: If query is list/array of strings (token->index)
            - list[str]: If query is list/array of integers (index->token)
        """
        # Scalar string: token -> index
        if isinstance(query, str):
            return self.stoi.get(query, self.default_index)
        
        # Scalar integer: index -> token
        elif isinstance(query, (int, np.integer)):
            if 0 <= query < len(self.itos):
                return self.itos[query]
            else:
                return self.default_token
        
        # Array: vectorized lookup
        elif isinstance(query, (list, np.ndarray)):
            if len(query) == 0:
                # Empty array: return appropriate empty result
                return []
            
            # Check first element to determine type
            first_elem = query[0]
            
            if isinstance(first_elem, str):
                # Array of strings: tokens -> indices
                return self.lookup_indices(query)
            elif isinstance(first_elem, (int, np.integer)):
                # Array of integers: indices -> tokens
                return self.lookup_tokens(query)
            else:
                raise TypeError(
                    f"Array elements must be str or int, got {type(first_elem)}"
                )
        
        else:
            raise TypeError(
                f"Query must be str, int, or array, got {type(query)}"
            )
    
    def __contains__(self, token: str) -> bool:
        """Check if token exists in vocabulary."""
        return token in self.tokens
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Vocab(size={len(self)}, default_token='{self.default_token}')"
    
    def __iter__(self):
        """Iterate over tokens in order."""
        return iter(self.itos)
    
    def get_token_set(self) -> set[str]:
        """Get set of all tokens.
        
        Returns:
            set[str]: Set of all tokens in vocabulary.
        """
        return self.tokens.copy()
    
    def to_dict(self) -> dict[str, int]:
        """Get copy of token-to-index mapping.
        
        Returns:
            dict[str, int]: Dictionary mapping tokens to indices.
        """
        return self.stoi.copy()
    
    def to_list(self) -> list[str]:
        """Get copy of ordered token list.
        
        Returns:
            list[str]: List of tokens in order.
        """
        return self.itos.copy()

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file.
        
        Args:
            path: Path to save vocabulary (JSON format).
        """
        data = {
            'itos': self.itos,
            'default_token': self.default_token,
            'default_index': self.default_index,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Vocab':
        """Load vocabulary from JSON file.
        
        Args:
            path: Path to load vocabulary from.
        
        Returns:
            Vocab: Loaded vocabulary instance.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            data['itos'],
            default_token=data.get('default_token', '<unk>'),
            default_index=data.get('default_index', -1),
        )
