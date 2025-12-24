"""Basic vocabulary class for token-index mapping."""

import json
from typing import Iterable

import numpy as np

# Avoid circular import: delay import of lookup functions to method level
# from perturblab.kernels.mapping import lookup_indices, lookup_tokens


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

    def __init__(self, tokens: list[str], default_token: str = "<unk>", default_index: int = -1):
        self._check_tokens(tokens)
        self.itos = tokens  # Index to string (ordered)
        self.stoi = {token: i for i, token in enumerate(tokens)}  # String to index
        self.tokens = set(self.itos)  # Token set for fast membership check
        self.default_token = default_token
        self.default_index = default_index

    @staticmethod
    def _check_tokens(tokens: list[str]) -> None:
        """Validate token list with comprehensive checks.
        
        Args:
            tokens (list[str]): Token list to validate.
            
        Raises:
            ValueError: If tokens is invalid (not list, not strings, has duplicates, empty strings).
        """
        if not isinstance(tokens, list):
            raise ValueError(f"tokens must be a list, got {type(tokens)}")
        
        if len(tokens) == 0:
            raise ValueError("tokens list cannot be empty")
        
        for i, token in enumerate(tokens):
            if not isinstance(token, str):
                raise ValueError(
                    f"All tokens must be strings. "
                    f"Token at index {i} is {type(token)}: {token}"
                )
            if len(token) == 0:
                raise ValueError(f"Empty string found at index {i}")
        
        # Check for duplicates
        if len(tokens) != len(set(tokens)):
            from collections import Counter
            counts = Counter(tokens)
            duplicates = [tok for tok, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate tokens found: {duplicates[:5]}... "
                f"(Total {len(duplicates)} duplicates)"
            )

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)

    def lookup_tokens(
        self, indices: list[int] | np.ndarray, fallback: str | None = None
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
        from perturblab.kernels.mapping import lookup_tokens

        return lookup_tokens(indices, self.itos, fallback)

    def lookup_indices(
        self, 
        tokens: Iterable[str], 
        fallback: int | None = None,
        warn_unknown: bool = False
    ) -> list[int]:
        """Map tokens to indices with optional warnings for unknown tokens.

        Args:
            tokens: String tokens to query.
            fallback: Index to return for unknown tokens.
                If None, uses self.default_index.
            warn_unknown (bool, optional): If True, log warning for unknown tokens.
                Defaults to False.

        Returns:
            list[int]: List of indices.
            
        Example:
            >>> vocab = Vocab(['A', 'B', 'C'])
            >>> vocab.lookup_indices(['A', 'D', 'B'], warn_unknown=True)
            # Warns: "1 unknown tokens: ['D']"
            # Returns: [0, -1, 1]
        """
        if fallback is None:
            fallback = self.default_index
        from perturblab.kernels.mapping import lookup_indices

        indices = list(lookup_indices(self.stoi, tokens, fallback))
        
        # Optional warning for unknown tokens
        if warn_unknown:
            token_list = list(tokens) if not isinstance(tokens, list) else tokens
            unknown = [tok for tok, idx in zip(token_list, indices) if idx == fallback]
            if unknown:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"{len(unknown)} unknown tokens (returning {fallback}): "
                    f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
                )
        
        return indices

    def __getitem__(
        self, query: str | int | list[str] | list[int] | np.ndarray
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
                raise TypeError(f"Array elements must be str or int, got {type(first_elem)}")

        else:
            raise TypeError(f"Query must be str, int, or array, got {type(query)}")

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

    def save(self, path: str, include_metadata: bool = True) -> None:
        """Save vocabulary to JSON file with optional metadata.

        Args:
            path (str): Path to save vocabulary (JSON format).
            include_metadata (bool, optional): If True, includes size and version info.
                Defaults to True.
        """
        import os
        
        data = {
            "itos": self.itos,
            "default_token": self.default_token,
            "default_index": self.default_index,
        }
        
        if include_metadata:
            data["metadata"] = {
                "size": len(self.itos),
                "version": "1.0",
                "class": self.__class__.__name__,
            }
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, verify_metadata: bool = True) -> "Vocab":
        """Load vocabulary from JSON file with validation.

        Args:
            path (str): Path to load vocabulary from.
            verify_metadata (bool, optional): If True, checks metadata if present.
                Defaults to True.

        Returns:
            Vocab: Loaded vocabulary instance.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If JSON format is invalid.
            RuntimeError: If metadata indicates incompatible version/class.
        """
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        
        # Validate required fields
        if "itos" not in data:
            raise ValueError("JSON file missing required field: 'itos'")
        
        if not isinstance(data["itos"], list):
            raise ValueError(f"'itos' must be a list, got {type(data['itos'])}")
        
        # Verify metadata if present and requested
        if verify_metadata and "metadata" in data:
            meta = data["metadata"]
            
            # Check class compatibility
            if "class" in meta and meta["class"] != cls.__name__:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Loading {meta['class']} vocabulary into {cls.__name__}. "
                    f"This may cause compatibility issues."
                )
            
            # Check size consistency
            if "size" in meta and meta["size"] != len(data["itos"]):
                raise RuntimeError(
                    f"Metadata size ({meta['size']}) doesn't match "
                    f"actual itos length ({len(data['itos'])}). File may be corrupted."
                )

        return cls(
            data["itos"],
            default_token=data.get("default_token", "<unk>"),
            default_index=data.get("default_index", -1),
        )
