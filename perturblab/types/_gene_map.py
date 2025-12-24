"""Gene mapping system for alias/symbol/ID resolution."""

import os
from typing import Literal, Optional

import pandas as pd

from perturblab.utils import get_logger

from ._gene_vocab import GeneVocab
from ._vocab import Vocab
from .math import BipartiteGraph

logger = get_logger()


class GeneMap:
    """Gene mapping system managing alias, symbol, and ID relationships.

    Maintains three vocabularies and their mapping relationships:
    - alias_vocab: All gene names (superset, most comprehensive)
    - symbol_vocab: Verified gene symbols (canonical names)
    - id_vocab: Gene IDs (unique identifiers, typically Ensembl IDs)

    Mapping constraints:
    - alias -> symbol: Must be surjective (every symbol has at least one alias)
    - symbol -> id: Must be injective (each symbol maps to unique ID)

    Important Notes:
        When using Ensembl data as the source for IDs:
        - Some gene names (e.g., SNORA70, Y_RNA) may correspond to multiple Ensembl IDs
          representing gene copies on different chromosomes or genomic locations
        - To maintain the injective constraint (symbol -> id), construction functions
          only keep the first encountered ID for each gene name
        - This means some Ensembl IDs may be excluded from the mapping
        - For human genes, using HGNC as the primary source (via build_human_gene_map_from_sources)
          is recommended as it provides curated, official nomenclature

    Args:
        alias_vocab: Vocabulary of all gene aliases.
        symbol_vocab: Vocabulary of verified gene symbols.
        id_vocab: Vocabulary of gene IDs (can be Vocab or GeneVocab).
        alias_to_symbol: Bipartite graph mapping aliases to symbols.
        symbol_to_id: Bipartite graph mapping symbols to IDs.
        build_fast_mapping: If True, builds cached mappings on first access
            for faster subsequent queries.
        meta: Optional DataFrame containing metadata for each gene ID.
            Index should be gene IDs matching id_vocab tokens.
    """

    def __init__(
        self,
        alias_vocab: GeneVocab,
        symbol_vocab: GeneVocab,
        id_vocab: Vocab,
        alias_to_symbol: BipartiteGraph,
        symbol_to_id: BipartiteGraph,
        build_fast_mapping: bool = True,
        meta: Optional[pd.DataFrame] = None,
    ):
        self.alias_vocab = alias_vocab
        self.symbol_vocab = symbol_vocab
        self.id_vocab = id_vocab
        self.alias_to_symbol = alias_to_symbol
        self.symbol_to_id = symbol_to_id
        self.build_fast_mapping = build_fast_mapping
        self.meta = meta

        # Lazy-initialized cached mappings
        self._alias_to_id_graph: BipartiteGraph | None = None
        self._symbol_to_alias_inverse: dict[int, list[int]] | None = None
        self._id_to_symbol_inverse: dict[int, list[int]] | None = None
        self._id_to_alias_inverse: dict[int, list[int]] | None = None

        self._validate_mappings()

    def _validate_mappings(self) -> None:
        """Validate mapping constraints."""
        # Check dimensions
        if self.alias_to_symbol.n_source != len(self.alias_vocab):
            raise ValueError(
                f"alias_to_symbol source dimension ({self.alias_to_symbol.n_source}) "
                f"must match alias_vocab size ({len(self.alias_vocab)})"
            )

        if self.alias_to_symbol.n_target != len(self.symbol_vocab):
            raise ValueError(
                f"alias_to_symbol target dimension ({self.alias_to_symbol.n_target}) "
                f"must match symbol_vocab size ({len(self.symbol_vocab)})"
            )

        if self.symbol_to_id.n_source != len(self.symbol_vocab):
            raise ValueError(
                f"symbol_to_id source dimension ({self.symbol_to_id.n_source}) "
                f"must match symbol_vocab size ({len(self.symbol_vocab)})"
            )

        if self.symbol_to_id.n_target != len(self.id_vocab):
            raise ValueError(
                f"symbol_to_id target dimension ({self.symbol_to_id.n_target}) "
                f"must match id_vocab size ({len(self.id_vocab)})"
            )

        # Validate alias_to_symbol is a function and surjective
        if not self.alias_to_symbol.is_function():
            raise ValueError(
                "alias_to_symbol must be a function (each alias maps to exactly one symbol)"
            )

        if not self.alias_to_symbol.is_surjective():
            raise ValueError(
                "alias_to_symbol must be surjective (every symbol must have at least one alias)"
            )

        # Validate symbol_to_id is injective
        if not self.symbol_to_id.is_function():
            raise ValueError("symbol_to_id must be a function (each symbol maps to exactly one ID)")

        if not self.symbol_to_id.is_injective():
            raise ValueError("symbol_to_id must be injective (each symbol maps to unique ID)")

    def _build_alias_to_id_graph(self) -> BipartiteGraph:
        """Build composed alias -> ID mapping graph."""
        if self._alias_to_id_graph is not None:
            return self._alias_to_id_graph

        # Compose alias -> symbol -> id
        edges = []
        alias_to_symbol_mapping = self.alias_to_symbol.get_function_mapping()
        symbol_to_id_mapping = self.symbol_to_id.get_function_mapping()

        for alias_idx, symbol_idx in alias_to_symbol_mapping.items():
            if symbol_idx in symbol_to_id_mapping:
                id_idx = symbol_to_id_mapping[symbol_idx]
                edges.append((alias_idx, id_idx))

        shape = (len(self.alias_vocab), len(self.id_vocab))
        self._alias_to_id_graph = BipartiteGraph(edges, shape=shape)
        return self._alias_to_id_graph

    def _get_symbol_to_alias_inverse(self) -> dict[int, list[int]]:
        """Get cached inverse mapping from symbol to aliases."""
        if self._symbol_to_alias_inverse is None:
            self._symbol_to_alias_inverse = self.alias_to_symbol.get_inverse_mapping()
        return self._symbol_to_alias_inverse

    def _get_id_to_symbol_inverse(self) -> dict[int, list[int]]:
        """Get cached inverse mapping from ID to symbol."""
        if self._id_to_symbol_inverse is None:
            self._id_to_symbol_inverse = self.symbol_to_id.get_inverse_mapping()
        return self._id_to_symbol_inverse

    def _get_id_to_alias_inverse(self) -> dict[int, list[int]]:
        """Get cached inverse mapping from ID to aliases."""
        if self._id_to_alias_inverse is not None:
            return self._id_to_alias_inverse

        # Build composed inverse mapping
        alias_to_id_graph = self._build_alias_to_id_graph()
        self._id_to_alias_inverse = alias_to_id_graph.get_inverse_mapping()
        return self._id_to_alias_inverse

    def standardize(self, alias: str) -> str | None:
        """Map any alias to its canonical symbol.

        Args:
            alias: Gene alias name.

        Returns:
            str | None: Canonical symbol name, or None if alias not found.
        """
        alias_idx = self.alias_vocab[alias]
        if alias_idx == self.alias_vocab.default_index:
            return None

        symbol_indices = self.alias_to_symbol.get_neighbors(alias_idx, check_bounds=False)
        if len(symbol_indices) == 0:
            return None

        return self.symbol_vocab[int(symbol_indices[0])]

    def id(self, alias: str) -> str | None:
        """Map any alias to its gene ID.

        Args:
            alias: Gene alias name.

        Returns:
            str | None: Gene ID, or None if alias has no corresponding ID.
        """
        alias_idx = self.alias_vocab[alias]
        if alias_idx == self.alias_vocab.default_index:
            return None

        if self.build_fast_mapping:
            # Use cached composed graph
            alias_to_id_graph = self._build_alias_to_id_graph()
            id_indices = alias_to_id_graph.get_neighbors(alias_idx, check_bounds=False)
        else:
            # Compose on-the-fly: alias -> symbol -> id
            symbol_indices = self.alias_to_symbol.get_neighbors(alias_idx, check_bounds=False)
            if len(symbol_indices) == 0:
                return None

            symbol_idx = int(symbol_indices[0])
            id_indices = self.symbol_to_id.get_neighbors(symbol_idx, check_bounds=False)

        if len(id_indices) == 0:
            return None

        return self.id_vocab[int(id_indices[0])]

    def symbol(self, gene_id: str) -> str | None:
        """Get symbol for a gene ID.

        Args:
            gene_id: Gene ID.

        Returns:
            str | None: Symbol name, or None if no symbol maps to this ID.
        """
        id_idx = self.id_vocab[gene_id]
        if id_idx == self.id_vocab.default_index:
            return None

        if self.build_fast_mapping:
            # Use cached inverse mapping
            inverse = self._get_id_to_symbol_inverse()
            symbol_indices = inverse.get(id_idx, [])
        else:
            # Build inverse on-the-fly
            inverse = self.symbol_to_id.get_inverse_mapping()
            symbol_indices = inverse.get(id_idx, [])

        if len(symbol_indices) == 0:
            return None

        # Should be exactly one due to injectivity
        return self.symbol_vocab[int(symbol_indices[0])]

    def symbol_aliases(self, symbol: str) -> list[str]:
        """Get all aliases for a symbol.

        Args:
            symbol: Gene symbol name.

        Returns:
            list[str]: List of aliases mapping to this symbol.
        """
        symbol_idx = self.symbol_vocab[symbol]
        if symbol_idx == self.symbol_vocab.default_index:
            return []

        if self.build_fast_mapping:
            # Use cached inverse mapping
            inverse = self._get_symbol_to_alias_inverse()
            alias_indices = inverse.get(symbol_idx, [])
        else:
            # Build inverse on-the-fly
            inverse = self.alias_to_symbol.get_inverse_mapping()
            alias_indices = inverse.get(symbol_idx, [])

        return [self.alias_vocab[idx] for idx in alias_indices]

    def id_aliases(self, gene_id: str) -> list[str]:
        """Get all aliases for a gene ID.

        Args:
            gene_id: Gene ID.

        Returns:
            list[str]: List of aliases mapping to this ID.
        """
        id_idx = self.id_vocab[gene_id]
        if id_idx == self.id_vocab.default_index:
            return []

        if self.build_fast_mapping:
            # Use cached inverse mapping
            inverse = self._get_id_to_alias_inverse()
            alias_indices = inverse.get(id_idx, [])
        else:
            # Compose on-the-fly: id -> symbol -> aliases
            id_to_symbol_inverse = self.symbol_to_id.get_inverse_mapping()
            symbol_indices = id_to_symbol_inverse.get(id_idx, [])

            if len(symbol_indices) == 0:
                return []

            symbol_to_alias_inverse = self.alias_to_symbol.get_inverse_mapping()
            alias_indices = []
            for symbol_idx in symbol_indices:
                alias_indices.extend(symbol_to_alias_inverse.get(symbol_idx, []))

        return [self.alias_vocab[idx] for idx in alias_indices]

    def batch_standardize(self, aliases: list[str]) -> list[str | None]:
        """Batch standardize multiple aliases to symbols.

        Args:
            aliases: List of gene aliases.

        Returns:
            list[str | None]: List of symbols.
        """
        return [self.standardize(alias) for alias in aliases]

    def batch_id(self, aliases: list[str]) -> list[str | None]:
        """Batch resolve multiple aliases to IDs.

        Args:
            aliases: List of gene aliases.

        Returns:
            list[str | None]: List of gene IDs.
        """
        return [self.id(alias) for alias in aliases]

    def query(
        self,
        query: str | list[str],
        query_type: Literal["alias", "id", "symbol"] = "alias",
    ) -> dict | list[dict]:
        """Query comprehensive gene information with vectorized support.

        This method provides a unified interface to query gene information
        by different identifiers (alias, ID, or symbol) and returns a complete
        dictionary with all related information. Uses pandas vectorized operations
        for efficient batch queries.

        Args:
            query: Gene identifier(s) to query. Can be:
                - str: Single identifier
                - list[str]: Multiple identifiers (vectorized query)
            query_type: Type of the query identifier:
                - 'alias': Query by any gene alias (default)
                - 'id': Query by gene ID
                - 'symbol': Query by gene symbol

        Returns:
            dict or list[dict]: Gene information dictionary/dictionaries containing:
                - 'query': Original query string
                - 'symbol': Canonical gene symbol (None if not found)
                - 'id': Gene ID (None if not found)
                - 'aliases': List of all aliases (empty if not found)
                - 'meta': Metadata dictionary (None if not available)

        Examples:
            >>> # Single query by alias
            >>> result = gene_map.query('p53', query_type='alias')
            >>> print(result['symbol'])  # 'TP53'
            >>>
            >>> # Vectorized query by ID
            >>> results = gene_map.query(['ENSG00000141510', 'ENSG00000012048'], query_type='id')
            >>> print([r['symbol'] for r in results])  # ['TP53', 'BRCA1']
            >>>
            >>> # Query with metadata
            >>> result = gene_map.query('TP53', query_type='symbol')
            >>> print(result['meta'])  # {'name': '...', 'chromosome': '17', ...}
        """
        # Handle vectorized input with pandas
        if isinstance(query, list):
            return self._query_vectorized(query, query_type)

        # Single query
        result = {
            "query": query,
            "symbol": None,
            "id": None,
            "aliases": [],
            "meta": None,
        }

        if query_type == "alias":
            # Query by alias
            symbol = self.standardize(query)
            if symbol:
                result["symbol"] = symbol
                result["id"] = self.id(query)
                result["aliases"] = self.symbol_aliases(symbol)

                # Get metadata if available
                gene_id = result["id"]
                if gene_id and self.meta is not None and gene_id in self.meta.index:
                    result["meta"] = self.meta.loc[gene_id].to_dict()

        elif query_type == "id":
            # Query by gene ID
            symbol = self.symbol(query)
            if symbol:
                result["symbol"] = symbol
                result["id"] = query
                result["aliases"] = self.symbol_aliases(symbol)

                # Get metadata if available
                if self.meta is not None and query in self.meta.index:
                    result["meta"] = self.meta.loc[query].to_dict()

        elif query_type == "symbol":
            # Query by symbol
            symbol_idx = self.symbol_vocab[query]
            if symbol_idx != self.symbol_vocab.default_index:
                result["symbol"] = query

                # Get ID
                id_indices = self.symbol_to_id.get_neighbors(symbol_idx, check_bounds=False)
                if len(id_indices) > 0:
                    gene_id = self.id_vocab[int(id_indices[0])]
                    result["id"] = gene_id

                    # Get metadata if available
                    if self.meta is not None and gene_id in self.meta.index:
                        result["meta"] = self.meta.loc[gene_id].to_dict()

                # Get aliases
                result["aliases"] = self.symbol_aliases(query)

        else:
            raise ValueError(
                f"Invalid query_type: {query_type}. Must be 'alias', 'id', or 'symbol'."
            )

        return result

    def _query_vectorized(
        self,
        queries: list[str],
        query_type: Literal["alias", "id", "symbol"],
    ) -> list[dict]:
        """Vectorized query implementation using pandas for efficiency.

        Args:
            queries: List of gene identifiers to query.
            query_type: Type of the query identifiers.

        Returns:
            list[dict]: List of gene information dictionaries.
        """
        # Create pandas Series for vectorized operations
        query_series = pd.Series(queries, index=range(len(queries)))

        # Initialize result DataFrame
        result_df = pd.DataFrame(
            {
                "query": queries,
                "symbol": None,
                "id": None,
            }
        )

        if query_type == "alias":
            # Vectorized alias -> symbol mapping
            symbols = self.batch_standardize(queries)
            result_df["symbol"] = symbols

            # Vectorized symbol -> id mapping
            ids = self.batch_id(queries)
            result_df["id"] = ids

        elif query_type == "id":
            # Vectorized id -> symbol mapping
            symbols = [self.symbol(q) for q in queries]
            result_df["symbol"] = symbols
            result_df["id"] = queries

        elif query_type == "symbol":
            # Queries are already symbols
            result_df["symbol"] = queries

            # Vectorized symbol -> id mapping
            ids = []
            for q in queries:
                symbol_idx = self.symbol_vocab[q]
                if symbol_idx != self.symbol_vocab.default_index:
                    id_indices = self.symbol_to_id.get_neighbors(symbol_idx, check_bounds=False)
                    if len(id_indices) > 0:
                        ids.append(self.id_vocab[int(id_indices[0])])
                    else:
                        ids.append(None)
                else:
                    ids.append(None)
            result_df["id"] = ids

        else:
            raise ValueError(
                f"Invalid query_type: {query_type}. Must be 'alias', 'id', or 'symbol'."
            )

        # Get aliases for each symbol (vectorized where possible)
        aliases_list = []
        for symbol in result_df["symbol"]:
            if symbol:
                aliases_list.append(self.symbol_aliases(symbol))
            else:
                aliases_list.append([])
        result_df["aliases"] = aliases_list

        # Get metadata (vectorized using pandas indexing)
        meta_list = []
        if self.meta is not None:
            for gene_id in result_df["id"]:
                if gene_id and gene_id in self.meta.index:
                    meta_list.append(self.meta.loc[gene_id].to_dict())
                else:
                    meta_list.append(None)
        else:
            meta_list = [None] * len(queries)
        result_df["meta"] = meta_list

        # Convert DataFrame to list of dictionaries
        return result_df.to_dict("records")

    def get_coverage_stats(self) -> dict:
        """Get statistics about mapping coverage.

        Returns:
            dict: Dictionary containing coverage statistics.
        """
        symbols_with_id = self.symbol_to_id.n_edges
        symbols_without_id = len(self.symbol_vocab) - symbols_with_id

        # Count unique IDs that have symbols
        id_indices_with_symbol = set()
        for _, id_idx in self.symbol_to_id.to_edge_list():
            id_indices_with_symbol.add(id_idx)

        ids_without_symbol = len(self.id_vocab) - len(id_indices_with_symbol)

        return {
            "n_aliases": len(self.alias_vocab),
            "n_symbols": len(self.symbol_vocab),
            "n_ids": len(self.id_vocab),
            "symbols_with_id": symbols_with_id,
            "symbols_without_id": symbols_without_id,
            "ids_with_symbol": len(id_indices_with_symbol),
            "ids_without_symbol": ids_without_symbol,
            "avg_aliases_per_symbol": (
                len(self.alias_vocab) / len(self.symbol_vocab) if len(self.symbol_vocab) > 0 else 0
            ),
        }

    def save(self, directory: str) -> None:
        """Save GeneMap to directory.

        Saves all components (vocabularies, graphs, and metadata) to separate files
        in the specified directory.

        Args:
            directory: Directory path to save the GeneMap.
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save vocabularies
        self.alias_vocab.save(os.path.join(directory, "alias_vocab.json"))
        self.symbol_vocab.save(os.path.join(directory, "symbol_vocab.json"))
        self.id_vocab.save(os.path.join(directory, "id_vocab.json"))

        # Save graphs
        self.alias_to_symbol.save(os.path.join(directory, "alias_to_symbol.npz"))
        self.symbol_to_id.save(os.path.join(directory, "symbol_to_id.npz"))

        # Save metadata if present
        if self.meta is not None:
            self.meta.to_csv(os.path.join(directory, "meta.csv"), index=True)

    @classmethod
    def load(cls, directory: str, build_fast_mapping: bool = True) -> "GeneMap":
        """Load GeneMap from directory.

        Args:
            directory: Directory path containing saved GeneMap components.
            build_fast_mapping: Whether to enable fast mapping cache.

        Returns:
            GeneMap: Loaded GeneMap instance.
        """
        # Load vocabularies
        alias_vocab = GeneVocab.load(os.path.join(directory, "alias_vocab.json"))
        symbol_vocab = GeneVocab.load(os.path.join(directory, "symbol_vocab.json"))
        id_vocab = Vocab.load(os.path.join(directory, "id_vocab.json"))

        # Load graphs
        alias_to_symbol = BipartiteGraph.load(os.path.join(directory, "alias_to_symbol.npz"))
        symbol_to_id = BipartiteGraph.load(os.path.join(directory, "symbol_to_id.npz"))

        # Load metadata if present
        meta_path = os.path.join(directory, "meta.csv")
        meta = None
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path, index_col=0)

        return cls(
            alias_vocab=alias_vocab,
            symbol_vocab=symbol_vocab,
            id_vocab=id_vocab,
            alias_to_symbol=alias_to_symbol,
            symbol_to_id=symbol_to_id,
            build_fast_mapping=build_fast_mapping,
            meta=meta,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GeneMap(aliases={len(self.alias_vocab)}, "
            f"symbols={len(self.symbol_vocab)}, "
            f"ids={len(self.id_vocab)})"
        )
