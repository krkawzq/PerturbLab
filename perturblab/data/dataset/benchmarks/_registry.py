"""Benchmark dataset registry.

Manages all dataset cards and provides unified query interface.
"""

from typing import Optional, Any
from collections import defaultdict
import anndata as ad

from ..cards import DatasetCard, H5ADDatasetCard, PerturbationType
from perturblab.utils import get_logger

logger = get_logger()


class BenchmarkRegistry:
    """Registry for managing benchmark dataset cards.
    
    Organizes dataset cards by name and source, provides search and lookup.
    """
    
    def __init__(self, cards: list[DatasetCard]):
        """Initialize registry from list of dataset cards."""
        self._cards = cards
        self._by_name = self._build_name_index()
        self._by_source = self._build_source_index()
    
    def _build_name_index(self) -> dict[str, list[DatasetCard]]:
        """Build index by dataset name (supports multiple sources per name)."""
        index = defaultdict(list)
        for card in self._cards:
            index[card.name].append(card)
        return dict(index)
    
    def _build_source_index(self) -> dict[str, list[DatasetCard]]:
        """Build index by source."""
        index = defaultdict(list)
        for card in self._cards:
            index[card.source].append(card)
        return dict(index)
    
    def list_cards(
        self,
        source: Optional[str] = None,
        perturbation_type: Optional[PerturbationType] = None
    ) -> list[DatasetCard]:
        """List dataset cards with optional filtering."""
        cards = self._cards
        
        if source:
            cards = [c for c in cards if c.source == source]
        
        if perturbation_type:
            cards = [c for c in cards if c.perturbation_type == perturbation_type]
        
        return cards
    
    def get_card(
        self,
        dataset_name: str,
        source: Optional[str] = None
    ) -> DatasetCard:
        """Get dataset card by name."""
        if dataset_name not in self._by_name:
            available = ', '.join(sorted(self._by_name.keys()))
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {available}"
            )
        
        variants = self._by_name[dataset_name]
        
        if source is None:
            return variants[0]
        
        for variant in variants:
            if variant.source == source:
                return variant
        
        available_sources = [v.source for v in variants]
        raise ValueError(
            f"Dataset '{dataset_name}' not available from source '{source}'. "
            f"Available sources: {', '.join(available_sources)}"
        )
    
    def list_sources(self, dataset_name: str) -> list[str]:
        """List available sources for a dataset."""
        if dataset_name not in self._by_name:
            return []
        return [c.source for c in self._by_name[dataset_name]]
    
    def list_names(self) -> list[str]:
        """List all unique dataset names."""
        return sorted(self._by_name.keys())


def list_benchmark(
    registry: BenchmarkRegistry,
    source: Optional[str] = None,
    perturbation_type: Optional[PerturbationType] = None,
    names_only: bool = False
) -> list[DatasetCard] | list[str]:
    """List available benchmark datasets."""
    cards = registry.list_cards(source=source, perturbation_type=perturbation_type)
    
    if names_only:
        return sorted(set(c.name for c in cards))
    
    return cards


def get_benchmark_info(
    registry: BenchmarkRegistry,
    dataset_name: str,
    source: Optional[str] = None
) -> DatasetCard:
    """Get dataset card (metadata)."""
    return registry.get_card(dataset_name, source=source)


def list_sources(registry: BenchmarkRegistry, dataset_name: str) -> list[str]:
    """List available sources for a dataset."""
    return registry.list_sources(dataset_name)


def load_benchmark(
    registry: BenchmarkRegistry,
    dataset_name: str,
    source: Optional[str] = None,
    force_download: bool = False
) -> ad.AnnData:
    """Load a benchmark dataset (H5AD only)."""
    card = registry.get_card(dataset_name, source=source)
    
    if not isinstance(card, H5ADDatasetCard):
        raise TypeError(
            f"load_benchmark only supports H5AD datasets. "
            f"Use load_dataset() for other formats."
        )
    
    return card.load(force_download=force_download)


def load_dataset(
    registry: BenchmarkRegistry,
    dataset_name: str,
    source: Optional[str] = None,
    force_download: bool = False,
    auto_fallback: bool = True,
    **kwargs
) -> Any:
    """Load any dataset with auto-fallback to mirrors."""
    if dataset_name not in registry._by_name:
        available = ', '.join(sorted(registry._by_name.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {available}"
        )
    
    available_cards = registry._by_name[dataset_name]
    
    if source is not None:
        card = registry.get_card(dataset_name, source=source)
        logger.info(f"Loading {dataset_name} from {source}")
        return card.load(force_download=force_download, **kwargs)
    
    # Try all sources with auto-fallback
    errors = []
    
    for i, card in enumerate(available_cards):
        try:
            logger.info(f"Attempting {dataset_name} from {card.source} ({i+1}/{len(available_cards)})")
            result = card.load(force_download=force_download, **kwargs)
            logger.info(f"Successfully loaded {dataset_name} from {card.source}")
            return result
        
        except Exception as e:
            error_msg = f"{card.source}: {str(e)}"
            errors.append(error_msg)
            logger.warning(f"Failed from {card.source}: {e}")
            
            if not auto_fallback:
                raise
            
            if i < len(available_cards) - 1:
                logger.info(f"Trying next source...")
            else:
                error_summary = "\n  - ".join(errors)
                raise ValueError(
                    f"Failed to load {dataset_name} from all {len(available_cards)} source(s):\n  - {error_summary}"
                )

