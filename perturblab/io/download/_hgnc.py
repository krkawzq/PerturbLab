"""HGNC (HUGO Gene Nomenclature Committee) downloader.

Downloads official gene symbol data from HGNC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from perturblab.utils import get_logger

from ._base import BaseDownloader, DownloadError, _prepare_path

logger = get_logger()

__all__ = ["HGNCDownloader"]

# HGNC REST API endpoints
HGNC_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json"
HGNC_COMPLETE = f"{HGNC_BASE_URL}/hgnc_complete_set.json"
HGNC_WITHDRAWN = f"{HGNC_BASE_URL}/withdrawn.json"


class HGNCDownloader(BaseDownloader):
    """HGNC gene nomenclature downloader namespace.

    Downloads official gene symbols and mappings from HGNC.
    Cannot be instantiated - use static methods directly.

    Examples
    --------
    >>> from perturblab.io.download import HGNCDownloader
    >>>
    >>> # Download complete gene set to file
    >>> path = HGNCDownloader.download(
    ...     target_path='/tmp/hgnc_genes.tsv',
    ...     format='tsv'
    ... )
    >>>
    >>> # Download specific columns
    >>> path = HGNCDownloader.download(
    ...     target_path='/tmp/hgnc_subset.csv',
    ...     columns=['hgnc_id', 'symbol', 'name'],
    ...     format='csv'
    ... )
    """

    @staticmethod
    def download(
        target_path: Path | str,
        *,
        columns: list[str] | None = None,
        format: Literal["json", "csv", "tsv", "parquet"] = "tsv",
        dataset: Literal["complete", "withdrawn"] = "complete",
    ) -> Path:
        """Download HGNC gene set to file.

        This is the core download method for HGNC data.

        Parameters
        ----------
        target_path : Path or str
            Target file path (required).
        columns : list of str, optional
            Specific columns to include. If None, includes all columns.
            Common columns: "hgnc_id", "symbol", "name", "locus_group",
            "locus_type", "alias_symbol", "prev_symbol", "entrez_id",
            "ensembl_gene_id", "gene_id"
        format : {"json", "csv", "tsv", "parquet"}, default="tsv"
            Output format.
        dataset : {"complete", "withdrawn"}, default="complete"
            Which dataset to download:
            - "complete": Complete gene set (current symbols)
            - "withdrawn": Withdrawn/obsolete symbols

        Returns
        -------
        Path
            Path to downloaded file.

        Raises
        ------
        DownloadError
            If download fails.

        Examples
        --------
        >>> # Download complete set as TSV
        >>> path = HGNCDownloader.download(
        ...     '/tmp/hgnc_genes.tsv',
        ...     format='tsv'
        ... )

        >>> # Download specific columns as CSV
        >>> path = HGNCDownloader.download(
        ...     '/tmp/hgnc_subset.csv',
        ...     columns=['symbol', 'name', 'entrez_id'],
        ...     format='csv'
        ... )
        """
        target_path = _prepare_path(target_path)

        # Determine URL
        url = HGNC_COMPLETE if dataset == "complete" else HGNC_WITHDRAWN

        logger.info(f"Downloading HGNC {dataset} dataset")

        try:
            # Download JSON data
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Extract gene data
            genes = data["response"]["docs"]
            df = pd.DataFrame(genes)

            # Filter columns if specified
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                if not available_cols:
                    raise DownloadError(
                        f"None of the specified columns found. " f"Available: {list(df.columns)}"
                    )
                df = df[available_cols]

            # Save to file
            HGNCDownloader._save_dataframe(df, target_path, format)

            logger.info(f"Downloaded {len(df)} genes to: {target_path}")
            return target_path

        except requests.RequestException as e:
            raise DownloadError(f"Failed to download HGNC data: {e}")
        except Exception as e:
            raise DownloadError(f"Failed to process HGNC data: {e}")

    @staticmethod
    def download_to_dataframe(
        *,
        columns: list[str] | None = None,
        dataset: Literal["complete", "withdrawn"] = "complete",
    ) -> pd.DataFrame:
        """Download HGNC gene set directly to DataFrame (in-memory).

        Parameters
        ----------
        columns : list of str, optional
            Specific columns to include.
        dataset : {"complete", "withdrawn"}, default="complete"
            Which dataset to download.

        Returns
        -------
        pd.DataFrame
            Gene data as DataFrame.

        Raises
        ------
        DownloadError
            If download fails.

        Examples
        --------
        >>> df = HGNCDownloader.download_to_dataframe()
        >>> print(df.shape)

        >>> df = HGNCDownloader.download_to_dataframe(
        ...     columns=['hgnc_id', 'symbol', 'name']
        ... )
        """
        url = HGNC_COMPLETE if dataset == "complete" else HGNC_WITHDRAWN

        logger.info(f"Downloading HGNC {dataset} dataset to DataFrame")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            genes = data["response"]["docs"]
            df = pd.DataFrame(genes)

            if columns:
                available_cols = [c for c in columns if c in df.columns]
                if not available_cols:
                    raise DownloadError(
                        f"None of the specified columns found. " f"Available: {list(df.columns)}"
                    )
                df = df[available_cols]

            logger.info(f"Downloaded {len(df)} genes")
            return df

        except requests.RequestException as e:
            raise DownloadError(f"Failed to download HGNC data: {e}")
        except Exception as e:
            raise DownloadError(f"Failed to process HGNC data: {e}")

    @staticmethod
    def _save_dataframe(df: pd.DataFrame, path: Path, format: str) -> None:
        """Save DataFrame to file in specified format."""
        if format == "json":
            df.to_json(path, orient="records", indent=2)
        elif format == "csv":
            df.to_csv(path, index=False)
        elif format == "tsv":
            df.to_csv(path, sep="\t", index=False)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise DownloadError(f"Unsupported format: {format}")
