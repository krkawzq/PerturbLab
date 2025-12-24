"""Ensembl genome annotation downloader.

Downloads gene annotations, GTF files, and FASTA sequences from Ensembl.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Literal

import pandas as pd

from perturblab.utils import get_logger

from ._base import BaseDownloader, DownloadError, _prepare_path
from ._http import HTTPDownloader

logger = get_logger()

__all__ = ["EnsemblDownloader"]

# Ensembl FTP base URLs
ENSEMBL_FTP = "https://ftp.ensembl.org/pub"
ENSEMBL_GENOMES_FTP = "https://ftp.ensemblgenomes.org/pub"


class EnsemblDownloader(BaseDownloader):
    """Ensembl genome annotation downloader namespace.

    Downloads GTF annotations and FASTA sequences from Ensembl.
    Cannot be instantiated - use static methods directly.

    Examples
    --------
    >>> from perturblab.io.download import EnsemblDownloader
    >>>
    >>> # Download from URL (recommended - exact file specification)
    >>> gtf_path = EnsemblDownloader.download(
    ...     'https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/Homo_sapiens.GRCh38.111.gtf.gz',
    ...     '/tmp/human.gtf'
    ... )
    >>>
    >>> # Parse GTF file
    >>> df = EnsemblDownloader.parse_gtf('/tmp/human.gtf')
    """

    @staticmethod
    def download(
        url: str,
        target_path: Path | str,
        *,
        decompress: bool = True,
    ) -> Path:
        """Download file from Ensembl by URL.

        This is the core download method. Due to the complexity of Ensembl's
        file naming conventions, it's recommended to provide the exact URL.

        You can browse Ensembl FTP to find files:
        - Main: https://ftp.ensembl.org/pub/
        - Genomes: https://ftp.ensemblgenomes.org/pub/

        Parameters
        ----------
        url : str
            Full URL to Ensembl file (required).
        target_path : Path or str
            Target file path (required).
        decompress : bool, default=True
            Whether to automatically decompress .gz files.

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
        >>> # Download GTF
        >>> url = 'https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/Homo_sapiens.GRCh38.111.gtf.gz'
        >>> gtf_path = EnsemblDownloader.download(
        ...     url,
        ...     '/tmp/human.gtf'
        ... )

        >>> # Download FASTA (keep compressed)
        >>> url = 'https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz'
        >>> fasta_path = EnsemblDownloader.download(
        ...     url,
        ...     '/tmp/human_cdna.fa.gz',
        ...     decompress=False
        ... )
        """
        target_path = _prepare_path(target_path)

        logger.info(f"Downloading from Ensembl: {url}")

        try:
            if decompress and url.endswith(".gz"):
                # Download to temp location first
                temp_gz = target_path.parent / f"{target_path.name}.temp.gz"
                HTTPDownloader.download(url, temp_gz, show_progress=True)

                # Decompress
                logger.info("Decompressing file...")
                with gzip.open(temp_gz, "rb") as f_in:
                    with open(target_path, "wb") as f_out:
                        f_out.write(f_in.read())

                temp_gz.unlink()
                logger.info(f"Decompressed to {target_path}")
                return target_path
            else:
                # Download directly
                return HTTPDownloader.download(url, target_path, show_progress=True)

        except Exception as e:
            raise DownloadError(f"Failed to download from Ensembl: {e}")

    @staticmethod
    def download_gtf(
        species: str,
        assembly: str,
        release: int,
        target_path: Path | str,
        *,
        decompress: bool = True,
        base_url: str = ENSEMBL_FTP,
    ) -> Path:
        """Download GTF annotation file for a species.

        Parameters
        ----------
        species : str
            Species name in title case with underscores (required).
            Example: "Homo_sapiens", "Mus_musculus"
        assembly : str
            Genome assembly name (required).
            Example: "GRCh38", "GRCm39"
        release : int
            Ensembl release number (required).
            Example: 111
        target_path : Path or str
            Target file path (required).
        decompress : bool, default=True
            Whether to decompress the .gz file.
        base_url : str, default=ENSEMBL_FTP
            Base URL for Ensembl FTP.
            Use ENSEMBL_GENOMES_FTP for non-vertebrate genomes.

        Returns
        -------
        Path
            Path to downloaded GTF file.

        Raises
        ------
        DownloadError
            If download fails.

        Examples
        --------
        >>> # Download human GTF
        >>> gtf_path = EnsemblDownloader.download_gtf(
        ...     'Homo_sapiens',
        ...     'GRCh38',
        ...     111,
        ...     '/tmp/human.gtf'
        ... )

        >>> # Download mouse GTF
        >>> gtf_path = EnsemblDownloader.download_gtf(
        ...     'Mus_musculus',
        ...     'GRCm39',
        ...     111,
        ...     '/tmp/mouse.gtf'
        ... )
        """
        # Construct URL
        species_lower = species.lower()
        url = (
            f"{base_url}/release-{release}/gtf/{species_lower}/"
            f"{species}.{assembly}.{release}.gtf.gz"
        )

        return EnsemblDownloader.download(url, target_path, decompress=decompress)

    @staticmethod
    def download_fasta(
        species: str,
        assembly: str,
        release: int,
        sequence_type: Literal["dna", "cdna", "cds", "ncrna", "pep"],
        target_path: Path | str,
        *,
        decompress: bool = True,
        base_url: str = ENSEMBL_FTP,
        all_sequences: bool = True,
    ) -> Path:
        """Download FASTA sequence file.

        Parameters
        ----------
        species : str
            Species name in title case with underscores (required).
        assembly : str
            Genome assembly name (required).
        release : int
            Ensembl release number (required).
        sequence_type : {"dna", "cdna", "cds", "ncrna", "pep"}
            Type of sequences (required):
            - "dna": Genomic DNA
            - "cdna": cDNA (transcripts)
            - "cds": Coding sequences
            - "ncrna": Non-coding RNA
            - "pep": Protein sequences
        target_path : Path or str
            Target file path (required).
        decompress : bool, default=True
            Whether to decompress the .gz file.
        base_url : str, default=ENSEMBL_FTP
            Base URL for Ensembl FTP.
        all_sequences : bool, default=True
            Whether to download ".all" version (all sequences).

        Returns
        -------
        Path
            Path to downloaded FASTA file.

        Raises
        ------
        DownloadError
            If download fails.

        Examples
        --------
        >>> # Download human cDNA
        >>> cdna_path = EnsemblDownloader.download_fasta(
        ...     'Homo_sapiens',
        ...     'GRCh38',
        ...     111,
        ...     'cdna',
        ...     '/tmp/human_cdna.fa'
        ... )

        >>> # Download mouse proteins
        >>> pep_path = EnsemblDownloader.download_fasta(
        ...     'Mus_musculus',
        ...     'GRCm39',
        ...     111,
        ...     'pep',
        ...     '/tmp/mouse_pep.fa'
        ... )
        """
        # Construct URL
        species_lower = species.lower()
        all_suffix = ".all" if all_sequences else ""

        url = (
            f"{base_url}/release-{release}/fasta/{species_lower}/{sequence_type}/"
            f"{species}.{assembly}.{sequence_type}{all_suffix}.fa.gz"
        )

        return EnsemblDownloader.download(url, target_path, decompress=decompress)

    @staticmethod
    def parse_gtf(
        gtf_path: Path | str,
        *,
        feature_type: str | None = None,
        attributes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Parse GTF file to DataFrame.

        Parameters
        ----------
        gtf_path : Path or str
            Path to GTF file (required).
        feature_type : str, optional
            Filter by feature type (e.g., "gene", "transcript", "exon").
            If None, includes all features.
        attributes : list of str, optional
            Specific attributes to extract from the 9th column.
            If None, extracts all attributes.

        Returns
        -------
        pd.DataFrame
            Parsed GTF data with columns:
            - seqname, source, feature, start, end, score, strand, frame
            - Plus extracted attributes (e.g., gene_id, gene_name, etc.)

        Examples
        --------
        >>> # Parse entire GTF
        >>> df = EnsemblDownloader.parse_gtf('/tmp/human.gtf')

        >>> # Parse only genes
        >>> genes = EnsemblDownloader.parse_gtf(
        ...     '/tmp/human.gtf',
        ...     feature_type='gene'
        ... )

        >>> # Extract specific attributes
        >>> df = EnsemblDownloader.parse_gtf(
        ...     '/tmp/human.gtf',
        ...     attributes=['gene_id', 'gene_name', 'gene_biotype']
        ... )
        """
        gtf_path = Path(gtf_path)
        logger.info(f"Parsing GTF file: {gtf_path}")

        records = []

        with open(gtf_path, "r") as f:
            for line in f:
                # Skip comments
                if line.startswith("#"):
                    continue

                fields = line.strip().split("\t")
                if len(fields) != 9:
                    continue

                seqname, source, feature, start, end, score, strand, frame, attrs = fields

                # Filter by feature type
                if feature_type and feature != feature_type:
                    continue

                # Parse attributes
                attr_dict = {}
                for attr in attrs.split(";"):
                    attr = attr.strip()
                    if not attr:
                        continue

                    match = re.match(r'(\w+)\s+"([^"]+)"', attr)
                    if match:
                        key, value = match.groups()
                        attr_dict[key] = value

                # Filter attributes if specified
                if attributes:
                    attr_dict = {k: v for k, v in attr_dict.items() if k in attributes}

                # Combine into record
                record = {
                    "seqname": seqname,
                    "source": source,
                    "feature": feature,
                    "start": int(start),
                    "end": int(end),
                    "score": score,
                    "strand": strand,
                    "frame": frame,
                    **attr_dict,
                }

                records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Parsed {len(df)} {feature_type or 'features'}")

        return df
