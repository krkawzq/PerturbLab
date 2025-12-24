"""Gene Ontology (GO) downloader.

Downloads GO ontologies and gene annotations.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Literal

import pandas as pd

from perturblab.utils import get_logger

from ._base import BaseDownloader, DownloadError, _prepare_path
from ._http import HTTPDownloader

logger = get_logger()

__all__ = ["GODownloader"]

# GO download URLs
GO_OBO_URL = "http://purl.obolibrary.org/obo/go.obo"
GO_BASIC_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
GO_ANNOTATION_BASE = "http://geneontology.org/gene-associations"


class GODownloader(BaseDownloader):
    """Gene Ontology downloader namespace.

    Downloads GO ontology files (.obo) and gene-GO annotations (.gaf).
    Cannot be instantiated - use static methods directly.

    Examples
    --------
    >>> from perturblab.io.download import GODownloader
    >>>
    >>> # Download GO ontology
    >>> obo_path = GODownloader.download(
    ...     '/tmp/go.obo',
    ...     version='basic'
    ... )
    >>>
    >>> # Download GO annotations
    >>> gaf_path = GODownloader.download_annotations(
    ...     'goa_human',
    ...     '/tmp/goa_human.gaf'
    ... )
    """

    @staticmethod
    def download(
        target_path: Path | str,
        *,
        version: Literal["full", "basic"] = "basic",
    ) -> Path:
        """Download GO ontology OBO file.

        This is the core download method for GO ontologies.

        Parameters
        ----------
        target_path : Path or str
            Target file path (required).
        version : {"full", "basic"}, default="basic"
            Ontology version:
            - "basic": Filtered version (recommended for most analyses)
            - "full": Complete ontology with all relationships

        Returns
        -------
        Path
            Path to downloaded OBO file.

        Raises
        ------
        DownloadError
            If download fails.

        Examples
        --------
        >>> # Download basic ontology (recommended)
        >>> obo_path = GODownloader.download(
        ...     '/tmp/go-basic.obo',
        ...     version='basic'
        ... )

        >>> # Download full ontology
        >>> obo_path = GODownloader.download(
        ...     '/tmp/go.obo',
        ...     version='full'
        ... )
        """
        target_path = _prepare_path(target_path)

        url = GO_BASIC_OBO_URL if version == "basic" else GO_OBO_URL

        logger.info(f"Downloading GO ontology ({version})")

        try:
            return HTTPDownloader.download(url, target_path, show_progress=True)
        except Exception as e:
            raise DownloadError(f"Failed to download GO ontology: {e}")

    @staticmethod
    def download_annotations(
        organism: str,
        target_path: Path | str,
        *,
        decompress: bool = True,
    ) -> Path:
        """Download GO gene annotations (GAF format).

        Parameters
        ----------
        organism : str
            Organism identifier (required). Common options:
            - "goa_human": Human (Homo sapiens)
            - "goa_mouse": Mouse (Mus musculus)
            - "goa_rat": Rat (Rattus norvegicus)
            - "fb": Fruit fly (Drosophila melanogaster)
            - "mgi": Mouse (from MGI)
            - "rgd": Rat (from RGD)
            - "sgd": Yeast (Saccharomyces cerevisiae)
            - "tair": Arabidopsis (Arabidopsis thaliana)
            - "wb": C. elegans (Caenorhabditis elegans)
            - "zfin": Zebrafish (Danio rerio)
        target_path : Path or str
            Target file path (required).
        decompress : bool, default=True
            Whether to decompress .gz files automatically.

        Returns
        -------
        Path
            Path to downloaded GAF file.

        Raises
        ------
        DownloadError
            If download fails.

        Examples
        --------
        >>> # Download human annotations
        >>> gaf_path = GODownloader.download_annotations(
        ...     'goa_human',
        ...     '/tmp/goa_human.gaf'
        ... )

        >>> # Download mouse annotations (keep compressed)
        >>> gaf_path = GODownloader.download_annotations(
        ...     'goa_mouse',
        ...     '/tmp/goa_mouse.gaf.gz',
        ...     decompress=False
        ... )
        """
        target_path = _prepare_path(target_path)

        # Construct URL
        url = f"{GO_ANNOTATION_BASE}/{organism}.gaf.gz"

        logger.info(f"Downloading GO annotations for {organism}")

        try:
            # Download to temp location first if we need to decompress
            if decompress:
                temp_gz = target_path.parent / f"{target_path.name}.temp.gz"
                HTTPDownloader.download(url, temp_gz, show_progress=True)

                # Decompress
                logger.info("Decompressing GAF file...")
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
            raise DownloadError(f"Failed to download GO annotations for {organism}: {e}")

    @staticmethod
    def parse_gaf(
        gaf_path: Path | str,
        *,
        evidence_codes: list[str] | None = None,
        aspect: Literal["P", "F", "C"] | None = None,
    ) -> pd.DataFrame:
        """Parse GAF (Gene Association File) to DataFrame.

        Parameters
        ----------
        gaf_path : Path or str
            Path to GAF file (required).
        evidence_codes : list of str, optional
            Filter by evidence codes (e.g., ["EXP", "IDA", "IPI"]).
            If None, includes all evidence codes.
            Common codes:
            - EXP: Inferred from Experiment
            - IDA: Inferred from Direct Assay
            - IPI: Inferred from Physical Interaction
            - IMP: Inferred from Mutant Phenotype
            - IGI: Inferred from Genetic Interaction
            - IEP: Inferred from Expression Pattern
            - ISS: Inferred from Sequence Similarity
            - TAS: Traceable Author Statement
            - IEA: Inferred from Electronic Annotation (less reliable)
        aspect : {"P", "F", "C"}, optional
            Filter by GO aspect:
            - "P": Biological Process
            - "F": Molecular Function
            - "C": Cellular Component

        Returns
        -------
        pd.DataFrame
            Parsed GAF data with columns:
            - DB, DB_Object_ID, DB_Object_Symbol, Qualifier, GO_ID,
              DB_Reference, Evidence_Code, With_From, Aspect, DB_Object_Name,
              DB_Object_Synonym, DB_Object_Type, Taxon, Date, Assigned_By,
              Annotation_Extension, Gene_Product_Form_ID

        Examples
        --------
        >>> # Parse all annotations
        >>> df = GODownloader.parse_gaf('/tmp/goa_human.gaf')

        >>> # Parse only experimental annotations
        >>> df = GODownloader.parse_gaf(
        ...     '/tmp/goa_human.gaf',
        ...     evidence_codes=['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP']
        ... )

        >>> # Parse only biological process annotations
        >>> df = GODownloader.parse_gaf(
        ...     '/tmp/goa_human.gaf',
        ...     aspect='P'
        ... )
        """
        gaf_path = Path(gaf_path)
        logger.info(f"Parsing GAF file: {gaf_path}")

        # GAF 2.2 column names
        columns = [
            "DB",
            "DB_Object_ID",
            "DB_Object_Symbol",
            "Qualifier",
            "GO_ID",
            "DB_Reference",
            "Evidence_Code",
            "With_From",
            "Aspect",
            "DB_Object_Name",
            "DB_Object_Synonym",
            "DB_Object_Type",
            "Taxon",
            "Date",
            "Assigned_By",
            "Annotation_Extension",
            "Gene_Product_Form_ID",
        ]

        records = []

        with open(gaf_path, "r") as f:
            for line in f:
                # Skip comments and header
                if line.startswith("!"):
                    continue

                fields = line.strip().split("\t")
                if len(fields) < 15:  # GAF 2.2 has at least 15 columns
                    continue

                # Pad to 17 columns if needed (for optional columns)
                while len(fields) < 17:
                    fields.append("")

                record = dict(zip(columns, fields[:17]))

                # Filter by evidence code
                if evidence_codes and record["Evidence_Code"] not in evidence_codes:
                    continue

                # Filter by aspect
                if aspect and record["Aspect"] != aspect:
                    continue

                records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Parsed {len(df)} annotations")

        return df

    @staticmethod
    def create_gene_go_mapping(
        gaf_path: Path | str,
        *,
        gene_column: str = "DB_Object_Symbol",
        evidence_codes: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Create a mapping from genes to GO terms.

        Parameters
        ----------
        gaf_path : Path or str
            Path to GAF file (required).
        gene_column : str, default="DB_Object_Symbol"
            Column to use as gene identifier.
            Options: "DB_Object_ID", "DB_Object_Symbol"
        evidence_codes : list of str, optional
            Filter by evidence codes.

        Returns
        -------
        dict[str, list[str]]
            Mapping from gene identifiers to lists of GO term IDs.

        Examples
        --------
        >>> # Create gene-to-GO mapping
        >>> gene_go = GODownloader.create_gene_go_mapping(
        ...     '/tmp/goa_human.gaf'
        ... )

        >>> # Get GO terms for a gene
        >>> gene_go['TP53']
        ['GO:0000122', 'GO:0000775', 'GO:0001701', ...]
        """
        df = GODownloader.parse_gaf(gaf_path, evidence_codes=evidence_codes)

        mapping = {}
        for _, row in df.iterrows():
            gene = row[gene_column]
            go_term = row["GO_ID"]

            if gene not in mapping:
                mapping[gene] = []

            if go_term not in mapping[gene]:
                mapping[gene].append(go_term)

        logger.info(f"Created mapping for {len(mapping)} genes")
        return mapping
