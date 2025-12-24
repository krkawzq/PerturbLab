"""Gene Ontology resources.

GO ontology files and gene-GO mappings.

Citation: Gene Ontology Consortium. http://geneontology.org
"""

from perturblab.data.resources import File

__all__ = ["GO_RESOURCES"]


# GO Resource Definitions
GO_RESOURCES = [
    # GO Basic Ontology (recommended for most analyses)
    File(
        key="go_basic",
        remote_config={
            "downloader": "GODownloader",
            "target_path": "go-basic.obo",  # Will be managed by cache
            "version": "basic",
        },
    ),
    # GO Full Ontology (includes all relationships)
    File(
        key="go_full",
        remote_config={
            "downloader": "GODownloader",
            "target_path": "go.obo",
            "version": "full",
        },
    ),
    # Gene2GO mapping from GEARS
    File(
        key="gene2go_gears",
        remote_config={
            "downloader": "HTTPDownloader",
            "url": "https://dataverse.harvard.edu/api/access/datafile/6153417",
        },
    ),
]
