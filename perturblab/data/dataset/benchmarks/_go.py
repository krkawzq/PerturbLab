"""GO-related dataset cards."""

from ..cards import PickleDatasetCard, OBODatasetCard


# GO Ontology Files
GO_DATASETS = [
    # GO Basic Ontology - Primary source
    OBODatasetCard(
        name='go_basic',
        url='http://current.geneontology.org/ontology/go-basic.obo',
        source='Gene Ontology Consortium',
        description='GO basic ontology (filtered, no inter-ontology links)',
        citation='Gene Ontology Consortium. http://geneontology.org',
    ),
    
    # GO Basic - EBI Mirror
    OBODatasetCard(
        name='go_basic',
        url='http://ftp.ebi.ac.uk/pub/databases/GO/goa/go-basic.obo',
        source='EBI Mirror',
        description='GO basic ontology (EBI mirror)',
        citation='Gene Ontology Consortium. http://geneontology.org',
    ),
    
    # GO Basic - NCBI Mirror
    OBODatasetCard(
        name='go_basic',
        url='ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_ontology/go-basic.obo',
        source='NCBI Mirror',
        description='GO basic ontology (NCBI mirror)',
        citation='Gene Ontology Consortium. http://geneontology.org',
    ),
    
    # GO Full Ontology - Primary
    OBODatasetCard(
        name='go_full',
        url='http://current.geneontology.org/ontology/go.obo',
        source='Gene Ontology Consortium',
        description='GO full ontology (includes all relationships)',
        citation='Gene Ontology Consortium. http://geneontology.org',
    ),
    
    # GO Full - EBI Mirror
    OBODatasetCard(
        name='go_full',
        url='http://ftp.ebi.ac.uk/pub/databases/GO/goa/go.obo',
        source='EBI Mirror',
        description='GO full ontology (EBI mirror)',
        citation='Gene Ontology Consortium. http://geneontology.org',
    ),
    
    # Gene2GO from GEARS
    PickleDatasetCard(
        name='gene2go_gears',
        url='https://dataverse.harvard.edu/api/access/datafile/6153417',
        source='GEARS',
        description='Gene-to-GO mapping from GEARS (pickle format)',
        citation='Roohani et al. (2023). Nature Biotechnology.',
    ),
]

