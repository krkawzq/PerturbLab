"""ScPerturb dataset cards."""

from dataclasses import dataclass

from ._base import H5ADDatasetCard


@dataclass
class ScPerturbCard(H5ADDatasetCard):
    """Dataset card for scPerturb datasets.
    
    scPerturb-specific dataset card with auto-construction of URL from Zenodo.
    
    Citation: Peidli et al. (2023). Nature Methods. 
              https://doi.org/10.1038/s41592-023-02144-y
    """
    
    zenodo_filename: str = ''
    
    def __post_init__(self):
        """Auto-construct URL and source."""
        object.__setattr__(self, 'source', 'scPerturb')
        
        if not self.url:
            zenodo_record = "13350497"
            url = f"https://zenodo.org/records/{zenodo_record}/files/{self.zenodo_filename}"
            object.__setattr__(self, 'url', url)

