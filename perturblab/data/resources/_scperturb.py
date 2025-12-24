"""scPerturb benchmark datasets.

Single-cell perturbation datasets from the scPerturb repository.

Citation: Peidli et al. (2023). scPerturb: harmonized single-cell perturbation data.
          Nature Methods. https://doi.org/10.1038/s41592-023-02144-y

Data: https://zenodo.org/record/13350497
"""

from perturblab.data.resources import h5adFile

__all__ = ["SCPERTURB_DATASETS"]

# Zenodo record for scPerturb
SCPERTURB_ZENODO_RECORD = "13350497"
ZENODO_BASE = f"https://zenodo.org/records/{SCPERTURB_ZENODO_RECORD}/files"


def _create_scperturb_resource(
    key: str,
    filename: str,
    description: str = "",
) -> h5adFile:
    """Helper to create scPerturb dataset resource.

    Parameters
    ----------
    key : str
        Dataset key.
    filename : str
        Zenodo filename.
    description : str
        Dataset description.

    Returns
    -------
    h5adFile
        Configured dataset resource.
    """
    url = f"{ZENODO_BASE}/{filename}?download=1"

    return h5adFile(
        key=key,
        remote_config={
            "downloader": "HTTPDownloader",
            "url": url,
            "show_progress": True,
        },
    )


# scPerturb Dataset Resources
SCPERTURB_DATASETS = [
    # ========================================================================
    # Norman et al. - Large-scale CRISPRi
    # ========================================================================
    _create_scperturb_resource(
        key="norman_2019",
        filename="NormanWeissman2019.h5ad",
        description="Large-scale CRISPRi screen in K562 cells (105,942 cells, 2.3GB)",
    ),
    _create_scperturb_resource(
        key="norman_2019_filtered",
        filename="NormanWeissman2019_filtered.h5ad",
        description="Large-scale CRISPRi screen in K562 cells (filtered)",
    ),
    # ========================================================================
    # Replogle et al. - CRISPRi screens
    # ========================================================================
    _create_scperturb_resource(
        key="replogle_2022_k562_essential",
        filename="ReplogleWeissman2022_K562_essential.h5ad",
        description="Essential genes CRISPRi screen in K562 (3.5GB)",
    ),
    _create_scperturb_resource(
        key="replogle_2022_k562_gwps",
        filename="ReplogleWeissman2022_K562_gwps.h5ad",
        description="Genome-wide pooled CRISPRi screen in K562",
    ),
    _create_scperturb_resource(
        key="replogle_2022_rpe1",
        filename="ReplogleWeissman2022_rpe1.h5ad",
        description="CRISPRi screen in RPE1 cells (2.8GB)",
    ),
    # ========================================================================
    # Dixit et al. - TF perturbations
    # ========================================================================
    _create_scperturb_resource(
        key="dixit_2016_k562_tfs_high_moi",
        filename="DixitRegev2016_K562_TFs_High_MOI.h5ad",
        description="TF perturbations in K562 (High MOI, 320MB)",
    ),
    _create_scperturb_resource(
        key="dixit_2016_k562_tfs_7days",
        filename="DixitRegev2016_K562_TFs_7_days.h5ad",
        description="TF perturbations in K562 (7 days, 257MB)",
    ),
    _create_scperturb_resource(
        key="dixit_2016_k562_tfs_13days",
        filename="DixitRegev2016_K562_TFs_13_days.h5ad",
        description="TF perturbations in K562 (13 days, 121MB)",
    ),
    # ========================================================================
    # Adamson et al. - CRISPR screens
    # ========================================================================
    _create_scperturb_resource(
        key="adamson_2016_10x010",
        filename="AdamsonWeissman2016_GSM2406681_10X010.h5ad",
        description="CRISPR screen in K562 (10X010, 471MB)",
    ),
    _create_scperturb_resource(
        key="adamson_2016_10x005",
        filename="AdamsonWeissman2016_GSM2406677_10X005.h5ad",
        description="CRISPR screen in K562 (10X005, 139MB)",
    ),
    _create_scperturb_resource(
        key="adamson_2016_10x001",
        filename="AdamsonWeissman2016_GSM2406675_10X001.h5ad",
        description="CRISPR screen in K562 (10X001, 35MB)",
    ),
    # ========================================================================
    # Srivatsan et al. - Chemical perturbations (sci-Plex)
    # ========================================================================
    _create_scperturb_resource(
        key="srivatsan_2020_sciplex2",
        filename="SrivatsanTrapnell2020_sciplex2.h5ad",
        description="Chemical perturbation screen (sci-Plex 2, A549/K562/MCF7, 388MB)",
    ),
    _create_scperturb_resource(
        key="srivatsan_2020_sciplex3",
        filename="SrivatsanTrapnell2020_sciplex3.h5ad",
        description="Chemical perturbation screen (sci-Plex 3, A549/K562/MCF7, 2.5GB)",
    ),
    _create_scperturb_resource(
        key="srivatsan_2020_sciplex4",
        filename="SrivatsanTrapnell2020_sciplex4.h5ad",
        description="Chemical perturbation screen (sci-Plex 4, A549/K562/MCF7, 253MB)",
    ),
    # ========================================================================
    # Frangieh et al. - RNA and Protein
    # ========================================================================
    _create_scperturb_resource(
        key="frangieh_2021_rna",
        filename="FrangiehIzar2021_RNA.h5ad",
        description="CRISPR screen in melanoma cells (RNA)",
    ),
    _create_scperturb_resource(
        key="frangieh_2021_protein",
        filename="FrangiehIzar2021_protein.h5ad",
        description="CRISPR screen in melanoma cells (protein)",
    ),
    # ========================================================================
    # Gasperini et al. - CRISPRi screens
    # ========================================================================
    _create_scperturb_resource(
        key="gasperini_2019_atscale",
        filename="GasperiniShendure2019_atscale.h5ad",
        description="CRISPRi screen at scale in K562",
    ),
    _create_scperturb_resource(
        key="gasperini_2019_high_moi",
        filename="GasperiniShendure2019_highMOI.h5ad",
        description="CRISPRi screen (high MOI) in K562",
    ),
    _create_scperturb_resource(
        key="gasperini_2019_low_moi",
        filename="GasperiniShendure2019_lowMOI.h5ad",
        description="CRISPRi screen (low MOI) in K562",
    ),
    # ========================================================================
    # Tian/Kampmann - CRISPRi/a in neurons and iPSC
    # ========================================================================
    _create_scperturb_resource(
        key="tian_2019_day7neuron",
        filename="TianKampmann2019_day7neuron.h5ad",
        description="CRISPRi screen in neurons (day 7, 269MB)",
    ),
    _create_scperturb_resource(
        key="tian_2019_ipsc",
        filename="TianKampmann2019_iPSC.h5ad",
        description="CRISPRi screen in iPSCs (351MB)",
    ),
    _create_scperturb_resource(
        key="tian_2021_crispra",
        filename="TianKampmann2021_CRISPRa.h5ad",
        description="CRISPRa screen (155MB)",
    ),
    _create_scperturb_resource(
        key="tian_2021_crispri",
        filename="TianKampmann2021_CRISPRi.h5ad",
        description="CRISPRi screen (289MB)",
    ),
    # ========================================================================
    # Joung/Zhang - Combinatorial and Atlas
    # ========================================================================
    _create_scperturb_resource(
        key="joung_2023_combinatorial",
        filename="JoungZhang2023_combinatorial.h5ad",
        description="Combinatorial perturbation screen",
    ),
    _create_scperturb_resource(
        key="joung_2023_atlas",
        filename="JoungZhang2023_atlas.h5ad",
        description="Perturbation atlas",
    ),
    # ========================================================================
    # Papalexi/Satija - ECCITE-seq
    # ========================================================================
    _create_scperturb_resource(
        key="papalexi_2021_eccite_rna",
        filename="PapalexiSatija2021_eccite_RNA.h5ad",
        description="ECCITE-seq perturbation screen (RNA)",
    ),
    _create_scperturb_resource(
        key="papalexi_2021_eccite_protein",
        filename="PapalexiSatija2021_eccite_protein.h5ad",
        description="ECCITE-seq perturbation screen (Protein)",
    ),
    _create_scperturb_resource(
        key="papalexi_2021_eccite_arrayed_rna",
        filename="PapalexiSatija2021_eccite_arrayed_RNA.h5ad",
        description="ECCITE-seq arrayed screen (RNA)",
    ),
    _create_scperturb_resource(
        key="papalexi_2021_eccite_arrayed_protein",
        filename="PapalexiSatija2021_eccite_arrayed_protein.h5ad",
        description="ECCITE-seq arrayed screen (Protein)",
    ),
    # ========================================================================
    # Additional Studies
    # ========================================================================
    _create_scperturb_resource(
        key="nadig_2024_hepg2",
        filename="NadigOConner2024_hepg2.h5ad",
        description="Perturbation screen in HepG2 cells",
    ),
    _create_scperturb_resource(
        key="nadig_2024_jurkat",
        filename="NadigOConner2024_jurkat.h5ad",
        description="Perturbation screen in Jurkat cells",
    ),
    _create_scperturb_resource(
        key="shifrut_2018",
        filename="ShifrutMarson2018.h5ad",
        description="CRISPR screen in T cells",
    ),
    _create_scperturb_resource(
        key="lara_astiaso_2023_exvivo",
        filename="LaraAstiasoHuntly2023_exvivo.h5ad",
        description="Ex vivo perturbation screen",
    ),
    _create_scperturb_resource(
        key="lara_astiaso_2023_invivo",
        filename="LaraAstiasoHuntly2023_invivo.h5ad",
        description="In vivo perturbation screen",
    ),
    _create_scperturb_resource(
        key="lara_astiaso_2023_leukemia",
        filename="LaraAstiasoHuntly2023_leukemia.h5ad",
        description="Leukemia perturbation screen",
    ),
    _create_scperturb_resource(
        key="aissa_2021",
        filename="AissaBenevolenskaya2021.h5ad",
        description="Perturbation screen (46MB)",
    ),
    _create_scperturb_resource(
        key="chang_2021",
        filename="ChangYe2021.h5ad",
        description="Perturbation screen (502MB)",
    ),
    _create_scperturb_resource(
        key="datlinger_2017",
        filename="DatlingerBock2017.h5ad",
        description="Pooled CRISPR screen with scRNA-seq (39MB)",
    ),
    _create_scperturb_resource(
        key="datlinger_2021",
        filename="DatlingerBock2021.h5ad",
        description="Pooled CRISPR screen (34MB)",
    ),
    _create_scperturb_resource(
        key="sunshine_2023",
        filename="SunshineHein2023.h5ad",
        description="Perturbation screen (744MB)",
    ),
    _create_scperturb_resource(
        key="gehring_2019",
        filename="GehringPachter2019.h5ad",
        description="Perturbation screen",
    ),
    _create_scperturb_resource(
        key="mcfarland_2020",
        filename="McFarlandTsherniak2020.h5ad",
        description="Perturbation screen",
    ),
    _create_scperturb_resource(
        key="schiebinger_2019_gse106340",
        filename="SchiebingerLander2019_GSE106340.h5ad",
        description="Developmental perturbation screen (GSE106340)",
    ),
    _create_scperturb_resource(
        key="schiebinger_2019_gse115943",
        filename="SchiebingerLander2019_GSE115943.h5ad",
        description="Developmental perturbation screen (GSE115943)",
    ),
    _create_scperturb_resource(
        key="schraivogel_2020_tap_chr8",
        filename="SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_8_screen.h5ad",
        description="TAP-seq screen on chromosome 8",
    ),
    _create_scperturb_resource(
        key="schraivogel_2020_tap_chr11",
        filename="SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_11_screen.h5ad",
        description="TAP-seq screen on chromosome 11",
    ),
    _create_scperturb_resource(
        key="weinreb_2020",
        filename="WeinrebKlein2020.h5ad",
        description="Developmental perturbation screen (229MB)",
    ),
    _create_scperturb_resource(
        key="wessels_2023",
        filename="WesselsSatija2023.h5ad",
        description="Perturbation screen (219MB)",
    ),
    _create_scperturb_resource(
        key="xie_2017",
        filename="XieHon2017.h5ad",
        description="Perturbation screen (117MB)",
    ),
    _create_scperturb_resource(
        key="xu_2023",
        filename="XuCao2023.h5ad",
        description="Perturbation screen (337MB)",
    ),
    _create_scperturb_resource(
        key="zhao_2021",
        filename="ZhaoSims2021.h5ad",
        description="Perturbation screen (587MB)",
    ),
    _create_scperturb_resource(
        key="santinha_2023",
        filename="SantinhaPlatt2023.h5ad",
        description="Perturbation screen",
    ),
    _create_scperturb_resource(
        key="liang_2023",
        filename="LiangWang2023.h5ad",
        description="Perturbation screen",
    ),
    _create_scperturb_resource(
        key="cui_2023",
        filename="CuiHacohen2023.h5ad",
        description="Perturbation screen (472MB)",
    ),
    _create_scperturb_resource(
        key="lotfollahi_2023",
        filename="LotfollahiTheis2023.h5ad",
        description="Perturbation screen",
    ),
]
