"""scPerturb dataset cards.

Complete list of datasets from scPerturb repository.
Based on: https://zenodo.org/records/13350497
Citation: Peidli et al. (2023). Nature Methods. https://doi.org/10.1038/s41592-023-02144-y

scPerturb contains 40+ h5ad files for RNA and protein single-cell perturbation datasets.
"""

from ..cards import ScPerturbCard, PerturbationType


# scPerturb Complete Dataset List
SCPERTURB_DATASETS = [
    # ========================================================================
    # Norman et al. - Large-scale CRISPRi
    # ========================================================================
    
    ScPerturbCard(
        name='Norman2019',
        zenodo_filename='NormanWeissman2019.h5ad',
        url='',
        description='Large-scale CRISPRi screen in K562 cells',
        citation='Norman et al. (2019). Science.',
        n_cells=105942,
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
        file_size_mb=2300.0,
    ),
    
    ScPerturbCard(
        name='Norman2019_filtered',
        zenodo_filename='NormanWeissman2019_filtered.h5ad',
        url='',
        description='Large-scale CRISPRi screen in K562 cells (filtered)',
        citation='Norman et al. (2019). Science.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
    ),
    
    # ========================================================================
    # Replogle et al. - CRISPRi screens
    # ========================================================================
    
    ScPerturbCard(
        name='Replogle2022_K562_essential',
        zenodo_filename='ReplogleWeissman2022_K562_essential.h5ad',
        url='',
        description='Essential genes CRISPRi screen in K562',
        citation='Replogle et al. (2022). Cell.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
        file_size_mb=3500.0,
    ),
    
    ScPerturbCard(
        name='Replogle2022_K562_gwps',
        zenodo_filename='ReplogleWeissman2022_K562_gwps.h5ad',
        url='',
        description='Genome-wide pooled CRISPRi screen in K562',
        citation='Replogle et al. (2022). Cell.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
    ),
    
    ScPerturbCard(
        name='Replogle2022_RPE1',
        zenodo_filename='ReplogleWeissman2022_rpe1.h5ad',
        url='',
        description='CRISPRi screen in RPE1 cells',
        citation='Replogle et al. (2022). Cell.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='RPE1',
        file_size_mb=2800.0,
    ),
    
    # ========================================================================
    # Dixit et al. - TF perturbations
    # ========================================================================
    
    ScPerturbCard(
        name='Dixit2016_K562_TFs_High_MOI',
        zenodo_filename='DixitRegev2016_K562_TFs_High_MOI.h5ad',
        url='',
        description='TF perturbations in K562 (High MOI)',
        citation='Dixit et al. (2016). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='K562',
        file_size_mb=319.7,
    ),
    
    ScPerturbCard(
        name='Dixit2016_K562_TFs_7days',
        zenodo_filename='DixitRegev2016_K562_TFs_7_days.h5ad',
        url='',
        description='TF perturbations in K562 (7 days)',
        citation='Dixit et al. (2016). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='K562',
        file_size_mb=256.7,
    ),
    
    ScPerturbCard(
        name='Dixit2016_K562_TFs_13days',
        zenodo_filename='DixitRegev2016_K562_TFs_13_days.h5ad',
        url='',
        description='TF perturbations in K562 (13 days)',
        citation='Dixit et al. (2016). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='K562',
        file_size_mb=121.2,
    ),
    
    # ========================================================================
    # Adamson et al. - CRISPR screens
    # ========================================================================
    
    ScPerturbCard(
        name='Adamson2016_10X010',
        zenodo_filename='AdamsonWeissman2016_GSM2406681_10X010.h5ad',
        url='',
        description='CRISPR screen in K562 (10X010)',
        citation='Adamson et al. (2016). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='K562',
        file_size_mb=471.3,
    ),
    
    ScPerturbCard(
        name='Adamson2016_10X005',
        zenodo_filename='AdamsonWeissman2016_GSM2406677_10X005.h5ad',
        url='',
        description='CRISPR screen in K562 (10X005)',
        citation='Adamson et al. (2016). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='K562',
        file_size_mb=139.1,
    ),
    
    ScPerturbCard(
        name='Adamson2016_10X001',
        zenodo_filename='AdamsonWeissman2016_GSM2406675_10X001.h5ad',
        url='',
        description='CRISPR screen in K562 (10X001)',
        citation='Adamson et al. (2016). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='K562',
        file_size_mb=34.6,
    ),
    
    # ========================================================================
    # Srivatsan et al. - Chemical perturbations (sci-Plex)
    # ========================================================================
    
    ScPerturbCard(
        name='Srivatsan2020_sciplex2',
        zenodo_filename='SrivatsanTrapnell2020_sciplex2.h5ad',
        url='',
        description='Chemical perturbation screen (sci-Plex 2)',
        citation='Srivatsan et al. (2020). Science.',
        perturbation_type=PerturbationType.CHEMICAL,
        cell_type='A549, K562, MCF7',
        control_label='DMSO',
        file_size_mb=388.3,
    ),
    
    ScPerturbCard(
        name='Srivatsan2020_sciplex3',
        zenodo_filename='SrivatsanTrapnell2020_sciplex3.h5ad',
        url='',
        description='Chemical perturbation screen (sci-Plex 3)',
        citation='Srivatsan et al. (2020). Science.',
        perturbation_type=PerturbationType.CHEMICAL,
        cell_type='A549, K562, MCF7',
        control_label='DMSO',
        file_size_mb=2500.0,
    ),
    
    ScPerturbCard(
        name='Srivatsan2020_sciplex4',
        zenodo_filename='SrivatsanTrapnell2020_sciplex4.h5ad',
        url='',
        description='Chemical perturbation screen (sci-Plex 4)',
        citation='Srivatsan et al. (2020). Science.',
        perturbation_type=PerturbationType.CHEMICAL,
        cell_type='A549, K562, MCF7',
        control_label='DMSO',
        file_size_mb=253.3,
    ),
    
    # ========================================================================
    # Frangieh et al. - RNA and Protein
    # ========================================================================
    
    ScPerturbCard(
        name='Frangieh2021_RNA',
        zenodo_filename='FrangiehIzar2021_RNA.h5ad',
        url='',
        description='CRISPR screen in melanoma cells (RNA)',
        citation='Frangieh et al. (2021). Nature Genetics.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='Melanoma',
        file_size_mb=75.6,
    ),
    
    ScPerturbCard(
        name='Frangieh2021_protein',
        zenodo_filename='FrangiehIzar2021_protein.h5ad',
        url='',
        description='CRISPR screen in melanoma cells (Protein)',
        citation='Frangieh et al. (2021). Nature Genetics.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='Melanoma',
        file_size_mb=24.7,
    ),
    
    # ========================================================================
    # Gasperini et al. - CRISPRi screens
    # ========================================================================
    
    ScPerturbCard(
        name='Gasperini2019_atscale',
        zenodo_filename='GasperiniShendure2019_atscale.h5ad',
        url='',
        description='CRISPRi screen at scale',
        citation='Gasperini et al. (2019). Cell.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
    ),
    
    ScPerturbCard(
        name='Gasperini2019_highMOI',
        zenodo_filename='GasperiniShendure2019_highMOI.h5ad',
        url='',
        description='CRISPRi screen (high MOI)',
        citation='Gasperini et al. (2019). Cell.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
    ),
    
    ScPerturbCard(
        name='Gasperini2019_lowMOI',
        zenodo_filename='GasperiniShendure2019_lowMOI.h5ad',
        url='',
        description='CRISPRi screen (low MOI)',
        citation='Gasperini et al. (2019). Cell.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='K562',
    ),
    
    # ========================================================================
    # Tian/Kampmann - CRISPRi/a in neurons and iPSC
    # ========================================================================
    
    ScPerturbCard(
        name='Tian2019_day7neuron',
        zenodo_filename='TianKampmann2019_day7neuron.h5ad',
        url='',
        description='CRISPRi screen in neurons (day 7)',
        citation='Tian et al. (2019). Science.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='Neuron',
        file_size_mb=268.9,
    ),
    
    ScPerturbCard(
        name='Tian2019_iPSC',
        zenodo_filename='TianKampmann2019_iPSC.h5ad',
        url='',
        description='CRISPRi screen in iPSCs',
        citation='Tian et al. (2019). Science.',
        perturbation_type=PerturbationType.CRISPRI,
        cell_type='iPSC',
        file_size_mb=350.8,
    ),
    
    ScPerturbCard(
        name='Tian2021_CRISPRa',
        zenodo_filename='TianKampmann2021_CRISPRa.h5ad',
        url='',
        description='CRISPRa screen',
        citation='Tian et al. (2021). Nature Genetics.',
        perturbation_type=PerturbationType.CRISPRA,
        file_size_mb=154.8,
    ),
    
    ScPerturbCard(
        name='Tian2021_CRISPRi',
        zenodo_filename='TianKampmann2021_CRISPRi.h5ad',
        url='',
        description='CRISPRi screen',
        citation='Tian et al. (2021). Nature Genetics.',
        perturbation_type=PerturbationType.CRISPRI,
        file_size_mb=289.4,
    ),
    
    # ========================================================================
    # Joung/Zhang - Combinatorial and Atlas
    # ========================================================================
    
    ScPerturbCard(
        name='Joung2023_combinatorial',
        zenodo_filename='JoungZhang2023_combinatorial.h5ad',
        url='',
        description='Combinatorial perturbation screen',
        citation='Joung et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Joung2023_atlas',
        zenodo_filename='JoungZhang2023_atlas.h5ad',
        url='',
        description='Perturbation atlas',
        citation='Joung et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    # ========================================================================
    # Papalexi/Satija - ECCITE-seq
    # ========================================================================
    
    ScPerturbCard(
        name='Papalexi2021_eccite_RNA',
        zenodo_filename='PapalexiSatija2021_eccite_RNA.h5ad',
        url='',
        description='ECCITE-seq perturbation screen (RNA)',
        citation='Papalexi et al. (2021). Nature Methods.',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Papalexi2021_eccite_protein',
        zenodo_filename='PapalexiSatija2021_eccite_protein.h5ad',
        url='',
        description='ECCITE-seq perturbation screen (Protein)',
        citation='Papalexi et al. (2021). Nature Methods.',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Papalexi2021_eccite_arrayed_RNA',
        zenodo_filename='PapalexiSatija2021_eccite_arrayed_RNA.h5ad',
        url='',
        description='ECCITE-seq arrayed screen (RNA)',
        citation='Papalexi et al. (2021). Nature Methods.',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Papalexi2021_eccite_arrayed_protein',
        zenodo_filename='PapalexiSatija2021_eccite_arrayed_protein.h5ad',
        url='',
        description='ECCITE-seq arrayed screen (Protein)',
        citation='Papalexi et al. (2021). Nature Methods.',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    # ========================================================================
    # Additional Studies
    # ========================================================================
    
    ScPerturbCard(
        name='Nadig2024_hepg2',
        zenodo_filename='NadigOConner2024_hepg2.h5ad',
        url='',
        description='Perturbation screen in HepG2 cells',
        citation='Nadig et al. (2024).',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='HepG2',
    ),
    
    ScPerturbCard(
        name='Nadig2024_jurkat',
        zenodo_filename='NadigOConner2024_jurkat.h5ad',
        url='',
        description='Perturbation screen in Jurkat cells',
        citation='Nadig et al. (2024).',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='Jurkat',
    ),
    
    ScPerturbCard(
        name='Shifrut2018',
        zenodo_filename='ShifrutMarson2018.h5ad',
        url='',
        description='CRISPR screen in T cells',
        citation='Shifrut et al. (2018). Cell.',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='T cell',
    ),
    
    ScPerturbCard(
        name='LaraAstia2023_exvivo',
        zenodo_filename='LaraAstiasoHuntly2023_exvivo.h5ad',
        url='',
        description='Ex vivo perturbation screen',
        citation='Lara-Astiaso et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='LaraAstia2023_invivo',
        zenodo_filename='LaraAstiasoHuntly2023_invivo.h5ad',
        url='',
        description='In vivo perturbation screen',
        citation='Lara-Astiaso et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='LaraAstia2023_leukemia',
        zenodo_filename='LaraAstiasoHuntly2023_leukemia.h5ad',
        url='',
        description='Leukemia perturbation screen',
        citation='Lara-Astiaso et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
        cell_type='Leukemia',
    ),
    
    ScPerturbCard(
        name='Aissa2021',
        zenodo_filename='AissaBenevolenskaya2021.h5ad',
        url='',
        description='Perturbation screen',
        citation='Aissa et al. (2021).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=45.9,
    ),
    
    ScPerturbCard(
        name='Chang2021',
        zenodo_filename='ChangYe2021.h5ad',
        url='',
        description='Perturbation screen',
        citation='Chang et al. (2021).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=501.8,
    ),
    
    ScPerturbCard(
        name='Datlinger2017',
        zenodo_filename='DatlingerBock2017.h5ad',
        url='',
        description='Pooled CRISPR screen with scRNA-seq',
        citation='Datlinger et al. (2017). Nature Methods.',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=39.1,
    ),
    
    ScPerturbCard(
        name='Datlinger2021',
        zenodo_filename='DatlingerBock2021.h5ad',
        url='',
        description='Pooled CRISPR screen',
        citation='Datlinger et al. (2021).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=33.6,
    ),
    
    ScPerturbCard(
        name='Sunshine2023',
        zenodo_filename='SunshineHein2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Sunshine et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=743.7,
    ),
    
    ScPerturbCard(
        name='Gehring2019',
        zenodo_filename='GehringPachter2019.h5ad',
        url='',
        description='Perturbation screen',
        citation='Gehring et al. (2019).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='McFarland2020',
        zenodo_filename='McFarlandTsherniak2020.h5ad',
        url='',
        description='Perturbation screen',
        citation='McFarland et al. (2020).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Schiebinger2019_GSE106340',
        zenodo_filename='SchiebingerLander2019_GSE106340.h5ad',
        url='',
        description='Developmental perturbation screen (GSE106340)',
        citation='Schiebinger et al. (2019). Cell.',
        perturbation_type=PerturbationType.GENETIC,
    ),
    
    ScPerturbCard(
        name='Schiebinger2019_GSE115943',
        zenodo_filename='SchiebingerLander2019_GSE115943.h5ad',
        url='',
        description='Developmental perturbation screen (GSE115943)',
        citation='Schiebinger et al. (2019). Cell.',
        perturbation_type=PerturbationType.GENETIC,
    ),
    
    ScPerturbCard(
        name='Schraivogel2020_TAP_chr8',
        zenodo_filename='SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_8_screen.h5ad',
        url='',
        description='TAP-seq screen on chromosome 8',
        citation='Schraivogel et al. (2020). Nature Biotechnology.',
        perturbation_type=PerturbationType.CRISPRI,
    ),
    
    ScPerturbCard(
        name='Schraivogel2020_TAP_chr11',
        zenodo_filename='SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_11_screen.h5ad',
        url='',
        description='TAP-seq screen on chromosome 11',
        citation='Schraivogel et al. (2020). Nature Biotechnology.',
        perturbation_type=PerturbationType.CRISPRI,
    ),
    
    ScPerturbCard(
        name='Weinreb2020',
        zenodo_filename='WeinrebKlein2020.h5ad',
        url='',
        description='Developmental perturbation screen',
        citation='Weinreb et al. (2020). Cell.',
        perturbation_type=PerturbationType.GENETIC,
        file_size_mb=228.5,
    ),
    
    ScPerturbCard(
        name='Wessels2023',
        zenodo_filename='WesselsSatija2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Wessels et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=219.4,
    ),
    
    ScPerturbCard(
        name='Xie2017',
        zenodo_filename='XieHon2017.h5ad',
        url='',
        description='Perturbation screen',
        citation='Xie et al. (2017).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=117.4,
    ),
    
    ScPerturbCard(
        name='Xu2023',
        zenodo_filename='XuCao2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Xu et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=336.6,
    ),
    
    ScPerturbCard(
        name='Zhao2021',
        zenodo_filename='ZhaoSims2021.h5ad',
        url='',
        description='Perturbation screen',
        citation='Zhao et al. (2021).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=586.9,
    ),
    
    ScPerturbCard(
        name='Santinha2023',
        zenodo_filename='SantinhaPlatt2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Santinha et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Liang2023',
        zenodo_filename='LiangWang2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Liang et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
    
    ScPerturbCard(
        name='Cui2023',
        zenodo_filename='CuiHacohen2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Cui et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
        file_size_mb=472.1,
    ),
    
    ScPerturbCard(
        name='Lotfollahi2023',
        zenodo_filename='LotfollahiTheis2023.h5ad',
        url='',
        description='Perturbation screen',
        citation='Lotfollahi et al. (2023).',
        perturbation_type=PerturbationType.CRISPR,
    ),
]
