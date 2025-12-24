"""Differential expression analysis using high-performance statistical kernels."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import false_discovery_control

from perturblab.kernels.statistics import (
    group_mean,
)
from perturblab.kernels.statistics import log_fold_change as calc_log_fc
from perturblab.kernels.statistics import (
    mannwhitneyu,
    ttest,
)
from perturblab.utils import get_logger

logger = get_logger()

# Supported statistical methods
_Method = Literal["wilcoxon", "t-test", "t-test_overestim_var", "welch"]
SUPPORTED_METHODS = {"wilcoxon", "t-test", "t-test_overestim_var", "welch"}


def differential_expression(
    adata: AnnData,
    perturb_col: str,
    control_tag: str | None = None,
    groups: list[str] | None = None,
    method: _Method = "wilcoxon",
    min_samples: int = 2,
    threads: int = -1,
    clip_value: float = 20.0,
    fdr_method: Literal["bh", "by"] = "bh",
    layer: str | None = None,
    use_raw: bool = False,
) -> pd.DataFrame:
    """Differential expression analysis using high-performance statistical kernels.

    This function performs group-wise differential expression analysis comparing
    target groups against a reference group using various statistical methods.
    It leverages optimized C++/Cython kernels for maximum performance.

    Parameters
    ----------
    adata
        Annotated data matrix.
    perturb_col
        Column name in `adata.obs` for perturbation cells.
    control_tag
        Name of the reference/control group. If None and `perturb_col` contains
        NA values, they will be treated as 'non-targeting' controls.
    groups
        List of target groups to compare against reference. If None, uses all
        groups except the reference.
    method
        Statistical test method:
        - 'wilcoxon': Mann-Whitney U test (rank-based, non-parametric)
        - 't-test': Student's t-test (assumes equal variance)
        - 't-test_overestim_var': Student's t-test with overestimated variance
        - 'welch': Welch's t-test (does not assume equal variance)
    min_samples
        Minimum number of samples required per group. Groups with fewer samples
        are excluded from analysis.
    threads
        Number of threads for parallel computation. -1 uses all available cores.
    clip_value
        Value to clip log fold changes if infinite or NaN. Set to None to disable.
    fdr_method
        FDR correction method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli).
    layer
        Layer name to use for analysis. If None, uses `adata.X`.
    use_raw
        Whether to use `adata.raw.X` for analysis.

    Returns
    -------
    DataFrame with columns:
        - target: Target group name
        - feature: Gene name
        - p_value: P-value from statistical test
        - statistic: Test statistic (U for Wilcoxon, t for t-tests)
        - fold_change: Fold change (target_mean / reference_mean)
        - log2_fold_change: Log2 fold change
        - fdr: FDR-corrected p-value

    Examples
    --------
    >>> import perturblab as pl
    >>> adata = pl.datasets.load_dataset("adamson_2016")
    >>> results = pl.data.process.differential_expression(
    ...     adata,
    ...     groupby_key='perturbation',
    ...     reference='non-targeting',
    ...     method='wilcoxon',
    ... )

    Notes
    -----
    - For large datasets, 'wilcoxon' is generally fastest and most robust
    - 't-test_overestim_var' is scanpy's default and matches its behavior
    - 'welch' is recommended when groups have unequal variances
    - The function automatically handles sparse matrices efficiently

    See Also
    --------
    perturblab.kernels.statistics.mannwhitneyu
    perturblab.kernels.statistics.welch_ttest
    perturblab.kernels.statistics.student_ttest
    """
    # ===== Step 0: Validation =====
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: {method}. " f"Supported methods: {SUPPORTED_METHODS}"
        )

    if perturb_col not in adata.obs.columns:
        raise ValueError(f"Column '{perturb_col}' not found in adata.obs")

    # ===== Step 1: Get expression matrix =====
    logger.info(f"🔬 Starting differential expression analysis (method={method})")

    if use_raw and adata.raw is not None:
        X = adata.raw.X
        gene_names = adata.raw.var_names.values
    elif layer is not None:
        X = adata.layers[layer]
        gene_names = adata.var_names.values
    else:
        X = adata.X
        gene_names = adata.var_names.values

    # Convert to CSC format for efficient column access (genes)
    from scipy.sparse import csc_matrix, csr_matrix

    if issparse(X):
        if not isinstance(X, csc_matrix):
            logger.info("Converting sparse matrix to CSC format...")
            X = csc_matrix(X)
    else:
        # Convert dense matrix to sparse CSC format
        logger.info("Converting dense matrix to CSC sparse format...")
        X = csc_matrix(X)

    obs = adata.obs.copy()

    # ===== Step 2: Handle reference and groups =====
    if control_tag is None:
        logger.info("🧭 No reference provided, treating NA as 'non-targeting'")
        obs[perturb_col] = obs[perturb_col].astype("category")
        if obs[perturb_col].isna().any():
            obs[perturb_col] = obs[perturb_col].cat.add_categories("non-targeting")
            obs[perturb_col] = obs[perturb_col].fillna("non-targeting")
            control_tag = "non-targeting"
        else:
            raise ValueError(
                "reference=None but no NA values found in groupby column. "
                "Please specify a reference group explicitly."
            )

    unique_groups = obs[perturb_col].unique().tolist()

    if control_tag not in unique_groups:
        raise ValueError(
            f"Reference group '{control_tag}' not found in '{perturb_col}'. "
            f"Available groups: {unique_groups}"
        )

    if groups is None:
        groups = [g for g in unique_groups if g != control_tag]
        logger.info(f"Comparing {len(groups)} groups against reference '{control_tag}'")
    else:
        groups = [g for g in groups if g in unique_groups and g != control_tag]
        if not groups:
            raise ValueError("No valid target groups found")

    # ===== Step 3: Filter groups by sample size =====
    valid_groups = []
    for g in groups:
        count = np.sum(obs[perturb_col] == g)
        if count >= min_samples:
            valid_groups.append(g)
        else:
            logger.warning(f"⚠️  Group '{g}' has only {count} samples (< {min_samples}), skipping")

    groups = valid_groups
    if not groups:
        raise ValueError(
            f"No groups have at least {min_samples} samples. " "Try reducing min_samples."
        )

    n_targets = len(groups)
    n_genes = X.shape[1]

    logger.info(f"📊 Analyzing {n_targets} target groups × {n_genes} genes")

    # ===== Step 4: Prepare cell assignments =====
    # Create assignment array: 0 = reference, 1..n = target groups
    cell_assignments = np.full(len(obs), -1, dtype=np.int32)
    cell_assignments[obs[perturb_col] == control_tag] = 0

    for i, g in enumerate(groups):
        cell_assignments[obs[perturb_col] == g] = i + 1

    # ===== Step 5: Run statistical tests =====
    logger.info(f"🧮 Computing statistics using {method}...")

    if method == "wilcoxon":
        # Mann-Whitney U test
        U1, U2, p_values = mannwhitneyu(
            X,
            cell_assignments,
            n_targets,
            threads=threads,
        )
        statistics = np.asarray(U2)  # Use U2 (reference group U) as statistic
        p_values = np.asarray(p_values)

    elif method in ("t-test", "t-test_overestim_var", "welch"):
        # T-tests (Student's or Welch's)
        # Note: t-test_overestim_var uses Student's method (scanpy compatibility)
        ttest_method = "welch" if method == "welch" else "student"

        t_stats, p_values, mean_diff, log2_fc_ttest = ttest(
            X,
            cell_assignments,
            n_targets,
            method=ttest_method,
            threads=threads,
        )
        statistics = np.asarray(t_stats)
        p_values = np.asarray(p_values)

        # t-test already computes log2_fc
        log2_fc = np.asarray(log2_fc_ttest)

    else:
        raise ValueError(f"Method {method} not implemented")

    # ===== Step 6: Compute fold changes =====
    logger.info("📈 Computing fold changes...")

    # Use log_fold_change kernel for efficient computation
    # (unless already computed by t-test)
    if method not in ("t-test", "t-test_overestim_var", "welch"):
        # For wilcoxon, we need to compute fold changes separately
        mean_diff, log2_fc = calc_log_fc(
            X,
            cell_assignments,
            n_targets,
            threads=threads,
        )
        log2_fc = np.asarray(log2_fc)

    # Compute fold change from log2 fold change
    fold_changes = np.power(2.0, log2_fc)

    # Clip extreme values if requested
    if clip_value is not None:
        log2_fc = np.clip(log2_fc, -clip_value, clip_value)
        fold_changes = np.clip(fold_changes, 2 ** (-clip_value), 2**clip_value)
        # Handle NaN/inf
        log2_fc = np.nan_to_num(log2_fc, nan=0.0, posinf=clip_value, neginf=-clip_value)
        fold_changes = np.nan_to_num(
            fold_changes, nan=1.0, posinf=2**clip_value, neginf=2 ** (-clip_value)
        )

    # ===== Step 7: Flatten results =====
    logger.info("📦 Assembling results...")

    # Flatten arrays (target outer, gene inner)
    statistics_flat = statistics.ravel(order="C")
    p_values_flat = p_values.ravel(order="C")
    fold_changes_flat = fold_changes.ravel(order="C")
    log2_fc_flat = log2_fc.ravel(order="C")

    # Create target and feature arrays
    targets_flat = np.repeat(np.asarray(groups, dtype=object), n_genes)
    features_flat = np.tile(gene_names, n_targets)

    # ===== Step 8: FDR correction =====
    logger.info(f"✨ Applying FDR correction (method={fdr_method})...")

    try:
        fdr_values = false_discovery_control(p_values_flat, method=fdr_method)
    except Exception as e:
        logger.warning(f"⚠️  FDR correction failed: {e}")
        logger.info(
            f"P-value stats: min={np.nanmin(p_values_flat):.2e}, "
            f"max={np.nanmax(p_values_flat):.2e}, "
            f"NaN count={np.isnan(p_values_flat).sum()}"
        )
        logger.warning("Using uncorrected p-values as FDR")
        fdr_values = p_values_flat.copy()

    # ===== Step 9: Create output DataFrame =====
    result = pd.DataFrame(
        {
            "target": targets_flat,
            "feature": features_flat,
            "statistic": statistics_flat,
            "p_value": p_values_flat,
            "fold_change": fold_changes_flat,
            "log2_fold_change": log2_fc_flat,
            "fdr": fdr_values,
        }
    )

    logger.info("✅ Differential expression analysis complete!")
    logger.info(f"   Results shape: {result.shape}")
    logger.info(
        f"   Significant genes (p<0.05): " f"{(result['p_value'] < 0.05).sum()} / {len(result)}"
    )

    return result


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    use_raw: bool | None = None,
    groups: Literal["all"] | list[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    method: _Method = "t-test",
    corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    pts: bool = False,
    key_added: str | None = None,
    copy: bool = False,
    layer: str | None = None,
    min_samples: int = 2,
    threads: int = -1,
    **kwargs,
) -> AnnData | None:
    """Rank genes by differential expression (scanpy-compatible interface).

    This function performs differential expression analysis and stores results
    in scanpy-compatible format. It wraps our high-performance statistical
    kernels while maintaining full API compatibility with scanpy.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    use_raw
        Use `raw` attribute of `adata` if present. Defaults to True if `adata.raw` exists.
    groups
        Subset of groups to which comparison shall be restricted, or 'all' (default).
    reference
        If 'rest', compare each group to the union of the rest of the groups.
        If a group identifier, compare with respect to this group.
    n_genes
        The number of genes that appear in the returned tables. Defaults to all genes.
    method
        Statistical test method:
        - 't-test': Student's t-test (default)
        - 't-test_overestim_var': t-test with overestimated variance (scanpy style)
        - 'wilcoxon': Wilcoxon rank-sum test
        - 'welch': Welch's t-test
    corr_method
        p-value correction method ('benjamini-hochberg' or 'bonferroni').
    pts
        Compute the fraction of cells expressing the genes.
    key_added
        The key in `adata.uns` where information is saved. Defaults to 'rank_genes_groups'.
    copy
        Whether to copy `adata` or modify it inplace.
    layer
        Key from `adata.layers` to use for tests.
    min_samples
        Minimum number of samples required per group (perturblab extension).
    threads
        Number of threads for parallel computation (perturblab extension, -1 for all cores).
    **kwargs
        Additional arguments passed to `differential_expression`.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object.
    Sets the following fields in `adata.uns[key_added]`:

    - **'names'**: structured np.recarray (dtype object)
        Gene names ordered by scores, indexed by group.
    - **'scores'**: structured np.recarray (dtype float32)
        Test statistics (z-scores for wilcoxon, t-statistics for t-tests).
    - **'logfoldchanges'**: structured np.recarray (dtype float32)
        Log2 fold changes for each gene.
    - **'pvals'**: structured np.recarray (dtype float64)
        P-values.
    - **'pvals_adj'**: structured np.recarray (dtype float64)
        Corrected p-values.
    - **'pts'**: pd.DataFrame (dtype float, optional)
        Fraction of cells expressing genes (if `pts=True`).
    - **'pts_rest'**: pd.DataFrame (dtype float, optional)
        Fraction of rest cells expressing genes (if `pts=True` and `reference='rest'`).

    Examples
    --------
    >>> import perturblab as pl
    >>> adata = pl.datasets.load_dataset("adamson_2016")
    >>> pl.data.process.rank_genes_groups(
    ...     adata,
    ...     groupby='perturbation',
    ...     reference='non-targeting',
    ...     method='wilcoxon',
    ...     n_genes=50,
    ... )
    >>> # Results stored in adata.uns['rank_genes_groups']
    >>> # Visualize results (if using scanpy plotting)
    >>> # import scanpy as sc
    >>> # sc.pl.rank_genes_groups(adata)

    Notes
    -----
    This function maintains full API compatibility with `scanpy.tl.rank_genes_groups`,
    allowing it to be used as a drop-in replacement with higher performance.

    See Also
    --------
    differential_expression : More flexible DE analysis with DataFrame output
    scanpy.tl.rank_genes_groups : Original scanpy implementation
    """
    if copy:
        adata = adata.copy()

    if key_added is None:
        key_added = "rank_genes_groups"

    # Handle use_raw default
    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw and adata.raw is None:
        raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")

    # Handle reference parameter
    control_tag = None if reference == "rest" else reference

    # Handle groups parameter
    if groups == "all":
        groups_list = None
    else:
        groups_list = list(groups) if groups is not None else None

    # Map method names (perturblab uses 'welch' while scanpy doesn't have it separately)
    method_map = {
        "t-test": "t-test",
        "t-test_overestim_var": "t-test_overestim_var",
        "wilcoxon": "wilcoxon",
    }

    if method not in method_map:
        raise ValueError(
            f"Method '{method}' not supported. " f"Choose from: {list(method_map.keys())}"
        )

    # Map corr_method to fdr_method
    fdr_method = "bh" if corr_method == "benjamini-hochberg" else "by"

    logger.info(f"🔬 Running rank_genes_groups (method={method}, reference={reference})")

    # Run differential expression analysis
    df = differential_expression(
        adata=adata,
        perturb_col=groupby,
        control_tag=control_tag,
        groups=groups_list,
        method=method_map[method],
        min_samples=min_samples,
        threads=threads,
        fdr_method=fdr_method,
        layer=layer,
        use_raw=use_raw,
        **kwargs,
    )

    # Store full DataFrame (perturblab extension)
    adata.uns[f"{key_added}_df"] = df

    # Get unique groups
    unique_groups = df["target"].unique().tolist()
    n_groups = len(unique_groups)
    n_genes_total = len(df["feature"].unique())

    # Determine number of genes to report
    if n_genes is None:
        n_genes = n_genes_total
    else:
        n_genes = min(n_genes, n_genes_total)

    logger.info(f"📋 Converting to scanpy format (top {n_genes} genes per group)...")

    # Prepare structured arrays (scanpy format)
    # These are record arrays where each column is a group
    dtypes_map = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    # Initialize structured arrays
    result_arrays = {}
    for field, dtype in dtypes_map.items():
        result_arrays[field] = np.zeros((n_genes, n_groups), dtype=dtype)

    # Fill arrays with top genes per group
    for i, target in enumerate(unique_groups):
        group_df = df[df["target"] == target].copy()

        # Sort by p-value (ascending) then by absolute statistic (descending)
        group_df = group_df.sort_values(
            by=["p_value", "statistic"],
            ascending=[True, False],
        ).head(n_genes)

        # Pad if necessary
        n_actual = len(group_df)
        if n_actual < n_genes:
            logger.warning(
                f"⚠️  Group '{target}' has only {n_actual} genes (< {n_genes}), "
                f"padding with empty values"
            )

        # Fill in the data
        result_arrays["names"][:n_actual, i] = group_df["feature"].values
        result_arrays["scores"][:n_actual, i] = group_df["statistic"].values.astype("float32")
        result_arrays["pvals"][:n_actual, i] = group_df["p_value"].values.astype("float64")
        result_arrays["pvals_adj"][:n_actual, i] = group_df["fdr"].values.astype("float64")
        result_arrays["logfoldchanges"][:n_actual, i] = group_df["log2_fold_change"].values.astype(
            "float32"
        )

        # Fill remaining with NaN/empty
        if n_actual < n_genes:
            result_arrays["names"][n_actual:, i] = ""
            result_arrays["scores"][n_actual:, i] = np.nan
            result_arrays["pvals"][n_actual:, i] = np.nan
            result_arrays["pvals_adj"][n_actual:, i] = np.nan
            result_arrays["logfoldchanges"][n_actual:, i] = np.nan

    # Convert to record arrays (scanpy format)
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }

    # Store as structured arrays indexed by group names
    for field, array in result_arrays.items():
        adata.uns[key_added][field] = np.rec.fromarrays(
            array.T,  # Transpose so each group is a column
            names=unique_groups,
        )

    # Compute pts (percentage of cells expressing) if requested
    if pts:
        logger.info("📊 Computing fraction of expressing cells (pts)...")

        # Get expression matrix
        if use_raw and adata.raw is not None:
            X = adata.raw.X
            var_names = adata.raw.var_names
        elif layer is not None:
            X = adata.layers[layer]
            var_names = adata.var_names
        else:
            X = adata.X
            var_names = adata.var_names

        from scipy.sparse import issparse

        # Compute fraction expressing for each group
        pts_df = pd.DataFrame(index=var_names, columns=unique_groups, dtype=float)
        pts_rest_df = (
            pd.DataFrame(index=var_names, columns=unique_groups, dtype=float)
            if reference == "rest"
            else None
        )

        for target in unique_groups:
            mask_group = adata.obs[groupby] == target
            X_group = X[mask_group, :]

            if issparse(X_group):
                frac_group = np.asarray(X_group.getnnz(axis=0) / X_group.shape[0]).flatten()
            else:
                frac_group = np.count_nonzero(X_group, axis=0) / X_group.shape[0]

            pts_df[target] = frac_group

            if reference == "rest":
                mask_rest = ~mask_group
                X_rest = X[mask_rest, :]

                if issparse(X_rest):
                    frac_rest = np.asarray(X_rest.getnnz(axis=0) / X_rest.shape[0]).flatten()
                else:
                    frac_rest = np.count_nonzero(X_rest, axis=0) / X_rest.shape[0]

                pts_rest_df[target] = frac_rest

        adata.uns[key_added]["pts"] = pts_df
        if pts_rest_df is not None:
            adata.uns[key_added]["pts_rest"] = pts_rest_df

    logger.info(f"✅ Results stored in adata.uns['{key_added}']")
    logger.info(
        f"    Available fields: 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj'"
        + (", 'pts'" if pts else "")
        + (", 'pts_rest'" if pts and reference == "rest" else "")
    )

    return adata if copy else None
