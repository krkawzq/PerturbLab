"""Classification and clustering evaluation metrics.

This module provides metrics to evaluate cell type annotation, clustering quality,
and label transfer accuracy.

Migrated and adapted from scanpy.metrics.

Copyright Notice
----------------
The confusion_matrix function in this module is adapted from the scanpy package
(https://github.com/scverse/scanpy).

Original Copyright:
    Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

Modifications for PerturbLab:
    - Enhanced documentation with perturbation-specific examples
    - Improved error messages and validation
    - Consistent logging with PerturbLab style
    - Added comprehensive usage examples for label transfer evaluation

References
----------
scanpy: https://github.com/scverse/scanpy
Original implementation: scanpy/metrics/_metrics.py
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from natsort import natsorted
from pandas.api.types import CategoricalDtype
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "confusion_matrix",
]


def confusion_matrix(
    orig: pd.Series | np.ndarray | Sequence | str,
    new: pd.Series | np.ndarray | Sequence | str,
    data: pd.DataFrame | None = None,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Create a labeled confusion matrix comparing two sets of labels.
    
    This is useful for:
    - Comparing clustering results to ground truth cell types
    - Evaluating label transfer accuracy
    - Assessing annotation consistency across methods
    - Comparing predicted perturbation responses to actual responses
    
    Parameters
    ----------
    orig
        Original labels. Can be:
        - Array-like of labels
        - Column name in `data` DataFrame
    new
        New labels (predictions). Can be:
        - Array-like of labels
        - Column name in `data` DataFrame
    data
        Optional DataFrame to extract labels from.
    normalize
        If True, normalize confusion matrix by row (original labels).
        This shows the proportion of each original label assigned to each new label.
    
    Returns
    -------
    pd.DataFrame
        Confusion matrix with:
        - Rows: Original labels
        - Columns: New labels
        - Values: Counts (if normalize=False) or proportions (if normalize=True)
    
    Examples
    --------
    Compare clustering results to ground truth cell types:
    
    >>> import perturblab as pl
    >>> import pandas as pd
    >>> 
    >>> # From arrays
    >>> true_labels = ['T-cell', 'B-cell', 'T-cell', 'B-cell', 'T-cell']
    >>> pred_labels = ['T-cell', 'B-cell', 'T-cell', 'T-cell', 'T-cell']
    >>> cm = pl.metrics.confusion_matrix(true_labels, pred_labels, normalize=True)
    >>> print(cm)
    #              T-cell    B-cell
    # T-cell        1.00      0.00
    # B-cell        0.50      0.50
    
    From DataFrame:
    
    >>> obs = pd.DataFrame({
    ...     'cell_type': ['T-cell', 'B-cell', 'T-cell', 'B-cell'],
    ...     'cluster': ['C1', 'C2', 'C1', 'C2']
    ... })
    >>> cm = pl.metrics.confusion_matrix('cell_type', 'cluster', data=obs)
    
    Visualize with seaborn:
    
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> cm = pl.metrics.confusion_matrix(true_labels, pred_labels)
    >>> sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    >>> plt.title('Cell Type Annotation Accuracy')
    >>> plt.show()
    
    Evaluate label transfer:
    
    >>> # After label transfer from reference to query
    >>> cm = pl.metrics.confusion_matrix(
    ...     adata.obs['true_labels'],
    ...     adata.obs['transferred_labels'],
    ...     normalize=True
    ... )
    >>> # Diagonal values show correct transfer rate per cell type
    >>> transfer_accuracy = np.diag(cm).mean()
    >>> print(f"Average transfer accuracy: {transfer_accuracy:.2%}")
    
    Notes
    -----
    When normalize=True:
    - Each row sums to 1.0
    - Values show: "of all cells with original label X, what fraction were assigned new label Y?"
    - Diagonal values indicate correct classification rate
    - Off-diagonal values indicate confusion between labels
    
    When normalize=False:
    - Values are raw counts
    - Useful when original label frequencies are important
    
    See Also
    --------
    sklearn.metrics.confusion_matrix : Underlying implementation
    """
    # Extract labels from DataFrame if provided
    if data is not None:
        if isinstance(orig, str):
            if orig not in data.columns:
                raise KeyError(f"Column '{orig}' not found in data")
            orig = data[orig]
        if isinstance(new, str):
            if new not in data.columns:
                raise KeyError(f"Column '{new}' not found in data")
            new = data[new]
    
    # Coerce to pandas Series for consistent handling
    orig = pd.Series(orig) if not isinstance(orig, pd.Series) else orig
    new = pd.Series(new) if not isinstance(new, pd.Series) else new
    
    # Validate same length
    if len(orig) != len(new):
        raise ValueError(
            f"Original and new labels must have same length, "
            f"got {len(orig)} and {len(new)}"
        )
    
    # Get all unique labels (union of both label sets)
    unique_labels = pd.unique(np.concatenate((orig.values, new.values)))
    
    # Compute confusion matrix
    mtx = sklearn_confusion_matrix(orig, new, labels=unique_labels)
    
    # Normalize by row if requested
    if normalize:
        row_sums = mtx.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        mtx = np.divide(mtx, row_sums, where=row_sums != 0, out=mtx.astype(float))
    
    # Create labeled DataFrame
    orig_name = "Original labels" if orig.name is None else orig.name
    new_name = "New Labels" if new.name is None else new.name
    
    df = pd.DataFrame(
        mtx,
        index=pd.Index(unique_labels, name=orig_name),
        columns=pd.Index(unique_labels, name=new_name),
    )
    
    # Filter to only labels that appear in each set (respecting categorical order if present)
    if isinstance(orig.dtype, CategoricalDtype):
        orig_idx = orig.cat.categories
    else:
        orig_idx = natsorted(pd.unique(orig))
    
    if isinstance(new.dtype, CategoricalDtype):
        new_idx = new.cat.categories
    else:
        new_idx = natsorted(pd.unique(new))
    
    df = df.loc[np.array(orig_idx), np.array(new_idx)]
    
    return df

