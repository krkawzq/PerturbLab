"""Type stubs for _mannwhitneyu Cython module."""

import numpy as np
from numpy.typing import NDArray

def group_mean_csc(
    data: NDArray,
    indices: NDArray,
    indptr: NDArray,
    group_id: NDArray[np.int32],
    n_groups: int,
    include_zeros: bool = True,
    threads: int = -1,
) -> NDArray[np.float64]: ...
def mannwhitneyu_csc(
    data: NDArray,
    indices: NDArray,
    indptr: NDArray,
    group_id: NDArray[np.int32],
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = True,
    threads: int = -1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
