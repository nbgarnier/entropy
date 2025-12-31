# entropy_pure - Pure Python implementation of entropy estimation
# Based on k-NN algorithms from Kraskov, Stogbauer, Grassberger (2004)
#
# This module provides efficient Python implementations without requiring
# compilation of C/C++ code.

from .core import (
    compute_entropy,
    compute_entropy_rate,
    compute_MI,
    compute_TE,
    compute_PMI,
    compute_PTE,
    compute_DI,
    compute_relative_entropy,
    compute_entropy_increments,
    compute_regularity_index,
)

from .commons import (
    get_last_info,
    get_extra_info,
    set_sampling,
    get_sampling,
    set_Theiler,
    choose_algorithm,
    multithreading,
)

from .tools import (
    reorder,
    embed,
    crop,
    compute_over_scales,
)

from .masks import (
    mask_finite,
    mask_NaN,
    mask_clean,
    retain_from_mask,
)

from .others import (
    compute_entropy_Renyi,
    compute_complexities,
    surrogate,
)

__version__ = "4.2.0-pure"
__all__ = [
    # Core functions
    "compute_entropy",
    "compute_entropy_rate",
    "compute_MI",
    "compute_TE",
    "compute_PMI",
    "compute_PTE",
    "compute_DI",
    "compute_relative_entropy",
    "compute_entropy_increments",
    "compute_regularity_index",
    # Commons
    "get_last_info",
    "get_extra_info",
    "set_sampling",
    "get_sampling",
    "set_Theiler",
    "choose_algorithm",
    "multithreading",
    # Tools
    "reorder",
    "embed",
    "crop",
    "compute_over_scales",
    # Masks
    "mask_finite",
    "mask_NaN",
    "mask_clean",
    "retain_from_mask",
    # Others
    "compute_entropy_Renyi",
    "compute_complexities",
    "surrogate",
]
