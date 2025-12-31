"""
Masking utilities for handling missing data.
"""

import numpy as np
from typing import Union


def no_NaN(x: np.ndarray) -> bool:
    """
    Check if there are no NaN values in the data.

    Parameters
    ----------
    x : np.ndarray
        Data array

    Returns
    -------
    bool
        True if all values are finite, False otherwise
    """
    return np.isfinite(x).all()


def mask_NaN(x: np.ndarray) -> np.ndarray:
    """
    Create a mask corresponding to NaN values in the data.

    Parameters
    ----------
    x : np.ndarray
        Data array

    Returns
    -------
    np.ndarray
        Mask of type int8, where 1 indicates NaN
    """
    mask_nan = np.isnan(x).astype('i1')
    return mask_clean(mask_nan)


def mask_finite(x: np.ndarray) -> np.ndarray:
    """
    Create a mask corresponding to finite (valid) values in the data.

    Parameters
    ----------
    x : np.ndarray
        Data array

    Returns
    -------
    np.ndarray
        Mask of type int8, where 1 indicates finite value
    """
    mask = np.isfinite(x).astype('i1')
    return mask_clean(mask)


def mask_clean(x: np.ndarray) -> np.ndarray:
    """
    Make any nd-array a compatible mask for the code.

    At any given time t, if the mask has a True value in one dimension,
    then the resulting mask will also have a True value at that time t (AND logic).

    Parameters
    ----------
    x : np.ndarray
        Mask array

    Returns
    -------
    np.ndarray
        1D mask of type int8
    """
    y = x.astype('i1')
    if y.ndim > 1:
        if y.shape[0] > y.shape[1]:
            y = y.T
        # Apply AND logic across dimensions
        for j in range(y.shape[0] - 1):
            y[0, :] = y[0, :] * y[j + 1, :]
        y = y[0, :]
    return y.flatten()


def retain_from_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract data points corresponding to valid mask values.

    Parameters
    ----------
    x : np.ndarray
        Data array
    mask : np.ndarray
        Mask array

    Returns
    -------
    np.ndarray
        Filtered data containing only points where mask > 0
    """
    y = np.array(mask).astype('i1')
    ind = np.where(y > 0)[0]
    out = np.array(x)[:, ind].copy()
    return out
