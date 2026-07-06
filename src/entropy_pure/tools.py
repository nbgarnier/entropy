"""
Data processing utilities for entropy computation.
"""

import numpy as np
from typing import Optional, Tuple, Callable, Any
from time import time

from . import commons


def reorder(x: np.ndarray) -> np.ndarray:
    """
    Make any nd-array compatible with entropy functions.

    Ensures time dimension is the second dimension and data is C-contiguous.

    Parameters
    ----------
    x : np.ndarray
        Any nd-array

    Returns
    -------
    np.ndarray
        Well-aligned and ordered nd-array
    """
    if x.ndim == 1:
        x = x.reshape((1, x.size))
    elif x.ndim == 2:
        if x.shape[0] > x.shape[1]:
            x = x.T

    if x.flags['C_CONTIGUOUS']:
        return x
    else:
        return x.copy()


def reorder_2d(x: np.ndarray, nx: int = -1, ny: int = -1, d: int = -1) -> np.ndarray:
    """
    Make any nd-array compatible with 2D entropy functions (images).

    Parameters
    ----------
    x : np.ndarray
        Any nd-array
    nx, ny : int
        Image dimensions (optional)
    d : int
        Number of components (optional)

    Returns
    -------
    np.ndarray
        Well-aligned nd-array
    """
    if x.ndim == 1:
        if nx > 0 and ny > 0:
            x = np.reshape(x, (nx, -1), order='F')
    elif x.ndim == 2:
        pass  # Nothing to do
    elif x.ndim == 3:
        x = np.reshape(x, (x.shape[0], x.shape[1]), order='F')
    else:
        print("order > 3 not supported")

    if x.flags['C_CONTIGUOUS']:
        return x
    else:
        return x.copy()


def embed_python(x: np.ndarray, m: int = 1, stride: int = 1,
                 i_window: int = 0) -> np.ndarray:
    """
    Time-embed an nd-array (Python reference implementation).

    Note: This function is for testing purposes. Use embed() instead.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_dims, n_pts)
    m : int
        Embedding dimension
    stride : int
        Distance between successive points
    i_window : int
        Which window to return (0 <= i_window < stride)

    Returns
    -------
    np.ndarray
        Embedded data
    """
    x = reorder(x)
    npts = x.shape[1]
    mx = x.shape[0]

    npts_new = (npts - npts % stride) // stride - (m - 1)
    n_windows = stride

    if i_window >= n_windows:
        raise ValueError("i_window must be less than stride!")

    n = mx * m
    x_new = np.zeros((n, npts_new))

    for i in range(npts_new):
        for d in range(mx):
            for l in range(m):
                x_new[d + l * mx, i] = x[d, i_window + n_windows * i + stride * (m - 1 - l)]

    return x_new


def embed(x: np.ndarray, n_embed: int = 1, stride: int = 1,
          i_window: int = 0, n_embed_max: int = -1) -> np.ndarray:
    """
    Causal time-embed an nd-array.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_dims, n_pts)
    n_embed : int
        Embedding dimension
    stride : int
        Distance between successive points
    i_window : int
        Which window to return (0 <= i_window < stride)
    n_embed_max : int
        Maximum embedding dimension for time reference (-1 uses n_embed)

    Returns
    -------
    np.ndarray
        Embedded data of shape (n_dims * n_embed, n_pts_new)
    """
    x = reorder(x)
    npts = x.shape[1]
    nx = x.shape[0]

    if npts < nx:
        raise ValueError("please transpose x")
    if i_window >= stride:
        raise ValueError("i_window must be smaller than stride")

    if n_embed_max <= 0:
        n_embed_max = n_embed

    # Number of points after embedding
    nb_pts_new = (npts - stride * (n_embed_max - 1) - i_window) // stride
    if nb_pts_new <= 0:
        raise ValueError("Not enough points for embedding")

    output = np.zeros((nx * n_embed, nb_pts_new), dtype=np.float64)

    for i in range(nb_pts_new):
        t = i_window + i * stride + stride * (n_embed_max - 1)
        for d in range(nx):
            for l in range(n_embed):
                t_idx = t - l * stride - stride * (n_embed_max - n_embed)
                output[d + l * nx, i] = x[d, t_idx]

    return output


def crop(x: np.ndarray, npts_new: int, i_window: int = 0) -> np.ndarray:
    """
    Crop an nd-array in time.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_dims, n_pts)
    npts_new : int
        New size in time
    i_window : int
        Starting point in time

    Returns
    -------
    np.ndarray
        Cropped data
    """
    x = reorder(x)
    npts = x.shape[1]
    nx = x.shape[0]

    if npts < nx:
        raise ValueError("please transpose x")
    if i_window + npts_new > npts:
        raise ValueError(f"i_window ({i_window}) + npts_new ({npts_new}) is larger than npts ({npts})")

    return x[:, i_window:i_window + npts_new].copy()


def FIR_LP(x: np.ndarray, tau: int = 2, fr: float = 1.0) -> np.ndarray:
    """
    Low-pass filter a signal using local averaging.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_dims, n_pts)
    tau : int
        Time-scale for local averaging
    fr : float
        Subsampling factor (keeps fr points per tau interval)

    Returns
    -------
    np.ndarray
        Low-pass filtered signal
    """
    x = reorder(x)
    npts = x.shape[1]
    nx = x.shape[0]

    if npts < nx:
        raise ValueError("please transpose x")
    if tau < 1:
        raise ValueError("tau must be positive")

    npts_new = int(npts * fr // tau)

    output = np.zeros((nx, npts_new), dtype=np.float64)

    for d in range(nx):
        for i in range(npts_new):
            start = int(i * tau / fr)
            end = min(start + tau, npts)
            output[d, i] = np.mean(x[d, start:end])

    return output


def compute_over_scales(func: Callable, tau_set: np.ndarray, *args,
                        verbosity_timing: int = 1, get_samplings: int = 0,
                        get_extra: int = 0, **kwargs) -> Tuple:
    """
    Run an estimation over a range of time-scales.

    Parameters
    ----------
    func : Callable
        Function to run (e.g., compute_MI)
    tau_set : np.ndarray
        Array of stride values
    verbosity_timing : int
        0 for no output, 1-2 for progress dots/details
    get_samplings : int
        If 1, return sampling parameters
    get_extra : int
        If 1, return std of data
    *args, **kwargs
        Arguments to pass to func

    Returns
    -------
    Tuple
        (results, std) and optionally (samplings, input_std)

    Example
    -------
    >>> tau_set = np.arange(1, 20)
    >>> MI, MI_std = compute_over_scales(compute_MI, tau_set, x, y, N_eff=1973)
    """
    # Test call to determine output shape
    test = func(*args, stride=int(tau_set[0]), **kwargs)

    if isinstance(test, list):
        N_results = len(test)
        test_2 = test[0]
        if isinstance(test_2, np.ndarray):
            res = np.zeros((test_2.size, N_results, len(tau_set)), dtype=float)
        else:
            res = np.zeros((N_results, len(tau_set)), dtype=float)
    else:
        res = np.zeros(len(tau_set), dtype=float)

    res_std = np.zeros(tau_set.shape, dtype=float)
    if get_samplings:
        samplings = np.zeros((7, len(tau_set)), dtype=float)
    if get_extra:
        input_std = np.zeros((2, len(tau_set)), dtype=float)

    for i, tau in enumerate(tau_set):
        if verbosity_timing > 1:
            print(f"{i+1} / {len(tau_set)} tau = {tau}", end='')
        elif verbosity_timing > 0:
            print(".", end='', flush=True)

        time1 = time()
        result = func(*args, stride=int(tau), **kwargs)
        res[..., i] = np.atleast_2d(result)
        res_std[i] = commons.get_last_info()[0]

        if get_samplings:
            samplings[..., i] = commons.get_last_sampling()
        if get_extra:
            input_std[..., i] = commons.get_extra_info()

        time2 = time()
        if verbosity_timing > 2:
            print(f" -> {res[..., i]}\t elapsed time: {time2-time1:.3f}")
        if verbosity_timing > 1:
            print()

    if verbosity_timing == 1:
        print()

    if get_extra:
        if get_samplings:
            return res, res_std, samplings, input_std
        else:
            return res, res_std, input_std
    else:
        if get_samplings:
            return res, res_std, samplings
        else:
            return res, res_std
