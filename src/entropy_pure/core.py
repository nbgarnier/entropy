"""
Core entropy computation functions using k-NN algorithms.

This module implements:
- Shannon entropy (Kozachenko-Leonenko / Grassberger estimator)
- Mutual Information (Kraskov-Stogbauer-Grassberger algorithms)
- Transfer Entropy
- Partial Mutual Information
- Partial Transfer Entropy
- Directed Information
- Relative Entropy (KL divergence)

References:
- Kozachenko, L.F., Leonenko, N.N. (1987)
- Kraskov, A., Stogbauer, H., Grassberger, P. (2004) PRE 69, 066138
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Union
import warnings

from . import commons


def _embed_data(x: np.ndarray, n_embed: int, stride: int,
                offset: int = 0, n_embed_max: int = -1) -> np.ndarray:
    """
    Causal time-embedding of data.

    Parameters
    ----------
    x : np.ndarray
        Data array of shape (n_dims, n_pts)
    n_embed : int
        Embedding dimension
    stride : int
        Time lag between embedded points
    offset : int
        Starting offset
    n_embed_max : int
        Maximum embedding dimension for time reference

    Returns
    -------
    np.ndarray
        Embedded data of shape (n_dims * n_embed, n_pts_new)
    """
    if n_embed_max < 0:
        n_embed_max = n_embed

    n_dims, n_pts = x.shape
    n_pts_new = (n_pts - stride * (n_embed_max - 1) - offset) // stride
    if n_pts_new <= 0:
        raise ValueError("Not enough points for embedding with given parameters")

    n_total = n_dims * n_embed
    x_new = np.zeros((n_total, n_pts_new), dtype=np.float64)

    for i in range(n_pts_new):
        t = offset + i * stride + stride * (n_embed_max - 1)
        for d in range(n_dims):
            for l in range(n_embed):
                x_new[d + l * n_dims, i] = x[d, t - l * stride]

    return x_new


def _compute_increments(x: np.ndarray, order: int, stride: int,
                        inc_type: int = 1) -> np.ndarray:
    """
    Compute increments of the signal.

    Parameters
    ----------
    x : np.ndarray
        Data array of shape (n_dims, n_pts)
    order : int
        Order of increments
    stride : int
        Time lag
    inc_type : int
        1 for regular increments, 2 for averaged increments

    Returns
    -------
    np.ndarray
        Increments
    """
    n_dims, n_pts = x.shape
    n_pts_new = n_pts - order * stride

    if n_pts_new <= 0:
        raise ValueError("Not enough points for increments")

    result = np.zeros((n_dims, n_pts_new), dtype=np.float64)

    if inc_type == 1:
        # Regular increments
        for d in range(n_dims):
            for i in range(n_pts_new):
                t = i + order * stride
                if order == 1:
                    result[d, i] = x[d, t] - x[d, t - stride]
                else:
                    # Higher order increments (difference of differences)
                    delta = x[d, t]
                    sign = 1
                    for k in range(1, order + 1):
                        sign *= -1
                        coeff = 1
                        for j in range(k):
                            coeff *= (order - j) / (j + 1)
                        delta += sign * coeff * x[d, t - k * stride]
                    result[d, i] = delta
    else:
        # Averaged increments
        for d in range(n_dims):
            for i in range(n_pts_new):
                t = i + order * stride
                avg = 0.0
                for k in range(stride):
                    avg += x[d, t - k]
                avg /= stride
                result[d, i] = x[d, t] - avg

    return result


def _entropy_knn(x: np.ndarray, k: int = 5) -> float:
    """
    Compute Shannon entropy using k-NN (Kozachenko-Leonenko estimator).

    Uses the formula:
    H = n * mean(log(2 * epsilon)) + psi(N) - psi(k)

    where epsilon is the distance to the k-th nearest neighbor.

    Parameters
    ----------
    x : np.ndarray
        Data array of shape (n_dims, n_pts) or (n_pts,) for 1D
    k : int
        Number of neighbors

    Returns
    -------
    float
        Entropy estimate in nats
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape

    if n_pts <= k:
        warnings.warn(f"Not enough points ({n_pts}) for k={k} neighbors")
        return np.nan

    # Build k-d tree (transpose so points are rows)
    tree = KDTree(x.T)

    # Find k+1 nearest neighbors (including self)
    distances, _ = tree.query(x.T, k=k+1, workers=-1)

    # Get distance to k-th neighbor (excluding self which is at distance 0)
    epsilon = distances[:, k]

    # Filter out zero distances
    valid = epsilon > 0
    n_valid = np.sum(valid)

    if n_valid == 0:
        return np.nan

    # Kozachenko-Leonenko estimator
    # H = d * mean(log(2*epsilon)) + psi(N) - psi(k)
    h = n_dims * np.mean(np.log(2.0 * epsilon[valid]))
    h += digamma(n_valid) - digamma(k)

    # Store info for get_last_info
    commons._last_info['n_errors'] = n_pts - n_valid
    commons._last_info['n_eff_local'] = n_valid

    return h


def _mi_knn(x: np.ndarray, y: np.ndarray, k: int = 5,
            algo: int = 1) -> Tuple[float, float]:
    """
    Compute Mutual Information using k-NN (Kraskov-Stogbauer-Grassberger).

    Parameters
    ----------
    x : np.ndarray
        First variable, shape (n_dims_x, n_pts)
    y : np.ndarray
        Second variable, shape (n_dims_y, n_pts)
    k : int
        Number of neighbors
    algo : int
        Algorithm (1 or 2 from KSG paper)

    Returns
    -------
    Tuple[float, float]
        MI estimates from algo 1 and algo 2
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n_x, n_pts = x.shape
    n_y = y.shape[0]

    if y.shape[1] != n_pts:
        raise ValueError("x and y must have same number of points")

    if n_pts <= k:
        return np.nan, np.nan

    # Joint space
    z = np.vstack([x, y])

    # Build trees
    tree_z = KDTree(z.T)
    tree_x = KDTree(x.T)
    tree_y = KDTree(y.T)

    # Find k-th neighbor distances in joint space
    dist_z, _ = tree_z.query(z.T, k=k+1, workers=-1)
    epsilon = dist_z[:, k]  # Distance to k-th neighbor (L-inf norm approximated by L2)

    # For KSG, we use Chebyshev (L-infinity) distance
    # scipy KDTree with p=np.inf
    tree_z_inf = KDTree(z.T)
    tree_x_inf = KDTree(x.T)
    tree_y_inf = KDTree(y.T)

    # Query with infinity norm
    dist_z_inf, _ = tree_z_inf.query(z.T, k=k+1, p=np.inf, workers=-1)
    epsilon = dist_z_inf[:, k]

    # Count neighbors in marginal spaces
    n_x_arr = np.zeros(n_pts, dtype=np.int32)
    n_y_arr = np.zeros(n_pts, dtype=np.int32)

    for i in range(n_pts):
        eps = epsilon[i]
        if eps > 0:
            # Count points within epsilon in each marginal (strict inequality for algo 1)
            n_x_arr[i] = tree_x_inf.query_ball_point(x.T[i], eps - 1e-10, p=np.inf, return_length=True) - 1
            n_y_arr[i] = tree_y_inf.query_ball_point(y.T[i], eps - 1e-10, p=np.inf, return_length=True) - 1

    valid = epsilon > 0
    n_valid = np.sum(valid)

    if n_valid == 0:
        return np.nan, np.nan

    # KSG Algorithm 1:
    # I = psi(k) - mean(psi(n_x+1) + psi(n_y+1)) + psi(N)
    I1 = digamma(k) - np.mean(digamma(n_x_arr[valid] + 1) + digamma(n_y_arr[valid] + 1)) + digamma(n_valid)

    # KSG Algorithm 2 (using epsilon/2):
    n_x_arr2 = np.zeros(n_pts, dtype=np.int32)
    n_y_arr2 = np.zeros(n_pts, dtype=np.int32)

    for i in range(n_pts):
        eps = epsilon[i]
        if eps > 0:
            # Use epsilon (not strict) for algo 2
            n_x_arr2[i] = tree_x_inf.query_ball_point(x.T[i], eps, p=np.inf, return_length=True) - 1
            n_y_arr2[i] = tree_y_inf.query_ball_point(y.T[i], eps, p=np.inf, return_length=True) - 1

    # KSG Algorithm 2:
    # I = psi(k) - 1/k - mean(psi(n_x) + psi(n_y)) + psi(N)
    I2 = digamma(k) - 1.0/k - np.mean(digamma(n_x_arr2[valid]) + digamma(n_y_arr2[valid])) + digamma(n_valid)

    commons._last_info['n_errors'] = n_pts - n_valid
    commons._last_info['n_eff_local'] = n_valid

    return I1, I2


def _cmi_knn(x: np.ndarray, y: np.ndarray, z: np.ndarray,
             k: int = 5) -> Tuple[float, float]:
    """
    Compute Conditional Mutual Information I(X;Y|Z) using k-NN.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    z : np.ndarray
        Conditioning variable
    k : int
        Number of neighbors

    Returns
    -------
    Tuple[float, float]
        CMI estimates
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    n_pts = x.shape[1]

    if n_pts <= k:
        return np.nan, np.nan

    # Joint spaces
    xyz = np.vstack([x, y, z])
    xz = np.vstack([x, z])
    yz = np.vstack([y, z])

    # Build trees with infinity norm
    tree_xyz = KDTree(xyz.T)
    tree_xz = KDTree(xz.T)
    tree_yz = KDTree(yz.T)
    tree_z = KDTree(z.T)

    # Find k-th neighbor in full joint space
    dist_xyz, _ = tree_xyz.query(xyz.T, k=k+1, p=np.inf, workers=-1)
    epsilon = dist_xyz[:, k]

    # Count neighbors in conditional spaces
    n_xz = np.zeros(n_pts, dtype=np.int32)
    n_yz = np.zeros(n_pts, dtype=np.int32)
    n_z = np.zeros(n_pts, dtype=np.int32)

    for i in range(n_pts):
        eps = epsilon[i]
        if eps > 0:
            n_xz[i] = tree_xz.query_ball_point(xz.T[i], eps - 1e-10, p=np.inf, return_length=True) - 1
            n_yz[i] = tree_yz.query_ball_point(yz.T[i], eps - 1e-10, p=np.inf, return_length=True) - 1
            n_z[i] = tree_z.query_ball_point(z.T[i], eps - 1e-10, p=np.inf, return_length=True) - 1

    valid = (epsilon > 0) & (n_xz > 0) & (n_yz > 0) & (n_z > 0)
    n_valid = np.sum(valid)

    if n_valid == 0:
        return np.nan, np.nan

    # I(X;Y|Z) = psi(k) - mean(psi(n_xz+1) + psi(n_yz+1) - psi(n_z+1))
    I1 = digamma(k) - np.mean(digamma(n_xz[valid] + 1) + digamma(n_yz[valid] + 1) - digamma(n_z[valid] + 1))
    I2 = I1  # Same formula for both algorithms in conditional case

    commons._last_info['n_errors'] = n_pts - n_valid

    return I1, I2


def _gaussian_entropy(x: np.ndarray) -> float:
    """
    Compute entropy assuming Gaussian distribution.

    H = 0.5 * log((2*pi*e)^d * det(Cov))

    Parameters
    ----------
    x : np.ndarray
        Data of shape (n_dims, n_pts)

    Returns
    -------
    float
        Entropy estimate
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape

    if n_pts <= n_dims:
        return np.nan

    cov = np.cov(x)
    if n_dims == 1:
        det = float(cov)
    else:
        det = np.linalg.det(cov)

    if det <= 0:
        return np.nan

    h = 0.5 * n_dims * (1 + np.log(2 * np.pi)) + 0.5 * np.log(det)
    return h


def _gaussian_mi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute MI assuming Gaussian distribution.

    Parameters
    ----------
    x, y : np.ndarray
        Data arrays

    Returns
    -------
    float
        MI estimate
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    H_x = _gaussian_entropy(x)
    H_y = _gaussian_entropy(y)
    H_xy = _gaussian_entropy(np.vstack([x, y]))

    return H_x + H_y - H_xy


def _sample_realizations(n_pts: int, n_embed: int, stride: int,
                         theiler: int, n_eff: int, n_real: int,
                         theiler_type: int = 4) -> List[Tuple[int, np.ndarray]]:
    """
    Generate sampling parameters for multiple realizations.

    Parameters
    ----------
    n_pts : int
        Number of points
    n_embed : int
        Embedding dimension
    stride : int
        Stride for embedding
    theiler : int
        Theiler window
    n_eff : int
        Effective number of points per realization
    n_real : int
        Number of realizations
    theiler_type : int
        Type of Theiler prescription (1-4)

    Returns
    -------
    List[Tuple[int, np.ndarray]]
        List of (offset, indices) for each realization
    """
    # Compute available points after embedding
    pts_offset = stride * (n_embed - 1)
    n_available = n_pts - pts_offset

    if n_available <= 0:
        return []

    # Handle automatic Theiler
    if theiler < 0:
        theiler = stride

    # Compute maximum number of points per realization
    n_eff_max = n_available // theiler
    if n_eff_max < 1:
        n_eff_max = n_available
        theiler = 1

    # Adjust n_eff
    if n_eff <= 0 or n_eff > n_eff_max:
        n_eff = n_eff_max

    # Compute maximum number of realizations
    n_real_max = stride
    if n_real <= 0 or n_real > n_real_max:
        n_real = min(n_real_max, stride)

    realizations = []

    if theiler_type <= 2:
        # Uniform sampling
        for r in range(n_real):
            offset = r
            indices = np.arange(n_eff) * theiler
            realizations.append((offset, indices))
    else:
        # Random sampling
        for r in range(n_real):
            offset = r
            indices = np.sort(np.random.choice(n_eff_max, size=min(n_eff, n_eff_max), replace=False)) * theiler
            realizations.append((offset, indices))

    # Store sampling info
    commons._last_samp['type'] = theiler_type
    commons._last_samp['Theiler'] = theiler
    commons._last_samp['N_eff'] = n_eff
    commons._last_samp['N_real'] = n_real

    return realizations


def compute_entropy(x: np.ndarray, n_embed: int = 1, stride: int = 1,
                    Theiler: int = 0, N_eff: int = 0, N_real: int = 0,
                    k: int = 5, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Shannon entropy H of a signal using k-NN.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_dims, n_pts) or (n_pts,)
    n_embed : int
        Embedding dimension (default=1)
    stride : int
        Stride for embedding (default=1)
    Theiler : int
        Theiler scale. If 0, uses default. If <0, automatic.
    N_eff : int
        Effective number of points (default=4096 or max available)
    N_real : int
        Number of realizations (default=10)
    k : int
        Number of neighbors. If -1, uses Gaussian approximation.
    mask : np.ndarray, optional
        Boolean mask for valid points

    Returns
    -------
    float
        Entropy estimate in nats
    """
    # Ensure 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape

    # Apply mask if provided
    if mask is not None and len(mask) > 1:
        valid_idx = np.where(mask > 0)[0]
        x = x[:, valid_idx]
        n_pts = x.shape[1]

    # Get default sampling parameters
    if Theiler == 0:
        Theiler = commons._samp_default['Theiler']
    if N_eff == 0:
        N_eff = commons._samp_default['N_eff']
    if N_real == 0:
        N_real = commons._samp_default['N_real']

    # Gaussian approximation
    if k == -1:
        if n_embed > 1:
            x_emb = _embed_data(x, n_embed, stride)
        else:
            x_emb = x
        return _gaussian_entropy(x_emb)

    # Generate realizations
    realizations = _sample_realizations(n_pts, n_embed, stride, Theiler, N_eff, N_real,
                                        commons._samp_default['type'])

    if not realizations:
        return np.nan

    # Compute entropy for each realization
    results = []
    total_errors = 0
    total_pts = 0
    data_stds = []

    for offset, indices in realizations:
        # Extract and embed data for this realization
        pts_offset = stride * (n_embed - 1)
        selected_pts = pts_offset + offset + indices
        selected_pts = selected_pts[selected_pts < n_pts]

        if len(selected_pts) < 2 * k:
            continue

        # Build embedded data
        n_total = n_dims * n_embed
        x_real = np.zeros((n_total, len(selected_pts)), dtype=np.float64)

        for i, t in enumerate(selected_pts):
            for d in range(n_dims):
                for l in range(n_embed):
                    x_real[d + l * n_dims, i] = x[d, t - l * stride]

        # Compute std of the data
        data_stds.append(np.std(x_real))

        # Compute entropy
        H = _entropy_knn(x_real, k)
        if not np.isnan(H):
            results.append(H)
            total_errors += commons._last_info.get('n_errors', 0)
            total_pts += commons._last_info.get('n_eff_local', 0)

    if not results:
        return np.nan

    # Average over realizations
    avg = np.mean(results)
    std = np.std(results) if len(results) > 1 else 0.0

    # Store info
    commons._last_info['std'] = std
    commons._last_info['std2'] = 0.0
    commons._last_info['n_errors'] = total_errors
    commons._last_info['n_eff_local'] = total_pts // len(results) if results else 0
    commons._last_info['n_eff'] = total_pts
    commons._last_info['n_real'] = len(results)

    commons._extra_info['data_std'] = np.mean(data_stds) if data_stds else 0.0
    commons._extra_info['data_std_std'] = np.std(data_stds) if len(data_stds) > 1 else 0.0

    return avg


def compute_entropy_increments(x: np.ndarray, inc_type: int = 1, order: int = 1,
                               stride: int = 1, Theiler: int = 0, N_eff: int = 0,
                               N_real: int = 0, k: int = 5,
                               mask: Optional[np.ndarray] = None) -> float:
    """
    Compute entropy of the increments of a signal.

    Parameters
    ----------
    x : np.ndarray
        Signal
    inc_type : int
        1 for regular increments, 2 for averaged increments
    order : int
        Order of increments
    stride : int
        Stride for increments
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    float
        Entropy estimate of the increments
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Compute increments
    x_inc = _compute_increments(x, order, stride, inc_type)

    # Compute entropy of increments
    return compute_entropy(x_inc, n_embed=1, stride=1, Theiler=Theiler,
                          N_eff=N_eff, N_real=N_real, k=k, mask=mask)


def compute_entropy_rate(x: np.ndarray, method: int = 2, m: int = 1,
                         stride: int = 1, Theiler: int = 0, N_eff: int = 0,
                         N_real: int = 0, k: int = 5,
                         mask: Optional[np.ndarray] = None) -> float:
    """
    Compute entropy rate of a signal.

    Parameters
    ----------
    x : np.ndarray
        Signal
    method : int
        0: H(m)/m, 1: H(m+1)-H(m), 2: H(1)-MI(x,x^(m))
    m : int
        Embedding dimension
    stride : int
        Stride for embedding
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    float
        Entropy rate estimate
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if method == 0:
        # H(m) / m
        H_m = compute_entropy(x, n_embed=m, stride=stride, Theiler=Theiler,
                             N_eff=N_eff, N_real=N_real, k=k, mask=mask)
        return H_m / m

    elif method == 1:
        # H(m+1) - H(m)
        H_m = compute_entropy(x, n_embed=m, stride=stride, Theiler=Theiler,
                             N_eff=N_eff, N_real=N_real, k=k, mask=mask)
        H_m1 = compute_entropy(x, n_embed=m+1, stride=stride, Theiler=Theiler,
                              N_eff=N_eff, N_real=N_real, k=k, mask=mask)
        return H_m1 - H_m

    else:  # method == 2
        # H(1) - MI(x_t, x_past)
        H_1 = compute_entropy(x, n_embed=1, stride=stride, Theiler=Theiler,
                             N_eff=N_eff, N_real=N_real, k=k, mask=mask)

        # Embed for past
        if m > 1:
            x_past = _embed_data(x, m, stride)
            # Current point
            offset = stride * m
            x_curr = x[:, offset:]
            n_pts = min(x_curr.shape[1], x_past.shape[1])
            x_curr = x_curr[:, :n_pts]
            x_past = x_past[:, :n_pts]

            MI_vals = compute_MI(x_curr, x_past, n_embed_x=1, n_embed_y=1, stride=1,
                                Theiler=Theiler, N_eff=N_eff, N_real=N_real, k=k)
            MI = MI_vals[0]
        else:
            MI = 0.0

        return H_1 - MI


def compute_MI(x: np.ndarray, y: np.ndarray, n_embed_x: int = 1,
               n_embed_y: int = 1, stride: int = 1, Theiler: int = 0,
               N_eff: int = 0, N_real: int = 0, k: int = 5,
               mask: Optional[np.ndarray] = None) -> List[float]:
    """
    Compute Mutual Information MI(x, y).

    Parameters
    ----------
    x, y : np.ndarray
        Signals
    n_embed_x, n_embed_y : int
        Embedding dimensions
    stride : int
        Stride for embedding
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    List[float]
        [MI_algo1, MI_algo2]
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n_pts = min(x.shape[1], y.shape[1])
    x = x[:, :n_pts]
    y = y[:, :n_pts]

    # Get defaults
    if Theiler == 0:
        Theiler = commons._samp_default['Theiler']
    if N_eff == 0:
        N_eff = commons._samp_default['N_eff']
    if N_real == 0:
        N_real = commons._samp_default['N_real']

    # Gaussian approximation
    if k == -1:
        if n_embed_x > 1:
            x_emb = _embed_data(x, n_embed_x, stride)
        else:
            x_emb = x
        if n_embed_y > 1:
            y_emb = _embed_data(y, n_embed_y, stride)
        else:
            y_emb = y
        n_pts_emb = min(x_emb.shape[1], y_emb.shape[1])
        return [_gaussian_mi(x_emb[:, :n_pts_emb], y_emb[:, :n_pts_emb]), np.nan]

    n_embed_max = max(n_embed_x, n_embed_y)
    realizations = _sample_realizations(n_pts, n_embed_max, stride, Theiler, N_eff, N_real,
                                        commons._samp_default['type'])

    if not realizations:
        return [np.nan, np.nan]

    results1 = []
    results2 = []

    for offset, indices in realizations:
        pts_offset = stride * (n_embed_max - 1)
        selected_pts = pts_offset + offset + indices
        selected_pts = selected_pts[selected_pts < n_pts]

        if len(selected_pts) < 2 * k:
            continue

        # Embed x
        nx = x.shape[0]
        n_x_total = nx * n_embed_x
        x_real = np.zeros((n_x_total, len(selected_pts)), dtype=np.float64)
        for i, t in enumerate(selected_pts):
            for d in range(nx):
                for l in range(n_embed_x):
                    t_idx = t - (n_embed_max - n_embed_x) * stride - l * stride
                    if 0 <= t_idx < n_pts:
                        x_real[d + l * nx, i] = x[d, t_idx]

        # Embed y
        ny = y.shape[0]
        n_y_total = ny * n_embed_y
        y_real = np.zeros((n_y_total, len(selected_pts)), dtype=np.float64)
        for i, t in enumerate(selected_pts):
            for d in range(ny):
                for l in range(n_embed_y):
                    t_idx = t - (n_embed_max - n_embed_y) * stride - l * stride
                    if 0 <= t_idx < n_pts:
                        y_real[d + l * ny, i] = y[d, t_idx]

        I1, I2 = _mi_knn(x_real, y_real, k)
        if not np.isnan(I1):
            results1.append(I1)
        if not np.isnan(I2):
            results2.append(I2)

    avg1 = np.mean(results1) if results1 else np.nan
    avg2 = np.mean(results2) if results2 else np.nan

    commons._last_info['std'] = np.std(results1) if len(results1) > 1 else 0.0
    commons._last_info['std2'] = np.std(results2) if len(results2) > 1 else 0.0
    commons._last_info['n_real'] = len(results1)

    return [avg1, avg2]


def compute_TE(x: np.ndarray, y: np.ndarray, n_embed_x: int = 1,
               n_embed_y: int = 1, stride: int = 1, Theiler: int = 0,
               N_eff: int = 0, N_real: int = 0, lag: int = 1, k: int = 5,
               mask: Optional[np.ndarray] = None) -> List[float]:
    """
    Compute Transfer Entropy TE(x -> y).

    TE(X->Y) = I(Y_future; X_past | Y_past)

    Parameters
    ----------
    x, y : np.ndarray
        Signals
    n_embed_x, n_embed_y : int
        Embedding dimensions
    stride : int
        Stride for embedding
    lag : int
        Lag for future point
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    List[float]
        [TE_algo1, TE_algo2]
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n_pts = min(x.shape[1], y.shape[1])
    x = x[:, :n_pts]
    y = y[:, :n_pts]

    if Theiler == 0:
        Theiler = commons._samp_default['Theiler']
    if N_eff == 0:
        N_eff = commons._samp_default['N_eff']
    if N_real == 0:
        N_real = commons._samp_default['N_real']

    # Gaussian approximation
    if k == -1:
        # TE = H(Y_future | Y_past) - H(Y_future | X_past, Y_past)
        # Using Gaussian formula
        offset = stride * max(n_embed_x, n_embed_y) + lag
        n_pts_te = n_pts - offset

        if n_pts_te < 10:
            return [np.nan, np.nan]

        # Y_future
        y_fut = y[:, offset:offset+n_pts_te]
        # Y_past (embedded)
        y_past = _embed_data(y, n_embed_y, stride, n_embed_max=max(n_embed_x, n_embed_y))
        y_past = y_past[:, lag:lag+n_pts_te]
        # X_past (embedded)
        x_past = _embed_data(x, n_embed_x, stride, n_embed_max=max(n_embed_x, n_embed_y))
        x_past = x_past[:, lag:lag+n_pts_te]

        # H(Y_fut | Y_past) = H(Y_fut, Y_past) - H(Y_past)
        H_yfy = _gaussian_entropy(np.vstack([y_fut, y_past]))
        H_y = _gaussian_entropy(y_past)

        # H(Y_fut | X_past, Y_past)
        H_yfxy = _gaussian_entropy(np.vstack([y_fut, x_past, y_past]))
        H_xy = _gaussian_entropy(np.vstack([x_past, y_past]))

        TE = (H_yfy - H_y) - (H_yfxy - H_xy)
        return [TE, np.nan]

    # k-NN based TE
    n_embed_max = max(n_embed_x, n_embed_y)
    offset = stride * n_embed_max + lag
    n_pts_te = n_pts - offset

    if n_pts_te < 2 * k:
        return [np.nan, np.nan]

    # Build data matrices
    # Y_future
    y_fut = y[:, offset:offset+n_pts_te]
    # Y_past (embedded)
    ny = y.shape[0]
    y_past = np.zeros((ny * n_embed_y, n_pts_te), dtype=np.float64)
    for i in range(n_pts_te):
        t = offset + i
        for d in range(ny):
            for l in range(n_embed_y):
                y_past[d + l * ny, i] = y[d, t - lag - l * stride]

    # X_past (embedded)
    nx = x.shape[0]
    x_past = np.zeros((nx * n_embed_x, n_pts_te), dtype=np.float64)
    for i in range(n_pts_te):
        t = offset + i
        for d in range(nx):
            for l in range(n_embed_x):
                x_past[d + l * nx, i] = x[d, t - lag - l * stride]

    # TE = I(Y_fut; X_past | Y_past)
    I1, I2 = _cmi_knn(y_fut, x_past, y_past, k)

    commons._last_info['n_real'] = 1

    return [I1, I2]


def compute_PMI(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                n_embed_x: int = 1, n_embed_y: int = 1, n_embed_z: int = 1,
                stride: int = 1, Theiler: int = 0, N_eff: int = 0,
                N_real: int = 0, k: int = 5,
                mask: Optional[np.ndarray] = None) -> List[float]:
    """
    Compute Partial Mutual Information PMI = MI(x, y | z).

    Parameters
    ----------
    x, y, z : np.ndarray
        Signals (z is conditioning variable)
    n_embed_x, n_embed_y, n_embed_z : int
        Embedding dimensions
    stride : int
        Stride for embedding
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    List[float]
        [PMI_algo1, PMI_algo2]
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    n_pts = min(x.shape[1], y.shape[1], z.shape[1])
    x = x[:, :n_pts]
    y = y[:, :n_pts]
    z = z[:, :n_pts]

    if k == -1:
        # Gaussian approximation
        # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        if n_embed_x > 1:
            x = _embed_data(x, n_embed_x, stride)
        if n_embed_y > 1:
            y = _embed_data(y, n_embed_y, stride)
        if n_embed_z > 1:
            z = _embed_data(z, n_embed_z, stride)
        n_pts = min(x.shape[1], y.shape[1], z.shape[1])

        MI_xyz = _gaussian_mi(x[:, :n_pts], np.vstack([y[:, :n_pts], z[:, :n_pts]]))
        MI_xz = _gaussian_mi(x[:, :n_pts], z[:, :n_pts])
        return [MI_xyz - MI_xz, np.nan]

    # Embed data
    n_embed_max = max(n_embed_x, n_embed_y, n_embed_z)
    offset = stride * (n_embed_max - 1)
    n_pts_eff = n_pts - offset

    if n_pts_eff < 2 * k:
        return [np.nan, np.nan]

    # Build embedded data
    nx, ny, nz = x.shape[0], y.shape[0], z.shape[0]
    x_emb = np.zeros((nx * n_embed_x, n_pts_eff), dtype=np.float64)
    y_emb = np.zeros((ny * n_embed_y, n_pts_eff), dtype=np.float64)
    z_emb = np.zeros((nz * n_embed_z, n_pts_eff), dtype=np.float64)

    for i in range(n_pts_eff):
        t = offset + i
        for d in range(nx):
            for l in range(n_embed_x):
                x_emb[d + l * nx, i] = x[d, t - l * stride]
        for d in range(ny):
            for l in range(n_embed_y):
                y_emb[d + l * ny, i] = y[d, t - l * stride]
        for d in range(nz):
            for l in range(n_embed_z):
                z_emb[d + l * nz, i] = z[d, t - l * stride]

    I1, I2 = _cmi_knn(x_emb, y_emb, z_emb, k)

    return [I1, I2]


def compute_PTE(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                n_embed_x: int = 1, n_embed_y: int = 1, n_embed_z: int = 1,
                stride: int = 1, lag: int = 1, Theiler: int = 0,
                N_eff: int = 0, N_real: int = 0, k: int = 5,
                mask: Optional[np.ndarray] = None) -> List[float]:
    """
    Compute Partial Transfer Entropy PTE = TE(x -> y | z).

    Parameters
    ----------
    x, y, z : np.ndarray
        Signals (z is conditioning variable)
    n_embed_x, n_embed_y, n_embed_z : int
        Embedding dimensions
    stride : int
        Stride for embedding
    lag : int
        Lag for future point
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    List[float]
        [PTE_algo1, PTE_algo2]
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    n_pts = min(x.shape[1], y.shape[1], z.shape[1])
    n_embed_max = max(n_embed_x, n_embed_y, n_embed_z)
    offset = stride * n_embed_max + lag
    n_pts_eff = n_pts - offset

    if n_pts_eff < 2 * k:
        return [np.nan, np.nan]

    # Y_future
    y_fut = y[:, offset:offset+n_pts_eff]

    # Condition: Y_past, Z_past
    ny, nz, nx = y.shape[0], z.shape[0], x.shape[0]
    cond_dim = ny * n_embed_y + nz * n_embed_z
    cond = np.zeros((cond_dim, n_pts_eff), dtype=np.float64)

    for i in range(n_pts_eff):
        t = offset + i
        idx = 0
        for d in range(ny):
            for l in range(n_embed_y):
                cond[idx, i] = y[d, t - lag - l * stride]
                idx += 1
        for d in range(nz):
            for l in range(n_embed_z):
                cond[idx, i] = z[d, t - lag - l * stride]
                idx += 1

    # X_past
    x_past = np.zeros((nx * n_embed_x, n_pts_eff), dtype=np.float64)
    for i in range(n_pts_eff):
        t = offset + i
        for d in range(nx):
            for l in range(n_embed_x):
                x_past[d + l * nx, i] = x[d, t - lag - l * stride]

    # PTE = I(Y_fut; X_past | Y_past, Z_past)
    I1, I2 = _cmi_knn(y_fut, x_past, cond, k)

    return [I1, I2]


def compute_DI(x: np.ndarray, y: np.ndarray, N: int, stride: int = 1,
               Theiler: int = 0, N_eff: int = 0, N_real: int = 0,
               k: int = 5, mask: Optional[np.ndarray] = None) -> List[float]:
    """
    Compute Directed Information DI(x -> y).

    DI(X->Y) = sum_{t=1}^{N} I(Y_t; X^{t-1} | Y^{t-1})

    Parameters
    ----------
    x, y : np.ndarray
        Signals
    N : int
        Embedding dimension for both
    stride : int
        Stride for embedding
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    List[float]
        [DI_algo1, DI_algo2]
    """
    # DI is approximated by TE with embedding N
    return compute_TE(x, y, n_embed_x=N, n_embed_y=N, stride=stride,
                     Theiler=Theiler, N_eff=N_eff, N_real=N_real,
                     lag=stride, k=k, mask=mask)


def compute_relative_entropy(x: np.ndarray, y: np.ndarray, n_embed_x: int = 1,
                             n_embed_y: int = 1, stride: int = 1,
                             Theiler: int = 0, N_eff: int = 0, N_real: int = 0,
                             k: int = 5, do_KLdiv: int = 1) -> float:
    """
    Compute relative entropy (KL divergence) between two distributions.

    Parameters
    ----------
    x, y : np.ndarray
        Signals from two distributions
    n_embed_x, n_embed_y : int
        Embedding dimensions
    stride : int
        Stride for embedding
    do_KLdiv : int
        1 for KL divergence, 0 for cross-entropy
    Theiler, N_eff, N_real, k
        Same as compute_entropy

    Returns
    -------
    float
        KL divergence or cross-entropy estimate
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # Embed data
    if n_embed_x > 1:
        x = _embed_data(x, n_embed_x, stride)
    if n_embed_y > 1:
        y = _embed_data(y, n_embed_y, stride)

    n_x, n_pts_x = x.shape
    n_y, n_pts_y = y.shape

    if n_x != n_y:
        raise ValueError("x and y must have same dimensionality after embedding")

    n_dims = n_x

    if k == -1:
        # Gaussian approximation
        # KL(P||Q) = 0.5 * (log(det(Σ_q)/det(Σ_p)) + tr(Σ_q^{-1}Σ_p) + (μ_q-μ_p)^T Σ_q^{-1} (μ_q-μ_p) - d)
        mu_x = np.mean(x, axis=1)
        mu_y = np.mean(y, axis=1)
        cov_x = np.cov(x)
        cov_y = np.cov(y)

        if n_dims == 1:
            cov_x = np.array([[cov_x]])
            cov_y = np.array([[cov_y]])

        try:
            inv_cov_y = np.linalg.inv(cov_y)
            det_x = np.linalg.det(cov_x)
            det_y = np.linalg.det(cov_y)

            if det_x <= 0 or det_y <= 0:
                return np.nan

            diff = mu_y - mu_x
            KL = 0.5 * (np.log(det_y / det_x) + np.trace(inv_cov_y @ cov_x)
                       + diff @ inv_cov_y @ diff - n_dims)
        except np.linalg.LinAlgError:
            return np.nan

        if do_KLdiv == 1:
            return KL
        else:
            H_x = _gaussian_entropy(x)
            return H_x + KL

    # k-NN based KL divergence
    # D_KL(P||Q) ≈ (d/n) * sum_i log(ν_k(x_i) / ρ_k(x_i)) + log(m/(n-1))
    # where ρ_k is distance to k-th neighbor in P, ν_k is distance to k-th neighbor in Q

    tree_x = KDTree(x.T)
    tree_y = KDTree(y.T)

    # Distance to k-th neighbor in x (from x)
    dist_x, _ = tree_x.query(x.T, k=k+1, workers=-1)
    rho = dist_x[:, k]

    # Distance to k-th neighbor in y (from x)
    dist_y, _ = tree_y.query(x.T, k=k, workers=-1)
    nu = dist_y[:, k-1]

    valid = (rho > 0) & (nu > 0)
    n_valid = np.sum(valid)

    if n_valid == 0:
        return np.nan

    # KL divergence estimate
    KL = n_dims * np.mean(np.log(nu[valid] / rho[valid])) + np.log(n_pts_y / (n_pts_x - 1))

    if do_KLdiv == 1:
        return KL
    else:
        # Cross-entropy = H(P) + KL(P||Q)
        H_x = _entropy_knn(x, k)
        return H_x + KL


def compute_regularity_index(x: np.ndarray, stride: int = 1, Theiler: int = 0,
                             N_eff: int = 0, N_real: int = 0, k: int = 5,
                             mask: Optional[np.ndarray] = None) -> List[float]:
    """
    Compute regularity index.

    Δ(x, τ) = H(δ_τ x) - h^{(τ)}(x)

    Parameters
    ----------
    x : np.ndarray
        Signal
    stride : int
        Stride (time scale)
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    List[float]
        [regularity_algo1, regularity_algo2]
    """
    # H(increments)
    H_inc = compute_entropy_increments(x, inc_type=1, order=1, stride=stride,
                                       Theiler=Theiler, N_eff=N_eff, N_real=N_real,
                                       k=k, mask=mask)

    # Entropy rate
    h_rate = compute_entropy_rate(x, method=2, m=1, stride=stride,
                                  Theiler=Theiler, N_eff=N_eff, N_real=N_real,
                                  k=k, mask=mask)

    delta = H_inc - h_rate
    return [delta, delta]
