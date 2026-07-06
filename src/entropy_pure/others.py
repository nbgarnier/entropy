"""
Additional entropy measures: Renyi entropy, complexities, and surrogates.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma, digamma
from typing import Optional, Tuple, List, Union

from . import commons
from .core import _embed_data, _compute_increments, _sample_realizations


def _renyi_knn(x: np.ndarray, q: float, k: int = 5) -> float:
    """
    Compute Renyi entropy using k-NN.

    H_q = (1/(1-q)) * log(sum_i p_i^q)

    Using the k-NN estimator from Leonenko et al.

    Parameters
    ----------
    x : np.ndarray
        Data of shape (n_dims, n_pts)
    q : float
        Order of Renyi entropy (must not be 1)
    k : int
        Number of neighbors

    Returns
    -------
    float
        Renyi entropy estimate
    """
    if abs(q - 1.0) < 1e-10:
        raise ValueError("q must not be 1 for Renyi entropy (use Shannon instead)")

    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape

    if n_pts <= k:
        return np.nan

    # Build k-d tree
    tree = KDTree(x.T)

    # Find k+1 nearest neighbors
    distances, _ = tree.query(x.T, k=k+1, workers=-1)
    epsilon = distances[:, k]

    valid = epsilon > 0
    n_valid = np.sum(valid)

    if n_valid == 0:
        return np.nan

    # Volume of unit ball in d dimensions
    V_d = (np.pi ** (n_dims / 2)) / gamma(n_dims / 2 + 1)

    # Renyi entropy estimator
    # Using Leonenko et al. (2008) formula
    rho = epsilon[valid]

    # Estimate of p^{q-1} using k-NN density
    # p(x_i) ~ k / (n * V_d * rho^d)
    density_est = k / (n_valid * V_d * (2 * rho) ** n_dims)

    # For H_q = 1/(1-q) * log(integral p^q dx)
    # ~ 1/(1-q) * log(1/n * sum_i p(x_i)^{q-1})
    if q > 1:
        log_term = np.log(np.mean(density_est ** (q - 1)))
    else:
        # For q < 1, we need to be careful with the estimator
        log_term = np.log(np.mean(density_est ** (q - 1)))

    H_q = log_term / (1 - q)

    return H_q


def compute_entropy_Renyi(x: np.ndarray, q: float, inc_type: int = 0,
                          n_embed: int = 1, stride: int = 1,
                          Theiler: int = 0, N_eff: int = 0, N_real: int = 0,
                          k: int = 5, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Renyi entropy H_q of order q.

    H_q = (1/(1-q)) * log(integral p(x)^q dx)

    Parameters
    ----------
    x : np.ndarray
        Signal
    q : float
        Order of Renyi entropy (must not be 1)
    inc_type : int
        0 for entropy of signal, 1 for increments, 2 for averaged increments
    n_embed : int
        Embedding dimension
    stride : int
        Stride for embedding
    Theiler, N_eff, N_real, k, mask
        Same as compute_entropy

    Returns
    -------
    float
        Renyi entropy estimate
    """
    if abs(q - 1.0) < 1e-10:
        raise ValueError("you want Renyi entropy of order 1, please use Shannon entropy instead")

    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Apply mask
    if mask is not None and len(mask) > 1:
        valid_idx = np.where(mask > 0)[0]
        x = x[:, valid_idx]

    # Compute increments if requested
    if inc_type > 0:
        x = _compute_increments(x, order=1, stride=stride, inc_type=inc_type)
        n_embed = 1  # Already processed

    # Embed data
    if n_embed > 1:
        x = _embed_data(x, n_embed, stride)

    # Get defaults
    if Theiler == 0:
        Theiler = commons._samp_default['Theiler']
    if N_eff == 0:
        N_eff = commons._samp_default['N_eff']
    if N_real == 0:
        N_real = commons._samp_default['N_real']

    return _renyi_knn(x, q, k)


def compute_complexities_old(x: np.ndarray, n_embed: int = 1, stride: int = 1,
                             r: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ApEn and SampEn complexities (kernel estimates).
    OLD VERSION - NO ENHANCED SAMPLINGS

    Parameters
    ----------
    x : np.ndarray
        Signal
    n_embed : int
        Maximum embedding dimension
    stride : int
        Stride for embedding
    r : float
        Radius parameter (as fraction of std)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ApEn, SampEn) arrays of size n_embed+1
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape

    if n_dims > 1:
        raise ValueError("for now, x can only be with dim[0]=1")

    x = x.flatten()
    r_abs = r * np.std(x)

    ApEn = np.zeros(n_embed + 1)
    SampEn = np.zeros(n_embed + 1)

    for m in range(n_embed + 1):
        # Create embedded vectors
        N = n_pts - m * stride
        if N < 10:
            ApEn[m] = np.nan
            SampEn[m] = np.nan
            continue

        # Build embedded data
        X = np.zeros((m + 1, N))
        for i in range(N):
            for j in range(m + 1):
                X[j, i] = x[i + j * stride]

        # Count matches using Chebyshev distance
        C = np.zeros(N)
        C_self = np.zeros(N)  # Including self-match

        for i in range(N):
            # Count matches for point i
            count = 0
            count_self = 0
            for j in range(N):
                # Chebyshev distance
                d = np.max(np.abs(X[:, i] - X[:, j]))
                if d <= r_abs:
                    count_self += 1
                    if i != j:
                        count += 1
            C[i] = count
            C_self[i] = count_self

        # Phi values for ApEn and SampEn
        if m == 0:
            phi_m = np.mean(np.log(C_self / N))
            ApEn[0] = 0  # By definition
        else:
            # For SampEn: count of pairs where template of length m matches
            phi_m_samp = np.sum(C) / (N * (N - 1)) if N > 1 else 0

            if m >= 1 and phi_m_samp > 0:
                SampEn[m] = -np.log(phi_m_samp / phi_prev) if phi_prev > 0 else np.nan
            phi_prev = phi_m_samp

            # For ApEn
            phi_m_apen = np.mean(np.log(C_self / N)) if np.all(C_self > 0) else np.nan
            if m >= 1:
                ApEn[m] = phi_apen_prev - phi_m_apen if not np.isnan(phi_m_apen) else np.nan
            phi_apen_prev = phi_m_apen

        if m == 0:
            phi_prev = np.sum(C) / (N * (N - 1)) if N > 1 else 0
            phi_apen_prev = np.mean(np.log(C_self / N)) if np.all(C_self > 0) else np.nan

    return ApEn, SampEn


def compute_complexities(x: np.ndarray, n_embed: int = 1, stride: int = 1,
                         r: float = 0.2, Theiler: int = 0, N_eff: int = 0,
                         N_real: int = 0, mask: Optional[np.ndarray] = None,
                         do_correlation_integrals: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ApEn and SampEn complexities (kernel estimates).

    Parameters
    ----------
    x : np.ndarray
        Signal
    n_embed : int
        Maximum embedding dimension
    stride : int
        Stride for embedding
    r : float
        Radius parameter (as fraction of std)
    Theiler, N_eff, N_real, mask
        Sampling parameters
    do_correlation_integrals : bool
        If True, return correlation integrals instead

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ApEn, SampEn) or (Cd, logCd) if do_correlation_integrals
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape

    # Apply mask
    if mask is not None and len(mask) > 1:
        valid_idx = np.where(mask > 0)[0]
        x = x[:, valid_idx]
        n_pts = x.shape[1]

    x = x.flatten()
    r_abs = r * np.std(x)

    ApEn = np.zeros(n_embed + 1)
    SampEn = np.zeros(n_embed + 1)

    phi_m_prev = 1.0
    phi_apen_prev = 0.0

    for m in range(n_embed + 1):
        N = n_pts - m * stride
        if N < 10:
            ApEn[m] = np.nan
            SampEn[m] = np.nan
            continue

        # Build embedded data
        X = np.zeros((m + 1, N))
        for i in range(N):
            for j in range(m + 1):
                X[j, i] = x[i + j * stride]

        # Use KDTree for efficient neighbor counting with Chebyshev distance
        if m + 1 > 1:
            tree = KDTree(X.T)
        else:
            tree = KDTree(X.T.reshape(-1, 1))

        # Count matches for each point
        C = np.zeros(N)
        C_self = np.zeros(N)

        for i in range(N):
            if m + 1 > 1:
                query_pt = X[:, i]
            else:
                query_pt = X[:, i].reshape(1)

            # Count within radius r_abs (Chebyshev)
            count = tree.query_ball_point(query_pt, r_abs, p=np.inf, return_length=True)
            C_self[i] = count
            C[i] = count - 1  # Exclude self

        # Compute phi values
        # For ApEn
        valid_self = C_self > 0
        if np.any(valid_self):
            phi_m_apen = np.mean(np.log(C_self[valid_self] / N))
        else:
            phi_m_apen = np.nan

        # For SampEn
        total_pairs = np.sum(C)
        phi_m_samp = total_pairs / (N * (N - 1)) if N > 1 else 0

        if m == 0:
            ApEn[0] = 0
            SampEn[0] = 0
        else:
            # ApEn[m] = phi^m - phi^{m+1} (computed at next iteration)
            ApEn[m] = phi_apen_prev - phi_m_apen if not np.isnan(phi_m_apen) else np.nan

            # SampEn[m] = -log(phi^{m+1} / phi^m)
            if phi_m_samp > 0 and phi_m_prev > 0:
                SampEn[m] = -np.log(phi_m_samp / phi_m_prev)
            else:
                SampEn[m] = np.nan

        phi_m_prev = phi_m_samp
        phi_apen_prev = phi_m_apen

    if do_correlation_integrals:
        Cd = np.zeros(n_embed + 1)
        logCd = np.zeros(n_embed + 1)
        Cd[0] = np.exp(-SampEn[0]) if not np.isnan(SampEn[0]) else 1.0
        logCd[0] = -ApEn[0]
        for m in range(n_embed):
            if not np.isnan(SampEn[m + 1]):
                Cd[m + 1] = Cd[m] * np.exp(-SampEn[m + 1])
            else:
                Cd[m + 1] = np.nan
            logCd[m + 1] = logCd[m] - ApEn[m + 1]
        return Cd, logCd

    return ApEn, SampEn


def surrogate(x: np.ndarray, method: int = 0, N_steps: int = 7) -> np.ndarray:
    """
    Create a surrogate version of the data.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape (n_dims, n_pts)
    method : int
        Surrogate method:
        - 0: shuffle (random permutation in time)
        - 1: randomize phase with unwindowed FFT
        - 2: randomize phase with windowed FFT (buggy)
        - 3: randomize phase with Gaussian null hypothesis
        - 4: improved surrogate (same PDF and PSD)
        - 5: Gaussianize the signal
    N_steps : int
        Number of steps for method 4

    Returns
    -------
    np.ndarray
        Surrogate data with same shape as input
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dims, n_pts = x.shape
    z = x.copy()

    if method == 0:
        # Shuffle in time (preserve joint coordinates)
        perm = np.random.permutation(n_pts)
        z = z[:, perm]

    elif method == 1:
        # Randomize phase with FFT
        for d in range(n_dims):
            fft_x = np.fft.fft(z[d, :])
            magnitudes = np.abs(fft_x)
            phases = np.random.uniform(0, 2 * np.pi, n_pts)
            # Ensure conjugate symmetry for real output
            if n_pts % 2 == 0:
                phases[n_pts // 2] = 0
            phases[0] = 0
            phases[n_pts // 2 + 1:] = -phases[1:n_pts // 2][::-1]
            fft_new = magnitudes * np.exp(1j * phases)
            z[d, :] = np.real(np.fft.ifft(fft_new))

    elif method == 2:
        # Randomize phase with windowed FFT (simplified)
        window = np.hanning(n_pts)
        for d in range(n_dims):
            x_windowed = z[d, :] * window
            fft_x = np.fft.fft(x_windowed)
            magnitudes = np.abs(fft_x)
            phases = np.random.uniform(0, 2 * np.pi, n_pts)
            phases[0] = 0
            if n_pts % 2 == 0:
                phases[n_pts // 2] = 0
            phases[n_pts // 2 + 1:] = -phases[1:n_pts // 2][::-1]
            fft_new = magnitudes * np.exp(1j * phases)
            z[d, :] = np.real(np.fft.ifft(fft_new)) / window
            z[d, :] = np.nan_to_num(z[d, :], nan=0.0, posinf=0.0, neginf=0.0)

    elif method == 3:
        # AAFT (Amplitude Adjusted Fourier Transform)
        for d in range(n_dims):
            # Sort original data
            x_sorted = np.sort(z[d, :])
            # Create Gaussian surrogate
            gaussian = np.random.randn(n_pts)
            gaussian_sorted = np.sort(gaussian)
            # Rank transform
            ranks = np.argsort(np.argsort(z[d, :]))
            y = gaussian_sorted[ranks]
            # Phase randomize the Gaussian
            fft_y = np.fft.fft(y)
            magnitudes = np.abs(fft_y)
            phases = np.random.uniform(0, 2 * np.pi, n_pts)
            phases[0] = 0
            if n_pts % 2 == 0:
                phases[n_pts // 2] = 0
            phases[n_pts // 2 + 1:] = -phases[1:n_pts // 2][::-1]
            fft_new = magnitudes * np.exp(1j * phases)
            y_surr = np.real(np.fft.ifft(fft_new))
            # Rescale to original amplitude distribution
            ranks_surr = np.argsort(np.argsort(y_surr))
            z[d, :] = x_sorted[ranks_surr]

    elif method == 4:
        # Improved surrogate (iterative AAFT)
        for d in range(n_dims):
            x_sorted = np.sort(z[d, :])
            fft_orig = np.fft.fft(z[d, :])
            magnitudes_orig = np.abs(fft_orig)

            # Initialize with shuffled data
            y = z[d, np.random.permutation(n_pts)]

            for _ in range(N_steps):
                # Match spectrum
                fft_y = np.fft.fft(y)
                phases_y = np.angle(fft_y)
                fft_new = magnitudes_orig * np.exp(1j * phases_y)
                y = np.real(np.fft.ifft(fft_new))

                # Match amplitude distribution
                ranks = np.argsort(np.argsort(y))
                y = x_sorted[ranks]

            z[d, :] = y

    elif method == 5:
        # Gaussianize (same PSD and dependences)
        means = np.mean(z, axis=1)
        stds = np.std(z, axis=1)

        for d in range(n_dims):
            # Rank transform to uniform, then to Gaussian
            from scipy.stats import norm
            ranks = (np.argsort(np.argsort(z[d, :])) + 0.5) / n_pts
            z[d, :] = norm.ppf(ranks)
            # Restore mean and std
            z[d, :] = z[d, :] * stds[d] + means[d]

    return z
