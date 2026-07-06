"""
Benchmark module for RR interval entropy analysis.

Computes Sample Entropy, Transfer Entropy, and Entropy Rate over multiple scales.
Outputs results as CSV files for comparison with original compiled code.

Usage:
    from entropy_pure.benchmark_rr import run_benchmark, process_rr_array

    # Single array
    results = process_rr_array(rr_data, scales=[1, 2, 4, 8, 16])

    # Batch processing
    run_benchmark(
        rr_arrays={'file1': arr1, 'file2': arr2},
        scales=[1, 2, 4, 8, 16],
        output_file='results.csv'
    )
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import csv
import time
from dataclasses import dataclass, field, asdict

from .core import compute_entropy, compute_entropy_rate, compute_MI, compute_TE
from .others import compute_complexities
from .tools import compute_over_scales
from . import commons


@dataclass
class EntropyMetrics:
    """Container for entropy metrics computed over scales."""
    # Scale values
    scales: np.ndarray = field(default_factory=lambda: np.array([]))

    # Sample Entropy (SampEn)
    sampen_values: np.ndarray = field(default_factory=lambda: np.array([]))
    sampen_max: float = np.nan
    sampen_auc: float = np.nan
    sampen_mean: float = np.nan

    # Entropy Rate (h)
    entropy_rate_values: np.ndarray = field(default_factory=lambda: np.array([]))
    entropy_rate_max: float = np.nan
    entropy_rate_auc: float = np.nan
    entropy_rate_mean: float = np.nan

    # Transfer Entropy (TE) - for bivariate analysis
    te_values: np.ndarray = field(default_factory=lambda: np.array([]))
    te_max: float = np.nan
    te_auc: float = np.nan
    te_mean: float = np.nan

    # Reverse TE (for bidirectional analysis)
    te_reverse_values: np.ndarray = field(default_factory=lambda: np.array([]))
    te_reverse_max: float = np.nan
    te_reverse_auc: float = np.nan

    # Shannon Entropy
    entropy_values: np.ndarray = field(default_factory=lambda: np.array([]))
    entropy_max: float = np.nan
    entropy_auc: float = np.nan

    # Computation metadata
    file_id: str = ""
    n_points: int = 0
    computation_time: float = 0.0


def _compute_auc(values: np.ndarray, scales: np.ndarray) -> float:
    """Compute area under curve using trapezoidal rule."""
    valid = ~np.isnan(values)
    if np.sum(valid) < 2:
        return np.nan
    # Use trapezoid (NumPy >= 2.0) or trapz (legacy)
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return trapz_func(values[valid], scales[valid])


def _compute_max(values: np.ndarray) -> float:
    """Compute maximum ignoring NaN."""
    valid = ~np.isnan(values)
    if not np.any(valid):
        return np.nan
    return np.max(values[valid])


def _compute_mean(values: np.ndarray) -> float:
    """Compute mean ignoring NaN."""
    return np.nanmean(values)


def process_rr_array(
    rr: np.ndarray,
    rr2: Optional[np.ndarray] = None,
    scales: Union[List[int], np.ndarray] = None,
    m_sampen: int = 2,
    r_sampen: float = 0.2,
    m_entropy_rate: int = 2,
    m_te: int = 1,
    k: int = 5,
    N_eff: int = 4096,
    N_real: int = 3,
    Theiler: int = 4,
    compute_te: bool = False,
    file_id: str = "",
    verbose: bool = False
) -> EntropyMetrics:
    """
    Process a single RR interval array and compute entropy metrics.

    Parameters
    ----------
    rr : np.ndarray
        RR interval time series (1D array)
    rr2 : np.ndarray, optional
        Second RR series for Transfer Entropy (e.g., from different lead)
    scales : list or np.ndarray
        Time scales (stride values) to compute over
        Default: [1, 2, 4, 8, 16, 32]
    m_sampen : int
        Embedding dimension for Sample Entropy (default: 2)
    r_sampen : float
        Tolerance for Sample Entropy as fraction of std (default: 0.2)
    m_entropy_rate : int
        Embedding dimension for Entropy Rate (default: 2)
    m_te : int
        Embedding dimension for Transfer Entropy (default: 1)
    k : int
        Number of neighbors for k-NN estimators (default: 5)
    N_eff : int
        Effective number of points per realization (default: 4096)
    N_real : int
        Number of realizations (default: 3)
    Theiler : int
        Theiler window (-1 for auto)
    compute_te : bool
        Whether to compute Transfer Entropy (requires rr2)
    file_id : str
        Identifier for this file/array
    verbose : bool
        Print progress information

    Returns
    -------
    EntropyMetrics
        Dataclass containing all computed metrics
    """
    start_time = time.time()

    if scales is None:
        scales = np.array([1, 2, 4, 8, 16, 32])
    scales = np.asarray(scales)

    # Ensure 2D shape (1, n_pts)
    if rr.ndim == 1:
        rr = rr.reshape(1, -1)

    n_pts = rr.shape[1]

    # Set sampling parameters
    commons.set_sampling(Theiler=Theiler, N_eff=N_eff, N_real=N_real)

    metrics = EntropyMetrics(
        scales=scales,
        file_id=file_id,
        n_points=n_pts
    )

    # 1. Compute Sample Entropy over scales
    if verbose:
        print(f"  Computing SampEn (m={m_sampen}, r={r_sampen})...")

    sampen_results = np.zeros(len(scales))
    for i, tau in enumerate(scales):
        try:
            # Coarse-grain the signal at this scale
            if tau > 1:
                n_coarse = n_pts // tau
                rr_coarse = np.mean(rr[0, :n_coarse*tau].reshape(-1, tau), axis=1)
            else:
                rr_coarse = rr[0, :]

            # Compute SampEn at embedding m
            _, sampen = compute_complexities(
                rr_coarse, n_embed=m_sampen, stride=1, r=r_sampen,
                N_eff=N_eff, N_real=N_real
            )
            sampen_results[i] = sampen[m_sampen] if m_sampen < len(sampen) else np.nan
        except Exception as e:
            if verbose:
                print(f"    SampEn error at scale {tau}: {e}")
            sampen_results[i] = np.nan

    metrics.sampen_values = sampen_results
    metrics.sampen_max = _compute_max(sampen_results)
    metrics.sampen_auc = _compute_auc(sampen_results, scales)
    metrics.sampen_mean = _compute_mean(sampen_results)

    # 2. Compute Entropy Rate over scales
    if verbose:
        print(f"  Computing Entropy Rate (m={m_entropy_rate})...")

    try:
        h_values, _ = compute_over_scales(
            compute_entropy_rate, scales, rr,
            m=m_entropy_rate, k=k, N_eff=N_eff, N_real=N_real,
            verbosity_timing=0
        )
        metrics.entropy_rate_values = h_values
        metrics.entropy_rate_max = _compute_max(h_values)
        metrics.entropy_rate_auc = _compute_auc(h_values, scales)
        metrics.entropy_rate_mean = _compute_mean(h_values)
    except Exception as e:
        if verbose:
            print(f"    Entropy Rate error: {e}")
        metrics.entropy_rate_values = np.full(len(scales), np.nan)

    # 3. Compute Shannon Entropy over scales
    if verbose:
        print(f"  Computing Shannon Entropy...")

    try:
        entropy_values, _ = compute_over_scales(
            compute_entropy, scales, rr,
            n_embed=m_entropy_rate, k=k, N_eff=N_eff, N_real=N_real,
            verbosity_timing=0
        )
        metrics.entropy_values = entropy_values
        metrics.entropy_max = _compute_max(entropy_values)
        metrics.entropy_auc = _compute_auc(entropy_values, scales)
    except Exception as e:
        if verbose:
            print(f"    Entropy error: {e}")
        metrics.entropy_values = np.full(len(scales), np.nan)

    # 4. Compute Transfer Entropy (if bivariate data provided)
    if compute_te and rr2 is not None:
        if verbose:
            print(f"  Computing Transfer Entropy (m={m_te})...")

        if rr2.ndim == 1:
            rr2 = rr2.reshape(1, -1)

        try:
            # TE: rr -> rr2
            te_forward = np.zeros(len(scales))
            te_reverse = np.zeros(len(scales))

            for i, tau in enumerate(scales):
                try:
                    te_result = compute_TE(
                        rr, rr2, n_embed_x=m_te, n_embed_y=m_te,
                        stride=tau, k=k, N_eff=N_eff, N_real=N_real
                    )
                    te_forward[i] = te_result[0]  # Algorithm 1

                    # Reverse direction
                    te_result_rev = compute_TE(
                        rr2, rr, n_embed_x=m_te, n_embed_y=m_te,
                        stride=tau, k=k, N_eff=N_eff, N_real=N_real
                    )
                    te_reverse[i] = te_result_rev[0]
                except:
                    te_forward[i] = np.nan
                    te_reverse[i] = np.nan

            metrics.te_values = te_forward
            metrics.te_max = _compute_max(te_forward)
            metrics.te_auc = _compute_auc(te_forward, scales)
            metrics.te_mean = _compute_mean(te_forward)

            metrics.te_reverse_values = te_reverse
            metrics.te_reverse_max = _compute_max(te_reverse)
            metrics.te_reverse_auc = _compute_auc(te_reverse, scales)

        except Exception as e:
            if verbose:
                print(f"    TE error: {e}")
            metrics.te_values = np.full(len(scales), np.nan)
            metrics.te_reverse_values = np.full(len(scales), np.nan)

    metrics.computation_time = time.time() - start_time

    return metrics


def run_benchmark(
    rr_arrays: Dict[str, np.ndarray],
    rr2_arrays: Optional[Dict[str, np.ndarray]] = None,
    scales: Union[List[int], np.ndarray] = None,
    output_file: str = "entropy_results.csv",
    output_detail_file: Optional[str] = None,
    m_sampen: int = 2,
    r_sampen: float = 0.2,
    m_entropy_rate: int = 2,
    m_te: int = 1,
    k: int = 5,
    N_eff: int = 4096,
    N_real: int = 3,
    Theiler: int = 4,
    verbose: bool = True
) -> List[EntropyMetrics]:
    """
    Run benchmark on a batch of RR interval arrays.

    Parameters
    ----------
    rr_arrays : dict
        Dictionary mapping file_id -> numpy array
    rr2_arrays : dict, optional
        Dictionary mapping file_id -> second RR array for TE computation
    scales : list or np.ndarray
        Time scales to compute over (default: [1, 2, 4, 8, 16, 32])
    output_file : str
        Path to output CSV file with summary metrics
    output_detail_file : str, optional
        Path to detailed CSV with values at each scale
    m_sampen, r_sampen, m_entropy_rate, m_te, k, N_eff, N_real, Theiler
        Parameters for entropy computations
    verbose : bool
        Print progress

    Returns
    -------
    List[EntropyMetrics]
        List of metric results for each file
    """
    if scales is None:
        scales = np.array([1, 2, 4, 8, 16, 32])
    scales = np.asarray(scales)

    compute_te = rr2_arrays is not None

    results = []
    n_files = len(rr_arrays)

    if verbose:
        print(f"Processing {n_files} files...")
        print(f"Scales: {scales}")
        print(f"Parameters: m_sampen={m_sampen}, r={r_sampen}, m_h={m_entropy_rate}, k={k}")
        print(f"Sampling: N_eff={N_eff}, N_real={N_real}")
        print("-" * 60)

    start_total = time.time()

    for idx, (file_id, rr) in enumerate(rr_arrays.items()):
        if verbose:
            print(f"[{idx+1}/{n_files}] {file_id} ({rr.size} pts)")

        rr2 = rr2_arrays.get(file_id) if rr2_arrays else None

        metrics = process_rr_array(
            rr=rr,
            rr2=rr2,
            scales=scales,
            m_sampen=m_sampen,
            r_sampen=r_sampen,
            m_entropy_rate=m_entropy_rate,
            m_te=m_te,
            k=k,
            N_eff=N_eff,
            N_real=N_real,
            Theiler=Theiler,
            compute_te=compute_te,
            file_id=file_id,
            verbose=verbose
        )

        results.append(metrics)

        if verbose:
            print(f"    SampEn: max={metrics.sampen_max:.4f}, AUC={metrics.sampen_auc:.4f}")
            print(f"    h:      max={metrics.entropy_rate_max:.4f}, AUC={metrics.entropy_rate_auc:.4f}")
            if compute_te:
                print(f"    TE:     max={metrics.te_max:.4f}, AUC={metrics.te_auc:.4f}")
            print(f"    Time: {metrics.computation_time:.2f}s")

    total_time = time.time() - start_total

    if verbose:
        print("-" * 60)
        print(f"Total time: {total_time:.1f}s ({total_time/n_files:.2f}s/file)")

    # Write summary CSV
    _write_summary_csv(results, output_file, scales, verbose)

    # Write detailed CSV if requested
    if output_detail_file:
        _write_detail_csv(results, output_detail_file, scales, verbose)

    return results


def _write_summary_csv(
    results: List[EntropyMetrics],
    output_file: str,
    scales: np.ndarray,
    verbose: bool
):
    """Write summary metrics to CSV."""

    headers = [
        'file_id', 'n_points', 'computation_time',
        'sampen_max', 'sampen_auc', 'sampen_mean',
        'entropy_rate_max', 'entropy_rate_auc', 'entropy_rate_mean',
        'entropy_max', 'entropy_auc',
        'te_max', 'te_auc', 'te_mean',
        'te_reverse_max', 'te_reverse_auc'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header with scale info
        writer.writerow([f"# Scales: {[int(s) for s in scales]}"])
        writer.writerow(headers)

        for m in results:
            row = [
                m.file_id, m.n_points, f"{m.computation_time:.3f}",
                f"{m.sampen_max:.6f}", f"{m.sampen_auc:.6f}", f"{m.sampen_mean:.6f}",
                f"{m.entropy_rate_max:.6f}", f"{m.entropy_rate_auc:.6f}", f"{m.entropy_rate_mean:.6f}",
                f"{m.entropy_max:.6f}", f"{m.entropy_auc:.6f}",
                f"{m.te_max:.6f}", f"{m.te_auc:.6f}", f"{m.te_mean:.6f}",
                f"{m.te_reverse_max:.6f}", f"{m.te_reverse_auc:.6f}"
            ]
            writer.writerow(row)

    if verbose:
        print(f"\nSummary written to: {output_file}")


def _write_detail_csv(
    results: List[EntropyMetrics],
    output_file: str,
    scales: np.ndarray,
    verbose: bool
):
    """Write detailed per-scale values to CSV."""

    # Build headers
    base_cols = ['file_id', 'metric']
    scale_cols = [f"scale_{s}" for s in scales]
    headers = base_cols + scale_cols + ['max', 'auc', 'mean']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for m in results:
            # SampEn row
            sampen_row = [m.file_id, 'SampEn']
            sampen_row += [f"{v:.6f}" for v in m.sampen_values]
            sampen_row += [f"{m.sampen_max:.6f}", f"{m.sampen_auc:.6f}", f"{m.sampen_mean:.6f}"]
            writer.writerow(sampen_row)

            # Entropy Rate row
            h_row = [m.file_id, 'EntropyRate']
            h_row += [f"{v:.6f}" for v in m.entropy_rate_values]
            h_row += [f"{m.entropy_rate_max:.6f}", f"{m.entropy_rate_auc:.6f}", f"{m.entropy_rate_mean:.6f}"]
            writer.writerow(h_row)

            # Shannon Entropy row
            e_row = [m.file_id, 'Entropy']
            e_row += [f"{v:.6f}" for v in m.entropy_values]
            e_row += [f"{m.entropy_max:.6f}", f"{m.entropy_auc:.6f}", ""]
            writer.writerow(e_row)

            # TE row (if computed)
            if len(m.te_values) > 0 and not np.all(np.isnan(m.te_values)):
                te_row = [m.file_id, 'TE_forward']
                te_row += [f"{v:.6f}" for v in m.te_values]
                te_row += [f"{m.te_max:.6f}", f"{m.te_auc:.6f}", f"{m.te_mean:.6f}"]
                writer.writerow(te_row)

                te_rev_row = [m.file_id, 'TE_reverse']
                te_rev_row += [f"{v:.6f}" for v in m.te_reverse_values]
                te_rev_row += [f"{m.te_reverse_max:.6f}", f"{m.te_reverse_auc:.6f}", ""]
                writer.writerow(te_rev_row)

    if verbose:
        print(f"Details written to: {output_file}")


def load_rr_files(
    file_paths: List[str],
    column: int = 0,
    delimiter: str = None,
    skip_header: int = 0
) -> Dict[str, np.ndarray]:
    """
    Load RR interval files from disk.

    Parameters
    ----------
    file_paths : list of str
        Paths to RR files (text/csv with one RR value per line)
    column : int
        Column index to read (for multi-column files)
    delimiter : str
        Column delimiter (default: whitespace)
    skip_header : int
        Number of header lines to skip

    Returns
    -------
    dict
        Mapping from filename -> numpy array
    """
    rr_arrays = {}

    for path in file_paths:
        path = Path(path)
        try:
            data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
            if data.ndim > 1:
                data = data[:, column]
            rr_arrays[path.stem] = data
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    return rr_arrays


def generate_synthetic_rr(
    n_files: int = 10,
    n_points: int = 3000,
    mean_rr: float = 800,
    std_rr: float = 50,
    seed: int = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic RR interval data for testing.

    Creates RR series with realistic temporal correlations.

    Parameters
    ----------
    n_files : int
        Number of synthetic files to generate
    n_points : int
        Points per file
    mean_rr : float
        Mean RR interval in ms
    std_rr : float
        Standard deviation
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Mapping from file_id -> numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    rr_arrays = {}

    for i in range(n_files):
        # Generate correlated noise (1/f-like)
        white = np.random.randn(n_points)
        # Simple AR(1) process for temporal correlation
        rr = np.zeros(n_points)
        rr[0] = mean_rr + std_rr * white[0]
        phi = 0.9  # AR coefficient
        for t in range(1, n_points):
            rr[t] = mean_rr * (1 - phi) + phi * rr[t-1] + std_rr * np.sqrt(1 - phi**2) * white[t]

        rr_arrays[f"synthetic_{i+1:03d}"] = rr

    return rr_arrays


# Convenience function for quick testing
def quick_test(verbose: bool = True):
    """
    Run a quick test with synthetic data.

    Returns
    -------
    List[EntropyMetrics]
        Results from synthetic benchmark
    """
    if verbose:
        print("Generating 5 synthetic RR files (3000 pts each)...")

    rr_arrays = generate_synthetic_rr(n_files=5, n_points=3000, seed=42)

    results = run_benchmark(
        rr_arrays=rr_arrays,
        scales=[1, 2, 4, 8, 16],
        output_file='/tmp/entropy_test_summary.csv',
        output_detail_file='/tmp/entropy_test_detail.csv',
        N_eff=2048,
        N_real=2,
        verbose=verbose
    )

    return results


if __name__ == "__main__":
    quick_test()
