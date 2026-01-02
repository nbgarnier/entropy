# entropy_pure

Pure Python implementation of information-theoretic measures for time series analysis.

This is a refactored version of the [entropy](https://github.com/nbgarnier/entropy) library that runs entirely in Python without requiring compilation of C/C++ code.

## Features

### Core Entropy Measures
- **Shannon Entropy** - Kozachenko-Leonenko / Grassberger k-NN estimator
- **Mutual Information** - Kraskov-Stogbauer-Grassberger algorithms (KSG1 & KSG2)
- **Transfer Entropy** - Information flow between time series
- **Partial Mutual Information** - Conditional MI
- **Partial Transfer Entropy** - Conditional TE
- **Directed Information** - Causal information measures
- **Relative Entropy** - KL divergence
- **Entropy Rate** - Rate of information generation

### Complexity Measures
- **Sample Entropy (SampEn)** - Regularity measure
- **Approximate Entropy (ApEn)** - Complexity measure
- **Renyi Entropy** - Generalized entropy of order q

### Analysis Tools
- **Multi-scale analysis** - Compute measures over time scales
- **RR interval benchmarking** - Batch processing for cardiac data
- **Surrogate generation** - Shuffle, phase randomization, AAFT

## Installation

```bash
# From source
cd src/entropy_pure
pip install .

# Or install in development mode
pip install -e .
```

## Dependencies

- numpy >= 1.20
- scipy >= 1.7

## Quick Start

```python
import numpy as np
from entropy_pure import compute_entropy, compute_MI, compute_TE, compute_entropy_rate

# Generate some data
np.random.seed(42)
x = np.random.randn(1, 10000)  # Shape: (n_dims, n_points)
y = np.random.randn(1, 10000)

# Shannon entropy
H = compute_entropy(x, n_embed=1, stride=1, k=5)
print(f"Entropy: {H:.4f}")

# Entropy rate
h = compute_entropy_rate(x, m=2, k=5)
print(f"Entropy rate: {h:.4f}")

# Mutual Information
MI = compute_MI(x, y, k=5)
print(f"Mutual Information: {MI[0]:.4f}")

# Transfer Entropy (with causal relationship)
z = np.zeros((1, 10000))
z[0, 1:] = 0.7 * x[0, :-1] + 0.3 * np.random.randn(9999)
TE = compute_TE(x, z, n_embed_x=1, n_embed_y=1, lag=1, k=5)
print(f"Transfer Entropy: {TE[0]:.4f}")
```

## Multi-Scale Analysis

Compute entropy measures over multiple time scales:

```python
import numpy as np
from entropy_pure import compute_entropy_rate, compute_over_scales, set_sampling

# Configure sampling
set_sampling(Theiler=4, N_eff=4096, N_real=3)

# Generate data
x = np.random.randn(1, 50000)

# Define scales
scales = np.array([1, 2, 4, 8, 16, 32])

# Compute entropy rate over scales
h_values, h_std = compute_over_scales(
    compute_entropy_rate, scales, x,
    m=2, k=5, verbosity_timing=0
)

print("Scale | Entropy Rate")
for s, h in zip(scales, h_values):
    print(f"  {s:3d} | {h:.4f}")
```

## RR Interval / ECG Analysis

The `benchmark_rr` module provides specialized tools for heart rate variability analysis:

### Single File Processing

```python
import numpy as np
from entropy_pure import process_rr_array

# Load RR intervals (in ms)
rr = np.loadtxt('rr_intervals.txt')

# Compute entropy metrics
metrics = process_rr_array(
    rr,
    scales=[1, 2, 4, 8, 16, 32],
    m_sampen=2,        # Embedding dimension for SampEn
    r_sampen=0.2,      # Tolerance (fraction of std)
    m_entropy_rate=2,  # Embedding for entropy rate
    k=5,               # Number of neighbors
    N_eff=4096,        # Points per realization
    N_real=3           # Number of realizations
)

# Access results
print(f"Sample Entropy - max: {metrics.sampen_max:.4f}, AUC: {metrics.sampen_auc:.4f}")
print(f"Entropy Rate   - max: {metrics.entropy_rate_max:.4f}, AUC: {metrics.entropy_rate_auc:.4f}")
```

### Batch Processing with CSV Output

```python
from entropy_pure import run_benchmark, load_rr_files

# Option 1: Load from files
rr_arrays = load_rr_files(['patient_001.txt', 'patient_002.txt'])

# Option 2: Pass numpy arrays directly
rr_arrays = {
    'patient_001': np.load('patient_001_rr.npy'),
    'patient_002': np.load('patient_002_rr.npy'),
    'patient_003': rr_data_array,  # Any numpy array
}

# Run benchmark
results = run_benchmark(
    rr_arrays=rr_arrays,
    scales=[1, 2, 4, 8, 16, 32],
    output_file='results_summary.csv',
    output_detail_file='results_detail.csv',
    m_sampen=2,
    r_sampen=0.2,
    m_entropy_rate=2,
    k=5,
    N_eff=4096,
    N_real=3,
    verbose=True
)
```

### Transfer Entropy (Bivariate Analysis)

```python
from entropy_pure import run_benchmark

# Two signals per subject (e.g., different ECG leads)
rr1_arrays = {'subj1': rr_lead1, 'subj2': rr_lead1_2}
rr2_arrays = {'subj1': rr_lead2, 'subj2': rr_lead2_2}

results = run_benchmark(
    rr_arrays=rr1_arrays,
    rr2_arrays=rr2_arrays,  # Enables TE computation
    output_file='te_results.csv'
)

# Results include TE in both directions
for r in results:
    print(f"{r.file_id}: TE_fwd={r.te_max:.4f}, TE_rev={r.te_reverse_max:.4f}")
```

### CSV Output Format

**Summary CSV** (`results_summary.csv`):
```
file_id,n_points,computation_time,sampen_max,sampen_auc,sampen_mean,entropy_rate_max,entropy_rate_auc,...
patient_001,3000,0.73,2.06,28.83,1.75,5.28,78.49,...
patient_002,3000,0.71,2.16,29.61,1.81,5.32,79.07,...
```

**Detail CSV** (`results_detail.csv`):
```
file_id,metric,scale_1,scale_2,scale_4,scale_8,scale_16,max,auc,mean
patient_001,SampEn,1.37,1.53,1.75,2.06,2.05,2.06,28.83,1.75
patient_001,EntropyRate,4.48,5.28,5.24,5.25,5.28,5.28,78.49,5.10
```

### Synthetic Data for Testing

```python
from entropy_pure import generate_synthetic_rr, run_benchmark

# Generate realistic synthetic RR data
rr_arrays = generate_synthetic_rr(
    n_files=10,
    n_points=3000,
    mean_rr=800,  # ms
    std_rr=50,
    seed=42
)

# Run benchmark
results = run_benchmark(rr_arrays, output_file='synthetic_test.csv')
```

## Sample Entropy and Complexity

```python
from entropy_pure import compute_complexities

# Compute ApEn and SampEn for embedding dimensions 0 to m
rr = np.random.randn(5000)
ApEn, SampEn = compute_complexities(
    rr,
    n_embed=3,   # Max embedding dimension
    stride=1,
    r=0.2        # Tolerance as fraction of std
)

print(f"SampEn(m=1): {SampEn[1]:.4f}")
print(f"SampEn(m=2): {SampEn[2]:.4f}")
print(f"SampEn(m=3): {SampEn[3]:.4f}")
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `compute_entropy(x, n_embed, stride, k)` | Shannon entropy (k-NN) |
| `compute_entropy_rate(x, method, m, k)` | Entropy rate |
| `compute_MI(x, y, n_embed_x, n_embed_y, k)` | Mutual Information (KSG) |
| `compute_TE(x, y, n_embed_x, n_embed_y, lag, k)` | Transfer Entropy |
| `compute_PMI(x, y, z, k)` | Partial MI (conditional on z) |
| `compute_PTE(x, y, z, lag, k)` | Partial TE |
| `compute_DI(x, y, N, k)` | Directed Information |
| `compute_relative_entropy(x, y, k)` | KL divergence |
| `compute_entropy_increments(x, inc_type, order, k)` | Increments entropy |
| `compute_regularity_index(x, order, k)` | Regularity index |

### Complexity Measures

| Function | Description |
|----------|-------------|
| `compute_complexities(x, n_embed, r)` | ApEn and SampEn |
| `compute_entropy_Renyi(x, q, k)` | Renyi entropy of order q |

### Configuration

| Function | Description |
|----------|-------------|
| `set_sampling(Theiler, N_eff, N_real)` | Set sampling parameters |
| `get_sampling()` | Get current sampling parameters |
| `get_last_info()` | Get info from last computation |
| `choose_algorithm(algo, version)` | Select MI algorithm (1 or 2) |

### Tools

| Function | Description |
|----------|-------------|
| `compute_over_scales(func, scales, *args)` | Run function over time scales |
| `reorder(x)` | Ensure correct array orientation |
| `embed(x, n_embed, stride)` | Time-delay embedding |
| `crop(x, npts_new, i_window)` | Crop time series |

### Masking

| Function | Description |
|----------|-------------|
| `mask_finite(x)` | Create mask for finite values |
| `mask_NaN(x)` | Create mask excluding NaN |
| `mask_clean(x)` | Combined cleaning mask |
| `retain_from_mask(x, mask)` | Apply mask to data |

### Surrogates

| Function | Description |
|----------|-------------|
| `surrogate(x, method)` | Generate surrogate data |

Methods: 0=shuffle, 1=phase randomize, 2=windowed FFT, 3=AAFT, 4=iterative AAFT, 5=Gaussianize

### Benchmark Utilities

| Function | Description |
|----------|-------------|
| `run_benchmark(rr_arrays, ...)` | Batch process RR files |
| `process_rr_array(rr, ...)` | Process single RR array |
| `load_rr_files(file_paths)` | Load RR files from disk |
| `generate_synthetic_rr(n_files, n_points)` | Generate test data |
| `EntropyMetrics` | Dataclass for results |

## Parameters

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k` | Number of nearest neighbors | 5 |
| `n_embed` | Embedding dimension | 1 |
| `stride` | Time lag between embedded points | 1 |
| `Theiler` | Theiler window (see below) | 4 |
| `N_eff` | Effective points per realization | 4096 |
| `N_real` | Number of realizations | 3-10 |

### Theiler Prescriptions

| Value | Name | Description |
|-------|------|-------------|
| 1 | `legacy` | tau_Theiler = stride, uniform sampling |
| 2 | `smart` | tau_Theiler = max, uniform sampling |
| 3 | `random` | tau_Theiler = stride, random sampling |
| 4 | `adapted` | Adaptive tau_Theiler (recommended) |

## Performance

This implementation uses `scipy.spatial.KDTree` for efficient nearest neighbor queries. Performance characteristics:

| Dataset Size | Typical Time | Notes |
|--------------|--------------|-------|
| 3,000 pts | ~0.7 sec/file | Standard ECG recording |
| 50,000 pts | ~0.5 sec/file | With N_eff=4096 sampling |
| 3M pts | ~0.25 sec/file | Long recording with sampling |

### Optimization Tips

1. **Reduce N_eff**: Lower effective points (e.g., 2048) for faster computation
2. **Reduce N_real**: Fewer realizations (e.g., 1-2) for quick estimates
3. **Parallel processing**: Use `process_files_parallel()` for batch jobs
4. **Gaussian approximation**: Use `k=-1` when normality assumption holds

### Batch Processing

```python
from entropy_pure import process_files_parallel

# Process multiple files in parallel
results = process_files_parallel(
    file_dict={'f1': arr1, 'f2': arr2, ...},
    scales=[1, 2, 4, 8],
    n_workers=4
)
```

## Comparison with Original C/C++ Library

| Aspect | entropy (C/C++) | entropy_pure (Python) |
|--------|-----------------|----------------------|
| Installation | Requires compilation | pip install |
| Dependencies | GSL, FFTW, ANN | numpy, scipy |
| Speed | Fastest | ~1.5-3x slower |
| Portability | Platform-specific | Cross-platform |
| API | Identical | Identical |

The pure Python version produces numerically equivalent results to the C/C++ version.

## References

- Kozachenko, L.F., Leonenko, N.N. (1987) - Entropy estimation
- Kraskov, A., Stogbauer, H., Grassberger, P. (2004) PRE 69, 066138 - MI estimation
- Schreiber, T. (2000) PRL 85, 461 - Transfer entropy
- Richman, J.S., Moorman, J.R. (2000) - Sample entropy

## License

BSD-3-Clause (same as original entropy library)

## Citing

To cite this work, please use the original entropy library DOI:
[![DOI](https://zenodo.org/badge/635707956.svg)](https://doi.org/10.5281/zenodo.13218642)
