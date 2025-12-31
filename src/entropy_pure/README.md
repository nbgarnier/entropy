# entropy_pure

Pure Python implementation of information-theoretic measures for time series analysis.

This is a refactored version of the [entropy](https://github.com/nbgarnier/entropy) library that runs entirely in Python without requiring compilation of C/C++ code.

## Features

- **Shannon Entropy** - Kozachenko-Leonenko / Grassberger k-NN estimator
- **Mutual Information** - Kraskov-Stogbauer-Grassberger algorithms (KSG1 & KSG2)
- **Transfer Entropy** - Information flow between time series
- **Partial Mutual Information** - Conditional MI
- **Partial Transfer Entropy** - Conditional TE
- **Directed Information** - Causal information measures
- **Relative Entropy** - KL divergence
- **Renyi Entropy** - Generalized entropy of order q
- **Complexity Measures** - ApEn, SampEn

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
from entropy_pure import compute_entropy, compute_MI, compute_TE

# Generate some data
np.random.seed(42)
x = np.random.randn(1, 10000)  # Shape: (n_dims, n_points)
y = np.random.randn(1, 10000)

# Shannon entropy
H = compute_entropy(x, n_embed=1, stride=1, k=5)
print(f"Entropy: {H:.4f}")

# Mutual Information
MI = compute_MI(x, y, k=5)
print(f"Mutual Information: {MI[0]:.4f}")

# Transfer Entropy (with causal relationship)
z = np.zeros((1, 10000))
z[0, 1:] = 0.7 * x[0, :-1] + 0.3 * np.random.randn(9999)
TE = compute_TE(x, z, n_embed_x=1, n_embed_y=1, lag=1, k=5)
print(f"Transfer Entropy: {TE[0]:.4f}")
```

## API Reference

### Core Functions

- `compute_entropy(x, n_embed=1, stride=1, k=5)` - Shannon entropy
- `compute_MI(x, y, n_embed_x=1, n_embed_y=1, k=5)` - Mutual Information
- `compute_TE(x, y, n_embed_x=1, n_embed_y=1, lag=1, k=5)` - Transfer Entropy
- `compute_PMI(x, y, z, k=5)` - Partial Mutual Information (conditioning on z)
- `compute_PTE(x, y, z, lag=1, k=5)` - Partial Transfer Entropy
- `compute_DI(x, y, N, k=5)` - Directed Information
- `compute_relative_entropy(x, y, k=5)` - KL divergence
- `compute_entropy_rate(x, method=2, m=1, k=5)` - Entropy rate
- `compute_entropy_increments(x, inc_type=1, order=1, k=5)` - Increments entropy

### Configuration

- `set_sampling(Theiler=4, N_eff=4096, N_real=10)` - Set sampling parameters
- `get_sampling()` - Get current sampling parameters
- `get_last_info()` - Get info from last computation
- `choose_algorithm(algo=1, version=1)` - Select MI algorithm

### Tools

- `reorder(x)` - Ensure correct array orientation
- `embed(x, n_embed, stride)` - Time-delay embedding
- `crop(x, npts_new, i_window)` - Crop time series
- `mask_finite(x)` - Create mask for valid data

### Others

- `compute_entropy_Renyi(x, q, k=5)` - Renyi entropy of order q
- `compute_complexities(x, n_embed, r=0.2)` - ApEn and SampEn
- `surrogate(x, method=0)` - Generate surrogate data

## Parameters

### Common Parameters

- `k`: Number of nearest neighbors (default: 5). Use k=-1 for Gaussian approximation.
- `n_embed`: Embedding dimension for time-delay embedding
- `stride`: Time lag between embedded points
- `Theiler`: Theiler window to avoid temporal correlations
- `N_eff`: Effective number of points per realization
- `N_real`: Number of realizations

### Theiler Prescriptions

1. `legacy` - tau_Theiler = stride, uniform sampling
2. `smart` - tau_Theiler = max, uniform sampling
3. `random` - tau_Theiler = stride, random sampling
4. `adapted` - Adaptive tau_Theiler (default)

## Performance

This pure Python implementation uses scipy.spatial.KDTree for efficient nearest neighbor queries. While it may be slower than the C/C++ version for very large datasets, it provides good performance for typical use cases and requires no compilation.

For large datasets (>100k points), consider:
- Using smaller `N_eff`
- Reducing `N_real`
- Using Gaussian approximation (`k=-1`) when appropriate

## References

- Kozachenko, L.F., Leonenko, N.N. (1987) - Entropy estimation
- Kraskov, A., Stogbauer, H., Grassberger, P. (2004) PRE 69, 066138 - MI estimation
- Schreiber, T. (2000) PRL 85, 461 - Transfer entropy

## License

BSD-3-Clause (same as original entropy library)
