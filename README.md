# entropy

An efficient C/C++ library integrated with Python and Matlab to estimate various entropies and many other quantities from information theory, using nearest neighbors estimates.

## What's New in This Fork

This fork adds **entropy_pure** - a complete pure Python reimplementation that requires no compilation:

| Feature | Original (C/C++) | This Fork (entropy_pure) |
|---------|------------------|--------------------------|
| Installation | `./configure && make` | `pip install .` |
| Dependencies | GSL, FFTW, ANN, compilers | numpy, scipy only |
| Platform | Requires platform-specific build | Cross-platform (Win/Mac/Linux) |
| Speed | Maximum performance | ~1.5-3x slower (still efficient) |
| API compatibility | Original | 100% identical |

**New features in entropy_pure:**
- **RR interval benchmarking** - Batch process cardiac data with CSV output
- **Multi-scale analysis** - Built-in `compute_over_scales()` for time-scale analysis
- **Sample Entropy / ApEn** - Complexity measures via `compute_complexities()`
- **Parallel processing** - `process_files_parallel()` for batch jobs

Both versions produce numerically equivalent results.

---

## Quick Start: Pure Python (Recommended for Most Users)

```bash
cd src/entropy_pure
pip install .
```

```python
import numpy as np
from entropy_pure import (
    compute_entropy, compute_entropy_rate, compute_MI, compute_TE,
    compute_complexities, compute_over_scales, run_benchmark
)

# Basic entropy
x = np.random.randn(1, 10000)
H = compute_entropy(x)
h = compute_entropy_rate(x, m=2)

# Sample Entropy
ApEn, SampEn = compute_complexities(x, n_embed=2, r=0.2)

# Multi-scale analysis
scales = np.array([1, 2, 4, 8, 16])
h_values, _ = compute_over_scales(compute_entropy_rate, scales, x, m=2)

# Batch RR interval processing
rr_arrays = {'patient1': rr1, 'patient2': rr2}
results = run_benchmark(rr_arrays, output_file='results.csv')
```

See [src/entropy_pure/README.md](src/entropy_pure/README.md) for complete documentation.

---

## C/C++ Version (Maximum Performance)

Use this if you need maximum speed and can compile on your platform.

### Compilation and Installation

1. Run `./configure` and install missing dependencies:
   - Linux: `apt install libtool-bin fftw3-dev libgsl-dev`
   - macOS: `brew install gsl fftw`
   - macOS ARM (older versions):
     ```bash
     ./configure CFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib
     ```

2. Build: `make matlab` or `make python` (or both)

### Matlab Version

- Fully supported as of 2023/10/09
- Run `make matlab` after configure
- Specify Matlab if needed: `./configure MATLAB=/Applications/MATLAB_R2023a.app`
- Add `/bin/matlab` to your Matlab path

### Python Version

```python
import numpy
import entropy.entropy as entropy

x = numpy.random.randn(1, 100000)
H = entropy.compute_entropy(x)
```

See examples in `bin/python/`.

---

## Available Functions

Both versions provide:

| Function | Description |
|----------|-------------|
| `compute_entropy` | Shannon entropy (k-NN estimator) |
| `compute_entropy_rate` | Entropy rate |
| `compute_MI` | Mutual Information (KSG algorithm) |
| `compute_TE` | Transfer Entropy |
| `compute_PMI` | Partial Mutual Information |
| `compute_PTE` | Partial Transfer Entropy |
| `compute_DI` | Directed Information |
| `compute_relative_entropy` | KL divergence |
| `compute_entropy_Renyi` | Renyi entropy |
| `compute_complexities` | ApEn and SampEn |

## Notes

This library provides continuous entropy estimates using nearest neighbors. It relies on the [ANN library](http://www.cs.umd.edu/~mount/ANN/) by David Mount and Sunil Arya, which has been patched and included in the source tree.

## Documentation

See the [documentation webpage](https://perso.ens-lyon.fr/nicolas.garnier/files/html/) for detailed function help.

## Citing

To cite this work, please use this DOI: [![DOI](https://zenodo.org/badge/635707956.svg)](https://doi.org/10.5281/zenodo.13218642)
