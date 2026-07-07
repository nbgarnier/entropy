# entropy

"entropy" is an efficient C/C++ library integrated with Python and Matlab to estimate various entropies and many other quantities from information theory, using nearest neighbors estimates.

A "pure" Python version is also provided (see [entropy_pure](https://github.com/nbgarnier/entropy/tree/main/src/entropy_pure)), which may be easier to install on your system as it does not require compilation.

# compilation and installation of the C library for use with Matlab or Python
- run ./configure and eventuallly solve the issues by installing missing programs and libraries (e.g.: "apt install libtool-bin fftw3-dev" on Linux if asked to do so)
   - on macos, if you are using brew, you just need to install gsl and fftw: "brew install gsl fftw"
   - on macos with ARM/Apple processors, if you are using brew and a recent version of the library (>=4.1.1), configuration should detect your homebrew installation correctly. For older versions (before 4.1.1), you need to explicitly indicate the locations of brew libraries; this is done by invoking configure with:
<pre><code>
./configure CFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib
</code></pre>
- then run either "make matlab" or "make python" (or both) to produce the library.
  
## Matlab version
- The Matlab version is fully supported as of 2023/10/09.
- "make matlab" should be working fine, as long as the configure step has correctly detected your matlab installation. To ensure the correct Matlab version is used (if you have several versions installed, or on MacOs if the auto-detection fails), re-run ./configure using the MATLAB option, e.g. on MacOs:
<pre><code>
./configure MATLAB=/Applications/MATLAB_R2023a.app
</code></pre>
- Matlab binaries and scripts are located in the subdirectory /bin/matlab ; you should add this path to your matlab environement in order to be able to run the functions provided by the library.

## Python version
- "make python" will both compile the library and install it in your python path, which depends on you current environment. You should select your environment first, then run "./configure" and "make python", in order to have the library and its functions available in your favored environment.
- there are examples in the bin/python subdirectory: please look at them to learn how to import and use the library, which should be as easy as:
<pre><code>
import numpy
import entropy.entropy as entropy

x = numpy.random.randn(1,100000)
H = entropy.compute_entropy(x)
</code></pre>

# pure Python version

No compilation is required. The library is installed with pip:
```bash
cd src/entropy_pure
pip install .
```

This pure Python version can be used as a drop-in replacement of the C/Python library in Python scripts:
```python
import numpy as np
import entropy_pure as entropy
...
```

o used as in the following example:
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

See [entropy_pure](https://github.com/nbgarnier/entropy/tree/main/src/entropy_pure)

# notes

this library provides continuous entropies estimates using nearest neighbors. It relies on the [ANN library](http://www.cs.umd.edu/~mount/ANN/) by David Mount and Sunil Arya, which has been patched and included in the source tree.

# documentation

See help on functions on the [documentation webpage](https://perso.ens-lyon.fr/nicolas.garnier/files/html/).

See Python examples in `bin/python/`.


# citing

to cite this work, please use this DOI: [![DOI](https://zenodo.org/badge/635707956.svg)](https://doi.org/10.5281/zenodo.13218642)



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

### Validation Results

The pure Python implementation was systematically validated against the C/ANN reference on 251 five-minute RR-interval segments from ambulatory ECG recordings. Three implementation discrepancies were identified and corrected:

1. **Zero-distance handling** -- the ANN library excludes all zero-distance neighbors, not just the query point itself; the Python code now replicates this behavior.
2. **L-infinity norm** -- ANN is compiled with the Chebyshev ($L_\infty$) norm; `scipy.spatial.KDTree.query` is now called with `p=np.inf` to match.
3. **Entropy-rate embedding alignment** -- at time scales $\tau > 1$, the marginal entropy and mutual information terms must share a unified embedding; the Python code now mirrors the C `Theiler_embed` layout.

After all corrections, Shannon entropy $H(m{=}2)$ matches to machine precision ($R^2 = 1.0000$) and entropy rate $h(m{=}2)$ agrees to within 0.06% ($R^2 = 0.9998$) across time scales $\tau \in \{1, 2, 4\}$.

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
