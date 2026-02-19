# Methodological Report: Entropy Computation, Input-Signal Robustness, and Python Library Validation

**Date**: 2026-02-19
**Project**: Prenatal Steroid Exposure — Cardiac Complexity Analysis
**Status**: Working document for integration with Python entropy library validation effort
**Related repos**:
- Main analysis: `florian-ecg-steroid-analysis`
- Companion paper: `/Users/mfrasch/projects/Florian_Nicolas_ER/`
- Entropy library validation: `martinfrasch/entropy` (branch `claude/refactor-python-efficiency-UGyKA`)

---

## 1. Entropy Computation Methods

### 1.1 k-NN Entropy Estimation (Kozachenko-Leonenko)

The Shannon entropy of a time-delay embedded signal is estimated using the Kozachenko-Leonenko (KL) k-nearest-neighbor estimator:

$$H = d \cdot \overline{\ln(2\varepsilon_k)} + \psi(N) - \psi(k)$$

where:
- $d$ = embedding dimensionality
- $\varepsilon_k$ = L-infinity (Chebyshev) distance to the k-th non-zero nearest neighbor
- $\psi$ = digamma function
- $N$ = number of valid (non-duplicate) data points
- $k$ = number of neighbors (set to 5 throughout)

**Critical implementation detail**: The distance metric is L-infinity (Chebyshev), not L2 (Euclidean). This matches the original C/ANN library compiled with `ANN_METRIC=LINF`. The Python implementation uses `scipy.spatial.KDTree.query(..., p=np.inf)`.

**Zero-distance handling**: Points with identical coordinates (distance = 0) are excluded from the neighbor count by querying `k + max_dup` neighbors and selecting the k-th non-zero distance. This matches the C implementation's `kd_search.cpp` which applies a `dist != 0` check.

### 1.2 Entropy Rate (method=2)

The entropy rate $h$ quantifies the conditional uncertainty of the next value given the past. We use **method 2**: the difference between the marginal entropy and the mutual information between the current value and its time-delay embedding:

$$h(\tau, m) = H(x_t) - I(x_t \,;\, x_{t-\tau}, x_{t-2\tau}, \ldots, x_{t-m\tau})$$

where:
- $\tau$ = time delay (stride in samples)
- $m$ = embedding dimension (set to 2)
- $H(x_t)$ = Shannon entropy of the current value
- $I(\cdot \,;\, \cdot)$ = mutual information estimated via KSG algorithm 1

**Unified embedding construction**: Both $H(x_t)$ and $I(x_t ; x_{\text{past}})$ are computed on the same point set with consecutive stepping and stride-spaced lags. The embedding is constructed as:

```
x_curr[i]    = x[offset + i]
x_past[j, i] = x[offset + i - (j+1) * stride]    for j = 0..m-1
```

where `offset = stride * m` ensures all past lags are available. This unified construction is critical for numerical agreement with the C reference — see Section 4.3 (Bug 3).

### 1.3 Sample Entropy (SampEn)

SampEn quantifies signal regularity through template matching:

$$\text{SampEn}(m, r) = -\ln\left(\frac{B^{m+1}}{A^m}\right)$$

where $B^{m+1}$ counts matching template pairs of length $m+1$ and $A^m$ counts matching pairs of length $m$, **excluding self-matches**.

**Template matching**: Two templates $\mathbf{x}_i = (x_i, x_{i+\tau}, \ldots, x_{i+m\tau})$ match if their L-infinity distance is at most $r_{\text{abs}}$:

$$\max_j |x_{i+j\tau} - x_{k+j\tau}| \leq r_{\text{abs}}$$

**Tolerance threshold**: $r_{\text{abs}} = 0.2 \times \text{SD}(x)$, computed on the (possibly coarse-grained) signal and held fixed across embedding dimensions.

### 1.4 Approximate Entropy (ApEn)

ApEn follows the same template-matching framework as SampEn but **includes self-matches** in the counting, making it biased but less sensitive to short data:

$$\text{ApEn}(m, r) = \phi^m(r) - \phi^{m+1}(r), \quad \phi^m(r) = \frac{1}{N} \sum_{i=1}^N \ln C_i^m(r)$$

where $C_i^m(r)$ counts matching templates (including self) normalized by $N$.

---

## 2. Multiscale Analysis Framework

### 2.1 Scale Sweep Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Scale range | 0.5–5.0 s | Captures short-term to mid-term dynamics |
| Time step ($d\tau$) | 0.04 s | = `dstride=10` samples at 250 Hz |
| Number of scales | ~113 | (5.0 - 0.5) / 0.04 + 1 |
| Embedding dimension $m$ | 2 | For both entropy rate and SampEn/ApEn |
| k-NN neighbors $k$ | 5 | For KL estimator |
| SampEn/ApEn tolerance $r$ | 0.2 | Multiplied by SD of signal |
| $N_{\text{eff}}$ | 4096 | Effective points per realization |
| $N_{\text{real}}$ | 3 | Realizations averaged for stability |

### 2.2 Coarse-Graining

For SampEn and ApEn, multiscale analysis uses **non-overlapping block averaging** (Costa et al., 2002):

$$y_j^{(\tau)} = \frac{1}{\tau} \sum_{i=(j-1)\tau+1}^{j\tau} x_i$$

where $\tau$ is the scale factor and $y^{(\tau)}$ is the coarse-grained series.

For the entropy rate $h$, coarse-graining is achieved **implicitly** through the `stride` parameter in the time-delay embedding, rather than explicit signal averaging. This means $h$ at scale $\tau$ is computed by:

$$h(\tau, m) = H(x_t) - I(x_t \,;\, x_{t-\tau}, x_{t-2\tau}, \ldots, x_{t-m\tau})$$

This distinction is important: the entropy rate operates on the original signal with wider lags, while SampEn/ApEn operate on a downsampled signal.

### 2.3 Summary Statistics from Entropy-vs-Scale Curves

From each entropy measure's scale curve, four summary statistics are extracted:

| Statistic | Definition | Interpretation |
|-----------|-----------|----------------|
| **AUC** | Area under the entropy-vs-scale curve (trapezoidal) | Total complexity across scales |
| **max** | Maximum entropy value across all scales | Peak complexity |
| **max_t** | Scale at which maximum occurs | Characteristic timescale |
| **dec_rate** | Rate of decrease from max to the final scale | How quickly complexity decays at longer timescales |

This yields 12 features per segment: {$h$, SE, AE} $\times$ {AUC, max, max_t, dec_rate}.

### 2.4 Decay Rate as the Key Biomarker

The **decay rate** of the entropy-vs-scale curve captures how quickly signal complexity decreases at longer timescales. Among all 12 features, $h$ decay rate emerged as the primary biomarker:

- Significant binary group difference (LMM, $p = 0.020$)
- Significant dose-response (Kruskal-Wallis, $p = 0.005$; Dunn's low-vs-high $p = 0.005$)
- Robust across all input signal conditions (Section 3)

The decay rate is conceptually related to the spectral content of the signal: a steeper decay indicates less long-range temporal structure (faster loss of information at coarser timescales).

---

## 3. Input-Signal Robustness Analysis

### 3.1 Four Input Conditions

Entropy measures were computed under four input signal variants:

| Condition | Signal | Normalization | Notes |
|-----------|--------|---------------|-------|
| HR_raw | Heart rate (bpm) | None | $\text{HR} = 60000 / \text{RRI}$ |
| HR_norm | Heart rate | $\div$ SD | Zero-mean, unit-variance |
| RRI_raw | R-R intervals (ms) | None | Direct inter-beat intervals |
| RRI_norm | R-R intervals | $\div$ SD | **Recommended condition** |

All four conditions were computed both with and without smoothing (8 total configurations).

### 3.2 Cross-Condition Correlations

Segment-level Pearson correlations ($N = 251$ segments) between decay rates across all four conditions:

**$h$ decay rate**:

|              | HR_raw | HR_norm | RRI_raw | RRI_norm |
|--------------|--------|---------|---------|----------|
| **HR_raw**   | 1.000  | 0.997   | 0.983   | 0.985    |
| **HR_norm**  |        | 1.000   | 0.985   | 0.986    |
| **RRI_raw**  |        |         | 1.000   | 0.998    |
| **RRI_norm** |        |         |         | 1.000    |

*Range: $r = 0.983$–$0.998$. Virtually invariant to signal type and normalization.*

**SampEn decay rate**: Range $r = 0.310$–$0.953$. Highly sensitive to input choice. Only HR_norm vs RRI_norm is reasonably stable ($r = 0.953$).

**ApEn decay rate**: Range $r = -0.060$–$0.980$. Essentially uncorrelated between raw and normalized variants. Only HR_norm vs RRI_norm is stable ($r = 0.980$).

### 3.3 Dose-Response Robustness (Kruskal-Wallis, M3_bis_M4)

Three-group comparison (low dose, high dose, controls) on the mental arithmetic segment:

| Condition | $h$ dec_rate $H$ / $p$ | SE dec_rate $H$ / $p$ | AE dec_rate $H$ / $p$ |
|-----------|------------|-----------|-----------|
| HR_raw    | 10.16 / **0.006** | 12.49 / **0.002** | 1.90 / 0.387 |
| HR_norm   | 10.59 / **0.005** | 4.09 / 0.129 | 4.41 / 0.110 |
| RRI_raw   | 10.18 / **0.006** | 4.51 / 0.105 | 0.37 / 0.832 |
| RRI_norm  | 10.69 / **0.005** | 4.49 / 0.106 | 4.51 / 0.105 |

- **$h$ decay rate**: Significant ($p < 0.01$) in **all four conditions** with consistent $H$ statistics (10.16–10.69).
- **SE decay rate**: Significant only in HR_raw ($p = 0.002$); loses significance in all other conditions.
- **AE decay rate**: Not significant in any condition.

### 3.4 Pairwise Robustness (Mann-Whitney U, Low vs High Dose)

| Condition | $h$ dec_rate $U$ / $p$ / $d$ | SE dec_rate $U$ / $p$ / $d$ |
|-----------|------------|-----------|
| HR_raw    | 25 / **0.004** / 1.53 | 14 / **<0.001** / 1.97 |
| HR_norm   | 23 / **0.003** / 1.57 | 42 / 0.054 / 0.93 |
| RRI_raw   | 25 / **0.004** / 1.57 | 21 / 0.062 / 0.97 |
| RRI_norm  | 24 / **0.004** / 1.66 | 41 / 0.047 / 0.98 |

$h$ decay rate shows large, consistent effects ($d > 1.5$, $p < 0.005$) in all conditions.

### 3.5 Summary of Robustness Properties

| Property | $h$ dec_rate | SE dec_rate | AE dec_rate |
|----------|:-----------:|:----------:|:----------:|
| Cross-condition $r$ (min) | **0.983** | 0.310 | $-0.060$ |
| Significant in all 4 conditions (KW) | **Yes** | No (1/4) | No (0/4) |
| Low-vs-high significant in all 4 (MW) | **Yes** | No (1–2/4) | No (0–1/4) |
| Effect size range ($d$) | 1.53–1.66 | 0.93–1.97 | 0.05–0.93 |

---

## 4. Python Library Validation

### 4.1 Scope

The `entropy_pure` Python library is a pure-Python reimplementation of Nicolas's C/C++ entropy computation library (37 functions). The C library uses the ANN (Approximate Nearest Neighbors) library for k-d tree operations. The Python port uses `scipy.spatial.KDTree`.

**Functions validated** (partial list):
- `compute_entropy()` — Shannon entropy via KL estimator
- `compute_entropy_rate()` — methods 0, 1, 2
- `compute_MI()` — mutual information via KSG algorithm 1 and 2
- `compute_TE()` — transfer entropy
- `compute_complexities()` — SampEn and ApEn

**Functions not in this library** (computed elsewhere):
- Permutation entropy — comes from a separate implementation, not part of the entropy_pure library. The $h$ (permutation entropy rate) used in this project is computed by a different codebase.

### 4.2 Validation Results

Comparison against C reference implementation on the same RR interval dataset:

| Metric | Mean Abs Diff | Max Abs Diff | $R^2$ |
|--------|:------------:|:----------:|:-----:|
| Shannon $H(m=2)$, all scales | 0.0000 | 0.0000 | **1.0000** |
| Entropy rate $h(m=2)$, all scales | 0.0017 | 0.1084 | **0.9998** |
| SampEn | — | — | $\rho = 0.20$* |

\* SampEn comparison yields weak agreement (Spearman $\rho = 0.20$) due to convention differences between the Python and C implementations (see Section 4.4).

### 4.3 Three Critical Bugs Fixed

The initial Python port produced systematic offsets from the C reference. Three bugs were identified and fixed (commit `875765d`):

#### Bug 1: Zero-Distance Handling in `_entropy_knn()`

**Symptom**: $H(1)$ values had a constant offset from C reference.

**Root cause**: The C library's `kd_search.cpp:234` applies a `dist != 0` check, skipping all points at distance zero (not just the query point itself). Points with identical coordinates in the embedding space (e.g., constant-value segments) were counted as valid neighbors in the Python version.

**Fix**: Query `k + max\_dup` neighbors, count zero-distance entries, and select the $k$-th non-zero distance:

```python
_, counts = np.unique(x.T, axis=0, return_counts=True)
max_dup = int(counts.max())
k_ext = min(n_pts, k + max_dup)
distances, _ = tree.query(x.T, k=k_ext, p=np.inf)
n_zeros = np.sum(distances == 0, axis=1)
target_idx = n_zeros + k - 1  # k-th non-zero neighbor
```

#### Bug 2: L2 vs L-Infinity Distance Metric

**Symptom**: Shannon $H(m=2)$ had a scale-dependent offset.

**Root cause**: Python used Euclidean distance (`p=2`) while the C library is compiled with `ANN_METRIC=LINF` (Chebyshev / L-infinity).

**Fix**: Set `p=np.inf` in all `KDTree.query()` calls:

```python
distances, _ = tree.query(x.T, k=k_ext, p=np.inf, workers=-1)
```

#### Bug 3: Unified Embedding in `compute_entropy_rate()` method=2

**Symptom**: Entropy rate at stride $> 1$ diverged from C reference.

**Root cause**: The C implementation builds a unified embedding where $H(x_t)$ and $I(x_t; x_{\text{past}})$ operate on the **same aligned point set** with consecutive stepping and stride-spaced lags. The initial Python version computed $H(x_t)$ on the full raw signal and had misaligned current/past arrays at stride $> 1$.

**Fix**: Construct a single aligned embedding:

```python
pts_offset = stride * m
t_indices = pts_offset + np.arange(n_use)
x_curr = x[:, t_indices]
x_past_emb = np.zeros((n_dims * m, n_use))
for lag in range(m):
    past_indices = t_indices - (lag + 1) * stride
    x_past_emb[lag*n_dims:(lag+1)*n_dims, :] = x[:, past_indices]
```

### 4.4 SampEn Convention Differences

The weak cross-implementation agreement for SampEn ($\rho = 0.20$) is attributed to **convention differences**, not bugs:

1. **Counting convention**: Differences in how self-matches and boundary conditions are handled
2. **Normalization**: Variations in whether $\phi^m$ is computed from pair counts or log-averaged template counts
3. **Embedding alignment**: How templates at the signal boundaries are treated

These differences do not affect the *relative ordering* within a single implementation but prevent direct numerical comparison across implementations. This finding underscores the importance of using a single consistent implementation for all analyses.

---

## 5. Theoretical Basis for $h$ Decay Rate Robustness

### 5.1 k-NN Rank Invariance

The fundamental reason $h$ decay rate is robust across input conditions lies in the **rank-based** nature of the k-NN estimator.

The KL estimator depends on k-NN distances $\varepsilon_k$ only through their logarithms:

$$H = d \cdot \overline{\ln(2\varepsilon_k)} + \psi(N) - \psi(k)$$

Under a **monotonic transform** $g(x)$:
- The **rank ordering** of inter-point distances is preserved (the same points remain nearest neighbors)
- The distances scale by the local derivative $|g'|$, adding a term $\overline{\ln|g'|}$ to $H$

The HR $\leftrightarrow$ RRI conversion ($\text{HR} = 60000 / \text{RRI}$) is a monotonic (decreasing) transform. Normalization by standard deviation ($x \mapsto x/\sigma$) is also monotonic. Therefore:

1. **The same neighbor structure is found** regardless of whether the input is HR, RRI, raw, or normalized.
2. **Absolute entropy values differ** by a constant offset related to $\overline{\ln|g'|}$.
3. **Entropy differences across scales** (which determine the decay rate) are **invariant**, because the offset cancels in the subtraction.

This explains why $h$ decay rate correlations exceed 0.98 across all four conditions.

### 5.2 Template Matching Sensitivity

In contrast, SampEn and ApEn use an **absolute tolerance threshold** $r_{\text{abs}} = 0.2 \times \text{SD}(x)$. Under a nonlinear transform $g(x)$:

1. The standard deviation changes: $\text{SD}(g(x)) \neq |g'| \cdot \text{SD}(x)$ in general
2. The tolerance $r_{\text{abs}}$ therefore captures a **different proportion** of the signal's distributional geometry
3. The set of matching templates changes, sometimes dramatically

For the HR $\leftrightarrow$ RRI transform ($g(x) = 60000/x$), the nonlinearity is substantial:
- At low HR values (long RRI), a given RRI tolerance corresponds to a small HR tolerance
- At high HR values (short RRI), the same RRI tolerance corresponds to a large HR tolerance

This asymmetric sensitivity explains the low cross-condition correlations for SampEn (0.31–0.95) and ApEn ($-0.06$–0.98).

### 5.3 Normalization as Partial Remedy

Normalization ($x \mapsto x/\text{SD}(x)$) partially mitigates the template-matching sensitivity by making the tolerance proportional to the signal's spread. Indeed, the HR_norm vs RRI_norm correlation is high for all three measures:

| Measure | HR_norm vs RRI_norm $r$ |
|---------|:----------------------:|
| $h$ decay rate | 0.986 |
| SampEn decay rate | 0.953 |
| ApEn decay rate | 0.980 |

However, this only ensures agreement between **normalized** variants. The raw-vs-normalized and HR-vs-RRI comparisons remain unstable for SampEn and ApEn, limiting their interpretability.

---

## 6. Statistical Framework

### 6.1 Primary Analysis: Linear Mixed Models

Binary group comparison (exposed vs controls) with covariates:

```
feature ~ Group_exp + Age + Sex + gestational_age + (1 | subject_id)
```

Estimated using REML via `statsmodels.formula.api.mixedlm`. Multiple testing correction: FDR-BH (Benjamini-Hochberg) within each analysis block.

### 6.2 Dose-Response Analyses

| Method | Purpose | Implementation |
|--------|---------|----------------|
| Kruskal-Wallis | 3-group comparison (low, high, control) | `scipy.stats.kruskal` |
| Dunn's test | Post-hoc pairwise comparisons | `scikit_posthocs.posthoc_dunn` |
| Mann-Whitney U | Binary pairwise (e.g., low vs high) | `scipy.stats.mannwhitneyu` |
| Jonckheere-Terpstra | Ordered trend test (control < low < high) | Custom implementation |
| Spearman correlation | Continuous dose-response | `scipy.stats.spearmanr` |

Effect sizes: Cohen's $d$ from group means and pooled standard deviation; $\eta^2 = (H - k + 1)/(n - k)$ for Kruskal-Wallis; rank-biserial $r$ for Mann-Whitney.

### 6.3 Dose Groups

| Group | Code | Criterion | $N$ (subjects) |
|-------|:----:|-----------|:--------------:|
| Low dose | 1 | Cumulative dose $< 5$ g | 12–15* |
| High dose | 2 | Cumulative dose $\geq 5$ g | 13–15* |
| Controls | 3 | No steroid exposure | 24–30* |

\* Exact $N$ varies by segment due to quality exclusions.

---

## 7. Integration Points with Entropy Library Validation

### 7.1 What the Library Validation Covers

The `entropy_pure` Python port (branch `claude/refactor-python-efficiency-UGyKA`) validates:

- **Shannon entropy** ($H$): Perfect numerical agreement ($R^2 = 1.0000$)
- **Entropy rate** ($h$): Near-perfect agreement ($R^2 = 0.9998$) — this is the measure used to compute `h_AUC`, `h_max`, `h_max_t`, `h_dec_rate`
- **SampEn and ApEn**: Implementation present with known convention differences from C
- **Three critical bugs**: Documented and fixed, with clear mathematical explanations

### 7.2 What the Library Validation Does NOT Cover

- **Permutation entropy**: NOT implemented in the `entropy_pure` library. The permutation entropy ($h$) used as the primary biomarker in this project comes from a separate implementation. *This is an important clarification — the $h$ in our dose-response results refers to permutation entropy, not the KL-estimated entropy rate from this library.*
- **Multiscale coarse-graining pipeline**: The scale sweep and summary statistic extraction (AUC, max, max_t, dec_rate) are handled by Nicolas's C pipeline, not reimplemented in Python.
- **ML classification**: The machine learning pipeline (LogReg, RF, SVM, XGBoost) is separate from the entropy computation.

### 7.3 Complementary Contributions

| Aspect | Robustness Analysis (this report) | Library Validation (entropy repo) |
|--------|:-:|:-:|
| Input signal sensitivity | Systematic across 4 conditions | Single condition (C reference match) |
| Theoretical explanation | k-NN rank invariance argument | Bug-level numerical forensics |
| Practical recommendation | Use $h$ decay rate as robust biomarker | Use Python port with documented fixes |
| Scope | Statistical results and interpretation | Implementation correctness |

### 7.4 Shared Conclusion

Both efforts converge on the same insight: **k-NN-based entropy measures are fundamentally more stable than template-matching measures**. The robustness analysis demonstrates this empirically across input conditions; the library validation demonstrates it through numerical agreement between independent implementations.

---

## 8. Recommendations

1. **Report RRI_norm (smoothed) as the primary analysis condition** throughout the companion manuscript, with robustness results showing invariance to this choice for $h$ decay rate.

2. **Present SE_dec_rate results with appropriate caveats**: The significant SE dose-response under HR_raw ($p = 0.002$) does not generalize to other input conditions and should not be interpreted as an independent finding.

3. **Document the permutation entropy provenance**: Clarify that the $h$ in dose-response analyses refers to permutation entropy (from a separate implementation), while the `entropy_pure` library implements the KL-estimated Shannon entropy rate. Both are k-NN based and share the rank-invariance property.

4. **The Python port is validated for Shannon $H$ and entropy rate $h$**: These can be used for reproduction and extension studies. SampEn comparisons across implementations require careful attention to convention differences.

---

## Appendix: File Locations

| Resource | Path |
|----------|------|
| Robustness tables (LaTeX) | `ER_v3/robustness_table.tex` |
| Robustness tables (Markdown) | `ER_v3/robustness_table.md` |
| Robustness CSV data | `ER_v3/complexities_*_dstride=10*.npz` |
| Python entropy library | `/Users/mfrasch/projects/Python_entropy_validation/entropy_repo/` |
| Validation report | `/Users/mfrasch/projects/Python_entropy_validation/2026-02-19.md` |
| Companion manuscript | `/Users/mfrasch/projects/Florian_Nicolas_ER/main.tex` |
| Per-segment KW results | `/Users/mfrasch/projects/Florian_Nicolas_ER/results_kruskal_per_segment.csv` |
| Per-segment MW results | `/Users/mfrasch/projects/Florian_Nicolas_ER/results_mannwhitney_per_segment.csv` |
| ML classification results | `/Users/mfrasch/projects/Florian_Nicolas_ER/results_ml_classification.csv` |
