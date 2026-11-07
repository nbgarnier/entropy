#!/usr/bin/env python3
"""
Generate test signals X (target) and Y (source) with NONLINEAR dependence,
for testing transfer entropy TE(Y -> X) and surrogate-based bias correction.

Convention: Y drives X (coupling Y -> X). With coupling = 0 the true TE(Y->X) is 0,
which lets you check that the estimated bias is correctly removed.

Systems:
  1. henon     : unidirectionally coupled Henon maps (chaotic, deterministic)
  2. quadratic : stochastic AR with quadratic coupling X_{t+1} ~ Y_{t-d}^2
                 (zero linear correlation, so only nonlinear measures see it)
  3. logistic  : unidirectionally coupled logistic maps (chaotic)

Usage:
  python generate_test_signals.py --system henon --n 5000 --coupling 0.4 --out data.csv
"""
import numpy
import numpy as np
import entropy.entropy as entropy
import entropy.tools as tools
#from time import time

def henon_coupled(npts, coupling=0.4, seed=0, transient=1000):
    """
    y_{t+1} = 1.4 - y_t^2 + 0.3 y_{t-1}                        (driver / source)
    x_{t+1} = 1.4 - (C*y_t*x_t + (1-C)*x_t^2) + 0.3 x_{t-1}    (response / target)
    Synchronization / divergence can occur for large C; values up to ~0.7 are fine.
    """
    rng = np.random.default_rng(seed)
    total = npts + transient
    x = np.zeros(total)
    y = np.zeros(total)
    x[:2] = rng.uniform(-0.1, 0.1, 2)
    y[:2] = rng.uniform(-0.1, 0.1, 2)
    for t in range(1, total - 1):
        y[t + 1] = 1.4 - y[t] ** 2 + 0.3 * y[t - 1]
        x[t + 1] = (1.4 - (coupling * y[t] * x[t] + (1 - coupling) * x[t] ** 2)
                    + 0.3 * x[t - 1])
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise RuntimeError("Henon system diverged; lower the coupling or change the seed.")
    return x[transient:], y[transient:]


def quadratic_stochastic(npts, coupling=0.5, delay=1, noise=0.3, seed=0, transient=500):
    """
    y_{t+1} = 0.6 y_t + 0.3 * sin(2*y_t) + noise_y          (nonlinear AR source)
    x_{t+1} = 0.5 x_t + coupling * (y_{t-delay+1}^2 - 1) + noise_x
    The quadratic term makes the Y->X dependence invisible to linear correlation
    and Granger causality, but detectable by transfer entropy.
    delay >= 1 is the lag (in samples) between Y and its effect on X.
    """
    rng = np.random.default_rng(seed)
    total = npts + transient
    x = np.zeros(total)
    y = np.zeros(total)
    ey = rng.normal(0, 1.0, total) * noise
    ex = rng.normal(0, 1.0, total) * noise
    for t in range(delay, total - 1):
        y[t + 1] = 0.6 * y[t] + 0.3 * np.sin(2 * y[t]) + ey[t]
        x[t + 1] = 0.5 * x[t] + coupling * (y[t - delay + 1] ** 2 - 1.0) + ex[t]
    return x[transient:], y[transient:]


def logistic_coupled(npts, coupling=0.3, seed=0, transient=1000, r=3.9):
    """
    y_{t+1} = f(y_t)
    x_{t+1} = (1-C) f(x_t) + C f(y_t),   f(u) = r u (1 - u)
    Stays in [0, 1] for 0 <= C <= 1. Strong C (>~0.4) leads to synchronization,
    where TE becomes ill-defined; keep C moderate.
    """
    rng = np.random.default_rng(seed)
    f = lambda u: r * u * (1 - u)
    total = npts + transient
    x = np.zeros(total)
    y = np.zeros(total)
    x[0], y[0] = rng.uniform(0.2, 0.8, 2)
    for t in range(total - 1):
        y[t + 1] = f(y[t])
        x[t + 1] = (1 - coupling) * f(x[t]) + coupling * f(y[t])
    return x[transient:], y[transient:]


SYSTEMS = {
    "henon": henon_coupled,
    "quadratic": quadratic_stochastic,
    "logistic": logistic_coupled,
}


def generate_coupled_signals(system="henon", npts=5000, coupling=0.4, **kwargs):

    x, y = SYSTEMS[system](npts, coupling, **kwargs)

    # Quick sanity check: linear correlation (can be ~0 even if TE > 0)
    r = np.corrcoef(x[1:], y[:-1])[0, 1]
    print(f"system={system}  npts={len(x)}  coupling={coupling}")
    print(f"mean/std X: {x.mean():.3f} / {x.std():.3f}   mean/std Y: {y.mean():.3f} / {y.std():.3f}")
    print(f"linear corr(X[t+1], Y[t]) = {r:+.3f}")

    return(x,y)


def compute_and_print_TE(x, y, lag=1, N_surrogates=20):
    res, surr = entropy.compute_TE(x, y, lag=lag, N_surrogates=N_surrogates)
    TE      = res[0]
    TE_std  = entropy.get_last_info()[0]
    TE_bias = np.mean(surr[0])
    print("lag=%2d\tTE(x->y) %1.4f +/- %1.4f with bias %1.4f" %(lag, TE, TE_std, TE_bias))

    res, surr = entropy.compute_TE(y, x, lag=lag, N_surrogates=N_surrogates)
    TE      = res[0]
    TE_std  = entropy.get_last_info()[0]
    TE_bias = np.mean(surr[0])
    print("\tTE(y->x) %1.4f +/- %1.4f with bias %1.4f" %(TE, TE_std, TE_bias))






def main():
    import matplotlib.pyplot as plt
    Fig=plt.figure(figsize=(12,8))
    P1 =Fig.add_subplot(1,1,1)
    P1.set_xlabel("lag")

    npts = 50000
    x,y = generate_coupled_signals("henon")
    x,y = generate_coupled_signals("quadratic", npts, coupling=0.5, delay=10, noise=0.3)
    x =tools.reorder(x)
    y =tools.reorder(y)

#    compute_and_print_TE(x, y, lag=1, N_surrogates=20)
#    compute_and_print_TE(x, y, lag=5, N_surrogates=20)
#    compute_and_print_TE(x, y, lag=10, N_surrogates=20)

    lag_set = numpy.arange(1, 25)      # a set of (time-)scales to examine
    TE, TE_std, TE_bias = tools.compute_over_lags(entropy.compute_TE, lag_set, verbosity_timing=1, x=x, y=y, N_surrogates=20)
    h = P1.errorbar(lag_set, TE[0], yerr=TE_std, label=r"TE($x\rightarrow y$)")
    my_color = h[0].get_color() 
    P1.plot(lag_set, TE[0]-TE_bias[0], ":", color=my_color)

    TE, TE_std, TE_bias = tools.compute_over_lags(entropy.compute_TE, lag_set, verbosity_timing=1, x=y, y=x, N_surrogates=20)
    h = P1.errorbar(lag_set, TE[0], yerr=TE_std, label=r"TE($y\rightarrow x$)")
    my_color = h[0].get_color() 
    P1.plot(lag_set, TE[0]-TE_bias[0], ":", color=my_color)
    
    P1.legend(fontsize=18,  bbox_to_anchor=(1.05, 1))
    Fig.savefig("TE.pdf", bbox_inches='tight');

if __name__ == "__main__":
    main()