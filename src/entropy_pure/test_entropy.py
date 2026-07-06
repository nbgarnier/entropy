"""
Test script for entropy_pure module.
"""
import numpy as np
import sys
import os

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entropy_pure import (
    compute_entropy,
    compute_MI,
    compute_TE,
    compute_entropy_rate,
    compute_relative_entropy,
    compute_PMI,
    set_sampling,
    get_sampling,
    get_last_info,
    reorder,
    embed,
    mask_finite,
    surrogate,
)


def test_entropy_gaussian():
    """Test Shannon entropy on Gaussian data."""
    print("Testing Shannon entropy on Gaussian data...")
    np.random.seed(42)

    # 1D Gaussian
    n_pts = 10000
    x = np.random.randn(1, n_pts)

    H = compute_entropy(x, n_embed=1, stride=1, k=5)

    # Theoretical entropy of standard Gaussian: 0.5 * log(2*pi*e) ≈ 1.42
    H_theory = 0.5 * (1 + np.log(2 * np.pi))
    print(f"  Computed H = {H:.4f}, Theory = {H_theory:.4f}")
    assert abs(H - H_theory) < 0.2, f"Entropy estimate too far from theory: {H} vs {H_theory}"

    # 2D Gaussian
    x2d = np.random.randn(2, n_pts)
    H_2d = compute_entropy(x2d, n_embed=1, stride=1, k=5)
    H_theory_2d = 2 * 0.5 * (1 + np.log(2 * np.pi))
    print(f"  Computed H (2D) = {H_2d:.4f}, Theory = {H_theory_2d:.4f}")
    assert abs(H_2d - H_theory_2d) < 0.3, f"2D entropy estimate too far from theory"

    print("  PASSED\n")


def test_mi_independent():
    """Test MI on independent Gaussians (should be ~0)."""
    print("Testing MI on independent data...")
    np.random.seed(42)

    n_pts = 5000
    x = np.random.randn(1, n_pts)
    y = np.random.randn(1, n_pts)

    MI = compute_MI(x, y, n_embed_x=1, n_embed_y=1, k=5)

    print(f"  MI (independent) = {MI[0]:.4f} (should be ~0)")
    assert abs(MI[0]) < 0.1, f"MI of independent variables should be ~0: {MI[0]}"

    print("  PASSED\n")


def test_mi_correlated():
    """Test MI on correlated data."""
    print("Testing MI on correlated data...")
    np.random.seed(42)

    n_pts = 5000
    x = np.random.randn(1, n_pts)
    noise = np.random.randn(1, n_pts)
    y = 0.8 * x + 0.6 * noise  # Correlation = 0.8

    MI = compute_MI(x, y, n_embed_x=1, n_embed_y=1, k=5)

    # MI of bivariate Gaussian with correlation rho:
    # I(X,Y) = -0.5 * log(1 - rho^2)
    rho = 0.8
    MI_theory = -0.5 * np.log(1 - rho**2)
    print(f"  MI (correlated) = {MI[0]:.4f}, Theory = {MI_theory:.4f}")
    assert abs(MI[0] - MI_theory) < 0.2, f"MI estimate differs from theory"

    print("  PASSED\n")


def test_te_causality():
    """Test TE on causal relationship."""
    print("Testing TE on causal data...")
    np.random.seed(42)

    n_pts = 5000
    x = np.random.randn(1, n_pts)
    noise = np.random.randn(1, n_pts)

    # y depends on past of x
    y = np.zeros((1, n_pts))
    y[0, 1:] = 0.7 * x[0, :-1] + 0.3 * noise[0, 1:]

    # TE from x to y should be positive
    TE_xy = compute_TE(x, y, n_embed_x=1, n_embed_y=1, lag=1, k=5)
    # TE from y to x should be close to 0
    TE_yx = compute_TE(y, x, n_embed_x=1, n_embed_y=1, lag=1, k=5)

    print(f"  TE(x->y) = {TE_xy[0]:.4f}, TE(y->x) = {TE_yx[0]:.4f}")
    assert TE_xy[0] > 0.1, f"TE(x->y) should be positive for causal relationship"
    assert TE_xy[0] > TE_yx[0], f"TE(x->y) should be greater than TE(y->x)"

    print("  PASSED\n")


def test_entropy_rate():
    """Test entropy rate computation."""
    print("Testing entropy rate...")
    np.random.seed(42)

    n_pts = 5000
    x = np.random.randn(1, n_pts)

    h = compute_entropy_rate(x, method=2, m=2, stride=1, k=5)
    H = compute_entropy(x, n_embed=1, stride=1, k=5)

    print(f"  Entropy rate (method 2) = {h:.4f}")
    print(f"  Shannon entropy H(1) = {H:.4f}")
    # For white noise, h ~ H
    assert abs(h - H) < 0.3, f"Entropy rate should be close to H for white noise"

    print("  PASSED\n")


def test_relative_entropy():
    """Test KL divergence computation."""
    print("Testing relative entropy (KL divergence)...")
    np.random.seed(42)

    n_pts = 5000
    # Two Gaussians with different means
    x = np.random.randn(1, n_pts)
    y = np.random.randn(1, n_pts) + 1.0  # Mean shifted by 1

    KL = compute_relative_entropy(x, y, n_embed_x=1, n_embed_y=1, k=5)

    # For same variance, different mean: KL = (mu_y - mu_x)^2 / (2 * sigma^2)
    # Here: KL ≈ 1^2 / 2 = 0.5
    print(f"  KL divergence = {KL:.4f} (theory ~0.5)")
    assert abs(KL - 0.5) < 0.3, f"KL divergence should be ~0.5"

    print("  PASSED\n")


def test_embedding():
    """Test time embedding function."""
    print("Testing embedding...")

    x = np.arange(10).reshape(1, -1).astype(float)
    x_emb = embed(x, n_embed=2, stride=1)

    print(f"  Original shape: {x.shape}, Embedded shape: {x_emb.shape}")
    assert x_emb.shape[0] == 2, "Embedded dimension should be 2"
    assert x_emb.shape[1] == 9, "Number of points should be n-1 for n_embed=2"

    # Check causal embedding
    # x_emb[0, :] should be x[0, 1:] (current)
    # x_emb[1, :] should be x[0, :-1] (past)
    np.testing.assert_array_equal(x_emb[0, :], x[0, 1:])
    np.testing.assert_array_equal(x_emb[1, :], x[0, :-1])

    print("  PASSED\n")


def test_masking():
    """Test masking functions."""
    print("Testing masking...")

    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0])
    mask = mask_finite(x)

    print(f"  Data: {x}")
    print(f"  Mask: {mask}")

    assert mask[0] == 1, "Finite value should have mask=1"
    assert mask[2] == 0, "NaN should have mask=0"
    assert mask[4] == 0, "Inf should have mask=0"

    print("  PASSED\n")


def test_surrogate():
    """Test surrogate data generation."""
    print("Testing surrogate generation...")
    np.random.seed(42)

    x = np.random.randn(1, 100)

    # Method 0: shuffle
    x_surr = surrogate(x, method=0)
    assert x_surr.shape == x.shape, "Surrogate should have same shape"
    assert not np.allclose(x_surr, x), "Shuffled surrogate should be different"
    np.testing.assert_almost_equal(np.sort(x.flatten()), np.sort(x_surr.flatten()),
                                   decimal=10, err_msg="Shuffle should preserve values")

    # Method 1: phase randomization
    x_surr_phase = surrogate(x, method=1)
    assert x_surr_phase.shape == x.shape, "Phase surrogate should have same shape"

    print("  PASSED\n")


def test_sampling_params():
    """Test sampling parameter functions."""
    print("Testing sampling parameters...")

    # Set sampling
    set_sampling(Theiler=4, N_eff=2048, N_real=5)
    params = get_sampling(verbosity=0)

    assert params[0] == 4, "Theiler type should be 4"
    assert params[2] == 2048, "N_eff should be 2048"
    assert params[3] == 5, "N_real should be 5"

    print("  PASSED\n")


def test_gaussian_approximation():
    """Test Gaussian approximation (k=-1)."""
    print("Testing Gaussian approximation...")
    np.random.seed(42)

    n_pts = 5000
    x = np.random.randn(1, n_pts)
    y = np.random.randn(1, n_pts)

    H_gauss = compute_entropy(x, k=-1)
    H_knn = compute_entropy(x, k=5)

    print(f"  Gaussian H = {H_gauss:.4f}, k-NN H = {H_knn:.4f}")
    assert abs(H_gauss - H_knn) < 0.2, "Gaussian and k-NN estimates should be close"

    MI_gauss = compute_MI(x, y, k=-1)
    MI_knn = compute_MI(x, y, k=5)

    print(f"  Gaussian MI = {MI_gauss[0]:.4f}, k-NN MI = {MI_knn[0]:.4f}")
    # Both should be close to 0 for independent data
    assert abs(MI_gauss[0]) < 0.1, "Gaussian MI should be ~0"

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing entropy_pure module")
    print("=" * 60 + "\n")

    test_entropy_gaussian()
    test_mi_independent()
    test_mi_correlated()
    test_te_causality()
    test_entropy_rate()
    test_relative_entropy()
    test_embedding()
    test_masking()
    test_surrogate()
    test_sampling_params()
    test_gaussian_approximation()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
