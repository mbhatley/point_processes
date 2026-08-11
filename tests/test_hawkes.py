import numpy as np
import pytest

from src.pysppmix import gen_hawkes, hawkes_intensity, int_meas_hawkes, fit_hawkes


def test_gen_hawkes_events_within_window_and_sorted():
    rng = np.random.default_rng(0)
    H = gen_hawkes(T_max=50, mu=1.0, alpha=0.5, beta=2.0, rng=rng)
    assert np.all(H.times >= 0) and np.all(H.times <= 50)
    assert np.all(np.diff(H.times) >= 0)


def test_gen_hawkes_rejects_unstable_branching_ratio():
    with pytest.raises(ValueError):
        gen_hawkes(T_max=10, mu=1.0, alpha=2.0, beta=1.0)  # alpha >= beta


def test_gen_hawkes_rejects_bad_params():
    with pytest.raises(ValueError):
        gen_hawkes(T_max=0, mu=1.0, alpha=0.5, beta=1.0)
    with pytest.raises(ValueError):
        gen_hawkes(T_max=10, mu=0, alpha=0.5, beta=1.0)
    with pytest.raises(ValueError):
        gen_hawkes(T_max=10, mu=1.0, alpha=-0.1, beta=1.0)


def test_hawkes_intensity_baseline_with_no_history():
    ints = hawkes_intensity(np.array([1.0, 2.0]), np.array([]), mu=3.0, alpha=0.5, beta=1.0)
    np.testing.assert_allclose(ints, [3.0, 3.0])


def test_hawkes_intensity_only_counts_past_events():
    hist = np.array([1.0, 5.0])  # 5.0 is in the future relative to t=2
    ints = hawkes_intensity(np.array([2.0]), hist, mu=1.0, alpha=1.0, beta=1.0)
    expected = 1.0 + 1.0 * np.exp(-1.0 * (2.0 - 1.0))  # only the t=1 event counts
    np.testing.assert_allclose(ints, [expected])


def test_int_meas_hawkes_matches_numeric_integration():
    hist = np.array([0.5, 1.5, 2.0])
    mu, alpha, beta = 1.5, 0.8, 2.0
    T = 5.0
    ts = np.linspace(0, T, 200001)
    numeric = np.trapezoid(hawkes_intensity(ts, hist, mu, alpha, beta), ts)
    closed_form = int_meas_hawkes(T, hist, mu, alpha, beta)
    assert closed_form == pytest.approx(numeric, rel=1e-3)


def test_fit_hawkes_recovers_known_parameters():
    rng = np.random.default_rng(42)
    true_mu, true_alpha, true_beta = 2.0, 1.0, 3.0
    H = gen_hawkes(T_max=400, mu=true_mu, alpha=true_alpha, beta=true_beta, rng=rng)

    fit = fit_hawkes(
        H.times, T_max=400, L=6000, start_vals=(1.0, 0.3, 1.5),
        hyperparams=(0.1, 0.1, 0.1), small_sig=0.12, rng=rng,
    )
    stats = fit.stats()
    lo, hi = stats["mu"].credible_set
    assert lo < true_mu < hi
    lo, hi = stats["alpha"].credible_set
    assert lo < true_alpha < hi
    lo, hi = stats["beta"].credible_set
    assert lo < true_beta < hi


def test_fit_hawkes_rejects_bad_args():
    hist = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fit_hawkes(hist, T_max=10, L=100, burnin=100)
    with pytest.raises(ValueError):
        fit_hawkes(hist, T_max=10, L=100, small_sig=0)
    with pytest.raises(ValueError):
        fit_hawkes(hist, T_max=10, L=100, hyperparams=(1, -1, 1))


def test_fit_hawkes_shapes():
    rng = np.random.default_rng(3)
    H = gen_hawkes(T_max=50, mu=1.0, alpha=0.3, beta=1.0, rng=rng)
    fit = fit_hawkes(H.times, T_max=50, L=300, rng=rng)
    assert fit.mu_gens.shape == (300,)
    assert fit.alpha_gens.shape == (300,)
    assert fit.beta_gens.shape == (300,)
    assert np.all(fit.mu_gens > 0) and np.all(fit.alpha_gens > 0) and np.all(fit.beta_gens > 0)
