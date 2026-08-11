import numpy as np
import pytest

from src.pysppmix import (
    NormMix,
    Window,
    to_int_surf,
    rsppmix,
    est_mix_damcmc,
    get_map_est,
    get_log_density_values,
    fix_label_switching,
    get_pm_est,
)


def make_well_separated_pattern(rng):
    true_mix = NormMix(
        ps=[0.5, 0.5],
        mus=[[-2.0, -2.0], [2.0, 2.0]],
        sigmas=[0.2 * np.eye(2), 0.2 * np.eye(2)],
    )
    surf = to_int_surf(true_mix, lam=150, win=Window((-5, 5), (-5, 5)))
    pp = rsppmix(surf, rng=rng)
    return true_mix, pp


def match_to_true(true_means, est_means):
    dists = np.linalg.norm(true_means[:, None, :] - est_means[None, :, :], axis=-1)
    order = [0, 1] if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0] else [1, 0]
    return order


# --- GetMAPEst ---

def test_get_map_est_recovers_well_separated_components():
    rng = np.random.default_rng(10)
    true_mix, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=300, rng=rng)

    surf = get_map_est(fit)
    assert surf.estimated
    assert surf.lam == pp.n

    true_means = np.array(true_mix.mus)
    est_means = np.array(surf.mus)
    order = match_to_true(true_means, est_means)
    for true_idx, est_idx in enumerate(order):
        np.testing.assert_allclose(est_means[est_idx], true_means[true_idx], atol=0.6)


def test_get_log_density_values_shape():
    rng = np.random.default_rng(11)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=50, rng=rng)
    vals = get_log_density_values(fit)
    assert vals.shape == (50,)
    assert np.all(np.isfinite(vals))


def test_get_map_est_rejects_bad_hyperparams():
    rng = np.random.default_rng(12)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=30, rng=rng)
    with pytest.raises(ValueError):
        get_map_est(fit, df0=1.0)
    with pytest.raises(ValueError):
        get_map_est(fit, sig0=-1)
    with pytest.raises(ValueError):
        get_map_est(fit, d=[1.0, 1.0, 1.0])  # wrong length


def test_get_map_est_all_burned_in_raises():
    rng = np.random.default_rng(13)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=10, rng=rng)
    with pytest.raises(ValueError):
        get_map_est(fit, burnin=10)


# --- FixLS_da ---

@pytest.mark.parametrize("method", ["ic", "sel"])
def test_fix_label_switching_orders_components_consistently(method):
    rng = np.random.default_rng(20)
    true_mix, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=200, rng=rng)

    fixed = fix_label_switching(fit, method=method)
    assert fixed.L == fit.L - int(0.1 * fit.L)

    # after fixing, per-iteration component identity should be far more stable
    # than before: check the sign of (mu_x for component 0) barely varies.
    x0 = fixed.genmus[:, 0, 0]
    assert x0.std() < 2.0  # well-separated means are ~4 apart; should not flip

    surf = get_pm_est(fixed, burnin=0)
    true_means = np.array(true_mix.mus)
    est_means = np.array(surf.mus)
    order = match_to_true(true_means, est_means)
    for true_idx, est_idx in enumerate(order):
        np.testing.assert_allclose(est_means[est_idx], true_means[true_idx], atol=0.6)


def test_fix_label_switching_rejects_bad_method():
    rng = np.random.default_rng(21)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=20, rng=rng)
    with pytest.raises(ValueError):
        fix_label_switching(fit, method="bogus")


def test_fix_label_switching_single_component_is_noop():
    rng = np.random.default_rng(22)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=1, L=20, rng=rng)
    fixed = fix_label_switching(fit, burnin=0)
    np.testing.assert_array_equal(fixed.genps, fit.genps)


def test_fix_label_switching_ic_orders_by_x():
    rng = np.random.default_rng(23)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=50, rng=rng)
    fixed = fix_label_switching(fit, method="ic", burnin=0)
    assert np.all(fixed.genmus[:, 0, 0] <= fixed.genmus[:, 1, 0])