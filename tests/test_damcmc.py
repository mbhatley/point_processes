import numpy as np
import pytest

from src.pysppmix import (
    NormMix,
    Window,
    to_int_surf,
    rsppmix,
    est_mix_damcmc,
    drop_realization,
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


def test_damcmc_recovers_well_separated_components():
    rng = np.random.default_rng(123)
    true_mix, pp = make_well_separated_pattern(rng)

    fit = est_mix_damcmc(pp, m=2, L=300, rng=rng)
    assert fit.genps.shape == (300, 2)
    assert fit.genmus.shape == (300, 2, 2)
    assert fit.gensigmas.shape == (300, 2, 2, 2)
    assert fit.genzs.shape == (300, pp.n)

    surf_est = get_pm_est(fit)
    assert surf_est.estimated

    # match recovered components to the true ones by nearest mean (label switching
    # is expected and not yet corrected -- see FixLS_da, not ported)
    true_means = np.array(true_mix.mus)
    est_means = np.array(surf_est.mus)
    dists = np.linalg.norm(true_means[:, None, :] - est_means[None, :, :], axis=-1)
    row_ind = [0, 1] if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0] else [1, 0]
    for true_idx, est_idx in enumerate(row_ind):
        np.testing.assert_allclose(est_means[est_idx], true_means[true_idx], atol=0.5)


def test_damcmc_rejects_bad_args():
    rng = np.random.default_rng(0)
    _, pp = make_well_separated_pattern(rng)

    with pytest.raises(ValueError):
        est_mix_damcmc(pp, m=0, L=10)
    with pytest.raises(ValueError):
        est_mix_damcmc(pp, m=2, L=0)
    with pytest.raises(ValueError):
        est_mix_damcmc(pp, m=2, L=10, hyper_da=(3, 1, -1))  # df0 negative
    with pytest.raises(ValueError):
        est_mix_damcmc(pp, m=10_000, L=10)  # more components than points


def test_drop_realization_by_count():
    rng = np.random.default_rng(1)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=50, rng=rng)
    burned = drop_realization(fit, 10)
    assert burned.L == 40
    np.testing.assert_array_equal(burned.genps, fit.genps[10:])


def test_drop_realization_by_mask():
    rng = np.random.default_rng(2)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=20, rng=rng)
    mask = np.zeros(20, dtype=bool)
    mask[::2] = True  # drop even iterations
    kept = drop_realization(fit, mask)
    assert kept.L == 10
    np.testing.assert_array_equal(kept.genps, fit.genps[~mask])


def test_damcmc_with_truncation_runs():
    rng = np.random.default_rng(3)
    true_mix, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=20, truncate=True, rng=rng)
    assert np.all(fit.approx_mass > 0)


def test_damcmc_useKmeans_runs():
    rng = np.random.default_rng(4)
    _, pp = make_well_separated_pattern(rng)
    fit = est_mix_damcmc(pp, m=2, L=20, use_kmeans=True, rng=rng)
    assert fit.L == 20
