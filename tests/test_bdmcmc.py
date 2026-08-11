import numpy as np
import pytest

from src.pysppmix import (
    NormMix,
    Window,
    to_int_surf,
    rsppmix,
    est_mix_bdmcmc,
    get_bd_table,
    get_bd_compfit,
    get_pm_est,
    get_map_est,
    fix_label_switching,
    drop_realization,
)


def make_two_component_pattern(rng):
    true_mix = NormMix(
        ps=[0.5, 0.5],
        mus=[[-3.0, -3.0], [3.0, 3.0]],
        sigmas=[0.2 * np.eye(2), 0.2 * np.eye(2)],
    )
    surf = to_int_surf(true_mix, lam=150, win=Window((-6, 6), (-6, 6)))
    pp = rsppmix(surf, rng=rng)
    return true_mix, pp


def test_bdmcmc_shapes_and_numcomp_range():
    rng = np.random.default_rng(0)
    _, pp = make_two_component_pattern(rng)
    fit = est_mix_bdmcmc(pp, m=5, L=400, rng=rng)

    assert fit.genps.shape == (400, 5)
    assert fit.genmus.shape == (400, 5, 2)
    assert fit.gensigmas.shape == (400, 5, 2, 2)
    assert fit.genzs.shape == (400, pp.n)
    assert fit.numcomp.shape == (400,)
    assert np.all(fit.numcomp >= 1) and np.all(fit.numcomp <= 5)
    # inactive component slots are zero-padded
    for l in range(400):
        k = fit.numcomp[l]
        np.testing.assert_array_equal(fit.genps[l, k:], 0)


def test_bdmcmc_recovers_true_number_of_components():
    rng = np.random.default_rng(1)
    _, pp = make_two_component_pattern(rng)
    fit = est_mix_bdmcmc(pp, m=5, L=1500, rng=rng)
    tab = get_bd_table(fit, show=False)
    assert tab["MAPcomp"] == 2


def test_bdmcmc_maxnumcomp_one_never_births():
    rng = np.random.default_rng(2)
    _, pp = make_two_component_pattern(rng)
    fit = est_mix_bdmcmc(pp, m=1, L=50, rng=rng)
    assert np.all(fit.numcomp == 1)


def test_bdmcmc_rejects_bad_args():
    rng = np.random.default_rng(3)
    _, pp = make_two_component_pattern(rng)
    with pytest.raises(ValueError):
        est_mix_bdmcmc(pp, m=0, L=10)
    with pytest.raises(ValueError):
        est_mix_bdmcmc(pp, m=3, L=0)
    with pytest.raises(ValueError):
        est_mix_bdmcmc(pp, m=3, L=10, lambda1=-1)
    with pytest.raises(ValueError):
        est_mix_bdmcmc(pp, m=3, L=10, hyper_bd=(3, 1, -1))


def test_get_bd_table_frequencies_sum_to_L_after_burnin():
    rng = np.random.default_rng(4)
    _, pp = make_two_component_pattern(rng)
    fit = est_mix_bdmcmc(pp, m=4, L=300, rng=rng)
    burned = drop_realization(fit, 30)
    tab = get_bd_table(burned, show=False)
    assert tab["FreqTab"].sum() == burned.L
    assert 1 <= tab["MAPcomp"] <= 4


def test_get_bd_compfit_returns_damcmc_result_with_matching_m():
    rng = np.random.default_rng(5)
    true_mix, pp = make_two_component_pattern(rng)
    fit = est_mix_bdmcmc(pp, m=5, L=1500, rng=rng)
    sub = get_bd_compfit(fit, num_comp=2)
    assert sub.m == 2
    assert sub.genps.shape[1] == 2
    assert sub.genmus.shape[1] == 2

    # downstream post-processing functions should all work unchanged
    pm = get_pm_est(sub, burnin=0)
    mp = get_map_est(sub, burnin=0)
    fixed = fix_label_switching(sub, burnin=0, method="ic")
    assert pm.m == 2 and mp.m == 2 and fixed.m == 2

    true_means = np.array(true_mix.mus)
    est_means = np.array(pm.mus)
    dists = np.linalg.norm(true_means[:, None, :] - est_means[None, :, :], axis=-1)
    order = [0, 1] if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0] else [1, 0]
    for true_idx, est_idx in enumerate(order):
        np.testing.assert_allclose(est_means[est_idx], true_means[true_idx], atol=0.8)


def test_get_bd_compfit_rejects_out_of_range_num_comp():
    rng = np.random.default_rng(6)
    _, pp = make_two_component_pattern(rng)
    fit = est_mix_bdmcmc(pp, m=3, L=100, rng=rng)
    with pytest.raises(ValueError):
        get_bd_compfit(fit, num_comp=10)
