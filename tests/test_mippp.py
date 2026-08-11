import numpy as np
import pytest

from src.pysppmix import (
    Window,
    PointPattern,
    rmippp_cond_mark,
    est_mippp_cond_mark,
    rmippp_cond_loc,
    est_mippp_cond_loc,
    get_stats,
)
from src.pysppmix.mippp import _neighbor_mask, _score_matrix

WIN = Window((-10, 10), (-10, 10))


# --- shared geometry helper ---

def test_score_matrix_hand_computed():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    data_marks = np.array([1, 1, 2])
    marks = np.array([1, 2])
    mask = _neighbor_mask(xy, xy, r=10.0, exclude_diag=True)
    scores = _score_matrix(mask, data_marks, marks)
    np.testing.assert_array_equal(scores, [[1, 1], [1, 1], [0, 2]])


def test_neighbor_mask_respects_radius_and_self_exclusion():
    xy = np.array([[0.0, 0.0], [0.05, 0.0], [5.0, 0.0]])
    mask = _neighbor_mask(xy, xy, r=0.1, exclude_diag=True)
    np.testing.assert_array_equal(mask, [[False, True, False], [True, False, False], [False, False, False]])


# --- cond_mark ---

def test_rmippp_cond_mark_shapes_and_mark_counts():
    rng = np.random.default_rng(0)
    sim = rmippp_cond_mark(lam=300, params=[0.2, 0.5, 0.3], window=WIN, rng=rng)
    assert sim.genMPP.marks is not None
    assert set(np.unique(sim.genMPP.marks)) <= {1, 2, 3}
    assert len(sim.ground_surfs) == 3
    assert len(sim.ground_pps) == 3
    assert sum(p.n for p in sim.ground_pps) == sim.genMPP.n


def test_rmippp_cond_mark_rejects_bad_params():
    with pytest.raises(ValueError):
        rmippp_cond_mark(params=[0.3, 0.3])
    with pytest.raises(ValueError):
        rmippp_cond_mark(params=[-0.5, 1.5])


def test_est_mippp_cond_mark_recovers_mark_distribution():
    rng = np.random.default_rng(1)
    true_params = np.array([0.2, 0.5, 0.3])
    sim = rmippp_cond_mark(lam=600, params=true_params, window=WIN, rng=rng)
    m_true = [s.m for s in sim.ground_surfs]

    fit = est_mippp_cond_mark(sim.genMPP, m=m_true, L=1000, rng=rng)
    assert fit.mark_dist.shape == (3,)
    assert fit.mark_dist == pytest.approx(true_params, abs=0.06)
    for j in range(3):
        lo, hi = get_stats(fit.gen_mark_ps[:, j]).credible_set
        assert lo < fit.mark_dist[j] < hi


def test_est_mippp_cond_mark_bdmcmc_path():
    rng = np.random.default_rng(2)
    sim = rmippp_cond_mark(lam=200, params=[0.4, 0.6], window=WIN, rng=rng)
    fit = est_mippp_cond_mark(sim.genMPP, m=5, L=500, rng=rng)  # scalar m -> BDMCMC
    assert fit.fit_bdmcmc
    assert len(fit.ground_fits) == 2
    assert fit.mark_dist.sum() == pytest.approx(1.0)


def test_est_mippp_cond_mark_rejects_bad_args():
    rng = np.random.default_rng(3)
    sim = rmippp_cond_mark(lam=100, params=[0.5, 0.5], window=WIN, rng=rng)
    with pytest.raises(ValueError):
        est_mippp_cond_mark(sim.genMPP, m=[3, 3, 3], L=100)  # wrong length
    with pytest.raises(ValueError):
        est_mippp_cond_mark(sim.genMPP, m=0, L=100)
    with pytest.raises(ValueError):
        est_mippp_cond_mark(sim.genMPP, m=5, hyper=[-1, 1], L=100)


def test_est_mippp_cond_mark_requires_marks():
    pp = PointPattern(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), window=WIN)
    with pytest.raises(ValueError):
        est_mippp_cond_mark(pp, m=[2], L=100)


# --- cond_loc ---

def test_rmippp_cond_loc_gamma_controls_rarity():
    rng = np.random.default_rng(0)
    small_win = Window((-3, 3), (-3, 3))
    sim = rmippp_cond_loc(gammas=[0.05, 3.0], r=0.5, window=small_win, L=20000, rng=rng)
    counts = np.bincount(sim.genMPP.marks)
    # mark 2 has a much larger gamma -> should be much rarer than mark 1
    assert counts[2] < counts[1]


def test_rmippp_cond_loc_rejects_bad_args():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        rmippp_cond_loc(gammas=[1.0], r=-1, rng=rng)


def test_est_mippp_cond_loc_includes_zero_for_independent_marks():
    rng = np.random.default_rng(1)
    n = 100
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    marks = rng.integers(1, 3, size=n)  # discrete uniform, independent of location
    pp = PointPattern(x=x, y=y, window=Window((0, 1), (0, 1)), marks=marks)

    # r kept small (average neighbor count ~3-6) -- pseudo-likelihood coverage
    # degrades in dense/large neighborhoods (see module docstring's "Known
    # limitation"); this isn't the regime that's testing here.
    for r in (0.1, 0.15):
        fit = est_mippp_cond_loc(pp, r=r, hyper=0.2, L=3000, rng=rng)
        for j in range(2):
            lo, hi = get_stats(fit.gen_gammas[fit.burnin :, j]).credible_set
            assert lo < 0 < hi


def test_est_mippp_cond_loc_recovers_known_gammas():
    rng = np.random.default_rng(9)
    win = Window((-3, 3), (-3, 3))
    sim = rmippp_cond_loc(gammas=[0.3, 0.6], r=0.4, window=win, L=30000, rng=rng)

    fit = est_mippp_cond_loc(sim.genMPP, r=sim.r, hyper=0.15, L=3000, rng=rng)
    for j in range(2):
        lo, hi = get_stats(fit.gen_gammas[fit.burnin :, j]).credible_set
        assert lo < sim.gammas[j] < hi


def test_est_mippp_cond_loc_rejects_bad_args():
    rng = np.random.default_rng(2)
    sim = rmippp_cond_loc(gammas=[0.3, 0.6], r=0.4, rng=rng)
    with pytest.raises(ValueError):
        est_mippp_cond_loc(sim.genMPP, r=-1, L=100)
    with pytest.raises(ValueError):
        est_mippp_cond_loc(sim.genMPP, r=0.4, hyper=0, L=100)
    with pytest.raises(ValueError):
        est_mippp_cond_loc(sim.genMPP, r=0.4, start_gamma=[1.0], L=100)  # wrong length


def test_est_mippp_cond_loc_prob_fields_and_prob_at_points_are_valid_distributions():
    rng = np.random.default_rng(3)
    sim = rmippp_cond_loc(gammas=[0.2, 0.5], r=0.4, rng=rng)
    fit = est_mippp_cond_loc(sim.genMPP, r=sim.r, hyper=0.2, L=500, rng=rng)

    fields = fit.prob_fields(LL=16)
    stacked = np.stack([f.z for f in fields], axis=-1)
    np.testing.assert_allclose(stacked.sum(axis=-1), 1.0, atol=1e-8)

    probs = fit.prob_at_points()
    assert probs.shape == (sim.genMPP.n, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)


def test_est_mippp_cond_loc_fit_ground_ippp():
    rng = np.random.default_rng(4)
    sim = rmippp_cond_loc(gammas=[0.3, 0.6], r=0.4, rng=rng)
    fit = est_mippp_cond_loc(sim.genMPP, r=sim.r, L=200, fit_ground_ippp=True, ground_m=2, rng=rng)
    assert fit.ground_fit is not None
    assert fit.ground_fit.m == 2
