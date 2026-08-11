import numpy as np
import pytest
from scipy.spatial.distance import pdist

from src.pysppmix import (
    Window,
    gen_markov_pp,
    gen_markov_pp_bd,
    markov_pp_bd,
    matern_hc_models,
    fit_strauss,
    bayesian_markov_pp_mh,
    count_pairs,
    min_dist,
    hppp,
)

WIN = Window((0, 1), (0, 1))


def test_count_pairs_known_points():
    pts = np.array([[0, 0], [0, 0.05], [0, 1.0]])
    assert count_pairs(pts, 0.1) == 1  # only the first pair is within 0.1
    assert count_pairs(pts, 2.0) == 3  # all 3 pairs within 2.0


def test_min_dist_known_points():
    pts = np.array([[0, 0], [0, 0.3], [0, 1.0]])
    assert min_dist(pts) == pytest.approx(0.3)


def test_min_dist_requires_two_points():
    with pytest.raises(ValueError):
        min_dist(np.array([[0, 0]]))


def test_gen_markov_pp_hardcore_respects_min_distance():
    rng = np.random.default_rng(0)
    pp = gen_markov_pp((0.1,), start_n=20, b=50, func_choice=1, window=WIN, L=1500, rng=rng)
    xy = np.column_stack([pp.x, pp.y])
    assert pdist(xy).min() >= 0.1 - 1e-9


def test_gen_markov_pp_strauss_runs():
    rng = np.random.default_rng(1)
    pp = gen_markov_pp((0.05, 0.5), start_n=15, b=60, func_choice=2, window=WIN, L=800, rng=rng)
    assert pp.n == 15


def test_gen_markov_pp_rejects_bad_args():
    with pytest.raises(ValueError):
        gen_markov_pp((0.1,), start_n=0, b=50, func_choice=1, window=WIN)
    with pytest.raises(ValueError):
        gen_markov_pp((0.1,), start_n=10, b=-1, func_choice=1, window=WIN)


def test_gen_markov_pp_bd_hardcore_respects_min_distance():
    rng = np.random.default_rng(2)
    pp = gen_markov_pp_bd((0.05,), start_n=15, b=80, move_setup=1, func_choice=1, window=WIN, L=1500, rng=rng)
    xy = np.column_stack([pp.x, pp.y])
    if pp.n > 1:
        assert pdist(xy).min() >= 0.05 - 1e-9


def test_gen_markov_pp_bd_save_last_iters():
    rng = np.random.default_rng(3)
    pp, saved = gen_markov_pp_bd(
        (0.05,), start_n=10, b=60, move_setup=1, func_choice=1, window=WIN, L=200, save_last_iters=20, rng=rng
    )
    assert len(saved) == 20
    assert all(isinstance(s, np.ndarray) and s.shape[1] == 2 for s in saved)


def test_gen_markov_pp_bd_change_only_keeps_n_fixed():
    rng = np.random.default_rng(4)
    pp = gen_markov_pp_bd((0.02,), start_n=12, b=60, move_setup=3, func_choice=1, window=WIN, L=500, rng=rng)
    assert pp.n == 12


def test_gen_markov_pp_bd_rejects_invalid_move_probs():
    with pytest.raises(ValueError):
        gen_markov_pp_bd((0.05,), start_n=10, b=60, move_setup=2, window=WIN, L=10, birth_prob=0.7, change_prob=0.5)


def test_markov_pp_bd_default_hardcore_respects_min_distance():
    rng = np.random.default_rng(5)
    pp = markov_pp_bd(main_params=(80.0,), inter_params=(0.05,), start_n=15, window=WIN, L=1500, rng=rng)
    xy = np.column_stack([pp.x, pp.y])
    if pp.n > 1:
        assert pdist(xy).min() >= 0.05 - 1e-9


def test_matern_hc_type1_thins_and_respects_no_new_points():
    rng = np.random.default_rng(6)
    base = hppp(150, WIN, rng=rng)
    res = matern_hc_models(base, type=1, r=0.05, rng=rng)
    assert res["thinned"].n <= base.n


def test_matern_hc_type2_and_type3():
    rng = np.random.default_rng(7)
    base = hppp(150, WIN, rng=rng)
    res2 = matern_hc_models(base, type=2, r=0.05, rng=rng)
    assert res2["thinned"].n <= base.n
    res3 = matern_hc_models(base, type=3, r=0.05, R_x=rng.gamma(1, 1, size=base.n), rng=rng)
    assert res3["thinned"].n <= base.n


def test_matern_hc_rejects_bad_type():
    with pytest.raises(ValueError):
        matern_hc_models(hppp(50, WIN), type=4)


def test_matern_hc_type3_requires_R_x():
    with pytest.raises(ValueError):
        matern_hc_models(hppp(50, WIN), type=3)


def make_hardcore_data(rng):
    return gen_markov_pp((0.05,), start_n=40, b=100, func_choice=1, window=WIN, L=1500, rng=rng)


def test_fit_strauss_shapes():
    rng = np.random.default_rng(8)
    pp = make_hardcore_data(rng)
    fit = fit_strauss(pp, L=500, rng=rng)
    assert fit.b_gens.shape == (500,)
    assert fit.r_gens.shape == (500,)
    assert fit.gamma_gens.shape == (500,)
    stats = fit.stats()
    assert set(stats) == {"b", "r", "gamma"}


def test_fit_strauss_rejects_bad_burnin():
    rng = np.random.default_rng(9)
    pp = make_hardcore_data(rng)
    with pytest.raises(ValueError):
        fit_strauss(pp, L=100, burnin=100, rng=rng)


def test_fit_strauss_requires_points():
    rng = np.random.default_rng(10)
    from pysppmix import PointPattern

    empty = PointPattern(x=np.empty(0), y=np.empty(0), window=WIN)
    with pytest.raises(ValueError):
        fit_strauss(empty, L=100, rng=rng)


@pytest.mark.parametrize(
    "thetas,model_choice",
    [
        ((0.05, 0), 1),
        ((0.05, 0), 2),  # sufficient-stat auxiliary ksi
        ((0.05, 0.03, 25), 2),  # fixed ksi_r/ksi_b
        ((0.05, -1), 2),  # random-walk MH
        ((0.05, 0), 3),  # hierarchical Bayes on beta
    ],
)
def test_bayesian_markov_pp_mh_all_model_choices_run(thetas, model_choice):
    rng = np.random.default_rng(11)
    pp = gen_markov_pp((0.05,), start_n=20, b=60, func_choice=1, window=WIN, L=800, rng=rng)
    fit = bayesian_markov_pp_mh(pp, thetas=thetas, b=60, model_choice=model_choice, L=200, rng=rng)
    assert fit.b_gens.shape == (200,)
    assert fit.r_gens.shape == (200,)
    if model_choice == 3:
        assert fit.beta_gens is not None
        assert np.all(np.isfinite(fit.r_gens))  # the fixed R bug (r_gens mostly 0) should NOT reproduce here
        assert not np.all(fit.r_gens[1:] == 0)


def test_bayesian_markov_pp_mh_rejects_bad_model_choice():
    rng = np.random.default_rng(12)
    pp = make_hardcore_data(rng)
    with pytest.raises(ValueError):
        bayesian_markov_pp_mh(pp, model_choice=4, L=100, rng=rng)
