import numpy as np
import pytest

from src.pysppmix import NormMix, Window, to_int_surf, rsppmix, gen_n_from_mix


def make_surf(lam=200):
    mix = NormMix(
        ps=[0.3, 0.7],
        mus=[[0.2, 0.2], [0.8, 0.8]],
        sigmas=[0.01 * np.eye(2), 0.01 * np.eye(2)],
    )
    return to_int_surf(mix, lam=lam, win=Window((0, 1), (0, 1)))


def test_gen_n_from_mix_shape():
    surf = make_surf()
    out = gen_n_from_mix(50, surf, rng=np.random.default_rng(0))
    assert out.shape == (50, 3)
    assert set(np.unique(out[:, 2])) <= {0, 1}


def test_gen_n_from_mix_zero_points():
    surf = make_surf()
    out = gen_n_from_mix(0, surf)
    assert out.shape == (0, 3)


def test_rsppmix_truncated_points_all_inside_window():
    surf = make_surf(lam=300)
    pp = rsppmix(surf, truncate=True, rng=np.random.default_rng(1))
    assert np.all(surf.window.contains(pp.x, pp.y))
    assert pp.n > 0


def test_rsppmix_zero_lambda_draw_returns_empty_pattern_not_error():
    # A tiny lambda makes a 0-point Poisson draw likely; it must not raise,
    # unlike the original R implementation (see pointprocess.py docstring).
    surf = make_surf(lam=1e-9)
    pp = rsppmix(surf, rng=np.random.default_rng(2))
    assert pp.n == 0


def test_rsppmix_rejects_plain_normmix():
    mix = NormMix(ps=[1.0], mus=[[0, 0]], sigmas=[np.eye(2)])
    with pytest.raises(TypeError):
        rsppmix(mix)


def test_rsppmix_marks():
    surf = make_surf(lam=100)
    pp = rsppmix(surf, marks=np.array([1, 2, 3]), rng=np.random.default_rng(3))
    assert pp.marks is not None
    assert set(np.unique(pp.marks)) <= {1, 2, 3}