import numpy as np
import pytest

from src.pysppmix import (
    NormMix,
    IntensitySurface,
    Window,
    to_int_surf,
    is_normmix,
    is_intensity_surface,
    approx_normmix,
    dnormmix,
    rnormmix,
)


def make_mix():
    return NormMix(
        ps=[0.3, 0.7],
        mus=[[0.2, 0.2], [0.8, 0.8]],
        sigmas=[0.01 * np.eye(2), 0.01 * np.eye(2)],
    )


def test_basic_construction():
    mix = make_mix()
    assert mix.m == 2
    assert is_normmix(mix)
    assert not is_intensity_surface(mix)


def test_ps_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        NormMix(ps=[0.3, 0.3], mus=[[0, 0], [1, 1]], sigmas=[np.eye(2), np.eye(2)])


def test_ps_must_be_nonnegative():
    with pytest.raises(ValueError, match="non-negative"):
        NormMix(ps=[1.3, -0.3], mus=[[0, 0], [1, 1]], sigmas=[np.eye(2), np.eye(2)])


def test_component_count_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        NormMix(ps=[1.0], mus=[[0, 0], [1, 1]], sigmas=[np.eye(2)])


def test_sigma_must_be_positive_definite():
    with pytest.raises(ValueError, match="positive definite"):
        NormMix(ps=[1.0], mus=[[0, 0]], sigmas=[[[1, 2], [2, 1]]])  # eigenvalues -1, 3


def test_sigma_must_be_2x2():
    with pytest.raises(ValueError, match="2x2"):
        NormMix(ps=[1.0], mus=[[0, 0]], sigmas=[np.eye(3)])


def test_to_int_surf_roundtrip():
    mix = make_mix()
    win = Window((0, 1), (0, 1))
    surf = to_int_surf(mix, lam=200, win=win)
    assert is_intensity_surface(surf)
    assert surf.lam == 200
    assert surf.window is win

    back = to_int_surf(surf, return_normmix=True)
    assert is_normmix(back) and not is_intensity_surface(back)


def test_to_int_surf_requires_lam_and_win():
    mix = make_mix()
    with pytest.raises(ValueError):
        to_int_surf(mix)


def test_intensity_surface_requires_positive_lambda():
    with pytest.raises(ValueError, match="greater than 0"):
        IntensitySurface(
            ps=[1.0], mus=[[0, 0]], sigmas=[np.eye(2)], lam=0, window=Window((0, 1), (0, 1))
        )


def test_approx_normmix_full_mass_when_window_much_larger():
    mix = make_mix()
    huge = Window((-100, 100), (-100, 100))
    mass = approx_normmix(mix, huge)
    np.testing.assert_allclose(mass, [1.0, 1.0], atol=1e-6)


def test_dnormmix_grid_shape_and_positivity():
    mix = make_mix()
    grid = dnormmix(mix, xlim=(0, 1), ylim=(0, 1), L=16, truncate=False)
    assert grid.z.shape == (16, 16)
    assert np.all(grid.z >= 0)


def test_dnormmix_requires_normmix_type():
    with pytest.raises(TypeError):
        dnormmix("not a mix", xlim=(0, 1), ylim=(0, 1))


def test_rnormmix_reproducible_with_rng():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    m1 = rnormmix(3, rng=rng1)
    m2 = rnormmix(3, rng=rng2)
    np.testing.assert_allclose(m1.ps, m2.ps)
    assert m1.m == 3


def test_rnormmix_rejects_bad_m():
    with pytest.raises(ValueError):
        rnormmix(0)


def test_rnormmix_dvec_length_mismatch_raises():
    with pytest.raises(ValueError, match="dvec"):
        rnormmix(3, dvec=[1, 1])