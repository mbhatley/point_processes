import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.pysppmix import (
    NormMix,
    Window,
    to_int_surf,
    rsppmix,
    dnormmix,
    get_mixture_limits,
    est_mix_damcmc,
    est_mix_bdmcmc,
    plot_density_grid,
    plot_normmix,
    plot_point_pattern,
    plotmix_2d,
    plot_surface_3d,
    plot_chains,
    plot_comp_dist,
    plot_hawkes,
    gen_hawkes,
    plot_region,
    plot_regions,
    EllipseWindow,
    PolygonWindow,
    random_ellipses,
    hppp,
)


def make_surf():
    mix = NormMix(
        ps=[0.3, 0.7],
        mus=[[0.2, 0.2], [0.8, 0.8]],
        sigmas=[0.01 * np.eye(2), 0.01 * np.eye(2)],
    )
    return mix, to_int_surf(mix, lam=150, win=Window((0, 1), (0, 1)))


def test_get_mixture_limits_applies_margin():
    mix, _ = make_surf()
    xlim, ylim = get_mixture_limits(mix, n_sigma=5)
    # component at mu=(0.2,0.2), sigma=0.01*I -> std=0.1, so xmin should be well below 0.2
    assert xlim[0] < 0.2 - 0.4
    assert xlim[1] > 0.8 + 0.4


def test_get_mixture_limits_rejects_non_normmix():
    with pytest.raises(TypeError):
        get_mixture_limits("not a mix")


def test_plot_density_grid_smoke():
    mix, surf = make_surf()
    grid = dnormmix(surf, L=32)
    ax = plot_density_grid(grid)
    assert ax.get_xlabel() == "x"
    ax2 = plot_density_grid(grid, contour=True, grayscale=True)
    assert ax2 is not None


def test_plot_normmix_smoke():
    mix, surf = make_surf()
    ax = plot_normmix(mix)
    assert "Density" in ax.get_title()
    ax2 = plot_normmix(surf, truncate=True, contour=True)
    assert "intensity" in ax2.get_title()


def test_plot_normmix_rejects_bad_type():
    with pytest.raises(TypeError):
        plot_normmix("nope")


def test_plot_point_pattern_smoke():
    _, surf = make_surf()
    pp = rsppmix(surf, rng=np.random.default_rng(0))
    ax = plot_point_pattern(pp, color_by_component=True)
    assert ax.get_legend() is not None
    ax2 = plot_point_pattern(pp, mus=np.array(surf.mus))
    assert ax2 is not None


def test_plotmix_2d_smoke():
    _, surf = make_surf()
    pp = rsppmix(surf, rng=np.random.default_rng(1))
    ax = plotmix_2d(surf, pp, color_by_component=True)
    assert "n=" in ax.get_title()
    assert ax.get_legend() is not None
    ax2 = plotmix_2d(surf)
    assert "n=" not in ax2.get_title()


def test_plotmix_2d_rejects_plain_normmix():
    mix, _ = make_surf()
    with pytest.raises(TypeError):
        plotmix_2d(mix)


def test_plot_surface_3d_smoke():
    mix, surf = make_surf()
    grid = dnormmix(surf, L=20)
    ax = plot_surface_3d(grid)
    assert ax.name == "3d"


def test_plot_chains_damcmc_smoke():
    rng = np.random.default_rng(2)
    _, surf = make_surf()
    pp = rsppmix(surf, rng=rng)
    fit = est_mix_damcmc(pp, m=2, L=60, rng=rng)
    figs = plot_chains(fit)
    assert set(figs.keys()) == {"p", "x", "y"}
    figs2 = plot_chains(fit, separate=False)
    assert set(figs2.keys()) == {"p", "x", "y"}


def test_plot_chains_bdmcmc_smoke():
    rng = np.random.default_rng(3)
    _, surf = make_surf()
    pp = rsppmix(surf, rng=rng)
    fit = est_mix_bdmcmc(pp, m=3, L=200, rng=rng)
    figs = plot_chains(fit)
    assert set(figs.keys()) == {"p", "x", "y"}


def test_plot_chains_rejects_bad_type():
    with pytest.raises(TypeError):
        plot_chains("nope")


def test_plot_comp_dist_smoke():
    rng = np.random.default_rng(4)
    _, surf = make_surf()
    pp = rsppmix(surf, rng=rng)
    fit = est_mix_bdmcmc(pp, m=3, L=200, rng=rng)
    ax1, ax2 = plot_comp_dist(fit)
    assert ax1.get_title().startswith("Distribution")
    assert ax2.get_title().startswith("Generated chain")


def test_plot_hawkes_smoke():
    rng = np.random.default_rng(5)
    H = gen_hawkes(T_max=20, mu=1.0, alpha=0.5, beta=1.0, rng=rng)
    ax = plot_hawkes(H, L=500)
    assert "Hawkes" in ax.get_title()
    assert ax.get_xlabel() == "Time"


def test_plot_region_ellipse_and_polygon_smoke():
    e = EllipseWindow(mu=[0, 0], A=np.eye(2), r=1.0)
    ax = plot_region(e)
    assert len(ax.patches) == 1

    poly = PolygonWindow(vertices=np.array([[0, 0], [1, 0], [0.5, 1]]))
    ax2 = plot_region(poly, boundary=False)
    assert len(ax2.patches) == 1


def test_plot_regions_smoke():
    rng = np.random.default_rng(6)
    base = hppp(10, Window((0, 5), (0, 5)), rng=rng)
    ellipses = random_ellipses(base, rng=rng)
    ax = plot_regions(ellipses, pattern=base)
    assert len(ax.patches) == base.n
