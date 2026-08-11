import numpy as np
import pytest

from src.pysppmix import (
    Window,
    EllipseWindow,
    PolygonWindow,
    hppp,
    ippp,
    random_ellipse_window,
    random_ellipses,
    random_polygon_window,
    random_polygons,
)


def test_ellipse_circle_geometry():
    e = EllipseWindow(mu=[0, 0], A=np.eye(2), r=2.0)
    assert e.area == pytest.approx(np.pi * 4)
    assert e.xrange == pytest.approx((-2.0, 2.0))
    assert e.yrange == pytest.approx((-2.0, 2.0))
    assert bool(e.contains(0, 0))
    assert not bool(e.contains(3, 0))


def test_ellipse_anisotropic_geometry():
    e = EllipseWindow(mu=[1, 1], A=np.array([[4.0, 0.0], [0.0, 1.0]]), r=1.0)
    assert e.area == pytest.approx(np.pi / 2)
    assert e.xrange == pytest.approx((0.5, 1.5))
    assert e.yrange == pytest.approx((0.0, 2.0))


def test_ellipse_boundary_matches_semi_axes():
    e = EllipseWindow(mu=[0, 0], A=np.array([[4.0, 0.0], [0.0, 1.0]]), r=1.0)
    b = e.boundary(L=1000)
    assert b[:, 0].max() == pytest.approx(0.5, abs=1e-3)
    assert b[:, 1].max() == pytest.approx(1.0, abs=1e-3)
    assert np.all(e.contains(b[:, 0] * 0.999, b[:, 1] * 0.999))  # just inside


def test_ellipse_rejects_bad_params():
    with pytest.raises(ValueError):
        EllipseWindow(mu=[0, 0], A=np.eye(2), r=-1)
    with pytest.raises(ValueError):
        EllipseWindow(mu=[0, 0], A=np.array([[1, 2], [3, 4]]), r=1)  # not symmetric
    with pytest.raises(ValueError):
        EllipseWindow(mu=[0, 0], A=np.array([[-1, 0], [0, 1]]), r=1)  # not PD


def test_ellipse_sample_uniform_all_inside():
    rng = np.random.default_rng(0)
    e = EllipseWindow(mu=[2, -1], A=np.array([[2.0, 0.5], [0.5, 1.0]]), r=1.5)
    x, y = e.sample_uniform(500, rng)
    assert x.shape == (500,)
    assert np.all(e.contains(x, y))


def test_polygon_square_geometry():
    sq = PolygonWindow(vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    assert sq.area == pytest.approx(1.0)
    assert sq.xrange == pytest.approx((0.0, 1.0))
    assert sq.yrange == pytest.approx((0.0, 1.0))
    assert bool(sq.contains(0.5, 0.5)[0])
    assert not bool(sq.contains(2.0, 2.0)[0])


def test_polygon_rejects_too_few_vertices():
    with pytest.raises(ValueError):
        PolygonWindow(vertices=np.array([[0, 0], [1, 1]]))


def test_polygon_from_polar_recenters_to_mu():
    lengths = np.array([1.0, 1.0, 1.0])
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    poly = PolygonWindow.from_polar(mu=[5, 5], lengths=lengths, angles=angles)
    np.testing.assert_allclose(poly.vertices.mean(axis=0), [5, 5], atol=1e-10)


def test_polygon_sample_uniform_all_inside():
    rng = np.random.default_rng(1)
    poly = PolygonWindow(vertices=np.array([[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]]))
    x, y = poly.sample_uniform(300, rng)
    assert np.all(poly.contains(x, y))


def test_hppp_ippp_work_over_ellipse():
    rng = np.random.default_rng(2)
    e = EllipseWindow(mu=[0, 0], A=np.eye(2), r=1.0)
    pp = hppp(50, e, rng=rng)
    assert np.all(e.contains(pp.x, pp.y))

    def intensity(x, y):
        return 100.0

    result = ippp(intensity, e, grid_size=30, rng=rng)
    assert np.all(e.contains(result["pp"].x, result["pp"].y))


def test_hppp_ippp_work_over_polygon():
    rng = np.random.default_rng(3)
    poly = PolygonWindow(vertices=np.array([[0, 0], [2, 0], [1, 2]]))
    pp = hppp(30, poly, rng=rng)
    assert np.all(poly.contains(pp.x, pp.y))


def test_random_ellipse_window_uses_wishart_when_A_given():
    rng = np.random.default_rng(4)
    e = random_ellipse_window(mu=[0, 0], r=1.0, A=np.eye(2), df=10, rng=rng)
    assert e.area > 0
    e_default = random_ellipse_window(mu=[0, 0], r=1.0, rng=rng)
    np.testing.assert_array_equal(e_default.A, np.eye(2))


def test_random_ellipses_uses_marks_as_radii():
    rng = np.random.default_rng(5)
    base = hppp(20, Window((0, 5), (0, 5)), rng=rng)
    from pysppmix import PointPattern

    marked = PointPattern(x=base.x, y=base.y, window=base.window, marks=np.full(base.n, 0.7))
    ellipses = random_ellipses(marked, rng=rng)
    assert len(ellipses) == base.n
    for e in ellipses:
        assert e.r == pytest.approx(0.7)


def test_random_polygon_window_num_vert_bounds():
    rng = np.random.default_rng(6)
    p = random_polygon_window(mu=[0, 0], r=1.0, num_vert=3, rng=rng)
    assert p.vertices.shape[0] == 3
    p2 = random_polygon_window(mu=[0, 0], r=1.0, num_vert=6, set_all=True, rng=rng)
    assert p2.vertices.shape[0] == 6


def test_random_polygons_matches_pattern_count():
    rng = np.random.default_rng(7)
    base = hppp(10, Window((0, 5), (0, 5)), rng=rng)
    polys = random_polygons(base, num_vert=5, rng=rng)
    assert len(polys) == base.n
