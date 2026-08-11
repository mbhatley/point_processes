## ####################################################
## 2026-08-12: Rev A. Sandbox
## ####################################################

from typing import Callable

import numpy as np

from .pointprocess import PointPattern
from .window import Window


def hppp(lam: float, window: Window, rng: np.random.Generator | None = None) -> PointPattern:
    """Simulate a homogeneous Poisson process over a window.

    :param lam: Intensity (expected points per unit area). Must be > 0.
    :param window: Region to sample over.
    :param rng: Random number generator; a fresh default generator is
        created if not given.
    :returns: The simulated point pattern.
    :raises ValueError: If ``lam`` is not > 0.
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    rng = rng or np.random.default_rng()
    n = rng.poisson(lam * window.area)
    x, y = window.sample_uniform(n, rng)
    return PointPattern(x=x, y=y, window=window)


def ippp(
    intensity: Callable[[float, float], float],
    window: Window,
    grid_size: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate an inhomogeneous Poisson process by thinning a dominating homogeneous process.

    The intensity function is evaluated on a ``grid_size`` x ``grid_size``
    grid to find its maximum, which bounds a homogeneous Poisson process;
    points are then kept independently with probability
    ``intensity(x, y) / max_intensity``.

    :param intensity: Intensity function taking ``(x, y)`` and returning a
        non-negative rate.
    :param window: Region to sample over.
    :param grid_size: Number of grid points per axis used to estimate the
        maximum intensity.
    :param rng: Random number generator; a fresh default generator is
        created if not given.
    :returns: A dict with ``max_intensity`` (the estimated bound),
        ``dominating`` (the pre-thinning :class:`PointPattern`), and ``pp``
        (the thinned result).
    :raises ValueError: If the intensity is <= 0 everywhere on the grid, or
        if the dominating draw has zero points.
    """
    rng = rng or np.random.default_rng()
    gx = np.linspace(window.xrange[0], window.xrange[1], grid_size)
    gy = np.linspace(window.yrange[0], window.yrange[1], grid_size)
    z = np.array([[intensity(x, y) for y in gy] for x in gx])
    max_lambda = z.max()
    if max_lambda <= 0:
        raise ValueError("intensity must exceed 0 somewhere on the grid.")

    dominating = hppp(max_lambda, window, rng=rng)
    if dominating.n == 0:
        raise ValueError("dominating HPPP draw had zero points -- try again.")

    vals = np.array([intensity(x, y) for x, y in zip(dominating.x, dominating.y)])
    keep = rng.random(dominating.n) < vals / max_lambda
    pp = PointPattern(x=dominating.x[keep], y=dominating.y[keep], window=window)
    return {"max_intensity": max_lambda, "dominating": dominating, "pp": pp}
