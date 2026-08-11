from dataclasses import dataclass

import numpy as np

from .normmix import IntensitySurface, is_intensity_surface
from .window import Window

_MAX_TRUNCATE_ATTEMPTS = 200


@dataclass
class PointPattern:
    """A 2D point pattern, with optional per-point component and mark labels.

    :ivar x: X coordinates of the points.
    :ivar y: Y coordinates of the points.
    :ivar window: Observation window the points were generated over.
    :ivar comp: Index of the mixture component that generated each point, if known.
    :ivar marks: Mark value attached to each point, if any.
    """

    x: np.ndarray
    y: np.ndarray
    window: Window
    comp: np.ndarray | None = None
    marks: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Number of points in the pattern.

        :returns: ``len(x)``.
        """
        return len(self.x)

    def __repr__(self) -> str:
        return f"Point pattern with {self.n} point(s) in {self.window!r}"


def gen_n_from_mix(n: int, mix: IntensitySurface, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a fixed number of points i.i.d. from a mixture, ignoring the window.

    :param n: Number of points to draw.
    :param mix: Mixture to draw from.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: Array of shape ``(n, 3)`` with columns x, y, and the generating component index.
    :raises ValueError: if ``n < 0``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = rng or np.random.default_rng()
    if n == 0:
        return np.empty((0, 3))

    comp = rng.choice(mix.m, size=n, replace=True, p=mix.ps)
    out = np.empty((n, 3))
    out[:, 2] = comp
    for k in range(mix.m):
        mask = comp == k
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        out[mask, :2] = rng.multivariate_normal(mix.mus[k], mix.sigmas[k], size=cnt)
    return out


def rsppmix(
    intsurf: IntensitySurface,
    truncate: bool = True,
    marks: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> PointPattern:
    """Simulate a point pattern from a Poisson process with a mixture intensity.

    :param intsurf: Intensity surface to simulate from.
    :param truncate: If ``True``, resample until exactly the drawn Poisson count of points fall
        inside the window. If ``False``, keep all drawn points (some may fall outside the window)
        and warn if any do.
    :param marks: Pool of possible mark values; if given, each point is independently assigned one
        uniformly at random.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: The simulated point pattern.
    :raises TypeError: if ``intsurf`` is not an ``IntensitySurface``.
    :raises ValueError: if ``marks`` is not a 1-D array.
    :raises RuntimeError: if ``truncate`` is ``True`` and enough in-window points could not be
        generated within the resampling attempt limit.
    """
    if not is_intensity_surface(intsurf):
        raise TypeError("intsurf must be an IntensitySurface (see to_int_surf()).")
    rng = rng or np.random.default_rng()
    win = intsurf.window

    n = rng.poisson(intsurf.lam)
    if n == 0:
        return PointPattern(x=np.empty(0), y=np.empty(0), window=win, comp=np.empty(0, dtype=int))

    spp = gen_n_from_mix(n, intsurf, rng=rng)

    if truncate:
        attempts = 0
        inside = win.contains(spp[:, 0], spp[:, 1])
        while inside.sum() < n:
            attempts += 1
            if attempts > _MAX_TRUNCATE_ATTEMPTS:
                raise RuntimeError(
                    f"Could not generate {n} points inside {win!r} after {_MAX_TRUNCATE_ATTEMPTS} "
                    "resampling attempts -- the mixture places negligible mass inside the window."
                )
            spp = np.vstack([spp, gen_n_from_mix(n, intsurf, rng=rng)])
            inside = win.contains(spp[:, 0], spp[:, 1])
        spp = spp[inside][:n]
    else:
        n_outside = int((~win.contains(spp[:, 0], spp[:, 1])).sum())
        if n_outside:
            import warnings

            warnings.warn(f"{n_outside} point(s) are outside the window.", stacklevel=2)

    point_marks = None
    if marks is not None:
        marks = np.asarray(marks)
        if marks.ndim != 1:
            raise ValueError("marks must be a 1-D array of possible mark values.")
        tags = rng.choice(len(marks), size=len(spp), replace=True)
        point_marks = marks[tags]

    return PointPattern(
        x=spp[:, 0], y=spp[:, 1], window=win, comp=spp[:, 2].astype(int), marks=point_marks
    )