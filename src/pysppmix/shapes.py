from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy import stats as scipy_stats

from .pointprocess import PointPattern

_MAX_REJECTION_ROUNDS = 200


@dataclass(frozen=True)
class EllipseWindow:
    """A (possibly rotated) ellipse: the Mahalanobis-distance level set ``{x : (x-mu)^T A (x-mu) <= r^2}``.

    This is the same form as a Gaussian confidence ellipse (``A =
    inv(Sigma)``), so it connects directly to a mixture component's
    covariance if one is used to derive the shape matrix.

    :ivar mu: Center of the ellipse.
    :ivar A: Symmetric positive-definite shape matrix.
    :ivar r: Mahalanobis radius.
    :raises ValueError: if ``A`` is not 2x2, symmetric, and positive definite, or if ``r <= 0``.
    """

    mu: np.ndarray
    A: np.ndarray
    r: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu", np.asarray(self.mu, dtype=float).reshape(2))
        A = np.asarray(self.A, dtype=float)
        if A.shape != (2, 2):
            raise ValueError(f"A must be 2x2, got shape {A.shape}")
        if not np.allclose(A, A.T, atol=1e-8):
            raise ValueError(f"A must be symmetric, got {A}")
        if np.any(np.linalg.eigvalsh(A) <= 0):
            raise ValueError(f"A must be positive definite, got eigenvalues {np.linalg.eigvalsh(A)}")
        object.__setattr__(self, "A", A)
        if self.r <= 0:
            raise ValueError(f"r must be > 0, got {self.r}")

    @property
    def _A_inv(self) -> np.ndarray:
        """Inverse of the shape matrix ``A``.

        :returns: ``inv(A)``.
        """
        return np.linalg.inv(self.A)

    @property
    def area(self) -> float:
        """Area of the ellipse.

        :returns: ``pi * r^2 / sqrt(det(A))``.
        """
        return float(np.pi * self.r**2 / np.sqrt(np.linalg.det(self.A)))

    @property
    def xrange(self) -> tuple[float, float]:
        """Bounding-box extent along x.

        :returns: ``(xmin, xmax)`` of the ellipse's bounding box.
        """
        half = self.r * np.sqrt(self._A_inv[0, 0])
        return (self.mu[0] - half, self.mu[0] + half)

    @property
    def yrange(self) -> tuple[float, float]:
        """Bounding-box extent along y.

        :returns: ``(ymin, ymax)`` of the ellipse's bounding box.
        """
        half = self.r * np.sqrt(self._A_inv[1, 1])
        return (self.mu[1] - half, self.mu[1] + half)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Test which points fall inside the ellipse.

        :param x: X coordinates to test.
        :param y: Y coordinates to test.
        :returns: Boolean mask, ``True`` where the point's Mahalanobis distance is within ``r``.
        """
        x = np.asarray(x, dtype=float) - self.mu[0]
        y = np.asarray(y, dtype=float) - self.mu[1]
        a, b, d = self.A[0, 0], self.A[0, 1], self.A[1, 1]
        quad = a * x**2 + 2 * b * x * y + d * y**2
        return quad <= self.r**2

    def boundary(self, L: int = 50) -> np.ndarray:
        """Trace the ellipse boundary with evenly spaced points.

        :param L: Number of boundary segments; ``L + 1`` points are returned (first and last coincide).
        :returns: Array of shape ``(L + 1, 2)`` of boundary point coordinates.
        """
        thetas = np.linspace(0, 2 * np.pi, L + 1)
        circle = self.r * np.column_stack([np.cos(thetas), np.sin(thetas)])
        # A = U^T U with upper-triangular U; boundary = mu + U^{-1} y
        U = np.linalg.cholesky(self.A).T
        U_inv = np.linalg.inv(U)
        return circle @ U_inv.T + self.mu

    def sample_uniform(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw points uniformly at random from inside the ellipse via rejection sampling in its bounding box.

        :param n: Number of points to draw.
        :param rng: Random number generator to draw from.
        :returns: The ``x`` and ``y`` coordinates of the drawn points.
        :raises RuntimeError: if ``n`` points could not be sampled within the rejection-sampling round limit.
        """
        xs = np.empty(0)
        ys = np.empty(0)
        for _ in range(_MAX_REJECTION_ROUNDS):
            need = n - len(xs)
            if need <= 0:
                break
            cx = rng.uniform(*self.xrange, need * 2)
            cy = rng.uniform(*self.yrange, need * 2)
            keep = self.contains(cx, cy)
            xs = np.concatenate([xs, cx[keep]])
            ys = np.concatenate([ys, cy[keep]])
        else:
            raise RuntimeError(f"Could not sample {n} points inside the ellipse in {_MAX_REJECTION_ROUNDS} rounds.")
        return xs[:n], ys[:n]

    def __repr__(self) -> str:
        return f"EllipseWindow(mu={self.mu.tolist()}, r={self.r})"


@dataclass(frozen=True)
class PolygonWindow:
    """A simple polygon given its vertices, in order.

    :ivar vertices: Polygon vertices, shape ``(N, 2)`` with ``N >= 3``.
    :raises ValueError: if fewer than 3 vertices are given, or the array isn't shape ``(N, 2)``.
    """

    vertices: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
            raise ValueError(f"vertices must be an (N, 2) array with N >= 3, got shape {v.shape}")
        object.__setattr__(self, "vertices", v)

    @classmethod
    def from_polar(cls, mu: np.ndarray, lengths: np.ndarray, angles: np.ndarray) -> "PolygonWindow":
        """Build a polygon from vertices given in polar coordinates around a center.

        Vertices are placed at ``(lengths[i], angles[i])`` and then
        recentered so their coordinate mean (not the polygon's true
        area-centroid) sits at ``mu``.

        :param mu: Target center for the recentered vertices.
        :param lengths: Radial distance of each vertex from the origin before recentering.
        :param angles: Angle (radians) of each vertex from the origin before recentering.
        :returns: The constructed polygon.
        """
        lengths = np.asarray(lengths, dtype=float)
        angles = np.asarray(angles, dtype=float)
        x = lengths * np.cos(angles)
        y = lengths * np.sin(angles)
        x = x - x.mean()
        y = y - y.mean()
        mu = np.asarray(mu, dtype=float)
        return cls(vertices=np.column_stack([x + mu[0], y + mu[1]]))

    @property
    def area(self) -> float:
        """Area of the polygon.

        :returns: Area via the shoelace formula.
        """
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    @property
    def xrange(self) -> tuple[float, float]:
        """Bounding-box extent along x.

        :returns: ``(xmin, xmax)`` of the polygon's vertices.
        """
        return (float(self.vertices[:, 0].min()), float(self.vertices[:, 0].max()))

    @property
    def yrange(self) -> tuple[float, float]:
        """Bounding-box extent along y.

        :returns: ``(ymin, ymax)`` of the polygon's vertices.
        """
        return (float(self.vertices[:, 1].min()), float(self.vertices[:, 1].max()))

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Test which points fall inside the polygon.

        :param x: X coordinates to test.
        :param y: Y coordinates to test.
        :returns: Boolean mask, ``True`` where the point falls inside the polygon.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        pts = np.column_stack([x, y])
        return Path(self.vertices).contains_points(pts)

    def boundary(self, L: int | None = None) -> np.ndarray:
        """Return the polygon's vertices.

        :param L: Unused; accepted for interface parity with :meth:`EllipseWindow.boundary`.
        :returns: The polygon's vertex array.
        """
        return self.vertices

    def sample_uniform(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw points uniformly at random from inside the polygon via rejection sampling in its bounding box.

        :param n: Number of points to draw.
        :param rng: Random number generator to draw from.
        :returns: The ``x`` and ``y`` coordinates of the drawn points.
        :raises RuntimeError: if ``n`` points could not be sampled within the rejection-sampling round limit.
        """
        xr, yr = self.xrange, self.yrange
        xs = np.empty(0)
        ys = np.empty(0)
        for _ in range(_MAX_REJECTION_ROUNDS):
            need = n - len(xs)
            if need <= 0:
                break
            cx = rng.uniform(*xr, need * 2)
            cy = rng.uniform(*yr, need * 2)
            keep = self.contains(cx, cy)
            xs = np.concatenate([xs, cx[keep]])
            ys = np.concatenate([ys, cy[keep]])
        else:
            raise RuntimeError(f"Could not sample {n} points inside the polygon in {_MAX_REJECTION_ROUNDS} rounds.")
        return xs[:n], ys[:n]

    def __repr__(self) -> str:
        return f"PolygonWindow({self.vertices.shape[0]} vertices)"


def random_ellipse_window(
    mu: np.ndarray, r: float, A: np.ndarray | None = None, df: float = 10, rng: np.random.Generator | None = None
) -> EllipseWindow:
    """Build a single ellipse centered at ``mu`` with a random shape matrix.

    :param mu: Center of the ellipse.
    :param r: Mahalanobis radius.
    :param A: Wishart scale matrix used to draw the shape matrix; if ``None`` the shape matrix is
        the identity (a plain circle).
    :param df: Degrees of freedom for the Wishart draw of the shape matrix.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: The generated ellipse.
    """
    rng = rng or np.random.default_rng()
    shape = scipy_stats.wishart.rvs(df=df, scale=A, random_state=rng) if A is not None else np.eye(2)
    return EllipseWindow(mu=mu, A=shape, r=r)


def random_ellipses(
    pp: PointPattern, A: np.ndarray | None = None, df: float = 10, rng: np.random.Generator | None = None
) -> list[EllipseWindow]:
    """Build one random ellipse centered at each point of a point pattern.

    Each ellipse's radius is that point's mark if ``pp`` has marks, otherwise drawn Uniform(0, 1).

    :param pp: Point pattern giving the ellipse centers (and, optionally, radii via marks).
    :param A: Wishart scale matrix used to draw each shape matrix; if ``None`` each shape matrix is
        the identity.
    :param df: Degrees of freedom for the Wishart draw of each shape matrix.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: One ellipse per point in ``pp``.
    """
    rng = rng or np.random.default_rng()
    radii = pp.marks if pp.marks is not None else rng.uniform(size=pp.n)
    return [random_ellipse_window((pp.x[i], pp.y[i]), radii[i], A=A, df=df, rng=rng) for i in range(pp.n)]


def random_polygon_window(
    mu: np.ndarray,
    r: float,
    num_vert: int = 3,
    set_all: bool = False,
    rng: np.random.Generator | None = None,
) -> PolygonWindow:
    """Build a single random polygon centered near ``mu``.

    :param mu: Approximate center of the polygon.
    :param r: Upper bound on vertex distance from ``mu``; vertex lengths are drawn Uniform(r/2, r).
    :param num_vert: Maximum number of vertices; the actual count is fixed at 3 if ``num_vert < 3``.
    :param set_all: If ``True``, always use exactly ``num_vert`` vertices instead of a random count
        between 3 and ``num_vert``.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: The generated polygon.
    """
    if num_vert < 3:
        num_vert = 3
    rng = rng or np.random.default_rng()

    if num_vert == 3:
        n_vert = 3
    elif set_all:
        n_vert = num_vert
    else:
        n_vert = int(rng.integers(3, num_vert + 1))

    lengths = rng.uniform(r / 2, r, n_vert)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vert))
    return PolygonWindow.from_polar(mu, lengths, angles)


def random_polygons(
    pp: PointPattern, num_vert: int = 3, set_all: bool = False, rng: np.random.Generator | None = None
) -> list[PolygonWindow]:
    """Build one random polygon centered at each point of a point pattern.

    Each polygon's radius is that point's mark if ``pp`` has marks, otherwise drawn Uniform(0, 1).

    :param pp: Point pattern giving the polygon centers (and, optionally, radii via marks).
    :param num_vert: Maximum number of vertices per polygon.
    :param set_all: If ``True``, always use exactly ``num_vert`` vertices instead of a random count.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: One polygon per point in ``pp``.
    """
    rng = rng or np.random.default_rng()
    radii = pp.marks if pp.marks is not None else rng.uniform(size=pp.n)
    return [
        random_polygon_window((pp.x[i], pp.y[i]), radii[i], num_vert=num_vert, set_all=set_all, rng=rng)
        for i in range(pp.n)
    ]