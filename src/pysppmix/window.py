from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    """An axis-aligned rectangular observation window ``[xmin, xmax] x [ymin, ymax]``.

    :ivar xrange: The ``(xmin, xmax)`` bounds of the window.
    :ivar yrange: The ``(ymin, ymax)`` bounds of the window.
    :raises ValueError: if either range is non-finite or not strictly increasing.
    """

    xrange: tuple[float, float]
    yrange: tuple[float, float]

    def __post_init__(self) -> None:
        xmin, xmax = self.xrange
        ymin, ymax = self.yrange
        if not (np.isfinite(xmin) and np.isfinite(xmax)):
            raise ValueError(f"xrange must be finite, got {self.xrange}")
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            raise ValueError(f"yrange must be finite, got {self.yrange}")
        if xmin >= xmax:
            raise ValueError(f"xrange must satisfy xmin < xmax, got {self.xrange}")
        if ymin >= ymax:
            raise ValueError(f"yrange must satisfy ymin < ymax, got {self.yrange}")

    @property
    def area(self) -> float:
        """Area of the window.

        :returns: Width times height.
        """
        return (self.xrange[1] - self.xrange[0]) * (self.yrange[1] - self.yrange[0])

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Test which points fall inside the window.

        :param x: X coordinates to test.
        :param y: Y coordinates to test.
        :returns: Boolean mask, ``True`` where the point falls inside the window (bounds inclusive).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (
            (x >= self.xrange[0])
            & (x <= self.xrange[1])
            & (y >= self.yrange[0])
            & (y <= self.yrange[1])
        )

    def sample_uniform(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw points uniformly at random from the window.

        :param n: Number of points to draw.
        :param rng: Random number generator to draw from.
        :returns: The ``x`` and ``y`` coordinates of the drawn points.
        """
        x = rng.uniform(self.xrange[0], self.xrange[1], n)
        y = rng.uniform(self.yrange[0], self.yrange[1], n)
        return x, y

    def __repr__(self) -> str:
        return f"Window(x=[{self.xrange[0]}, {self.xrange[1]}], y=[{self.yrange[0]}, {self.yrange[1]}])"