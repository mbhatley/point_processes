from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .window import Window

_PS_SUM_TOL = 1e-3


def _as_mu(mu) -> np.ndarray:
    """Validate and coerce a mixture component mean to a length-2 float array.

    :param mu: Candidate mean vector.
    :returns: The validated mean as a length-2 array.
    :raises ValueError: if ``mu`` does not have shape ``(2,)``.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if mu.shape != (2,):
        raise ValueError(f"each component mean must have length 2, got shape {mu.shape}")
    return mu


def _as_sigma(sigma) -> np.ndarray:
    """Validate and coerce a mixture component covariance to a 2x2 symmetric positive-definite array.

    :param sigma: Candidate covariance matrix.
    :returns: The validated covariance matrix.
    :raises ValueError: if ``sigma`` is not 2x2, symmetric, and positive definite.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (2, 2):
        raise ValueError(f"each component covariance must be 2x2, got shape {sigma.shape}")
    if not np.allclose(sigma, sigma.T, atol=1e-8):
        raise ValueError(f"covariance matrix must be symmetric, got {sigma}")
    eigvals = np.linalg.eigvalsh(sigma)
    if np.any(eigvals <= 0):
        raise ValueError(f"covariance matrix must be positive definite, got eigenvalues {eigvals}")
    return sigma


@dataclass
class NormMix:
    """A finite mixture of bivariate normal components.

    The number of components ``m`` is derived from ``len(ps)`` rather than
    stored as a separate field, so it can never drift out of sync with the
    component lists.

    :ivar ps: Mixing probabilities, one per component, summing to 1.
    :ivar mus: Component mean vectors, each length 2.
    :ivar sigmas: Component covariance matrices, each 2x2 and positive definite.
    :ivar estimated: Whether this mixture was estimated from data rather than specified directly.
    :raises ValueError: if the component counts of ``ps``/``mus``/``sigmas`` disagree, there are
        zero components, any probability is negative, or the probabilities don't sum to 1 (within
        tolerance).
    """

    ps: np.ndarray
    mus: list[np.ndarray]
    sigmas: list[np.ndarray]
    estimated: bool = False

    def __post_init__(self) -> None:
        self.ps = np.asarray(self.ps, dtype=float).reshape(-1)
        self.mus = [_as_mu(mu) for mu in self.mus]
        self.sigmas = [_as_sigma(s) for s in self.sigmas]

        if len(self.ps) != len(self.mus) or len(self.ps) != len(self.sigmas):
            raise ValueError(
                "Number of components mismatch: "
                f"len(ps)={len(self.ps)}, len(mus)={len(self.mus)}, len(sigmas)={len(self.sigmas)}"
            )
        if len(self.ps) == 0:
            raise ValueError("A mixture must have at least one component.")
        if np.any(self.ps < 0):
            raise ValueError(f"Component probabilities must be non-negative, got {self.ps}")
        if abs(self.ps.sum() - 1) > _PS_SUM_TOL:
            raise ValueError(f"Component probabilities must sum to 1, got sum={self.ps.sum()}")

    @property
    def m(self) -> int:
        """Number of mixture components.

        :returns: ``len(ps)``.
        """
        return len(self.ps)

    def __repr__(self) -> str:
        return f"Normal Mixture with {self.m} component(s)."

    def summary(self) -> pd.DataFrame:
        """Tabulate the mixture's components.

        :returns: One row per component with columns ``comp``, ``ps``, ``mu``, ``sigma``.
        """
        return pd.DataFrame(
            {
                "comp": np.arange(1, self.m + 1),
                "ps": self.ps,
                "mu": self.mus,
                "sigma": self.sigmas,
            }
        )


@dataclass
class IntensitySurface(NormMix):
    """A mixture intensity surface: a :class:`NormMix` scaled by an expected point count over a window.

    Named ``lam`` (not ``lambda``, a Python keyword) for the average number of points over the window.

    :ivar lam: Expected number of points over ``window``.
    :ivar window: Observation window the intensity is defined over.
    :raises ValueError: if ``window`` is not supplied, or ``lam <= 0``.
    :raises TypeError: if ``window`` is not a :class:`~pysppmix.window.Window`.
    """

    lam: float = 1.0
    window: Window | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.window is None:
            raise ValueError("An intensity surface requires a window.")
        if not isinstance(self.window, Window):
            raise TypeError(f"window must be a Window, got {type(self.window)}")
        if self.lam <= 0:
            raise ValueError(f"Intensity (lam) must be greater than 0, got {self.lam}")

    def __repr__(self) -> str:
        pre = "Estimated average" if self.estimated else "Expected"
        return f"{pre} number of points over the window: {self.lam}\n{self.window!r}\n{super().__repr__()}"


def is_normmix(x) -> bool:
    """Test whether a value is a :class:`NormMix`.

    :param x: Value to test.
    :returns: ``True`` if ``x`` is a ``NormMix`` (or subclass) instance.
    """
    return isinstance(x, NormMix)


def is_intensity_surface(x) -> bool:
    """Test whether a value is an :class:`IntensitySurface`.

    :param x: Value to test.
    :returns: ``True`` if ``x`` is an ``IntensitySurface`` instance.
    """
    return isinstance(x, IntensitySurface)


def to_int_surf(
    mix: NormMix,
    lam: float | None = None,
    win: Window | None = None,
    return_normmix: bool = False,
) -> NormMix:
    """Convert between a :class:`NormMix` and an :class:`IntensitySurface`, or restamp ``lam``/``win`` on one.

    :param mix: Mixture to convert or restamp.
    :param lam: Expected point count to use. Required when converting a plain ``NormMix``; when
        ``mix`` is already an ``IntensitySurface`` and this is omitted, its existing ``lam`` is kept.
    :param win: Observation window to use. Required when converting a plain ``NormMix``; when
        ``mix`` is already an ``IntensitySurface`` and this is omitted, its existing window is kept.
    :param return_normmix: If ``True``, return a plain ``NormMix`` (dropping ``lam``/``window``)
        instead of an ``IntensitySurface``.
    :returns: The converted or restamped mixture.
    :raises ValueError: if ``lam`` is not positive, or if converting a plain ``NormMix`` without both
        ``lam`` and ``win`` supplied.
    :raises TypeError: if ``mix`` is neither a ``NormMix`` nor an ``IntensitySurface``.
    """
    if is_intensity_surface(mix):
        new_lam = mix.lam if lam is None else lam
        new_win = mix.window if win is None else win
        if new_lam <= 0:
            raise ValueError("Intensity (lam) must be greater than 0.")
        intsurf = IntensitySurface(
            ps=mix.ps, mus=mix.mus, sigmas=mix.sigmas,
            estimated=mix.estimated, lam=new_lam, window=new_win,
        )
    elif is_normmix(mix):
        if return_normmix:
            return NormMix(ps=mix.ps, mus=mix.mus, sigmas=mix.sigmas, estimated=mix.estimated)
        if lam is None or win is None:
            raise ValueError("Both lam and win must be supplied to build an intensity surface.")
        intsurf = IntensitySurface(
            ps=mix.ps, mus=mix.mus, sigmas=mix.sigmas,
            estimated=mix.estimated, lam=lam, window=win,
        )
    else:
        raise TypeError("mix must be a NormMix or IntensitySurface.")

    if return_normmix:
        return NormMix(ps=intsurf.ps, mus=intsurf.mus, sigmas=intsurf.sigmas, estimated=intsurf.estimated)
    return intsurf


def get_mixture_limits(mix: NormMix, n_sigma: float = 5.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute default plot limits for a mixture.

    :param mix: Mixture to compute limits for.
    :param n_sigma: Margin around each component's mean, in standard deviations.
    :returns: Bounding box, as ``(xmin, xmax), (ymin, ymax)``, of every component's mean plus/minus
        ``n_sigma`` standard deviations.
    :raises TypeError: if ``mix`` is neither a ``NormMix`` nor an ``IntensitySurface``.
    """
    if not is_normmix(mix):
        raise TypeError("mix must be a NormMix or IntensitySurface.")
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for mu, sigma in zip(mix.mus, mix.sigmas):
        spread = n_sigma * np.sqrt(np.diag(sigma))
        xmins.append(mu[0] - spread[0])
        xmaxs.append(mu[0] + spread[0])
        ymins.append(mu[1] - spread[1])
        ymaxs.append(mu[1] + spread[1])
    return (min(xmins), max(xmaxs)), (min(ymins), max(ymaxs))


def approx_normmix(mix: NormMix, window: Window) -> np.ndarray:
    """Compute each mixture component's probability mass that falls inside a window.

    :param mix: Mixture whose components to integrate.
    :param window: Window to integrate over.
    :returns: Per-component probability mass inside ``window``, length ``mix.m``.
    """
    (xmin, xmax), (ymin, ymax) = window.xrange, window.yrange
    approx = np.empty(mix.m)
    for k in range(mix.m):
        rv = stats.multivariate_normal(mean=mix.mus[k], cov=mix.sigmas[k])
        approx[k] = (
            rv.cdf([xmax, ymax]) - rv.cdf([xmin, ymax]) - rv.cdf([xmax, ymin]) + rv.cdf([xmin, ymin])
        )
    return approx


@dataclass
class DensityGrid:
    """A regularly spaced evaluation of a density or intensity surface.

    :ivar x: Grid coordinates along the x-axis.
    :ivar y: Grid coordinates along the y-axis.
    :ivar z: Evaluated values, shape ``(len(y), len(x))``, with ``z[j, i]`` the value at ``(x[i], y[j])``.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def dnormmix(
    mix: NormMix,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    L: int = 128,
    truncate: bool = True,
) -> DensityGrid:
    """Evaluate a normal mixture's (or intensity surface's) density on a grid.

    :param mix: Mixture to evaluate.
    :param xlim: Grid bounds along x. Required for a plain ``NormMix``; ignored (the window's
        ``xrange`` is used instead) for an ``IntensitySurface``.
    :param ylim: Grid bounds along y. Required for a plain ``NormMix``; ignored (the window's
        ``yrange`` is used instead) for an ``IntensitySurface``.
    :param L: Grid resolution; the density is evaluated on an ``L`` x ``L`` grid.
    :param truncate: If ``True``, renormalize each component by the probability mass that falls
        inside the evaluation window so the density integrates to 1 over the window.
    :returns: The evaluated density grid.
    :raises TypeError: if ``mix`` is neither a ``NormMix`` nor an ``IntensitySurface``.
    :raises ValueError: if ``L < 2``, or ``xlim``/``ylim`` are omitted for a plain ``NormMix``.
    """
    if not is_normmix(mix):
        raise TypeError("mix must be a NormMix or IntensitySurface.")
    if L < 2:
        raise ValueError(f"L must be at least 2, got {L}")

    if is_intensity_surface(mix):
        xlim = mix.window.xrange
        ylim = mix.window.yrange
    elif xlim is None or ylim is None:
        raise ValueError("xlim and ylim are required when mix is a plain NormMix.")

    x = np.linspace(xlim[0], xlim[1], L)
    y = np.linspace(ylim[0], ylim[1], L)

    if truncate:
        window = mix.window if is_intensity_surface(mix) else Window(tuple(xlim), tuple(ylim))
        approx = approx_normmix(mix, window)
        approx = np.where(approx <= 0, 1.0, approx)  # guard against degenerate/zero mass
    else:
        approx = np.ones(mix.m)

    xx, yy = np.meshgrid(x, y)  # xx, yy shape (L, L), matches z[j, i] = f(x[i], y[j])
    z = np.zeros((L, L))
    for k in range(mix.m):
        rv = stats.multivariate_normal(mean=mix.mus[k], cov=mix.sigmas[k])
        pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        z += (mix.ps[k] / approx[k]) * rv.pdf(pts).reshape(L, L)

    if is_intensity_surface(mix):
        z = z * mix.lam

    return DensityGrid(x=x, y=y, z=z)


def rnormmix(
    m: int,
    sig0: float | None = None,
    Amat: np.ndarray | None = None,
    df: int = 10,
    rand_m: bool = False,
    window: Window = Window((0.0, 1.0), (0.0, 1.0)),
    dvec: np.ndarray | None = None,
    mu0: np.ndarray | None = None,
    Sigma0: np.ndarray | None = None,
    Bmat: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> NormMix:
    """Generate a random mixture.

    Component covariances are generated via an inverse-Wishart prior;
    means are drawn either uniformly over the window, from a normal prior
    centered at ``mu0``, or (when ``Bmat`` is given) jointly with each
    covariance via a normal-inverse-Wishart-style construction.

    :param m: Number of components (or, if ``rand_m``, the maximum number of components).
    :param sig0: Scale of each component's covariance. Defaults to ``0.1`` times the smaller of the
        window's width/height when omitted (and ``Bmat`` is not given).
    :param Amat: Shape matrix for the covariance-generating Wishart scale; identity if omitted.
    :param df: Degrees of freedom for the covariance-generating Wishart draws.
    :param rand_m: If ``True``, draw the actual component count uniformly from ``1..m``.
    :param window: Window used as the default mean-generating region and the default center ``mu0``.
    :param dvec: Dirichlet concentration parameters for the mixing probabilities, length ``m``.
        Defaults to all ones.
    :param mu0: Prior mean for the component means.
    :param Sigma0: Prior covariance for the component means.
    :param Bmat: If supplied, component covariances and means are generated jointly: covariances via
        Wishart(df, scale=inv(Bmat)), then each mean via a normal centered at ``mu0`` with covariance
        ``sig0**2 * inv(covariance)``. Requires ``mu0`` and ``sig0`` to also be given.
    :param rng: Random number generator to draw from; a new default generator is created if ``None``.
    :returns: The generated mixture.
    :raises ValueError: if ``m < 1``, ``dvec`` has the wrong length, or ``Bmat`` is given without
        both ``mu0`` and ``sig0``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    rng = rng or np.random.default_rng()
    xlim, ylim = window.xrange, window.yrange

    if rand_m:
        m = int(rng.integers(1, m + 1))

    if dvec is None:
        dvec = np.ones(m)
    else:
        dvec = np.asarray(dvec, dtype=float).reshape(-1)
        if len(dvec) != m:
            raise ValueError(f"dvec must have length m={m}, got {len(dvec)}")

    gen_ps = rng.dirichlet(dvec)

    if Bmat is None:
        if mu0 is None:
            mu0 = np.array([np.mean(xlim), np.mean(ylim)])
            if Sigma0 is None:
                gen_mu = np.column_stack(
                    [rng.uniform(xlim[0], xlim[1], m), rng.uniform(ylim[0], ylim[1], m)]
                )
            else:
                gen_mu = rng.multivariate_normal(mu0, Sigma0, size=m)
        else:
            mu0 = np.asarray(mu0, dtype=float)
            cov = np.eye(2) if Sigma0 is None else Sigma0
            gen_mu = rng.multivariate_normal(mu0, cov, size=m)

        if sig0 is None:
            sig0 = 0.1 * min(xlim[1] - xlim[0], ylim[1] - ylim[0])
        scale = (sig0**2) * (Amat if Amat is not None else np.eye(2))
        gen_sigma = np.stack([stats.wishart.rvs(df=df, scale=scale, random_state=rng) for _ in range(m)])
    else:
        if mu0 is None or sig0 is None:
            raise ValueError("When Bmat is supplied, mu0 and sig0 must also be passed.")
        mu0 = np.asarray(mu0, dtype=float)
        gen_sigma = np.stack(
            [stats.wishart.rvs(df=df, scale=np.linalg.inv(Bmat), random_state=rng) for _ in range(m)]
        )
        gen_mu = np.empty((m, 2))
        for i in range(m):
            gen_mu[i] = rng.multivariate_normal(mu0, (sig0**2) * np.linalg.inv(gen_sigma[i]))

    mus = [gen_mu[k] for k in range(m)]
    sigmas = [np.linalg.inv(gen_sigma[k]) for k in range(m)]
    return NormMix(ps=gen_ps, mus=mus, sigmas=sigmas)