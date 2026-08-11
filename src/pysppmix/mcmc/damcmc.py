import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans2

from ..normmix import NormMix, approx_normmix
from ..pointprocess import PointPattern
from ..window import Window


@dataclass
class DamcmcResult:
    """Posterior realizations from :func:`est_mix_damcmc`.

    Stored as plain arrays rather than nested per-iteration structures,
    since that is both simpler to validate and faster to slice with
    NumPy.

    :ivar data: The point pattern the mixture was fit to.
    :ivar m: Number of mixture components.
    :ivar L: Number of MCMC iterations.
    :ivar genps: Sampled mixture weights, shape ``(L, m)``.
    :ivar genmus: Sampled component means, shape ``(L, m, 2)``.
    :ivar gensigmas: Sampled component covariances, shape ``(L, m, 2, 2)``.
    :ivar genzs: Sampled component labels per point, shape ``(L, n)``.
    :ivar genlambdas: Sampled Poisson rate per iteration, shape ``(L,)``.
    :ivar approx_mass: Each component's approximate in-window probability
        mass per iteration, shape ``(L, m)``.
    :ivar hyper_da: The ``(d, sig0, df0)`` hyperparameters used to fit the
        chain.
    :ivar truncate: Whether component densities were truncated to the
        observation window during fitting.
    """

    data: PointPattern
    m: int
    L: int
    genps: np.ndarray       # (L, m)
    genmus: np.ndarray      # (L, m, 2)
    gensigmas: np.ndarray   # (L, m, 2, 2)
    genzs: np.ndarray       # (L, n) int
    genlambdas: np.ndarray  # (L,)
    approx_mass: np.ndarray  # (L, m)
    hyper_da: tuple[float, float, float]
    truncate: bool

    def __repr__(self) -> str:
        return (
            f"Normal Mixture fit via Data Augmentation MCMC\n"
            f"DAMCMC iterations: {self.L}\n"
            f"Number of components: {self.m}"
        )


def drop_realization(fit, drop=None):
    """Discard MCMC realizations, either as a burn-in count or via a boolean mask.

    Works generically on any MCMC-result dataclass (:class:`DamcmcResult`,
    :class:`~pysppmix.mcmc.bdmcmc.BdmcmcResult`, ...): every array field
    whose first axis has length ``fit.L`` is sliced consistently, and the
    result is rebuilt as the same concrete type.

    :param fit: The MCMC result to trim.
    :param drop: Number of initial iterations to discard (default: 10% of
        ``fit.L``), or a boolean mask of iterations to drop (``True`` =
        drop).
    :returns: A new result of the same type with the selected iterations
        removed.
    :raises ValueError: If a boolean mask's shape doesn't match
        ``(fit.L,)``.
    """
    if drop is None:
        drop = 0.1 * fit.L

    if isinstance(drop, (bool, np.bool_)) or (
        isinstance(drop, np.ndarray) and drop.dtype == bool
    ):
        mask = np.asarray(drop, dtype=bool)
        if mask.shape != (fit.L,):
            raise ValueError(f"boolean drop mask must have shape ({fit.L},), got {mask.shape}")
        keep = ~mask
    else:
        keep = np.arange(fit.L) >= int(drop)

    updates = {"L": int(keep.sum())}
    for f in dataclasses.fields(fit):
        val = getattr(fit, f.name)
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == fit.L:
            updates[f.name] = val[keep]
    return dataclasses.replace(fit, **updates)


# Added to df0 before it's used as an Inverse-Wishart degrees-of-freedom
# parameter in 2D: the literal default df0=1 is invalid (a 2D
# Inverse-Wishart requires df > 1 to be proper), so the sampler actually
# uses (df0 + DF_BASELINE).
DF_BASELINE = 4


def _initial_partition(xy: np.ndarray, m: int, use_kmeans: bool, rng: np.random.Generator) -> np.ndarray:
    n = xy.shape[0]
    if use_kmeans:
        _, labels = kmeans2(xy, m, seed=int(rng.integers(2**31 - 1)), minit="++")
        return labels.astype(int)
    centers = xy[rng.choice(n, size=m, replace=False)]
    dists = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=-1)
    return dists.argmin(axis=1)


def _gibbs_sweep(
    xy: np.ndarray,
    ps: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    d_vec: np.ndarray,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    Sigma0_inv: np.ndarray,
    Psi0: np.ndarray,
    df0_eff: float,
    win: Window,
    truncate: bool,
    rng: np.random.Generator,
):
    """Run one fixed-dimension Gibbs sweep, updating labels, weights, means, and covariances.

    Shared by :func:`est_mix_damcmc` and
    :func:`~pysppmix.mcmc.bdmcmc.est_mix_bdmcmc`, which runs this sweep at
    whatever component count its birth-death process currently has,
    between trans-dimensional moves.

    :param xy: Observed point locations, shape ``(n, 2)``.
    :param ps: Current mixture weights, shape ``(m,)``.
    :param mus: Current component means, shape ``(m, 2)``.
    :param sigmas: Current component covariances, shape ``(m, 2, 2)``.
    :param d_vec: Dirichlet concentration parameters, shape ``(m,)``.
    :param mu0: Prior mean for component means.
    :param Sigma0: Prior covariance for component means.
    :param Sigma0_inv: Inverse of ``Sigma0``.
    :param Psi0: Inverse-Wishart scale matrix for component covariances.
    :param df0_eff: Effective Inverse-Wishart degrees of freedom (already
        adjusted by ``DF_BASELINE``).
    :param win: Observation window, used to compute in-window mass when
        ``truncate`` is set.
    :param truncate: Whether to normalize each component's density by its
        in-window probability mass.
    :param rng: Random number generator.
    :returns: Updated ``(ps, mus, sigmas, z, counts, mass)`` after one
        sweep.
    """
    m = len(ps)
    n = xy.shape[0]

    if truncate:
        mix = NormMix(ps=ps, mus=list(mus), sigmas=list(sigmas))
        mass = approx_normmix(mix, win)
        mass = np.where(mass <= 1e-12, 1.0, mass)
    else:
        mass = np.ones(m)

    # 1. z_i | ps, mu, sigma
    logw = np.empty((n, m))
    for k in range(m):
        rv = stats.multivariate_normal(mean=mus[k], cov=sigmas[k])
        logw[:, k] = np.log(ps[k]) + rv.logpdf(xy) - np.log(mass[k])
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    u = rng.random(n)
    z = (u[:, None] > np.cumsum(w, axis=1)).sum(axis=1)  # inverse-CDF sampling, vectorized
    z = np.clip(z, 0, m - 1)

    counts = np.bincount(z, minlength=m)

    # 2. ps | z
    ps_new = rng.dirichlet(d_vec + counts)

    # 3. mu_k | z, sigma_k (old sigma)
    new_mus = np.empty((m, 2))
    for k in range(m):
        nk = counts[k]
        sigma_k_inv = np.linalg.inv(sigmas[k])
        if nk > 0:
            xbar_k = xy[z == k].mean(axis=0)
            prec_n = Sigma0_inv + nk * sigma_k_inv
            cov_n = np.linalg.inv(prec_n)
            mean_n = cov_n @ (Sigma0_inv @ mu0 + nk * sigma_k_inv @ xbar_k)
        else:
            cov_n, mean_n = Sigma0, mu0
        new_mus[k] = rng.multivariate_normal(mean_n, cov_n)

    # 4. sigma_k | z, mu_k (new mu)
    new_sigmas = np.empty((m, 2, 2))
    for k in range(m):
        nk = counts[k]
        if nk > 0:
            resid = xy[z == k] - new_mus[k]
            scatter = resid.T @ resid
        else:
            scatter = np.zeros((2, 2))
        new_sigmas[k] = stats.invwishart.rvs(df=df0_eff + nk, scale=Psi0 + scatter, random_state=rng)

    return ps_new, new_mus, new_sigmas, z, counts, mass


def est_mix_damcmc(
    pp: PointPattern,
    m: int,
    truncate: bool = False,
    L: int = 10000,
    hyper_da: tuple[float, float, float] = (3, 1, 1),
    use_kmeans: bool = False,
    rng: np.random.Generator | None = None,
) -> DamcmcResult:
    """Fit an m-component bivariate normal mixture intensity via Data Augmentation MCMC.

    Runs a Diebolt & Robert (1994) style Gibbs sampler that alternates,
    for ``L`` iterations:

    1. ``z_i | ps, mu, sigma`` -- assign each point to a component,
       categorically, with weight proportional to
       ``ps_k * N(x_i; mu_k, sigma_k)`` (divided by each component's
       in-window mass when ``truncate=True``).
    2. ``ps | z`` -- redraw mixture weights from
       ``Dirichlet(d + n_1, ..., d + n_m)``.
    3. ``mu_k | z, sigma_k`` -- redraw each component mean from its
       conjugate Normal posterior given an ``N(mu0, Sigma0)`` prior and
       the points currently assigned to it.
    4. ``sigma_k | z, mu_k`` -- redraw each component covariance from
       ``InverseWishart(df0 + DF_BASELINE, sig0^2*I + scatter_k)``.
       Component means and covariances have independent priors (the
       two-block scheme of Diebolt & Robert 1994), not a joint
       Normal-Inverse-Wishart prior.
    5. ``lambda | n`` -- redraw the Poisson rate from
       ``Gamma(1 + n, 1 + 1)``.

    ``df0`` is treated as degrees of freedom *above* ``DF_BASELINE``
    (rather than used directly), since a 2D Inverse-Wishart distribution
    requires ``df > 1`` to be proper. A component that loses all its
    points during a sweep has its mean and covariance redrawn from the
    prior for that sweep rather than left undefined.

    :param pp: Observed point pattern to fit against.
    :param m: Number of mixture components.
    :param truncate: Whether to normalize each component's density by its
        in-window probability mass.
    :param L: Number of MCMC iterations to run.
    :param hyper_da: Hyperparameters ``(d, sig0, df0)`` -- Dirichlet
        concentration, Inverse-Wishart scale, and degrees of freedom
        above ``DF_BASELINE``.
    :param use_kmeans: Whether to initialize component labels with
        k-means (otherwise, random centers).
    :param rng: Random number generator; a new default one is created if
        omitted.
    :returns: The fitted MCMC chain.
    :raises ValueError: If ``m < 1``, ``L < 1``, there are fewer points
        than components, or ``hyper_da`` contains an invalid value
        (``d <= 0``, ``sig0 <= 0``, or ``df0 < 0``).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if pp.n < m:
        raise ValueError(f"Need at least as many points ({pp.n}) as components (m={m}).")

    d, sig0, df0 = hyper_da
    if d <= 0:
        raise ValueError(f"hyper_da[0] (Dirichlet concentration d) must be > 0, got {d}")
    if sig0 <= 0:
        raise ValueError(f"hyper_da[1] (sig0) must be > 0, got {sig0}")
    if df0 < 0:
        raise ValueError(f"hyper_da[2] (df0) must be >= 0, got {df0}")

    rng = rng or np.random.default_rng()
    xy = np.column_stack([pp.x, pp.y])
    n = pp.n
    win = pp.window

    mu0 = xy.mean(axis=0)
    Sigma0 = np.cov(xy, rowvar=False)
    if n < 3 or np.any(np.linalg.eigvalsh(Sigma0) <= 1e-12):
        Sigma0 = np.eye(2) * max(np.var(xy), 1e-6)
    Sigma0_inv = np.linalg.inv(Sigma0)
    Psi0 = (sig0**2) * np.eye(2)
    d_vec = np.full(m, float(d))

    z = _initial_partition(xy, m, use_kmeans, rng)
    mus = np.array([xy[z == k].mean(axis=0) if np.any(z == k) else mu0 for k in range(m)])
    sigmas = np.array(
        [np.cov(xy[z == k], rowvar=False) if (z == k).sum() > 2 else Sigma0 for k in range(m)]
    )
    for k in range(m):
        if np.any(np.linalg.eigvalsh(sigmas[k]) <= 1e-10):
            sigmas[k] = Sigma0
    ps = np.full(m, 1.0 / m)

    genps = np.empty((L, m))
    genmus = np.empty((L, m, 2))
    gensigmas = np.empty((L, m, 2, 2))
    genzs = np.empty((L, n), dtype=int)
    genlambdas = np.empty(L)
    approx_mass = np.empty((L, m))

    lam_a0, lam_b0 = 1.0, 1.0  # Gamma(1, 1) prior on the Poisson rate: a weakly-informative filled-in default

    for it in range(L):
        ps, mus, sigmas, z, counts, mass = _gibbs_sweep(
            xy, ps, mus, sigmas, d_vec, mu0, Sigma0, Sigma0_inv, Psi0,
            df0 + DF_BASELINE, win, truncate, rng,
        )

        # 5. lambda | n
        lam = rng.gamma(shape=lam_a0 + n, scale=1.0 / (lam_b0 + 1.0))

        genps[it] = ps
        genmus[it] = mus
        gensigmas[it] = sigmas
        genzs[it] = z
        genlambdas[it] = lam
        approx_mass[it] = mass

    return DamcmcResult(
        data=pp,
        m=m,
        L=L,
        genps=genps,
        genmus=genmus,
        gensigmas=gensigmas,
        genzs=genzs,
        genlambdas=genlambdas,
        approx_mass=approx_mass,
        hyper_da=tuple(hyper_da),
        truncate=truncate,
    )