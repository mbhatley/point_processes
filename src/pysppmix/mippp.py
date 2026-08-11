from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from .normmix import DensityGrid, IntensitySurface, rnormmix, to_int_surf
from .pointprocess import PointPattern, gen_n_from_mix, rsppmix
from .window import Window
from .mcmc.damcmc import DamcmcResult, drop_realization, est_mix_damcmc
from .mcmc.bdmcmc import est_mix_bdmcmc, get_bd_table, get_bd_compfit
from .mcmc.labelswitch import fix_label_switching
from .mcmc.postprocess import get_pm_est


# ---------------------------------------------------------------------------
# cond_mark: locations and marks independent
# ---------------------------------------------------------------------------


@dataclass
class MippCondMarkFit:
    """Result of fitting a marked point pattern with independent locations and marks.

    :ivar marks: Sorted unique mark values.
    :ivar ground_fits: Per-mark fitted MCMC chain (DAMCMC or BDMCMC,
        label-switching corrected), one per entry in ``marks``.
    :ivar post_surf: Per-mark posterior-mean intensity surface, one per
        entry in ``marks``.
    :ivar gen_mark_ps: Posterior draws of the mark distribution, shape
        ``(L, nmarks)``, or ``None`` if ``fit_markdist=False``.
    :ivar mark_dist: Posterior mean mark distribution, shape ``(nmarks,)``,
        or ``None`` if ``fit_markdist=False``.
    :ivar pp: The observed marked point pattern that was fit.
    :ivar fit_bdmcmc: Whether each per-mark ground fit used BDMCMC (random
        number of components) rather than DAMCMC (fixed number).
    """

    marks: np.ndarray
    ground_fits: list[DamcmcResult]
    post_surf: list[IntensitySurface]
    gen_mark_ps: np.ndarray | None
    mark_dist: np.ndarray | None
    pp: PointPattern
    fit_bdmcmc: bool

    def __repr__(self) -> str:
        method = "BDMCMC" if self.fit_bdmcmc else "DAMCMC"
        return f"MIPPP fit conditional on mark ({method} ground process, {len(self.marks)} marks)"


def est_mippp_cond_mark(
    pp: PointPattern,
    m: int | Sequence[int] = 10,
    L: int = 50000,
    burnin: int | None = None,
    hyper_da: Sequence[tuple | None] | None = None,
    hyper: np.ndarray | None = None,
    fit_markdist: bool = True,
    truncate: bool = False,
    rng: np.random.Generator | None = None,
) -> MippCondMarkFit:
    """Fit a marked point pattern by modeling locations and marks independently.

    Splits the pattern by mark value and fits a separate mixture intensity
    surface (DAMCMC if ``m`` is a scalar, BDMCMC if ``m`` is a
    per-mark sequence) to each mark's sub-pattern, then optionally fits a
    Dirichlet-multinomial posterior for the mark distribution itself.
    Label-switching correction is always applied to each ground fit before
    computing its posterior mean, which is a no-op when no switching
    occurred.

    :param pp: Marked point pattern to fit (``pp.marks`` must not be None).
    :param m: Number of mixture components per mark. A scalar fits BDMCMC
        (random number of components, same starting ``m`` for every mark);
        a sequence of length ``nmarks`` fits DAMCMC with a fixed count per
        mark.
    :param L: Number of MCMC iterations per mark's ground fit.
    :param burnin: Number of initial iterations to discard per ground fit.
        Defaults to ``0.1 * L``.
    :param hyper_da: Per-mark DAMCMC/BDMCMC hyperparameters (one entry per
        mark, or ``None`` to use each fitter's default).
    :param hyper: Dirichlet concentration added to each mark's observed
        count when ``fit_markdist=True``; defaults to 1 per mark (a flat
        Dirichlet prior).
    :param fit_markdist: Whether to also fit the posterior mark
        distribution.
    :param truncate: Whether to truncate each ground fit's density to its
        window.
    :param rng: Random number generator; a fresh default generator is
        created if not given.
    :returns: The per-mark ground fits and, if requested, the mark
        distribution posterior.
    :raises ValueError: If ``pp`` has no marks, if ``m`` is not positive,
        if a sequence ``m``/``hyper`` has the wrong length, or if ``hyper``
        is not all positive.
    """
    if pp.marks is None:
        raise ValueError("pp has no marks.")
    if burnin is None:
        burnin = int(0.1 * L)

    rng = rng or np.random.default_rng()
    marks = np.unique(pp.marks)
    nmarks = len(marks)

    fit_bdmcmc = np.isscalar(m)
    if fit_bdmcmc:
        if m <= 0:
            raise ValueError(f"m must be > 0, got {m}")
        m_list = [int(m)] * nmarks
    else:
        m_arr = np.asarray(m)
        if len(m_arr) != nmarks:
            raise ValueError(f"m must have length nmarks={nmarks}, got {len(m_arr)}")
        if np.any(m_arr <= 0):
            raise ValueError(f"all entries of m must be > 0, got {m_arr}")
        m_list = m_arr.tolist()

    if hyper_da is None:
        hyper_da = [None] * nmarks
    if hyper is None:
        hyper_vec = np.ones(nmarks)
    else:
        hyper_vec = np.asarray(hyper, dtype=float)
        if np.any(hyper_vec <= 0):
            raise ValueError(f"hyper must be all positive, got {hyper_vec}")

    ground_fits = []
    post_surf = []
    for k in range(nmarks):
        mask = pp.marks == marks[k]
        subpattern = PointPattern(x=pp.x[mask], y=pp.y[mask], window=pp.window)

        if fit_bdmcmc:
            kwargs = {} if hyper_da[k] is None else {"hyper_bd": hyper_da[k]}
            bd_fit = est_mix_bdmcmc(subpattern, m=m_list[k], L=L, truncate=truncate, rng=rng, **kwargs)
            bd_fit = drop_realization(bd_fit, burnin)
            bd_fit = drop_realization(bd_fit, bd_fit.badgen)
            tab = get_bd_table(bd_fit, show=False)
            sub_fit = get_bd_compfit(bd_fit, tab["MAPcomp"], burnin=0)
        else:
            kwargs = {} if hyper_da[k] is None else {"hyper_da": hyper_da[k]}
            sub_fit = est_mix_damcmc(subpattern, m=m_list[k], L=L, truncate=truncate, rng=rng, **kwargs)
            sub_fit = drop_realization(sub_fit, burnin)

        fixed = fix_label_switching(sub_fit, burnin=0, method="sel")
        ground_fits.append(fixed)
        post_surf.append(get_pm_est(fixed, burnin=0))

    gen_mark_ps = None
    mark_dist = None
    if fit_markdist:
        mark_counts = np.array([(pp.marks == marks[k]).sum() for k in range(nmarks)], dtype=float)
        post_param = mark_counts + hyper_vec
        gen_mark_ps = rng.dirichlet(post_param, size=L)
        mark_dist = gen_mark_ps.mean(axis=0)

    return MippCondMarkFit(
        marks=marks, ground_fits=ground_fits, post_surf=post_surf,
        gen_mark_ps=gen_mark_ps, mark_dist=mark_dist, pp=pp, fit_bdmcmc=fit_bdmcmc,
    )


@dataclass
class MippCondMarkSim:
    """A simulated marked point pattern with independent locations and marks.

    :ivar ground_surfs: Per-mark random mixture intensity surface used to
        generate that mark's locations.
    :ivar ground_pps: Per-mark simulated point pattern.
    :ivar genMPP: The combined marked point pattern.
    :ivar params: Mark probabilities used to generate ``genMPP``.
    """

    ground_surfs: list[IntensitySurface]
    ground_pps: list[PointPattern]
    genMPP: PointPattern
    params: np.ndarray

    def __repr__(self) -> str:
        return f"Simulated MIPPP conditional on mark ({len(self.params)} marks, n={self.genMPP.n})"


def rmippp_cond_mark(
    lam: float = 500,
    params: Sequence[float] = (0.5, 0.5),
    window: Window = Window((-3.0, 3.0), (-3.0, 3.0)),
    rng: np.random.Generator | None = None,
) -> MippCondMarkSim:
    """Simulate a marked point pattern with an independent mark distribution and a random mixture ground process per mark.

    :param lam: Expected total number of points across all marks.
    :param params: Mark probabilities, must be non-negative and sum to 1.
    :param window: Region to simulate over.
    :param rng: Random number generator; a fresh default generator is
        created if not given.
    :returns: The simulated marked point pattern and the ground surfaces
        used to generate it.
    :raises ValueError: If ``params`` is negative anywhere or does not sum
        to 1.
    """
    params = np.asarray(params, dtype=float)
    if abs(params.sum() - 1) > 1e-8:
        raise ValueError(f"params must sum to 1, got sum={params.sum()}")
    if np.any(params < 0):
        raise ValueError(f"params must be non-negative, got {params}")

    rng = rng or np.random.default_rng()
    nmarks = len(params)
    n = rng.poisson(lam)
    gen_marks = rng.choice(nmarks, size=n, replace=True, p=params) + 1  # 1-indexed, matching R
    counts = np.bincount(gen_marks - 1, minlength=nmarks)

    ground_surfs = []
    ground_pps = []
    xs, ys, comps, marks_out = [], [], [], []
    for k in range(nmarks):
        true_mix = rnormmix(m=5, df=10, rand_m=True, window=window, rng=rng)
        surf = to_int_surf(true_mix, lam=max(counts[k], 1e-9), win=window)
        pts = gen_n_from_mix(int(counts[k]), surf, rng=rng)
        ground_surfs.append(surf)
        ground_pps.append(
            PointPattern(x=pts[:, 0], y=pts[:, 1], window=window, comp=pts[:, 2].astype(int))
        )
        xs.append(pts[:, 0]); ys.append(pts[:, 1]); comps.append(pts[:, 2])
        marks_out.append(np.full(int(counts[k]), k + 1))

    genMPP = PointPattern(
        x=np.concatenate(xs), y=np.concatenate(ys), window=window,
        comp=np.concatenate(comps).astype(int), marks=np.concatenate(marks_out),
    )
    return MippCondMarkSim(ground_surfs=ground_surfs, ground_pps=ground_pps, genMPP=genMPP, params=params)


# ---------------------------------------------------------------------------
# cond_loc: marks depend on neighbors' marks (auto-model / Gibbs field)
# ---------------------------------------------------------------------------


def _score_matrix(mask: np.ndarray, data_marks: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Compute the auto-model neighbor score for every query point and candidate mark.

    :param mask: Boolean neighbor indicator, shape ``(n_query, n_data)``;
        ``mask[q, i]`` is True if data point ``i`` is a neighbor of query
        point ``q``.
    :param data_marks: Mark value of each data point, shape ``(n_data,)``.
    :param marks: Candidate mark values, shape ``(nmarks,)``.
    :returns: For each query point and candidate mark ``j``, the sum over
        its neighbors ``i`` of ``(data_marks[i] - marks[j])**2``. Shape
        ``(n_query, nmarks)``.
    """
    diff2 = (data_marks[:, None] - marks[None, :]) ** 2
    return mask.astype(float) @ diff2


def _neighbor_mask(query_xy: np.ndarray, data_xy: np.ndarray, r: float, exclude_diag: bool) -> np.ndarray:
    """Compute a boolean neighbor indicator between two sets of locations.

    :param query_xy: Query locations, shape ``(n_query, 2)``.
    :param data_xy: Data locations, shape ``(n_data, 2)``.
    :param r: Neighborhood radius.
    :param exclude_diag: If True, zero out the diagonal (used when
        ``query_xy is data_xy`` to exclude a point from being its own
        neighbor).
    :returns: Boolean array, shape ``(n_query, n_data)``, True where the
        pair is within distance ``r``.
    """
    dists = cdist(query_xy, data_xy)
    mask = dists <= r
    if exclude_diag:
        np.fill_diagonal(mask, False)
    return mask


@dataclass
class MippCondLocFit:
    """Result of fitting a marked point pattern whose marks depend on neighboring marks.

    :ivar marks: Sorted unique mark values.
    :ivar gen_gammas: Random-walk Metropolis-Hastings draws of the
        per-mark smoothness parameters, shape ``(L, nmarks)``.
    :ivar r: Neighborhood radius used to define which points influence
        each other's mark.
    :ivar L: Number of MCMC iterations run.
    :ivar burnin: Number of initial iterations excluded from posterior
        summaries.
    :ivar pp: The observed marked point pattern that was fit.
    :ivar ground_fit: Fitted mixture intensity surface for the locations
        themselves, or ``None`` if not requested.
    """

    marks: np.ndarray
    gen_gammas: np.ndarray
    r: float
    L: int
    burnin: int
    pp: PointPattern
    ground_fit: DamcmcResult | None

    def mean_gammas(self) -> np.ndarray:
        """Compute the post-burn-in posterior mean of each mark's gamma.

        :returns: Posterior mean gammas, shape ``(nmarks,)``.
        """
        return self.gen_gammas[self.burnin :].mean(axis=0)

    def prob_fields(self, LL: int = 128) -> list[DensityGrid]:
        """Compute each mark's probability field over a grid, using the posterior mean gammas.

        :param LL: Number of grid points per axis.
        :returns: One probability field per mark value, in the same order
            as ``marks``.
        """
        gamma = self.mean_gammas()
        xr, yr = self.pp.window.xrange, self.pp.window.yrange
        xs = np.linspace(*xr, LL)
        ys = np.linspace(*yr, LL)
        xx, yy = np.meshgrid(xs, ys)
        grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

        data_xy = np.column_stack([self.pp.x, self.pp.y])
        data_mark_idx = np.searchsorted(self.marks, self.pp.marks)
        mask = _neighbor_mask(grid_xy, data_xy, self.r, exclude_diag=False)
        scores = _score_matrix(mask, self.marks[data_mark_idx], self.marks)  # (LL*LL, nmarks)

        log_q = -scores * gamma[None, :]
        log_q -= log_q.max(axis=1, keepdims=True)
        q = np.exp(log_q)
        q /= q.sum(axis=1, keepdims=True)

        return [DensityGrid(x=xs, y=ys, z=q[:, j].reshape(LL, LL)) for j in range(len(self.marks))]

    def prob_at_points(self) -> np.ndarray:
        """Compute leave-one-out mark probabilities at each observed data location, for model checking.

        :returns: Mark probabilities at each data point, shape
            ``(n, nmarks)``.
        """
        gamma = self.mean_gammas()
        data_xy = np.column_stack([self.pp.x, self.pp.y])
        data_mark_idx = np.searchsorted(self.marks, self.pp.marks)
        mask = _neighbor_mask(data_xy, data_xy, self.r, exclude_diag=True)
        scores = _score_matrix(mask, self.marks[data_mark_idx], self.marks)

        log_q = -scores * gamma[None, :]
        log_q -= log_q.max(axis=1, keepdims=True)
        q = np.exp(log_q)
        return q / q.sum(axis=1, keepdims=True)

    def __repr__(self) -> str:
        return f"MIPPP fit conditional on location (r={self.r:g}, {len(self.marks)} marks, L={self.L})"


def est_mippp_cond_loc(
    pp: PointPattern,
    r: float,
    hyper: float = 1.0,
    L: int = 10000,
    burnin: int | None = None,
    start_gamma: np.ndarray | None = None,
    fit_ground_ippp: bool = False,
    ground_m: int = 3,
    truncate: bool = False,
    rng: np.random.Generator | None = None,
) -> MippCondLocFit:
    """Fit a marked point pattern whose marks depend on their neighbors' marks.

    Each point's mark is modeled by an auto-model (Gibbs) field: the
    conditional probability of mark ``j`` at a location, given its
    neighbors within radius ``r``, is proportional to
    ``exp(-gamma_j * sum_{neighbors i} (mark_i - j)**2)``. A gamma near 0
    gives near-uniform mark probabilities regardless of neighbors; a
    larger ``gamma_j`` increasingly penalizes mark ``j`` unless neighbors
    agree with it, producing a smoothness/clustering field over
    integer-coded marks.

    Since the joint normalizing constant over all mark configurations is
    intractable, gammas are fit by pseudo-likelihood (the standard
    approach for auto-models, Besag 1975) using single-component
    (one gamma at a time) random-walk Metropolis-Hastings with an
    improper (flat) prior on each gamma; ``hyper`` is the random-walk
    proposal standard deviation, not a prior parameter, so a small value
    is recommended.

    Two known limitations apply to this fit:

    1. Pseudo-likelihood bias at large ``r``: under a known-null dataset
       (marks independent of location, true gamma = 0 for every mark),
       95% credible sets for gamma cover 0 less often than 95% of the
       time once neighborhoods become large/dense relative to the point
       count -- this is a property of the pseudo-likelihood target
       itself, not an MCMC convergence artifact (more iterations do not
       change it). A sweep across ``r`` (12 independent null datasets,
       n=100, 2 marks, unit square) shows nominal coverage around
       r=0.10-0.15 and degradation beyond that::

           r=0.05 (~0.7 avg neighbors):  20/24 gammas covered 0
           r=0.10 (~2.9 avg neighbors):  23/24 gammas covered 0   (nominal)
           r=0.15 (~6.2 avg neighbors):  22/24 gammas covered 0   (nominal)
           r=0.30 (~28  avg neighbors):  17/24 gammas covered 0   (degraded)
           r=0.50 (~78  avg neighbors):  degenerate

       Keep ``r`` small enough that a typical point's neighborhood is a
       modest fraction of the whole pattern.

    2. Slow mixing for rare marks: when one mark is much rarer than
       another, its gamma can become weakly bounded from above (once it
       is large enough to already suppress the rare mark everywhere, the
       pseudo-likelihood is nearly flat in further increases), and the
       fixed-step-size sampler can produce a trace that has genuinely not
       converged after several thousand iterations -- visible directly as
       block-averages that keep climbing rather than stabilizing. Unlike
       the coverage issue above, more iterations do eventually help here.
       Inspect the ``gen_gammas`` trace for rare marks before trusting
       their posterior mean.

    :param pp: Marked point pattern to fit (``pp.marks`` must not be
        None).
    :param r: Neighborhood radius. Must be > 0.
    :param hyper: Random-walk proposal standard deviation for each gamma.
        Must be > 0.
    :param L: Number of MCMC iterations.
    :param burnin: Number of initial iterations to exclude from posterior
        summaries. Defaults to ``0.1 * L``.
    :param start_gamma: Starting gamma values, shape ``(nmarks,)``.
        Defaults to all zeros.
    :param fit_ground_ippp: Whether to also fit a mixture intensity
        surface for the locations themselves.
    :param ground_m: Number of mixture components for the ground fit, if
        ``fit_ground_ippp`` is True.
    :param truncate: Whether to truncate the ground fit's density to its
        window.
    :param rng: Random number generator; a fresh default generator is
        created if not given.
    :returns: The fitted gamma chain and, if requested, the ground
        location fit.
    :raises ValueError: If ``pp`` has no marks, ``r`` or ``hyper`` is not
        > 0, or ``start_gamma`` has the wrong length.
    """
    if pp.marks is None:
        raise ValueError("pp has no marks.")
    if r <= 0:
        raise ValueError(f"r must be > 0, got {r}")
    if hyper <= 0:
        raise ValueError(f"hyper (RW proposal SD) must be > 0, got {hyper}")
    if burnin is None:
        burnin = int(0.1 * L)

    rng = rng or np.random.default_rng()
    marks = np.unique(pp.marks)
    nmarks = len(marks)
    mark_idx = np.searchsorted(marks, pp.marks)
    xy = np.column_stack([pp.x, pp.y])
    n = pp.n

    if start_gamma is None:
        gamma = np.zeros(nmarks)
    else:
        gamma = np.asarray(start_gamma, dtype=float).copy()
        if gamma.shape != (nmarks,):
            raise ValueError(f"start_gamma must have length nmarks={nmarks}, got shape {gamma.shape}")

    # Neighbor structure is fixed (observed data), so precompute once: scores[i, j]
    # = sum over neighbors of point i (leave-one-out) of (neighbor_mark - marks[j])^2.
    mask = _neighbor_mask(xy, xy, r, exclude_diag=True)
    scores = _score_matrix(mask, marks[mark_idx], marks)  # (n, nmarks)

    def pseudo_log_lik(g: np.ndarray) -> float:
        log_q = -scores * g[None, :]
        return float(np.sum(log_q[np.arange(n), mark_idx] - logsumexp(log_q, axis=1)))

    cur_pll = pseudo_log_lik(gamma)
    gen_gammas = np.empty((L, nmarks))
    for it in range(L):
        for j in range(nmarks):
            prop = gamma.copy()
            prop[j] = gamma[j] + rng.normal(0, hyper)
            prop_pll = pseudo_log_lik(prop)
            if np.log(rng.random()) < prop_pll - cur_pll:
                gamma = prop
                cur_pll = prop_pll
        gen_gammas[it] = gamma

    ground_fit = est_mix_damcmc(pp, m=ground_m, truncate=truncate, rng=rng) if fit_ground_ippp else None

    return MippCondLocFit(
        marks=marks, gen_gammas=gen_gammas, r=r, L=L, burnin=burnin, pp=pp, ground_fit=ground_fit,
    )


@dataclass
class MippCondLocSim:
    """A simulated marked point pattern whose marks depend on neighboring marks.

    :ivar surf: Intensity surface used to generate the locations.
    :ivar gammas: Per-mark smoothness parameters used to generate the
        marks.
    :ivar r: Neighborhood radius used to define which points influence
        each other's mark.
    :ivar genMPP: The simulated marked point pattern.
    :ivar marks: Mark values used, ``1..len(gammas)``.
    """

    surf: IntensitySurface
    gammas: np.ndarray
    r: float
    genMPP: PointPattern
    marks: np.ndarray

    def __repr__(self) -> str:
        return f"Simulated MIPPP conditional on location (r={self.r:g}, {len(self.marks)} marks, n={self.genMPP.n})"


def _nearest_neighbor_dists(xy: np.ndarray) -> np.ndarray:
    """Compute each point's distance to its nearest neighbor.

    :param xy: Locations, shape ``(n, 2)``.
    :returns: Nearest-neighbor distance for each point, shape ``(n,)``.
    """
    d = cdist(xy, xy)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def rmippp_cond_loc(
    surf: IntensitySurface | None = None,
    loc_pp: PointPattern | None = None,
    gammas: np.ndarray | None = None,
    r: float | None = None,
    truncate: bool = False,
    window: Window = Window((-3.0, 3.0), (-3.0, 3.0)),
    L: int = 50000,
    rng: np.random.Generator | None = None,
) -> MippCondLocSim:
    """Simulate a marked point pattern whose marks depend on their neighbors' marks.

    Locations come from an ordinary mixture-intensity IPPP (``surf``/
    ``loc_pp``, generated randomly if omitted). Marks are then drawn by
    single-site Gibbs sampling: repeatedly pick a random point and
    resample its mark from the auto-model conditional ``exp(-gamma_j *
    sum_{neighbors i} (mark_i - j)**2)`` given its neighbors' current
    marks, for ``L`` total single-point updates. This is the standard way
    to sample an auto-model's joint distribution when only its full
    conditionals are known explicitly.

    :param surf: Intensity surface for the locations. A random mixture
        surface is generated if not given (and ``loc_pp`` is also not
        given).
    :param loc_pp: Pre-generated locations. If not given, locations are
        drawn from ``surf``.
    :param gammas: Per-mark smoothness parameters. If not given, a random
        number of marks (1-10) is chosen and each gamma is drawn from
        ``Gamma(shape=10, rate=0.1)``.
    :param r: Neighborhood radius. If not given, drawn uniformly between
        the minimum and maximum nearest-neighbor distance among the
        locations. Must be > 0.
    :param truncate: Whether to truncate ``surf``'s density to its window
        when drawing ``loc_pp``.
    :param window: Region to simulate locations over, used only when
        ``surf``/``loc_pp`` are not given.
    :param L: Number of single-point Gibbs updates.
    :param rng: Random number generator; a fresh default generator is
        created if not given.
    :returns: The simulated marked point pattern and the parameters used
        to generate it.
    :raises ValueError: If fewer than 2 locations are available (a
        neighborhood cannot be defined), or if ``r`` is not > 0.
    """
    rng = rng or np.random.default_rng()

    if surf is None:
        base_mix = rnormmix(m=5, df=10, rand_m=True, window=window, rng=rng)
        surf = to_int_surf(base_mix, lam=200, win=window)
    if loc_pp is None:
        loc_pp = rsppmix(surf, truncate=truncate, rng=rng)

    n = loc_pp.n
    if n < 2:
        raise ValueError(f"Need at least 2 locations to define a neighborhood, got n={n}")
    xy = np.column_stack([loc_pp.x, loc_pp.y])

    if gammas is None:
        nmarks = int(rng.integers(1, 11))
        gammas = stats.gamma.rvs(a=10, scale=1 / 0.1, size=nmarks, random_state=rng)
    gammas = np.asarray(gammas, dtype=float)
    nmarks = len(gammas)
    marks = np.arange(1, nmarks + 1)

    if r is None:
        nn = _nearest_neighbor_dists(xy)
        r = rng.uniform(nn.min(), nn.max())
    if r <= 0:
        raise ValueError(f"r must be > 0, got {r}")

    full_mask = _neighbor_mask(xy, xy, r, exclude_diag=True)  # (n, n), fixed given locations
    mark_idx = rng.integers(0, nmarks, size=n)

    for _ in range(L):
        i = rng.integers(n)
        neigh = full_mask[i]
        scores_i = _score_matrix(neigh[None, :], marks[mark_idx], marks)[0]  # (nmarks,)
        log_q = -gammas * scores_i
        log_q -= log_q.max()
        q = np.exp(log_q)
        q /= q.sum()
        mark_idx[i] = rng.choice(nmarks, p=q)

    genMPP = PointPattern(x=loc_pp.x, y=loc_pp.y, window=loc_pp.window, marks=marks[mark_idx])
    return MippCondLocSim(surf=surf, gammas=gammas, r=r, genMPP=genMPP, marks=marks)
