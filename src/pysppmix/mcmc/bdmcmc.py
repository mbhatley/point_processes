from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..normmix import NormMix, approx_normmix
from ..pointprocess import PointPattern
from .damcmc import DF_BASELINE, DamcmcResult, _gibbs_sweep, _initial_partition, drop_realization


@dataclass
class BdmcmcResult:
    """Posterior realizations from :func:`est_mix_bdmcmc`.

    Component arrays are padded to ``maxnumcomp``: slots at index
    ``>= numcomp[l]`` are zero for iteration ``l``. ``genzs`` holds
    component labels in ``[0, numcomp[l])``.

    :ivar data: The point pattern the mixture was fit to.
    :ivar maxnumcomp: Maximum number of components allowed.
    :ivar L: Number of MCMC iterations.
    :ivar genps: Sampled mixture weights, shape ``(L, maxnumcomp)``.
    :ivar genmus: Sampled component means, shape ``(L, maxnumcomp, 2)``.
    :ivar gensigmas: Sampled component covariances, shape
        ``(L, maxnumcomp, 2, 2)``.
    :ivar genzs: Sampled component labels per point, shape ``(L, n)``.
    :ivar genlambdas: Sampled Poisson rate per iteration, shape ``(L,)``.
    :ivar approx_mass: Each active component's approximate in-window
        probability mass per iteration, shape ``(L, maxnumcomp)``.
    :ivar numcomp: Number of active components per iteration, shape
        ``(L,)``.
    :ivar badgen: Whether an iteration produced a degenerate realization
        (a near-zero weight or in-window mass), shape ``(L,)``.
    :ivar hyper_bd: The ``(d, sig0, df0)`` hyperparameters used to fit the
        chain.
    :ivar lambda1: Prior rate controlling the truncated-Poisson prior on
        the number of components.
    :ivar lambda2: Birth-rate parameter controlling the probability of
        proposing an additional component.
    :ivar truncate: Whether component densities were truncated to the
        observation window during fitting.
    """

    data: PointPattern
    maxnumcomp: int
    L: int
    genps: np.ndarray        # (L, maxnumcomp)
    genmus: np.ndarray       # (L, maxnumcomp, 2)
    gensigmas: np.ndarray    # (L, maxnumcomp, 2, 2)
    genzs: np.ndarray        # (L, n) int
    genlambdas: np.ndarray   # (L,)
    approx_mass: np.ndarray  # (L, maxnumcomp)
    numcomp: np.ndarray      # (L,) int -- number of active components per iteration
    badgen: np.ndarray       # (L,) bool -- degenerate realizations (see class docstring)
    hyper_bd: tuple[float, float, float]
    lambda1: float
    lambda2: float
    truncate: bool

    def __repr__(self) -> str:
        return (
            f"Normal Mixture fit via Birth-Death MCMC\n"
            f"BDMCMC iterations: {self.L}\n"
            f"Max number of components: {self.maxnumcomp}"
        )


def _birth_prob(m: int, maxm: int, lambda2: float) -> float:
    if m <= 1:
        return 1.0
    if m >= maxm:
        return 0.0
    return lambda2 / (lambda2 + m)


def _log_transdim_ratio(
    m_small: int, ps_small: np.ndarray, w_star: float, n: int,
    lambda1: float, lambda2: float, maxm: int, d: float,
) -> float:
    """Compute the log acceptance-ratio term for a birth move from ``m_small`` to ``m_small + 1`` components.

    (Used with the roles of "before" and "after" swapped, by the caller,
    for the corresponding death move.) Combines four terms: the
    truncated-Poisson prior ratio on the number of components, the
    Dirichlet(d) prior ratio on the weights, the likelihood ratio (every
    point's density scales by ``(1 - w_star)`` since the new component
    starts empty), and the birth/death proposal ratio. The Jacobian of the
    weight-rescaling transform exactly cancels the ``Beta(1, m_small)``
    proposal density for ``w_star``, so it does not appear as a separate
    term.

    :param m_small: Component count before the move.
    :param ps_small: Mixture weights before the move, shape
        ``(m_small,)``.
    :param w_star: Proposed weight for the new/removed component.
    :param n: Number of observed points.
    :param lambda1: Truncated-Poisson prior rate on the number of
        components.
    :param lambda2: Birth-rate parameter controlling the birth/death move
        probabilities.
    :param maxm: Maximum allowed number of components.
    :param d: Dirichlet concentration parameter.
    :returns: The log acceptance-ratio term ``log R``.
    """
    ps_big = np.concatenate([ps_small * (1 - w_star), [w_star]])
    log_prior_m = np.log(lambda1) - np.log(m_small + 1)
    log_dir = stats.dirichlet.logpdf(ps_big, np.full(m_small + 1, d)) - stats.dirichlet.logpdf(
        ps_small, np.full(m_small, d)
    )
    log_lik = n * np.log(1 - w_star)
    p_birth_small = _birth_prob(m_small, maxm, lambda2)
    p_death_big = 1 - _birth_prob(m_small + 1, maxm, lambda2)
    log_prop = np.log(p_death_big) - np.log(m_small + 1) - np.log(p_birth_small) + np.log(m_small)
    return log_prior_m + log_dir + log_lik + log_prop


def est_mix_bdmcmc(
    pp: PointPattern,
    m: int,
    truncate: bool = False,
    lambda1: float = 1.0,
    lambda2: float = 10.0,
    hyper_bd: tuple[float, float, float] = (1.0, 1.0, 1.0),
    L: int = 30000,
    use_kmeans: bool = False,
    rng: np.random.Generator | None = None,
) -> BdmcmcResult:
    """Fit a random-component (up to ``m``) bivariate normal mixture intensity via Birth-Death MCMC.

    Each iteration performs a fixed-dimension Gibbs sweep (identical
    model and steps to :func:`~pysppmix.mcmc.damcmc.est_mix_damcmc` at
    whatever component count the chain currently has), followed by one
    trans-dimensional move:

    * **Birth**, chosen with probability ``lambda2 / (lambda2 + m)``
      (forced to 1 at ``m=1`` and 0 at ``m=maxnumcomp``): propose a new,
      empty component with ``mu* ~ N(mu0, Sigma0)``,
      ``Sigma* ~ InverseWishart(df0 + DF_BASELINE, sig0^2*I)``, and
      weight ``w* ~ Beta(1, m)``, rescaling existing weights to
      ``ps_j' = ps_j * (1 - w*)``. Accepted with probability
      ``min(1, R)``, where ``R`` combines the prior ratio on the
      component count, the Dirichlet weight-prior ratio, the likelihood
      ratio ``(1 - w*)^n``, and the birth/death proposal ratio.
    * **Death**: only components with zero points currently assigned are
      candidates (removing an occupied component would require
      reassigning its points, which this sampler does not attempt). If
      none are empty, the move is a no-op for that iteration. Otherwise
      one empty component is chosen uniformly and removed with
      probability ``min(1, 1/R)``, using the same ``R`` as the birth move
      with "before" and "after" swapped.

    ``m`` is the maximum number of components (``maxnumcomp``); the
    actual number active at any iteration is recorded in the returned
    :class:`BdmcmcResult`. ``hyper_bd`` is ``(d, sig0, df0)``, the same
    convention (and same ``DF_BASELINE`` reconciliation) as
    :func:`~pysppmix.mcmc.damcmc.est_mix_damcmc`'s ``hyper_da``.

    Because a component can only be removed once it happens to lose every
    point back to the other components in the same sweep, a spurious
    extra component can occasionally get stuck for a long run before
    being able to die -- a known weakness of birth-death samplers
    generally. In an empirical check (15 seeds, two well-separated true
    components, ``L=3000``), 13/15 chains concentrated confidently on the
    true count while 2/15 got stuck at one extra component for the whole
    run. Treat a single chain's :func:`get_bd_table` mode as a starting
    point, not a final answer -- inspect
    :func:`~pysppmix.plotting.plot_comp_dist` (and ideally run more than
    one chain) before trusting it.

    :param pp: Observed point pattern to fit against.
    :param m: Maximum number of mixture components (``maxnumcomp``).
    :param truncate: Whether to normalize each component's density by its
        in-window probability mass.
    :param lambda1: Truncated-Poisson prior rate on the number of
        components.
    :param lambda2: Birth-rate parameter controlling how strongly the
        sampler favors proposing an additional component.
    :param hyper_bd: Hyperparameters ``(d, sig0, df0)`` -- Dirichlet
        concentration, Inverse-Wishart scale, and degrees of freedom
        above ``DF_BASELINE``.
    :param L: Number of MCMC iterations to run.
    :param use_kmeans: Whether to initialize component labels with
        k-means (otherwise, random centers).
    :param rng: Random number generator; a new default one is created if
        omitted.
    :returns: The fitted MCMC chain.
    :raises ValueError: If ``m < 1``, ``L < 1``, there are no data
        points, ``lambda1 <= 0``, ``lambda2 <= 0``, or ``hyper_bd``
        contains an invalid value (``d <= 0``, ``sig0 <= 0``, or
        ``df0 < 0``).
    """
    if m < 1:
        raise ValueError(f"m (maxnumcomp) must be >= 1, got {m}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if pp.n < 1:
        raise ValueError("Need at least one data point.")
    if lambda1 <= 0:
        raise ValueError(f"lambda1 must be > 0, got {lambda1}")
    if lambda2 <= 0:
        raise ValueError(f"lambda2 must be > 0, got {lambda2}")

    d, sig0, df0 = hyper_bd
    if d <= 0:
        raise ValueError(f"hyper_bd[0] (Dirichlet concentration d) must be > 0, got {d}")
    if sig0 <= 0:
        raise ValueError(f"hyper_bd[1] (sig0) must be > 0, got {sig0}")
    if df0 < 0:
        raise ValueError(f"hyper_bd[2] (df0) must be >= 0, got {df0}")

    rng = rng or np.random.default_rng()
    xy = np.column_stack([pp.x, pp.y])
    n = pp.n
    win = pp.window
    maxm = m

    mu0 = xy.mean(axis=0)
    Sigma0 = np.cov(xy, rowvar=False)
    if n < 3 or np.any(np.linalg.eigvalsh(Sigma0) <= 1e-12):
        Sigma0 = np.eye(2) * max(np.var(xy), 1e-6)
    Sigma0_inv = np.linalg.inv(Sigma0)
    Psi0 = (sig0**2) * np.eye(2)
    df0_eff = df0 + DF_BASELINE

    cur_m = min(3, maxm)
    z = _initial_partition(xy, cur_m, use_kmeans, rng)
    mus = np.array([xy[z == k].mean(axis=0) if np.any(z == k) else mu0 for k in range(cur_m)])
    sigmas = np.array(
        [np.cov(xy[z == k], rowvar=False) if (z == k).sum() > 2 else Sigma0 for k in range(cur_m)]
    )
    for k in range(cur_m):
        if np.any(np.linalg.eigvalsh(sigmas[k]) <= 1e-10):
            sigmas[k] = Sigma0
    ps = np.full(cur_m, 1.0 / cur_m)

    genps = np.zeros((L, maxm))
    genmus = np.zeros((L, maxm, 2))
    gensigmas = np.zeros((L, maxm, 2, 2))
    genzs = np.zeros((L, n), dtype=int)
    genlambdas = np.empty(L)
    approx_mass = np.zeros((L, maxm))
    numcomp = np.empty(L, dtype=int)
    badgen = np.zeros(L, dtype=bool)

    lam_a0, lam_b0 = 1.0, 1.0

    for it in range(L):
        d_vec = np.full(cur_m, float(d))
        ps, mus, sigmas, z, counts, mass = _gibbs_sweep(
            xy, ps, mus, sigmas, d_vec, mu0, Sigma0, Sigma0_inv, Psi0, df0_eff, win, truncate, rng
        )

        # --- trans-dimensional move (skipped entirely if maxnumcomp == 1: nothing to birth into) ---
        if maxm > 1 and rng.random() < _birth_prob(cur_m, maxm, lambda2):
            w_star = rng.beta(1, cur_m)
            mu_star = rng.multivariate_normal(mu0, Sigma0)
            sigma_star = stats.invwishart.rvs(df=df0_eff, scale=Psi0, random_state=rng)
            log_R = _log_transdim_ratio(cur_m, ps, w_star, n, lambda1, lambda2, maxm, d)
            if np.log(rng.random()) < log_R:
                ps = np.concatenate([ps * (1 - w_star), [w_star]])
                mus = np.vstack([mus, mu_star])
                sigmas = np.concatenate([sigmas, sigma_star[None]], axis=0)
                cur_m += 1
                # z unchanged: existing labels still refer to the same indices;
                # the new component (index cur_m-1) starts empty.
        elif cur_m > 1:
            empty = np.flatnonzero(counts == 0)
            if empty.size > 0:
                j = int(rng.choice(empty))
                w_star = ps[j]
                keep = np.array([i for i in range(cur_m) if i != j])
                ps_small = ps[keep] / (1 - w_star)
                log_R = _log_transdim_ratio(cur_m - 1, ps_small, w_star, n, lambda1, lambda2, maxm, d)
                if np.log(rng.random()) < -log_R:
                    ps = ps_small
                    mus = mus[keep]
                    sigmas = sigmas[keep]
                    z = np.where(z > j, z - 1, z)  # shift labels above the removed slot down by one
                    cur_m -= 1

        lam = rng.gamma(shape=lam_a0 + n, scale=1.0 / (lam_b0 + 1.0))

        genps[it, :cur_m] = ps
        genmus[it, :cur_m] = mus
        gensigmas[it, :cur_m] = sigmas
        genzs[it] = z
        genlambdas[it] = lam
        numcomp[it] = cur_m

        # mass from _gibbs_sweep is sized for the pre-move component count; the
        # trans-dimensional move above may have changed cur_m, so recompute here.
        if truncate:
            mix = NormMix(ps=ps, mus=list(mus), sigmas=list(sigmas))
            mass = approx_normmix(mix, win)
        else:
            mass = np.ones(cur_m)
        approx_mass[it, :cur_m] = mass
        badgen[it] = bool(np.any(mass[:cur_m] <= 1e-8)) or bool(np.any(ps <= 1e-8))

    return BdmcmcResult(
        data=pp,
        maxnumcomp=maxm,
        L=L,
        genps=genps,
        genmus=genmus,
        gensigmas=gensigmas,
        genzs=genzs,
        genlambdas=genlambdas,
        approx_mass=approx_mass,
        numcomp=numcomp,
        badgen=badgen,
        hyper_bd=tuple(hyper_bd),
        lambda1=lambda1,
        lambda2=lambda2,
        truncate=truncate,
    )


def get_bd_table(fit: BdmcmcResult, show: bool = True) -> dict:
    """Compute the frequency table of the number of active components across a chain, and the MAP/mean component count.

    :param fit: A fitted BDMCMC chain.
    :param show: Whether to print the frequency table.
    :returns: A dict with keys ``"MAPcomp"`` (the most frequent component
        count), ``"FreqTab"`` (frequency array indexed from 1
        component), and ``"MeanComp"`` (the frequency-weighted mean
        component count).
    """
    freq = np.bincount(fit.numcomp, minlength=fit.maxnumcomp + 1)[1:]  # index 0 -> 1 component
    if show:
        print("Frequency table for the number of components:")
        for k, count in enumerate(freq, start=1):
            if count:
                print(f"  {k}: {count}")
    map_comp = int(np.argmax(freq)) + 1
    mean_comp = float(np.average(np.arange(1, fit.maxnumcomp + 1), weights=freq))
    return {"MAPcomp": map_comp, "FreqTab": freq, "MeanComp": mean_comp}


def get_bd_compfit(fit: BdmcmcResult, num_comp: int | None = None, burnin: int | None = None) -> DamcmcResult:
    """Extract the sub-chain of iterations with exactly ``num_comp`` active components.

    Returns a plain :class:`~pysppmix.mcmc.damcmc.DamcmcResult`, so
    :func:`~pysppmix.mcmc.postprocess.get_pm_est`,
    :func:`~pysppmix.mcmc.mapest.get_map_est`, and
    :func:`~pysppmix.mcmc.labelswitch.fix_label_switching` all work on it
    unchanged.

    :param fit: A fitted BDMCMC chain.
    :param num_comp: Component count to extract; defaults to the MAP
        component count.
    :param burnin: Number of initial iterations to discard before
        selecting ``num_comp``; defaults to 10% of ``fit.L``.
    :returns: The sub-chain restricted to iterations with exactly
        ``num_comp`` active components.
    :raises ValueError: If ``num_comp`` is out of range, or no
        realizations have exactly ``num_comp`` components after burn-in.
    """
    if burnin is None:
        burnin = int(0.1 * fit.L)
    if burnin >= fit.L:
        burnin = int(0.1 * fit.L)
    fit = drop_realization(fit, burnin)

    if num_comp is None:
        num_comp = get_bd_table(fit, show=False)["MAPcomp"]
    if not (1 <= num_comp <= fit.maxnumcomp):
        raise ValueError(f"num_comp must be between 1 and {fit.maxnumcomp}, got {num_comp}")

    keep = fit.numcomp == num_comp
    if not np.any(keep):
        raise ValueError(f"No realizations with exactly {num_comp} component(s) after burn-in.")

    return DamcmcResult(
        data=fit.data,
        m=num_comp,
        L=int(keep.sum()),
        genps=fit.genps[keep][:, :num_comp],
        genmus=fit.genmus[keep][:, :num_comp],
        gensigmas=fit.gensigmas[keep][:, :num_comp],
        genzs=fit.genzs[keep],
        genlambdas=fit.genlambdas[keep],
        approx_mass=fit.approx_mass[keep][:, :num_comp],
        hyper_da=fit.hyper_bd,
        truncate=fit.truncate,
    )