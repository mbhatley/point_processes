from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from ..pointprocess import PointPattern
from ..stats import ChainStats, get_stats
from .simulate import gen_markov_pp_bd


def count_pairs(pts: np.ndarray, r: float) -> int:
    """Count the number of unordered point pairs within distance ``r``.

    :param pts: Array of shape ``(n, 2)`` of point coordinates.
    :param r: Distance threshold.
    :returns: Number of point pairs with pairwise distance less than ``r``.
    """
    pts = np.atleast_2d(pts)
    if pts.shape[0] < 2:
        return 0
    return int(np.sum(pdist(pts) < r))


def min_dist(pts: np.ndarray) -> float:
    """Compute the smallest pairwise distance in a point pattern.

    :param pts: Array of shape ``(n, 2)`` of point coordinates.
    :returns: The minimum pairwise Euclidean distance.
    :raises ValueError: If fewer than 2 points are given.
    """
    pts = np.atleast_2d(pts)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points to compute a minimum pairwise distance.")
    return float(pdist(pts).min())


@dataclass
class StraussFit:
    """MCMC chain output from fitting a Strauss process.

    :ivar b_gens: Sampled baseline-intensity values, length ``L``.
    :ivar r_gens: Sampled interaction-range values, length ``L``.
    :ivar gamma_gens: Sampled interaction-strength values, length ``L``.
    :ivar L: Total number of MCMC iterations.
    :ivar burnin: Number of initial iterations discarded as burn-in.
    """

    b_gens: np.ndarray
    r_gens: np.ndarray
    gamma_gens: np.ndarray
    L: int
    burnin: int

    def stats(self, alpha: float = 0.05) -> dict[str, ChainStats]:
        """Summarize the post-burn-in posterior for each parameter.

        :param alpha: Significance level for the credible interval.
        :returns: Mapping from parameter name (``"b"``, ``"r"``, ``"gamma"``) to its chain statistics.
        """
        return {
            "b": get_stats(self.b_gens[self.burnin :], alpha),
            "r": get_stats(self.r_gens[self.burnin :], alpha),
            "gamma": get_stats(self.gamma_gens[self.burnin :], alpha),
        }


def fit_strauss(
    pp: PointPattern,
    L: int = 50000,
    burnin: int | None = None,
    alpha_par: float = 1.0,
    beta_par: float | None = None,
    r_alpha: float = 1.0,
    r_beta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> StraussFit:
    """Fit a Strauss process to a point pattern via Bayesian MCMC.

    The baseline intensity ``b`` is drawn once per iteration directly from
    its closed-form Gamma posterior, conditionally independent of ``r``
    and ``gamma`` given the point count. The interaction range ``r`` and
    interaction strength ``gamma`` are each updated via an independence
    Metropolis-Hastings step, in that order, with the ``gamma`` step's
    pair-count statistic computed using the just-updated ``r``.
    ``r_alpha``/``r_beta`` parameterize the independence proposal (and,
    since they cancel in the acceptance ratio, implicitly the prior) for
    ``r``; ``gamma``'s implicit prior is Uniform(0, 1) with no tunable
    hyperparameters.

    :param pp: Observed point pattern to fit against.
    :param L: Number of MCMC iterations.
    :param burnin: Number of initial iterations to discard; defaults to ``0.1 * L``.
    :param alpha_par: Shape hyperparameter of ``b``'s Gamma prior.
    :param beta_par: Scale hyperparameter of ``b``'s Gamma prior; defaults to ``n / ((n + 1) * window_area)``.
    :param r_alpha: Shape parameter of the independence proposal (and implicit prior) for ``r``.
    :param r_beta: Scale parameter of the independence proposal (and implicit prior) for ``r``.
    :param rng: Random number generator; a new default generator is created if omitted.
    :returns: The fitted MCMC chain.
    :raises ValueError: If ``L <= burnin``, if ``pp`` has no points, or if ``pp`` has fewer than 2 points.
    """
    if burnin is None:
        burnin = int(0.1 * L)
    if L <= burnin:
        raise ValueError(f"L ({L}) must be > burnin ({burnin}).")
    n = pp.n
    if n <= 0:
        raise ValueError("pp has no points.")
    if n < 2:
        raise ValueError("Need at least 2 points to fit interaction parameters.")

    rng = rng or np.random.default_rng()
    pts = np.column_stack([pp.x, pp.y])
    window_area = pp.window.area
    beta = beta_par if beta_par is not None else n / ((n + 1) * window_area)

    b_gens = rng.gamma(shape=alpha_par + n, scale=beta, size=L)
    r_gens = np.zeros(L)
    gamma_gens = np.zeros(L)

    # gamma_gens[0] == 0 makes the very first iteration's ratios involve 0**x
    # and x/0 -- both mathematically well-defined here (0, 1, or inf) but
    # numpy warns about them by default; suppressed since it's expected.
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, L):
            prop_r = rng.gamma(shape=r_alpha, scale=r_beta)
            cur_count = count_pairs(pts, r_gens[i - 1])
            prop_count = count_pairs(pts, prop_r)
            mh_ratio = gamma_gens[i - 1] ** (prop_count - cur_count)
            if rng.random() < mh_ratio:
                r_gens[i] = prop_r
            else:
                r_gens[i] = r_gens[i - 1]

            prop_gamma = rng.random()
            count_for_gamma = count_pairs(pts, r_gens[i])
            mh_ratio = (prop_gamma / gamma_gens[i - 1]) ** count_for_gamma
            if rng.random() < mh_ratio:
                gamma_gens[i] = prop_gamma
            else:
                gamma_gens[i] = gamma_gens[i - 1]

    return StraussFit(b_gens=b_gens, r_gens=r_gens, gamma_gens=gamma_gens, L=L, burnin=burnin)


@dataclass
class HardCoreFit:
    """MCMC chain output from fitting a hard-core process.

    :ivar b_gens: Sampled baseline-intensity values, length ``L``.
    :ivar r_gens: Sampled interaction-range values, length ``L``.
    :ivar beta_gens: Sampled hyperprior-scale values, length ``L``; populated only when ``model_choice=3``.
    :ivar L: Total number of MCMC iterations.
    :ivar burnin: Number of initial iterations discarded as burn-in.
    :ivar model_choice: Which of the three normalizing-constant strategies (1, 2, or 3) was used.
    """

    b_gens: np.ndarray
    r_gens: np.ndarray
    beta_gens: np.ndarray | None
    L: int
    burnin: int
    model_choice: int

    def stats(self, alpha: float = 0.05) -> dict[str, ChainStats]:
        """Summarize the post-burn-in posterior for each parameter.

        :param alpha: Significance level for the credible interval.
        :returns: Mapping from parameter name (``"b"``, ``"r"``, and ``"beta"`` when populated) to its chain statistics.
        """
        out = {
            "b": get_stats(self.b_gens[self.burnin :], alpha),
            "r": get_stats(self.r_gens[self.burnin :], alpha),
        }
        if self.beta_gens is not None:
            out["beta"] = get_stats(self.beta_gens[self.burnin :], alpha)
        return out


def bayesian_markov_pp_mh(
    pp: PointPattern,
    thetas: np.ndarray = (0.01, 0.0),
    b: float = 50.0,
    model_choice: int = 1,
    L: int = 50000,
    burnin: int | None = None,
    alpha_par: float = 1.0,
    beta_par: float | None = None,
    rng: np.random.Generator | None = None,
) -> HardCoreFit:
    """Fit a hard-core process to a point pattern via Bayesian MCMC.

    The hard-core process's normalizing constant is intractable in ``r``,
    so three strategies are offered, selected by ``model_choice``:

    1. An improper/reference prior under which ``b`` has a closed-form
       Gamma posterior and ``r`` is drawn Uniform(0, minimum observed
       pairwise distance), ignoring the constant's ``r``-dependence.
    2. An exchange-algorithm-style Monte Carlo approximation of the ratio
       of normalizing constants, using auxiliary point patterns simulated
       at fixed hyperparameters ``ksi_r``/``ksi_b``. ``thetas[1]`` selects
       a fixed ``ksi_r`` (with ``thetas[2]`` as ``ksi_b``), ``0`` derives
       both from the data's own sufficient statistics, and ``-1`` switches
       ``b``'s update to a Gaussian random-walk proposal instead of an
       independence sampler.
    3. A hierarchical Bayes model that places a Gamma/Inverse-Gamma
       hyperprior on ``b``'s prior rate.

    For all three strategies, ``r_gens`` is drawn as ``L`` i.i.d.
    Uniform(0, minimum observed pairwise distance) samples, keeping the
    interaction-range chain identically distributed regardless of which
    strategy is used for ``b``.

    :param pp: Observed point pattern to fit against.
    :param thetas: Auxiliary hyperparameters; only ``thetas[1:]`` are used, and only when ``model_choice=2``.
    :param b: Initial baseline-intensity value.
    :param model_choice: Which normalizing-constant strategy to use (1, 2, or 3).
    :param L: Number of MCMC iterations.
    :param burnin: Number of initial iterations to discard; defaults to ``0.1 * L``.
    :param alpha_par: Shape hyperparameter of ``b``'s Gamma prior/proposal.
    :param beta_par: Scale hyperparameter of ``b``'s Gamma prior/proposal; defaults to ``n / ((n + 1) * window_area)``.
    :param rng: Random number generator; a new default generator is created if omitted.
    :returns: The fitted MCMC chain.
    :raises ValueError: If ``L <= burnin``, if ``pp`` has no points, if ``b < 0``, or if ``model_choice`` is not 1, 2, or 3.
    """
    if burnin is None:
        burnin = int(0.1 * L)
    if L <= burnin:
        raise ValueError(f"L ({L}) must be > burnin ({burnin}).")
    n = pp.n
    if n <= 0:
        raise ValueError("pp has no points.")
    if b < 0:
        raise ValueError(f"b must be >= 0, got {b}")
    if model_choice not in (1, 2, 3):
        raise ValueError(f"model_choice must be 1, 2, or 3, got {model_choice}")

    rng = rng or np.random.default_rng()
    pts = np.column_stack([pp.x, pp.y])
    window_area = pp.window.area
    beta = beta_par if beta_par is not None else n / ((n + 1) * window_area)
    mind = min_dist(pts)

    beta_gens = None

    if model_choice == 1:
        b_gens = rng.gamma(shape=alpha_par + n, scale=beta, size=L)
        r_gens = rng.uniform(0, mind, size=L)

    elif model_choice == 2:
        b_gens = np.zeros(L)
        r_gens = np.zeros(L)
        b_gens[0] = beta

        do_random_walk = False
        if len(thetas) > 1 and thetas[1] > 0:
            ksi_r, ksi_b = thetas[1], thetas[2]
        elif len(thetas) > 1 and thetas[1] == -1:
            do_random_walk = True
            beta = 100.0
            ksi_r, ksi_b = mind, n
        else:
            ksi_r, ksi_b = mind, n

        LL = 100
        _, aux_patterns = gen_markov_pp_bd(
            (ksi_r,), start_n=ksi_b, b=ksi_b, move_setup=1, func_choice=1,
            window=pp.window, L=5000, birth_prob=0.5, save_last_iters=LL, rng=rng,
        )
        aux_mindists = np.array([min_dist(ap) for ap in aux_patterns])
        aux_counts = np.array([ap.shape[0] for ap in aux_patterns])

        n_jumps = 0
        for i in range(1, L):
            prop_r = rng.uniform(0, mind)
            if do_random_walk:
                prop_b = rng.normal(r_gens[i - 1], beta)
                while prop_b <= 0:
                    prop_b = rng.normal(r_gens[i - 1], beta)
            else:
                prop_b = rng.gamma(shape=alpha_par + n, scale=beta)

            mask_prop = max(prop_r, ksi_r) < aux_mindists
            mask_cur = max(r_gens[i - 1], ksi_r) < aux_mindists
            const_prop = np.sum((prop_b / ksi_b) ** aux_counts[mask_prop])
            const_cur = np.sum((b_gens[i - 1] / ksi_b) ** aux_counts[mask_cur])

            energy_ratio = 1.0 if max(prop_r, r_gens[i - 1]) < mind else 0.0
            proposal_ratio = np.exp((b_gens[i - 1] - prop_b) / beta) if do_random_walk else 1.0

            if const_prop == 0:
                mh_ratio = np.inf
            else:
                mh_ratio = proposal_ratio * (prop_b / b_gens[i - 1]) ** n * (const_cur / const_prop) * energy_ratio

            if rng.random() < mh_ratio:
                n_jumps += 1
                b_gens[i] = prop_b
                r_gens[i] = prop_r
            else:
                b_gens[i] = b_gens[i - 1]
                r_gens[i] = r_gens[i - 1]

    else:  # model_choice == 3: hierarchical Bayes on beta
        b_gens = np.zeros(L)
        r_gens = np.zeros(L)
        beta_gens = np.zeros(L)
        beta_alpha, beta_beta = n, 10000.0

        b_gens[0] = beta
        beta_gens[0] = 1 / rng.gamma(shape=alpha_par + beta_alpha, scale=1 / (1 / beta_beta + b_gens[0]))
        for i in range(1, L):
            b_gens[i] = rng.gamma(shape=alpha_par + n, scale=beta_gens[i - 1])
            beta_gens[i] = 1 / rng.gamma(shape=alpha_par + beta_alpha, scale=1 / (1 / beta_beta + b_gens[i]))
        r_gens[:] = rng.uniform(0, mind, size=L)

    return HardCoreFit(b_gens=b_gens, r_gens=r_gens, beta_gens=beta_gens, L=L, burnin=burnin, model_choice=model_choice)