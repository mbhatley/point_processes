from dataclasses import dataclass

import numpy as np

from .stats import ChainStats, get_stats


@dataclass
class HawkesProcess:
    """A simulated or observed univariate exponential-kernel Hawkes process.

    :ivar times: Sorted event times in ``[0, T_max]``.
    :ivar mu: Baseline (immigrant) intensity.
    :ivar alpha: Self-excitation weight.
    :ivar beta: Excitation decay rate.
    :ivar T_max: Length of the observation window.
    """

    times: np.ndarray
    mu: float
    alpha: float
    beta: float
    T_max: float

    def __repr__(self) -> str:
        """Return a one-line human-readable summary of the process.

        :returns: Summary including the parameters and event count.
        """
        return (
            f"Hawkes process (mu={self.mu:.3g}, alpha={self.alpha:.3g}, beta={self.beta:.3g}, "
            f"T_max={self.T_max:.3g}): {len(self.times)} event(s)"
        )


def gen_hawkes(T_max: float, mu: float, alpha: float, beta: float, rng: np.random.Generator | None = None) -> HawkesProcess:
    """Simulate a Hawkes process exactly via the branching/cluster representation.

    Immigrants arrive as a homogeneous Poisson(mu) process on
    ``[0, T_max]``; each event (immigrant or offspring) independently
    spawns a Poisson(alpha/beta)-distributed number of children, each
    with waiting time ``~ Exponential(beta)`` after its parent,
    recursively, discarding anything past ``T_max`` (Hawkes & Oakes,
    1974). This produces exact draws with no thinning/rejection step.

    :param T_max: Length of the observation window.
    :param mu: Baseline (immigrant) intensity. Must be > 0.
    :param alpha: Self-excitation weight. Must be >= 0.
    :param beta: Excitation decay rate. Must be > 0 and strictly greater
        than ``alpha``, so the branching ratio ``alpha/beta < 1`` keeps
        the expected event count finite.
    :param rng: Random number generator; a fresh default generator is
        used if omitted.
    :returns: The simulated process.
    :raises ValueError: If ``T_max``, ``mu``, or ``beta`` is not
        positive, if ``alpha`` is negative, or if ``alpha >= beta``.
    """
    if T_max <= 0:
        raise ValueError(f"T_max must be > 0, got {T_max}")
    if mu <= 0 or beta <= 0:
        raise ValueError(f"mu and beta must be > 0, got mu={mu}, beta={beta}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if alpha >= beta:
        raise ValueError(
            f"alpha ({alpha}) must be < beta ({beta}) -- branching ratio alpha/beta must be < 1 "
            "for a stable (non-explosive) process."
        )
    rng = rng or np.random.default_rng()

    n_immigrants = rng.poisson(mu * T_max)
    events: list[float] = list(rng.uniform(0, T_max, n_immigrants))
    frontier = list(events)
    while frontier:
        t = frontier.pop()
        n_children = rng.poisson(alpha / beta)
        if n_children == 0:
            continue
        child_times = t + rng.exponential(1 / beta, n_children)
        for ct in child_times:
            if ct <= T_max:
                events.append(ct)
                frontier.append(ct)

    return HawkesProcess(times=np.sort(np.array(events)), mu=mu, alpha=alpha, beta=beta, T_max=T_max)


def hawkes_intensity(ts: np.ndarray, hist: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Evaluate the conditional intensity at each requested time.

    :param ts: Times at which to evaluate the intensity.
    :param hist: Past event times; only entries with ``h < t`` contribute
        to the intensity at time ``t``.
    :param mu: Baseline intensity.
    :param alpha: Self-excitation weight.
    :param beta: Excitation decay rate.
    :returns: Intensity value at each entry of ``ts``.
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    hist = np.asarray(hist, dtype=float)
    out = np.empty_like(ts)
    for i, t in enumerate(ts):
        past = hist[hist < t]
        out[i] = mu + alpha * np.sum(np.exp(-beta * (t - past)))
    return out


def int_meas_hawkes(T: float, hist: np.ndarray, mu: float, alpha: float, beta: float) -> float:
    """Compute the compensator (integral of the intensity over ``[0, T]``).

    Uses the closed-form expression for the exponential kernel,
    ``mu*T + (alpha/beta) * sum_i [1 - exp(-beta*(T - t_i))]``, so no
    numerical integration is needed.

    :param T: Upper limit of integration.
    :param hist: Past event times up to and including ``T``.
    :param mu: Baseline intensity.
    :param alpha: Self-excitation weight.
    :param beta: Excitation decay rate.
    :returns: The compensator value.
    """
    hist = np.asarray(hist, dtype=float)
    past = hist[hist <= T]
    return float(mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - past))))


def _log_likelihood(hist_sorted: np.ndarray, T_max: float, mu: float, alpha: float, beta: float) -> float:
    """Evaluate the exponential-Hawkes log-likelihood in O(n) time.

    Uses the recursive form of Ogata (1981): each event's contribution
    is accumulated via a running sum that only depends on the previous
    event, avoiding the naive O(n^2) double sum.

    :param hist_sorted: Event times, sorted ascending.
    :param T_max: Length of the observation window.
    :param mu: Baseline intensity.
    :param alpha: Self-excitation weight.
    :param beta: Excitation decay rate.
    :returns: The log-likelihood of the observed events under the model.
    """
    n = len(hist_sorted)
    if n == 0:
        return -mu * T_max
    A = np.empty(n)
    A[0] = 0.0
    for i in range(1, n):
        A[i] = np.exp(-beta * (hist_sorted[i] - hist_sorted[i - 1])) * (1 + A[i - 1])
    log_intensities = np.log(mu + alpha * A)
    compensator = int_meas_hawkes(T_max, hist_sorted, mu, alpha, beta)
    return float(np.sum(log_intensities) - compensator)


@dataclass
class HawkesFit:
    """Posterior samples from a Bayesian fit of a Hawkes process.

    :ivar mu_gens: Generated values of ``mu``, one per iteration.
    :ivar alpha_gens: Generated values of ``alpha``, one per iteration.
    :ivar beta_gens: Generated values of ``beta``, one per iteration.
    :ivar L: Total number of iterations.
    :ivar burnin: Number of leading iterations treated as burn-in by
        :meth:`stats`.
    """

    mu_gens: np.ndarray
    alpha_gens: np.ndarray
    beta_gens: np.ndarray
    L: int
    burnin: int

    def stats(self, alpha: float = 0.05) -> dict[str, ChainStats]:
        """Summarize each parameter's post-burn-in chain.

        :param alpha: Significance level for the credible interval.
        :returns: Summary statistics keyed by parameter name (``"mu"``,
            ``"alpha"``, ``"beta"``).
        """
        return {
            "mu": get_stats(self.mu_gens[self.burnin :], alpha),
            "alpha": get_stats(self.alpha_gens[self.burnin :], alpha),
            "beta": get_stats(self.beta_gens[self.burnin :], alpha),
        }


def fit_hawkes(
    hist: np.ndarray,
    T_max: float,
    L: int = 10000,
    burnin: int | None = None,
    start_vals: tuple[float, float, float] = (10.0, 0.5, 10.0),
    hyperparams: tuple[float, float, float] = (100.0, 100.0, 100.0),
    small_sig: float = 0.1,
    rng: np.random.Generator | None = None,
) -> HawkesFit:
    """Fit (mu, alpha, beta) to observed event times by Bayesian random-walk Metropolis-Hastings.

    Each parameter has an independent ``Exponential(rate=hyperparams[i])``
    prior. The sampler updates one parameter at a time using a
    random-walk proposal in log-space (std ``small_sig``, transformed
    back by exponentiating), which keeps every proposal positive and
    lets a single shared step size work across parameters living on
    different scales; the log-transform's Jacobian is included in the
    acceptance ratio, since a proposal symmetric in log-space is not
    symmetric on the original positive scale. The likelihood is
    evaluated in closed form (see :func:`int_meas_hawkes`), so no
    auxiliary-variable machinery is needed to handle an intractable
    normalizing constant.

    :param hist: Observed event times.
    :param T_max: Length of the observation window.
    :param L: Number of MCMC iterations to run.
    :param burnin: Number of leading iterations recorded as burn-in;
        defaults to 10% of ``L``.
    :param start_vals: Initial ``(mu, alpha, beta)`` values.
    :param hyperparams: Prior rate for each of ``(mu, alpha, beta)``.
    :param small_sig: Standard deviation of the log-space random-walk
        proposal.
    :param rng: Random number generator; a fresh default generator is
        used if omitted.
    :returns: The generated posterior chains.
    :raises ValueError: If ``L <= burnin``, if ``small_sig <= 0``, or if
        any entry of ``hyperparams`` is not positive.
    """
    if burnin is None:
        burnin = int(0.1 * L)
    if L <= burnin:
        raise ValueError(f"L ({L}) must be > burnin ({burnin}).")
    if small_sig <= 0:
        raise ValueError(f"small_sig must be > 0, got {small_sig}")
    if any(h <= 0 for h in hyperparams):
        raise ValueError(f"hyperparams must all be > 0, got {hyperparams}")

    rng = rng or np.random.default_rng()
    hist_sorted = np.sort(np.asarray(hist, dtype=float))
    rates = np.asarray(hyperparams, dtype=float)

    mu_gens = np.empty(L)
    alpha_gens = np.empty(L)
    beta_gens = np.empty(L)
    mu, alpha, beta = start_vals

    def log_post(mu_, alpha_, beta_) -> float:
        return _log_likelihood(hist_sorted, T_max, mu_, alpha_, beta_) - rates @ [mu_, alpha_, beta_]

    cur_log_post = log_post(mu, alpha, beta)

    for it in range(L):
        # RW-MH on log(mu), then log(alpha), then log(beta); the log-Jacobian
        # is folded into log_accept below.
        for idx, val in enumerate((mu, alpha, beta)):
            y = np.log(val) + rng.normal(0, small_sig)
            prop = np.exp(y)
            params = [mu, alpha, beta]
            params[idx] = prop
            prop_log_post = log_post(*params)
            log_accept = (prop_log_post + np.log(prop)) - (cur_log_post + np.log(val))
            if np.log(rng.random()) < log_accept:
                mu, alpha, beta = params
                cur_log_post = prop_log_post

        mu_gens[it] = mu
        alpha_gens[it] = alpha
        beta_gens[it] = beta

    return HawkesFit(mu_gens=mu_gens, alpha_gens=alpha_gens, beta_gens=beta_gens, L=L, burnin=burnin)
