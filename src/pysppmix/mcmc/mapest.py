import numpy as np
from scipy import stats

from ..normmix import IntensitySurface, NormMix, approx_normmix
from .damcmc import DamcmcResult, drop_realization


def get_log_density_values(fit: DamcmcResult, truncate: bool = False) -> np.ndarray:
    """Compute each post-burnin iteration's log-likelihood of the observed points under that iteration's mixture.

    :param fit: A fitted DAMCMC chain.
    :param truncate: Whether to normalize each component's density by its
        in-window probability mass.
    :returns: Per-iteration log-likelihood, shape ``(fit.L,)``.
    """
    xy = np.column_stack([fit.data.x, fit.data.y])
    n, m, L = fit.data.n, fit.m, fit.L
    out = np.empty(L)
    for l in range(L):
        ps, mus, sigmas = fit.genps[l], fit.genmus[l], fit.gensigmas[l]
        if truncate:
            mix = NormMix(ps=ps, mus=list(mus), sigmas=list(sigmas))
            mass = approx_normmix(mix, fit.data.window)
            mass = np.where(mass <= 1e-12, 1.0, mass)
        else:
            mass = np.ones(m)
        dens = np.zeros(n)
        for k in range(m):
            rv = stats.multivariate_normal(mean=mus[k], cov=sigmas[k])
            dens += (ps[k] / mass[k]) * rv.pdf(xy)
        out[l] = np.sum(np.log(np.clip(dens, 1e-300, None)))
    return out


def get_map_est(
    fit: DamcmcResult,
    burnin: int | None = None,
    truncate: bool = False,
    d: float | np.ndarray | None = None,
    mu0: np.ndarray | None = None,
    Sigma0: np.ndarray | None = None,
    df0: float = 10.0,
    sig0: float = 1.0,
) -> IntensitySurface:
    """Return the intensity surface at the highest-scoring post-burnin iteration.

    Each retained iteration is scored by
    ``log p(data | ps, mus, sigmas) + log p(ps, mus, sigmas)``: the data
    log-likelihood (the sampler conditions on ``n``, so the Poisson rate
    plays no role in comparing iterations -- the returned surface's rate
    is simply set to ``n``) plus the log-prior under independent
    ``mu_k ~ N(mu0, Sigma0)``, ``sigma_k ~ InverseWishart(df0, sig0^2*I)``,
    ``ps ~ Dirichlet(d)`` priors. These scoring priors are independent of
    whatever hyperparameters the chain was originally fit with -- they
    only control how a finished fit is re-scored.

    ``truncate`` renormalizes each component's density by its in-window
    probability mass when scoring.

    :param fit: A fitted DAMCMC chain.
    :param burnin: Number of initial iterations to discard; defaults to
        10% of ``fit.L``.
    :param truncate: Whether to normalize each component's density by its
        in-window probability mass when scoring.
    :param d: Dirichlet concentration weight(s) for the scoring prior;
        defaults to 1 for every component.
    :param mu0: Prior mean for component means; defaults to the data
        mean.
    :param Sigma0: Prior covariance for component means; defaults to the
        data covariance (or a fallback if degenerate).
    :param df0: Inverse-Wishart degrees of freedom for the scoring prior;
        must be > 1 for a proper 2D prior.
    :param sig0: Inverse-Wishart scale for the scoring prior.
    :returns: The intensity surface at the highest-scoring iteration.
    :raises ValueError: If burn-in discards every realization,
        ``sig0 <= 0``, ``df0 <= 1``, ``d`` has the wrong length, or any
        ``d`` weight is <= 0.
    """
    if burnin is None:
        burnin = int(0.1 * fit.L)
    fit = drop_realization(fit, burnin)
    if fit.L == 0:
        raise ValueError("burnin discards every realization -- nothing left to score.")

    m = fit.m
    n = fit.data.n
    xy = np.column_stack([fit.data.x, fit.data.y])

    if sig0 <= 0:
        raise ValueError(f"sig0 must be > 0, got {sig0}")
    if df0 <= 1:
        raise ValueError(f"df0 must be > 1 for a proper 2D Inverse-Wishart prior, got {df0}")

    if Sigma0 is None:
        Sigma0 = np.cov(xy, rowvar=False)
        if n < 3 or np.any(np.linalg.eigvalsh(Sigma0) <= 1e-12):
            Sigma0 = np.eye(2) * max(np.var(xy), 1e-6)
    if mu0 is None:
        mu0 = xy.mean(axis=0)

    if d is None:
        d_vec = np.ones(m)
    else:
        d_vec = np.broadcast_to(np.asarray(d, dtype=float), (m,)).copy()
        if d_vec.shape != (m,):
            raise ValueError(f"d must have length m={m}, got shape {d_vec.shape}")
    if np.any(d_vec <= 0):
        raise ValueError(f"d (Dirichlet weights) must be > 0, got {d_vec}")

    log_lik = get_log_density_values(fit, truncate=truncate)

    Psi0 = (sig0**2) * np.eye(2)
    mu_prior = stats.multivariate_normal(mean=mu0, cov=Sigma0)
    sigma_prior = stats.invwishart(df=df0, scale=Psi0)

    log_prior = np.empty(fit.L)
    for l in range(fit.L):
        ps_l = fit.genps[l]
        lp = stats.dirichlet.logpdf(ps_l / ps_l.sum(), d_vec)
        for k in range(m):
            lp += mu_prior.logpdf(fit.genmus[l, k])
            lp += sigma_prior.logpdf(fit.gensigmas[l, k])
        log_prior[l] = lp

    map_iter = int(np.argmax(log_lik + log_prior))

    return IntensitySurface(
        ps=fit.genps[map_iter],
        mus=list(fit.genmus[map_iter]),
        sigmas=list(fit.gensigmas[map_iter]),
        lam=float(n),
        window=fit.data.window,
        estimated=True,
    )