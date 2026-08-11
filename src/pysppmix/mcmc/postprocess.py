import numpy as np

from ..normmix import IntensitySurface
from .damcmc import DamcmcResult, drop_realization


def get_pm_est(fit: DamcmcResult, burnin: int | None = None) -> IntensitySurface:
    """Compute the posterior-mean intensity surface, averaging weights, means, covariances, and rate over post-burnin iterations.

    :param fit: A fitted DAMCMC chain.
    :param burnin: Number of initial iterations to discard; defaults to
        10% of ``fit.L``.
    :returns: The posterior-mean intensity surface.
    :raises ValueError: If burn-in discards every realization.
    """
    if burnin is None:
        burnin = int(0.1 * fit.L)
    fit = drop_realization(fit, burnin)
    if fit.L == 0:
        raise ValueError("burnin discards every realization -- nothing left to average.")

    post_ps = fit.genps.mean(axis=0)
    post_mus = list(fit.genmus.mean(axis=0))
    post_sigmas = list(fit.gensigmas.mean(axis=0))
    mean_lambda = float(fit.genlambdas.mean())

    return IntensitySurface(
        ps=post_ps, mus=post_mus, sigmas=post_sigmas,
        lam=mean_lambda, window=fit.data.window, estimated=True,
    )