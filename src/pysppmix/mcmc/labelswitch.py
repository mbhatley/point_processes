import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

from .damcmc import DamcmcResult, drop_realization


def _responsibilities(xy: np.ndarray, ps: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, mass=None) -> np.ndarray:
    m = len(ps)
    n = xy.shape[0]
    logw = np.empty((n, m))
    for k in range(m):
        rv = stats.multivariate_normal(mean=mus[k], cov=sigmas[k])
        logw[:, k] = np.log(ps[k]) + rv.logpdf(xy)
        if mass is not None:
            logw[:, k] -= np.log(mass[k])
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    return w


def _apply_perms(fit: DamcmcResult, perms: np.ndarray) -> DamcmcResult:
    """Permute a fitted chain's component ordering according to per-iteration permutations.

    :param fit: The chain to permute.
    :param perms: ``perms[l, k]`` is the old component index that becomes
        new slot ``k`` at iteration ``l``, shape ``(fit.L, fit.m)``.
    :returns: A new chain with components reordered consistently across
        every array field.
    """
    genps = np.take_along_axis(fit.genps, perms, axis=1)
    genmus = np.take_along_axis(fit.genmus, perms[:, :, None], axis=1)
    gensigmas = np.take_along_axis(fit.gensigmas, perms[:, :, None, None], axis=1)
    approx_mass = np.take_along_axis(fit.approx_mass, perms, axis=1)
    inv_perms = np.argsort(perms, axis=1)  # inv_perms[l, old_index] = new slot
    genzs = np.take_along_axis(inv_perms, fit.genzs, axis=1)

    return DamcmcResult(
        data=fit.data,
        m=fit.m,
        L=fit.L,
        genps=genps,
        genmus=genmus,
        gensigmas=gensigmas,
        genzs=genzs,
        genlambdas=fit.genlambdas,
        approx_mass=approx_mass,
        hyper_da=fit.hyper_da,
        truncate=fit.truncate,
    )


def _fix_ic(fit: DamcmcResult) -> DamcmcResult:
    perms = np.empty((fit.L, fit.m), dtype=int)
    for l in range(fit.L):
        perms[l] = np.lexsort((fit.genmus[l, :, 1], fit.genmus[l, :, 0]))
    return _apply_perms(fit, perms)


def _fix_sel(fit: DamcmcResult, max_iter: int) -> DamcmcResult:
    L, m, n = fit.L, fit.m, fit.data.n
    xy = np.column_stack([fit.data.x, fit.data.y])

    resp = np.empty((L, n, m))
    for l in range(L):
        mass = fit.approx_mass[l] if fit.truncate else None
        resp[l] = _responsibilities(xy, fit.genps[l], fit.genmus[l], fit.gensigmas[l], mass)

    perms = np.tile(np.arange(m), (L, 1))
    eps = 1e-12

    for _ in range(max_iter):
        permuted = np.take_along_axis(resp, perms[:, None, :], axis=2)  # (L, n, m)
        Qhat = permuted.mean(axis=0)  # (n, m)
        logQ = np.log(Qhat + eps)

        new_perms = np.empty_like(perms)
        for l in range(L):
            cost = -(logQ.T @ resp[l])  # cost[k_new, k_old]
            row_ind, col_ind = linear_sum_assignment(cost)
            new_perms[l, row_ind] = col_ind

        if np.array_equal(new_perms, perms):
            perms = new_perms
            break
        perms = new_perms

    return _apply_perms(fit, perms)


def fix_label_switching(
    fit: DamcmcResult,
    burnin: int | None = None,
    method: str = "ic",
    max_iter: int = 50,
) -> DamcmcResult:
    """Relabel mixture components across iterations to remove label switching.

    ``method="ic"`` (identifiability constraint) permutes each
    iteration's components into ascending order of their mean's
    x-coordinate (ties broken by y-coordinate) -- deterministic and
    non-iterative, since an ordering on a continuous coordinate
    essentially never has exact ties across distinct components.

    ``method="sel"`` uses Stephens' (2000) decision-theoretic relabeling:
    it alternates averaging each iteration's responsibility matrix (soft
    component-membership probabilities per point) into a running
    estimate, then finding, for each iteration independently, the
    permutation minimizing the KL-type loss between its own
    responsibilities and that running estimate -- solved exactly as a
    linear sum assignment problem via the Hungarian algorithm, repeated
    until the permutations stop changing.

    Both methods return a new chain (burn-in already applied) with every
    per-iteration array field permuted into a consistent labeling; call
    :func:`~pysppmix.mcmc.postprocess.get_pm_est` on the result afterward
    for a posterior-mean surface free of label switching.

    :param fit: A fitted DAMCMC chain.
    :param burnin: Number of initial iterations to discard; defaults to
        10% of ``fit.L``.
    :param method: Relabeling strategy, ``"ic"`` or ``"sel"``.
    :param max_iter: Maximum number of alternating-minimization passes
        for ``method="sel"``.
    :returns: The relabeled chain (unchanged if ``fit.m < 2``, since
        there is nothing to switch).
    :raises ValueError: If ``method`` is not ``"ic"`` or ``"sel"``, or
        burn-in discards every realization.
    """
    if method not in ("ic", "sel"):
        raise ValueError(f"method must be 'ic' or 'sel', got {method!r}")
    if burnin is None:
        burnin = int(0.1 * fit.L)
    fit = drop_realization(fit, burnin)
    if fit.L == 0:
        raise ValueError("burnin discards every realization -- nothing left to relabel.")
    if fit.m < 2:
        return fit  # nothing to switch

    if method == "ic":
        return _fix_ic(fit)
    return _fix_sel(fit, max_iter=max_iter)