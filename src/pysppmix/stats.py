from dataclasses import dataclass

import numpy as np


@dataclass
class ChainStats:
    """Posterior mean and equal-tailed credible interval for one scalar MCMC chain.

    :ivar mean: Posterior mean.
    :ivar credible_set: ``(low, high)`` bounds of the equal-tailed credible
        interval.
    :ivar confidence: Credible level as a percentage, e.g. ``95.0`` for a
        95% credible set.
    """

    mean: float
    credible_set: tuple[float, float]
    confidence: float

    def __repr__(self) -> str:
        lo, hi = self.credible_set
        return f"mean={self.mean:.4g}, {self.confidence:g}% credible set=[{lo:.4g}, {hi:.4g}]"


def get_stats(chain: np.ndarray, alpha: float = 0.05) -> ChainStats:
    """Compute the posterior mean and an equal-tailed credible interval for an MCMC chain.

    :param chain: 1-D array of scalar MCMC draws.
    :param alpha: Tail probability; the credible interval covers
        ``1 - alpha`` of the chain via its ``alpha/2`` and ``1 - alpha/2``
        quantiles. Must be in ``(0, 1)``.
    :returns: The chain's posterior mean and credible interval.
    :raises ValueError: If ``chain`` is empty or ``alpha`` is not in
        ``(0, 1)``.
    """
    chain = np.asarray(chain, dtype=float)
    if chain.size == 0:
        raise ValueError("chain must not be empty.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    lo, hi = np.quantile(chain, [alpha / 2, 1 - alpha / 2])
    return ChainStats(mean=float(chain.mean()), credible_set=(float(lo), float(hi)), confidence=100 * (1 - alpha))
