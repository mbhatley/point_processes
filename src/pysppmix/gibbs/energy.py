from typing import Callable

import numpy as np

FUNC_HARDCORE = 1
FUNC_STRAUSS = 2


def pairwise_func(x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray, func_choice: int) -> float:
    """Compute the pairwise interaction energy between two points.

    Hard-core (``func_choice=FUNC_HARDCORE``, ``thetas=[r]``): infinite
    energy (forbidden) if the pair is closer than ``r``, else 0.
    Strauss (``func_choice=FUNC_STRAUSS``, ``thetas=[r, gamma]``):
    ``-log(gamma)`` if the (distinct) pair is within ``r``, else 0.

    :param x1: First point, shape ``(2,)``.
    :param x2: Second point, shape ``(2,)``.
    :param thetas: Interaction parameters; ``[r]`` for hard-core,
        ``[r, gamma]`` for Strauss.
    :param func_choice: ``FUNC_HARDCORE`` or ``FUNC_STRAUSS``.
    :returns: Pairwise interaction energy.
    :raises ValueError: If ``func_choice`` is neither ``FUNC_HARDCORE`` nor
        ``FUNC_STRAUSS``.
    """
    dist = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    if func_choice == FUNC_HARDCORE:
        return np.inf if dist < thetas[0] else 0.0
    if func_choice == FUNC_STRAUSS:
        return -np.log(thetas[1]) if 0 < dist <= thetas[0] else 0.0
    raise ValueError(f"func_choice must be {FUNC_HARDCORE} (hard-core) or {FUNC_STRAUSS} (Strauss), got {func_choice}")


def papangelou_at_x(x: np.ndarray, pts: np.ndarray, b: float, thetas: np.ndarray, func_choice: int) -> float:
    """Compute the Papangelou conditional intensity at a point given the rest of the pattern.

    :param x: Point at which to evaluate the conditional intensity, shape
        ``(2,)``.
    :param pts: The rest of the pattern ``x`` is conditioned on, shape
        ``(n, 2)``.
    :param b: Baseline intensity.
    :param thetas: Interaction parameters, see :func:`pairwise_func`.
    :param func_choice: ``FUNC_HARDCORE`` or ``FUNC_STRAUSS``.
    :returns: The conditional intensity at ``x``.
    """
    pts = np.atleast_2d(pts)
    if pts.shape[0] == 0:
        return float(b)
    total = sum(pairwise_func(x, pts[i], thetas, func_choice) for i in range(pts.shape[0]))
    return b * np.exp(-total)


def default_hardcore_main_effect(x: np.ndarray, params: np.ndarray) -> float:
    """Default hard-core main-effect term used by :func:`get_general_energy`.

    :param x: Point at which to evaluate the main effect, shape ``(2,)``.
    :param params: ``[beta]``, the baseline-intensity parameter.
    :returns: ``-log(beta)``.
    """
    return -np.log(params[0])


def default_hardcore_pairwise_effect(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> float:
    """Default hard-core pairwise-effect term used by :func:`get_general_energy`.

    :param x1: First point, shape ``(2,)``.
    :param x2: Second point, shape ``(2,)``.
    :param params: ``[r]``, the hard-core exclusion radius.
    :returns: Infinite energy if the pair is closer than ``r``, else 0.
    """
    dist = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return np.inf if dist < params[0] else 0.0


def get_general_energy(
    pts: np.ndarray,
    main_effect: Callable[[np.ndarray, np.ndarray], float] = default_hardcore_main_effect,
    pairwise_effect: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = default_hardcore_pairwise_effect,
    main_params: np.ndarray = (50.0,),
    inter_params: np.ndarray = (0.1,),
) -> float:
    """Compute ``exp(-total main effect - total pairwise energy)`` over a pattern.

    :param pts: Point pattern, shape ``(n, 2)``.
    :param main_effect: Per-point main-effect callable ``(point, main_params) -> float``.
    :param pairwise_effect: Per-pair interaction callable ``(point, point, inter_params) -> float``.
    :param main_params: Parameters passed to ``main_effect``.
    :param inter_params: Parameters passed to ``pairwise_effect``.
    :returns: The unnormalized density value for the pattern.
    :raises ValueError: If ``pts`` is empty.
    """
    pts = np.atleast_2d(pts)
    n = pts.shape[0]
    if n == 0:
        raise ValueError("pts must contain at least one point.")
    pair_sum = sum(
        pairwise_effect(pts[i], pts[j], inter_params) for i in range(n) for j in range(i + 1, n)
    )
    main_sum = sum(main_effect(pts[i], main_params) for i in range(n))
    return float(np.exp(-main_sum - pair_sum))


def get_papangelou_at_x(
    x: np.ndarray,
    pts: np.ndarray,
    main_effect: Callable[[np.ndarray, np.ndarray], float] = default_hardcore_main_effect,
    pairwise_effect: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = default_hardcore_pairwise_effect,
    main_params: np.ndarray = (50.0,),
    inter_params: np.ndarray = (0.1,),
) -> float:
    """Compute the Papangelou conditional intensity at a point for the general-energy model.

    :param x: Point at which to evaluate the conditional intensity, shape
        ``(2,)``.
    :param pts: The rest of the pattern ``x`` is conditioned on, shape
        ``(n, 2)``.
    :param main_effect: Per-point main-effect callable, see
        :func:`get_general_energy`.
    :param pairwise_effect: Per-pair interaction callable, see
        :func:`get_general_energy`.
    :param main_params: Parameters passed to ``main_effect``.
    :param inter_params: Parameters passed to ``pairwise_effect``.
    :returns: The conditional intensity at ``x``.
    :raises ValueError: If ``pts`` is empty.
    """
    pts = np.atleast_2d(pts)
    if pts.shape[0] == 0:
        raise ValueError("pts must contain at least one point (the process being conditioned on).")
    pair_sum = sum(pairwise_effect(x, pts[i], inter_params) for i in range(pts.shape[0]))
    return float(np.exp(-main_effect(x, main_params) - pair_sum))
