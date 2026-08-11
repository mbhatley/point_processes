import numpy as np

from ..pointprocess import PointPattern
from ..window import Window
from .energy import (
    FUNC_HARDCORE,
    default_hardcore_main_effect,
    default_hardcore_pairwise_effect,
    get_general_energy,
    get_papangelou_at_x,
    pairwise_func,
    papangelou_at_x,
)

_MAX_REJECTION_ATTEMPTS = 5000


def _resolve_move_probs(move_setup: int, birth_prob: float, change_prob: float) -> tuple[float, float, float]:
    """Resolve birth/change/death move probabilities for a move-set configuration.

    :param move_setup: Which move set to use: 1 (birth/death only), 2 (birth/death/change with an explicit ``change_prob``), or 3 (change moves only).
    :param birth_prob: Probability of proposing a birth move.
    :param change_prob: Probability of proposing a change move; only used when ``move_setup=2``.
    :returns: The resolved ``(birth_prob, change_prob, death_prob)`` probabilities.
    :raises ValueError: If ``move_setup`` is not 1, 2, or 3, or if ``birth_prob + change_prob > 1`` under ``move_setup=2`` (leaving no probability mass for a death move).
    """
    if move_setup == 1:
        return birth_prob, 0.0, 1 - birth_prob
    if move_setup == 2:
        death_prob = 1 - birth_prob - change_prob
        if death_prob < 0:
            raise ValueError(
                f"birth_prob ({birth_prob}) + change_prob ({change_prob}) > 1 leaves no room for deaths."
            )
        return birth_prob, change_prob, death_prob
    if move_setup == 3:
        return 0.0, 1.0, 0.0
    raise ValueError(f"move_setup must be 1, 2, or 3, got {move_setup}")


def gen_markov_pp(
    thetas: np.ndarray,
    start_n: int = 10,
    b: float = 50.0,
    func_choice: int = FUNC_HARDCORE,
    window: Window = Window((0.0, 1.0), (0.0, 1.0)),
    L: int = 50000,
    rng: np.random.Generator | None = None,
) -> PointPattern:
    """Simulate a Gibbs point process via Ripley's rejection method.

    Resamples one point at a time: a point is dropped, a replacement is
    proposed uniformly at random over the window, and the replacement is
    accepted with probability equal to its Papangelou conditional
    intensity relative to an upper bound.

    :param thetas: Interaction parameters; ``thetas[0]`` is the interaction range, and ``thetas[1]`` is the Strauss interaction strength (only used when ``func_choice=2``).
    :param start_n: Initial number of points.
    :param b: Baseline intensity.
    :param func_choice: Interaction type: hard-core (``FUNC_HARDCORE``) or Strauss (``FUNC_STRAUSS``).
    :param window: Observation window to simulate over.
    :param L: Number of resampling iterations.
    :param rng: Random number generator; a new default generator is created if omitted.
    :returns: The simulated point pattern.
    :raises ValueError: If ``start_n <= 0`` or ``b < 0``.
    :raises RuntimeError: If a single point's rejection sampling exceeds the attempt limit, or if a computed acceptance probability falls outside ``[0, 1]``.
    """
    if start_n <= 0:
        raise ValueError(f"start_n must be > 0, got {start_n}")
    if b < 0:
        raise ValueError(f"b must be >= 0, got {b}")
    rng = rng or np.random.default_rng()
    xr, yr = window.xrange, window.yrange

    pts = np.column_stack([rng.uniform(*xr, start_n), rng.uniform(*yr, start_n)])
    gamma = thetas[1] if func_choice == 2 else None

    for _ in range(L):
        n = pts.shape[0]
        drop = rng.integers(n)
        rest = np.delete(pts, drop, axis=0)

        if func_choice == 2:
            max_papangelou = b if gamma < 1 else b * gamma ** (n - 1)
        else:
            max_papangelou = b

        for attempt in range(_MAX_REJECTION_ATTEMPTS):
            prop = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            total = sum(pairwise_func(prop, rest[i], thetas, func_choice) for i in range(rest.shape[0]))
            papangelou = b * np.exp(-total)
            prob = papangelou / max_papangelou
            if not (0 <= prob <= 1):
                raise RuntimeError(f"Papangelou probability out of [0,1]: {prob} (energy total={total})")
            if rng.random() < prob:
                break
        else:
            raise RuntimeError(
                f"Rejection sampling exceeded {_MAX_REJECTION_ATTEMPTS} attempts for one point -- "
                "try a smaller hard-core distance/gamma, or a larger window."
            )
        pts = np.vstack([rest, prop])

    return PointPattern(x=pts[:, 0], y=pts[:, 1], window=window)


def gen_markov_pp_bd(
    thetas: np.ndarray,
    start_n: int = 10,
    b: float = 50.0,
    move_setup: int = 1,
    func_choice: int = FUNC_HARDCORE,
    window: Window = Window((0.0, 1.0), (0.0, 1.0)),
    L: int = 50000,
    birth_prob: float = 1 / 3,
    change_prob: float = 1 / 3,
    save_last_iters: int = 0,
    rng: np.random.Generator | None = None,
):
    """Simulate a Gibbs point process via spatial birth-death-change Metropolis-Hastings.

    :param thetas: Interaction parameters passed through to the Papangelou conditional intensity.
    :param start_n: Initial number of points used to seed the process.
    :param b: Baseline intensity.
    :param move_setup: Which move set to use; see :func:`_resolve_move_probs`.
    :param func_choice: Interaction type: hard-core (``FUNC_HARDCORE``) or Strauss (``FUNC_STRAUSS``).
    :param window: Observation window to simulate over.
    :param L: Number of Metropolis-Hastings iterations.
    :param birth_prob: Probability of proposing a birth move.
    :param change_prob: Probability of proposing a change move; only used when ``move_setup=2``.
    :param save_last_iters: Number of trailing iterations' raw coordinate arrays to also return, e.g. for use as auxiliary patterns in an exchange-algorithm MCMC step.
    :param rng: Random number generator; a new default generator is created if omitted.
    :returns: The final realization, or ``(final_realization, saved_realizations)`` when ``save_last_iters > 0``.
    :raises ValueError: If ``start_n <= 0``, ``b < 0``, or ``save_last_iters > L``.
    """
    if start_n <= 0:
        raise ValueError(f"start_n must be > 0, got {start_n}")
    if b < 0:
        raise ValueError(f"b must be >= 0, got {b}")
    if save_last_iters > L:
        raise ValueError(f"save_last_iters ({save_last_iters}) must be <= L ({L})")

    rng = rng or np.random.default_rng()
    birth_p, change_p, death_p = _resolve_move_probs(move_setup, birth_prob, change_prob)
    window_area = window.area
    xr, yr = window.xrange, window.yrange

    pp0 = gen_markov_pp(thetas, start_n, b, func_choice, window, L=500, rng=rng)
    pts = np.column_stack([pp0.x, pp0.y])

    saved: list[np.ndarray] = []
    for k in range(L):
        n = pts.shape[0]
        u = rng.random()

        if u <= birth_p:
            propx = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            prop_pts = np.vstack([pts, propx])
            pap = papangelou_at_x(propx, pts, b, thetas, func_choice)
            mh_ratio = (window_area / (n + 1)) * pap
        elif u <= birth_p + death_p:
            if n == 1:
                mh_ratio = 0.0
                prop_pts = pts
            else:
                dele = rng.integers(n)
                rest = np.delete(pts, dele, axis=0)
                pap = papangelou_at_x(pts[dele], rest, b, thetas, func_choice)
                mh_ratio = (n / window_area) * (1 / pap) if pap > 0 else 0.0
                prop_pts = rest
        else:
            dele = rng.integers(n)
            rest = np.delete(pts, dele, axis=0)
            add_x = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            pap_add = papangelou_at_x(add_x, rest, b, thetas, func_choice)
            pap_del = papangelou_at_x(pts[dele], rest, b, thetas, func_choice)
            mh_ratio = pap_add / pap_del if pap_del > 0 else 0.0
            prop_pts = np.vstack([rest, add_x])

        if rng.random() < mh_ratio:
            pts = prop_pts

        if k >= L - save_last_iters:
            saved.append(pts.copy())

    pp = PointPattern(x=pts[:, 0], y=pts[:, 1], window=window)
    if save_last_iters > 0:
        return pp, saved
    return pp


def markov_pp_bd(
    main_params: np.ndarray = (50.0,),
    inter_params: np.ndarray = (0.1,),
    main_effect=default_hardcore_main_effect,
    pairwise_effect=default_hardcore_pairwise_effect,
    start_n: int = 10,
    move_setup: int = 1,
    window: Window = Window((0.0, 1.0), (0.0, 1.0)),
    L: int = 50000,
    birth_prob: float = 1 / 2,
    change_prob: float = 1 / 3,
    save_last_iters: int = 0,
    rng: np.random.Generator | None = None,
):
    """Simulate a Gibbs point process with an arbitrary main/pairwise energy via birth-death-change Metropolis-Hastings.

    The initial configuration is repeatedly redrawn uniformly at random
    until it satisfies the given energy constraint (has positive energy),
    since birth/death/change moves only ever reject *new* constraint
    violations and so could never recover from an invalid starting
    configuration on their own.

    :param main_params: Parameters passed to ``main_effect``.
    :param inter_params: Parameters passed to ``pairwise_effect``.
    :param main_effect: Per-point main-effect energy function.
    :param pairwise_effect: Per-pair interaction energy function.
    :param start_n: Initial number of points.
    :param move_setup: Which move set to use; see :func:`_resolve_move_probs`.
    :param window: Observation window to simulate over.
    :param L: Number of Metropolis-Hastings iterations.
    :param birth_prob: Probability of proposing a birth move.
    :param change_prob: Probability of proposing a change move; only used when ``move_setup=2``.
    :param save_last_iters: Number of trailing iterations' raw coordinate arrays to also return.
    :param rng: Random number generator; a new default generator is created if omitted.
    :returns: The final realization, or ``(final_realization, saved_realizations)`` when ``save_last_iters > 0``.
    :raises ValueError: If ``start_n <= 0`` or ``save_last_iters > L``.
    :raises RuntimeError: If no valid starting configuration is found within the attempt limit.
    """
    if start_n <= 0:
        raise ValueError(f"start_n must be > 0, got {start_n}")
    if save_last_iters > L:
        raise ValueError(f"save_last_iters ({save_last_iters}) must be <= L ({L})")

    rng = rng or np.random.default_rng()
    birth_p, change_p, death_p = _resolve_move_probs(move_setup, birth_prob, change_prob)
    window_area = window.area
    xr, yr = window.xrange, window.yrange

    # Redraw the initial configuration until it satisfies the energy
    # constraint -- a chance-invalid start could never self-repair, since
    # birth/death/change moves only ever reject new violations.
    for _ in range(_MAX_REJECTION_ATTEMPTS):
        pts = np.column_stack([rng.uniform(*xr, start_n), rng.uniform(*yr, start_n)])
        if get_general_energy(pts, main_effect, pairwise_effect, main_params, inter_params) > 0:
            break
    else:
        raise RuntimeError(
            f"Could not find a valid starting configuration of {start_n} points in "
            f"{_MAX_REJECTION_ATTEMPTS} attempts -- try a smaller start_n or weaker interaction."
        )

    saved: list[np.ndarray] = []
    for k in range(L):
        n = pts.shape[0]
        u = rng.random()

        if u <= birth_p:
            propx = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            pap = get_papangelou_at_x(propx, pts, main_effect, pairwise_effect, main_params, inter_params)
            prop_pts = np.vstack([pts, propx])
            mh_ratio = (window_area / (n + 1)) * pap
        elif u <= birth_p + death_p:
            if n == 1:
                mh_ratio = 0.0
                prop_pts = pts
            else:
                dele = rng.integers(n)
                rest = np.delete(pts, dele, axis=0)
                pap = get_papangelou_at_x(pts[dele], rest, main_effect, pairwise_effect, main_params, inter_params)
                mh_ratio = (n / window_area) * (1 / pap) if pap > 0 else 0.0
                prop_pts = rest
        else:
            dele = rng.integers(n)
            rest = np.delete(pts, dele, axis=0)
            add_x = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            pap_add = get_papangelou_at_x(add_x, rest, main_effect, pairwise_effect, main_params, inter_params)
            pap_del = get_papangelou_at_x(pts[dele], rest, main_effect, pairwise_effect, main_params, inter_params)
            mh_ratio = pap_add / pap_del if pap_del > 0 else 0.0
            prop_pts = np.vstack([rest, add_x])

        if rng.random() < mh_ratio:
            pts = prop_pts

        if k >= L - save_last_iters:
            saved.append(pts.copy())

    pp = PointPattern(x=pts[:, 0], y=pts[:, 1], window=window)
    if save_last_iters > 0:
        return pp, saved
    return pp


def matern_hc_models(
    pp: PointPattern,
    type: int = 1,
    r: float = 0.1,
    m_x: np.ndarray | None = None,
    R_x: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Apply Matern hard-core thinning to a point pattern.

    :param pp: Point pattern to thin.
    :param type: Thinning scheme: 1 (drop a point with an earlier neighbor within ``r``), 2 (drop a point with a higher-mark neighbor within ``r``), or 3 (as 2, but each point has its own radius ``R_x[i]``).
    :param r: Thinning radius, for types 1 and 2.
    :param m_x: Marks used to break ties for types 2/3; drawn from a standard normal distribution if omitted.
    :param R_x: Per-point thinning radii, required for type 3.
    :param rng: Random number generator; a new default generator is created if omitted.
    :returns: A dict with keys ``"pp"`` (input pattern), ``"thinned"`` (thinned pattern), and for types 2/3 also ``"m_x"`` (marks used) and, for type 3, ``"R_x"`` (radii used).
    :raises ValueError: If ``pp`` has no points, if ``type`` is not 1, 2, or 3, or if ``type=3`` and ``R_x`` is omitted.
    """
    if pp.n == 0:
        raise ValueError("pp has no points.")
    rng = rng or np.random.default_rng()
    xy = np.column_stack([pp.x, pp.y])
    n = pp.n
    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)

    if type == 1:
        keep = np.ones(n, dtype=bool)
        for i in range(1, n):
            if np.any(dists[i, :i] < r):
                keep[i] = False
        thinned = PointPattern(x=pp.x[keep], y=pp.y[keep], window=pp.window)
        return {"pp": pp, "thinned": thinned}

    if type in (2, 3):
        marks = m_x if m_x is not None else rng.standard_normal(n)
        radii = r if type == 2 else R_x
        if type == 3 and radii is None:
            raise ValueError("R_x is required for type 3.")
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            rad = radii if type == 2 else radii[i]
            close = (dists[i] < rad) & (np.arange(n) != i)
            if np.any(marks[close] < marks[i]):
                keep[i] = False
        thinned = PointPattern(x=pp.x[keep], y=pp.y[keep], window=pp.window)
        result = {"pp": pp, "thinned": thinned, "m_x": marks}
        if type == 3:
            result["R_x"] = radii
        return result

    raise ValueError(f"type must be 1, 2, or 3, got {type}")