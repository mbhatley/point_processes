import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from .normmix import NormMix, DensityGrid, is_normmix, is_intensity_surface, dnormmix, get_mixture_limits
from .pointprocess import PointPattern
from .mcmc.damcmc import DamcmcResult, drop_realization
from .mcmc.bdmcmc import BdmcmcResult, get_bd_table, get_bd_compfit
from .hawkes import HawkesProcess, hawkes_intensity

# Fixed, colorblind-validated categorical color order for adjacent-pair
# contexts such as lines/bars; colors repeat after 8 components.
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


def _categorical_color(k: int) -> str:
    """Look up the categorical color for a component index, cycling after 8.

    :param k: Zero-based component index.
    :returns: A hex color string.
    """
    return CATEGORICAL_COLORS[k % len(CATEGORICAL_COLORS)]


def _sequential_cmap(grayscale: bool = False) -> Colormap:
    """Build the sequential colormap used for density/intensity heatmaps.

    :param grayscale: If True, use a white-to-black ramp instead of blue.
    :returns: The colormap.
    """
    if grayscale:
        return LinearSegmentedColormap.from_list("pysppmix_gray", ["#ffffff", "#000000"])
    return LinearSegmentedColormap.from_list("pysppmix_blue", ["#fcfcfb", "#2a78d6", "#0b2b52"])


def plot_density_grid(
    grid: DensityGrid, contour: bool = False, grayscale: bool = False, ax: Axes | None = None, cmap: Colormap | None = None
) -> Axes:
    """Render a density/intensity grid as a heatmap or contour plot.

    :param grid: The grid of values to render.
    :param contour: If True, draw labeled contour lines instead of a
        filled heatmap.
    :param grayscale: If True, use a grayscale colormap.
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :param cmap: Colormap to use; defaults to the sequential blue (or
        grayscale) ramp.
    :returns: The axes the grid was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    cmap = cmap or _sequential_cmap(grayscale)
    if contour:
        cs = ax.contour(grid.x, grid.y, grid.z, cmap=cmap)
        ax.clabel(cs, inline=True, fontsize=8)
    else:
        im = ax.pcolormesh(grid.x, grid.y, grid.z, shading="auto", cmap=cmap)
        ax.figure.colorbar(im, ax=ax, label="Intensity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_normmix(
    mix: NormMix,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    L: int = 256,
    truncate: bool = False,
    contour: bool = False,
    grayscale: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot a mixture's density (or intensity) as a heatmap with component means overlaid.

    :param mix: The mixture (or intensity surface) to plot.
    :param xlim: X-axis limits; defaults to the surface's window (for an
        intensity surface) or to ``normmix.get_mixture_limits``.
    :param ylim: Y-axis limits; same default logic as ``xlim``.
    :param L: Grid resolution per axis.
    :param truncate: If True, renormalize the density/intensity to the
        plotted window.
    :param contour: If True, draw contour lines instead of a heatmap.
    :param grayscale: If True, use a grayscale colormap.
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :returns: The axes the mixture was drawn on.
    :raises TypeError: If ``mix`` is not a ``NormMix`` or
        ``IntensitySurface``.
    """
    if not is_normmix(mix):
        raise TypeError("mix must be a NormMix or IntensitySurface.")
    if is_intensity_surface(mix):
        xlim = mix.window.xrange
        ylim = mix.window.yrange
    elif xlim is None or ylim is None:
        xlim, ylim = get_mixture_limits(mix)

    grid = dnormmix(mix, xlim=xlim, ylim=ylim, L=L, truncate=truncate)
    ax = plot_density_grid(grid, contour=contour, grayscale=grayscale, ax=ax)
    mus = np.array(mix.mus)
    ax.scatter(mus[:, 0], mus[:, 1], marker="x", s=120, linewidths=2, c="red" if contour else "0.25")

    kind = "Poisson intensity surface" if is_intensity_surface(mix) else "Density"
    ax.set_title(f"{kind} ({mix.m} component{'s' if mix.m != 1 else ''})")
    return ax


def plot_point_pattern(
    pp: PointPattern,
    mus: np.ndarray | None = None,
    color_by_component: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Scatter-plot a point pattern, optionally colored by component and/or with means overlaid.

    :param pp: The point pattern to plot.
    :param mus: Component mean locations to overlay as markers.
    :param color_by_component: If True and ``pp.comp`` is set, color each
        point by its component label.
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :returns: The axes the pattern was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    if color_by_component and pp.comp is not None:
        for k in sorted(set(pp.comp.tolist())):
            mask = pp.comp == k
            ax.scatter(pp.x[mask], pp.y[mask], s=20, color=_categorical_color(k), label=f"Component {k + 1}")
        ax.legend(title="Component", fontsize=8)
    else:
        ax.scatter(pp.x, pp.y, s=20, color="black")

    if mus is not None:
        mus = np.asarray(mus)
        ax.scatter(mus[:, 0], mus[:, 1], marker="x", s=120, c="red", linewidths=2)

    ax.set_xlim(pp.window.xrange)
    ax.set_ylim(pp.window.yrange)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Point pattern (n={pp.n})")
    return ax


def plotmix_2d(
    intsurf,
    pattern: PointPattern | None = None,
    contour: bool = False,
    truncate: bool = True,
    L: int = 256,
    grayscale: bool = False,
    color_by_component: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot a Poisson intensity surface with an optional point-pattern overlay.

    :param intsurf: The intensity surface to plot.
    :param pattern: A point pattern to overlay as a scatter plot.
    :param contour: If True, draw contour lines instead of a heatmap.
    :param truncate: If True, renormalize the intensity to the surface's
        window.
    :param L: Grid resolution per axis.
    :param grayscale: If True, use a grayscale colormap.
    :param color_by_component: If True and ``pattern.comp`` is set, color
        each overlaid point by its component label.
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :returns: The axes the surface was drawn on.
    :raises TypeError: If ``intsurf`` is not an ``IntensitySurface``.
    """
    if not is_intensity_surface(intsurf):
        raise TypeError("intsurf must be an IntensitySurface (see to_int_surf()).")
    win = intsurf.window
    grid = dnormmix(intsurf, L=L, truncate=truncate)
    ax = plot_density_grid(grid, contour=contour, grayscale=grayscale, ax=ax)
    mus = np.array(intsurf.mus)
    ax.scatter(mus[:, 0], mus[:, 1], marker="x", s=120, linewidths=2, c="red" if contour else "0.25")

    if pattern is not None:
        if color_by_component and pattern.comp is not None:
            for k in sorted(set(pattern.comp.tolist())):
                mask = pattern.comp == k
                ax.scatter(
                    pattern.x[mask], pattern.y[mask], s=15, color=_categorical_color(k),
                    label=f"Component {k + 1}",
                )
            ax.legend(title="Component", fontsize=8)
        else:
            ax.scatter(pattern.x, pattern.y, s=15, color="black")
        title = f"Poisson intensity surface (lambda={intsurf.lam:.1f}, m={intsurf.m}, n={pattern.n})"
    else:
        title = f"Poisson intensity surface (lambda={intsurf.lam:.1f}, m={intsurf.m})"

    ax.set_xlim(win.xrange)
    ax.set_ylim(win.yrange)
    ax.set_title(title)
    return ax


def plot_surface_3d(grid: DensityGrid, grayscale: bool = False, ax: Axes | None = None) -> Axes:
    """Render a density/intensity grid as a static 3D surface.

    Uses matplotlib's 3D toolkit, which does not provide interactive
    rotation; the result is a fixed-view surface suitable for embedding
    in a notebook or saved figure.

    :param grid: The grid of values to render.
    :param grayscale: If True, use a grayscale colormap.
    :param ax: 3D axes to draw on; a new figure/axes is created if
        omitted.
    :returns: The axes the surface was drawn on.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    xx, yy = np.meshgrid(grid.x, grid.y)
    ax.plot_surface(xx, yy, grid.z, cmap=_sequential_cmap(grayscale), linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Density")
    return ax


def plot_chains(fit, burnin: int | None = None, separate: bool = True) -> dict[str, Figure]:
    """Plot trace plots of the generated mixing-probability and mean chains.

    If ``fit`` is a ``BdmcmcResult``, it is first reduced to its
    MAP-component-count sub-chain via ``get_bd_compfit``.

    :param fit: The MCMC fit whose chains should be plotted.
    :param burnin: Number of leading iterations to drop before plotting;
        defaults to 10% of ``fit.L``.
    :param separate: If True, draw one subplot per component in a
        ``ceil(sqrt(m))``-column grid; if False, overlay all components
        on a single axes per series.
    :returns: One figure per series (``"p"``, ``"x"``, ``"y"``), keyed
        by series name.
    :raises TypeError: If ``fit`` is not a ``DamcmcResult`` or
        ``BdmcmcResult``.
    """
    if isinstance(fit, BdmcmcResult):
        tab = get_bd_table(fit, show=False)
        fit = get_bd_compfit(fit, tab["MAPcomp"])
    elif not isinstance(fit, DamcmcResult):
        raise TypeError("fit must be a DamcmcResult or BdmcmcResult.")

    if burnin is None:
        burnin = int(0.1 * fit.L)
    fit = drop_realization(fit, burnin)
    m = fit.m
    iters = np.arange(1, fit.L + 1)

    series = [
        ("p", fit.genps, "Probability", "Generated mixture probabilities"),
        ("x", fit.genmus[:, :, 0], r"$\mu_x$", "Generated mixture mean, X-coordinate"),
        ("y", fit.genmus[:, :, 1], r"$\mu_y$", "Generated mixture mean, Y-coordinate"),
    ]

    figs: dict[str, Figure] = {}
    for key, data, ylabel, title in series:
        if separate:
            ncols = int(np.ceil(np.sqrt(m)))
            nrows = int(np.ceil(m / ncols))
            fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
            for k in range(m):
                a = axes.flat[k]
                a.plot(iters, data[:, k], color=_categorical_color(k), linewidth=1)
                a.set_title(f"Component {k + 1}", fontsize=9)
            for a in axes.flat[m:]:
                a.axis("off")
            fig.supxlabel("Iteration")
            fig.supylabel(ylabel)
            fig.suptitle(f"{title} (m={m}, L={fit.L})")
        else:
            fig, ax = plt.subplots()
            for k in range(m):
                ax.plot(iters, data[:, k], color=_categorical_color(k), linewidth=1, label=f"Component {k + 1}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} (m={m}, L={fit.L})")
            ax.legend(fontsize=8)
        figs[key] = fig
    return figs


def plot_comp_dist(
    fit: BdmcmcResult, burnin: int | None = None, axes: tuple[Axes, Axes] | None = None
) -> tuple[Axes, Axes]:
    """Plot the posterior distribution of the number of components and its trace.

    :param fit: The BDMCMC fit to summarize.
    :param burnin: Number of leading iterations to drop before
        tabulating; defaults to 10% of ``fit.L``.
    :param axes: A pair of axes ``(bar_chart, trace)`` to draw on; new
        ones are created if omitted.
    :returns: The bar-chart axes and the trace-plot axes.
    """
    if burnin is None:
        burnin = int(0.1 * fit.L)
    burned = drop_realization(fit, burnin)
    tab = get_bd_table(burned, show=False)
    freq = tab["FreqTab"]

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    xs = np.arange(1, fit.maxnumcomp + 1)
    ax1.bar(xs, freq, color=_categorical_color(0))
    ax1.set_xticks(xs)
    ax1.set_xlabel("Number of components")
    ax1.set_ylabel(f"Iterations, {burned.L} total")
    ax1.set_title("Distribution of the number of components")
    if freq.max() > 0:
        for x, f in zip(xs, freq):
            if f:
                ax1.text(x, f + 0.02 * freq.max(), str(int(f)), ha="center", fontsize=8)

    ax2.plot(np.arange(1, fit.L + 1), fit.numcomp, color=_categorical_color(0), linewidth=1)
    ax2.set_ylim(1, fit.maxnumcomp)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Number of components")
    ax2.set_title("Generated chain for the number of components")

    return ax1, ax2


def plot_hawkes(H: HawkesProcess, L: int = 10000, ax: Axes | None = None) -> Axes:
    """Plot a Hawkes process's intensity curve with event times marked as ticks.

    :param H: The Hawkes process to plot.
    :param L: Number of points used to evaluate the intensity curve.
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :returns: The axes the intensity curve was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.linspace(0, H.T_max, L)
    intensity = hawkes_intensity(x, H.times, H.mu, H.alpha, H.beta)
    ax.plot(x, intensity, color=_categorical_color(0), linewidth=1)
    tick_y = H.mu + 0.01 * (intensity.max() - intensity.min())
    ax.plot(H.times, np.full_like(H.times, tick_y), "|", color=_categorical_color(7), markersize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Hawkes intensity")
    ax.set_title(f"Hawkes process intensity (mu={H.mu:.3g}, alpha={H.alpha:.3g}, beta={H.beta:.3g}, T={H.T_max:.3g})")
    return ax


def plot_region(
    region,
    ax: Axes | None = None,
    color: str = "gray",
    boundary: bool = True,
    alpha: float = 1.0,
) -> Axes:
    """Draw a single region that provides a ``.boundary()`` method.

    :param region: The region to draw (e.g. an ``EllipseWindow`` or
        ``PolygonWindow``).
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :param color: Fill color for the region.
    :param boundary: If True, draw the outline; if False, fill only.
    :param alpha: Fill opacity.
    :returns: The axes the region was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    pts = region.boundary()
    patch = MplPolygon(pts, closed=True, facecolor=color, edgecolor="black" if boundary else "none", alpha=alpha)
    ax.add_patch(patch)
    xr, yr = region.xrange, region.yrange
    pad_x, pad_y = 0.1 * (xr[1] - xr[0]), 0.1 * (yr[1] - yr[0])
    ax.set_xlim(xr[0] - pad_x, xr[1] + pad_x)
    ax.set_ylim(yr[0] - pad_y, yr[1] + pad_y)
    ax.set_aspect("equal")
    return ax


def plot_regions(regions, pattern: PointPattern | None = None, ax: Axes | None = None, color: str = "gray") -> Axes:
    """Draw several regions at once, with axis limits spanning all of them.

    :param regions: The regions to draw.
    :param pattern: A point pattern to overlay as a scatter plot.
    :param ax: Axes to draw on; a new figure/axes is created if omitted.
    :param color: Fill color for each region.
    :returns: The axes the regions were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for region in regions:
        pts = region.boundary()
        patch = MplPolygon(pts, closed=True, facecolor=color, edgecolor="black")
        ax.add_patch(patch)
        xr, yr = region.xrange, region.yrange
        xmins.append(xr[0]); xmaxs.append(xr[1])
        ymins.append(yr[0]); ymaxs.append(yr[1])
    if pattern is not None:
        ax.scatter(pattern.x, pattern.y, s=15, color="black", zorder=3)
        xmins.append(pattern.window.xrange[0]); xmaxs.append(pattern.window.xrange[1])
        ymins.append(pattern.window.yrange[0]); ymaxs.append(pattern.window.yrange[1])
    if xmins:
        ax.set_xlim(min(xmins), max(xmaxs))
        ax.set_ylim(min(ymins), max(ymaxs))
    ax.set_aspect("equal")
    return ax
