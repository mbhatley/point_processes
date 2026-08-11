"""Interactive scratch space for exercising pysppmix functions one at a time.

Run the whole file (``python sandbox.py``) for a quick end-to-end smoke
test, or -- better, in an IDE -- run it cell-by-cell / drop into a
debugger on any line, or open a REPL and ``from sandbox import *`` then
call things directly:

    $ source .venv/bin/activate
    $ python -i sandbox.py
    >>> mix
    >>> dnormmix(mix, xlim=(0, 1), ylim=(0, 1), L=32).z.max()

As functions get ported from legacy_reference/sppmix_dump.R, add a small
section here exercising each one -- this file is meant to grow alongside
the port, not stay fixed.
"""
from src.pysppmix import (
    Window,
    NormMix,
    IntensitySurface,
    to_int_surf,
    is_normmix,
    is_intensity_surface,
    approx_normmix,
    dnormmix,
    rnormmix,
    rsppmix,
    gen_n_from_mix,
    est_mix_damcmc,
    drop_realization,
    get_pm_est,
    get_map_est,
    fix_label_switching,
    est_mix_bdmcmc,
    get_bd_table,
    get_bd_compfit,
    plot_normmix,
    plotmix_2d,
    plot_chains,
    plot_comp_dist,
    plot_hawkes,
    gen_markov_pp,
    fit_strauss,
    bayesian_markov_pp_mh,
    gen_hawkes,
    fit_hawkes,
    EllipseWindow,
    PolygonWindow,
    random_ellipses,
    random_polygons,
    plot_regions,
    hppp,
    ippp,
    plot_density_grid,
    rmippp_cond_mark,
    est_mippp_cond_mark,
    rmippp_cond_loc,
    est_mippp_cond_loc,
)
import matplotlib

matplotlib.use("Agg")  # headless: this script saves PNGs instead of showing windows
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_DIR = os.path.join(os.path.dirname(__file__), "sandbox_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

rng = np.random.default_rng(0)

# --- 1. Build a normmix by hand (port of demo/use_sppmix_objects.R) ---
mix = NormMix(
    ps=[0.3, 0.7],
    mus=[[0.2, 0.2], [0.8, 0.8]],
    sigmas=[0.01 * np.eye(2), 0.01 * np.eye(2)],
)
print(mix)
print(mix.summary())

# --- 2. Turn it into an intensity surface over a window ---
demo_truemix3comp = NormMix(
    ps=[0.2, 0.5, 0.3],
    mus=[[-0.3, -1.3], [0.1, 0.5], [0.7, 1.7]],
    sigmas=[0.3 * np.eye(2), 0.5 * np.eye(2), 0.2 * np.eye(2)],
)
intsurf = to_int_surf(demo_truemix3comp, lam=200, win=Window((-1, 1), (-2, 3)))
print(intsurf)

# --- 3. Evaluate its density on a grid ---
grid = dnormmix(intsurf, L=64)
print("density grid z shape:", grid.z.shape, "max:", grid.z.max())

# --- 4. Draw a random mixture and a point pattern from it ---
random_mix = rnormmix(m=3, sig0=0.1, df=5, window=Window((-3, 3), (-3, 3)), rng=rng)
print(random_mix)

pp = rsppmix(intsurf, rng=rng)
print(pp)
print("first 5 points:\n", np.column_stack([pp.x, pp.y, pp.comp])[:5])

# --- 5. Fit a DAMCMC model back to simulated data and check recovery ---
# (port of demo/use_sppmix_DAMCMC_noedgeeffects.R style workflow)
true_mix = NormMix(
    ps=[0.5, 0.5],
    mus=[[-2.0, -2.0], [2.0, 2.0]],
    sigmas=[0.2 * np.eye(2), 0.2 * np.eye(2)],
)
true_surf = to_int_surf(true_mix, lam=150, win=Window((-5, 5), (-5, 5)))
fit_pp = rsppmix(true_surf, rng=rng)
print(f"\nSimulated {fit_pp.n} points from a known 2-component mixture.")

fit = est_mix_damcmc(fit_pp, m=2, L=500, rng=rng)
print(fit)
pm_est = get_pm_est(fit)
print("Posterior mean estimate:", pm_est)
print("Recovered means (label order may differ from the true means above):")
print(np.array(pm_est.mus))

# --- 6. MAP estimate and label-switching correction ---
map_est = get_map_est(fit)
print("MAP estimate means:\n", np.array(map_est.mus))

fixed_ic = fix_label_switching(fit, method="ic")
print("PM estimate after IC relabeling:\n", np.array(get_pm_est(fixed_ic, burnin=0).mus))

fixed_sel = fix_label_switching(fit, method="sel")
print("PM estimate after SEL (Stephens 2000) relabeling:\n", np.array(get_pm_est(fixed_sel, burnin=0).mus))

# --- 7. BDMCMC: let the number of components be random and see if it finds 2 ---
bd_fit = est_mix_bdmcmc(fit_pp, m=5, L=1500, rng=rng)
print(bd_fit)
bd_tab = get_bd_table(bd_fit, show=True)
print("BDMCMC component-count table:", bd_tab)

bd_sub = get_bd_compfit(bd_fit, num_comp=bd_tab["MAPcomp"])
bd_pm = get_pm_est(bd_sub, burnin=0)
print(f"Posterior mean surface for the MAP ({bd_tab['MAPcomp']}-component) sub-chain:\n", np.array(bd_pm.mus))

# --- 8. Plotting (saved to sandbox_plots/ since this runs headlessly) ---
ax = plot_normmix(intsurf, contour=True)
ax.figure.savefig(f"{PLOT_DIR}/normmix_contour.png", dpi=110)
plt.close(ax.figure)

ax = plotmix_2d(intsurf, pp, color_by_component=True)
ax.figure.savefig(f"{PLOT_DIR}/plotmix_2d.png", dpi=110)
plt.close(ax.figure)

chain_figs = plot_chains(fit)
chain_figs["p"].savefig(f"{PLOT_DIR}/chains_p.png", dpi=110)
for f in chain_figs.values():
    plt.close(f)

ax1, ax2 = plot_comp_dist(bd_fit)
ax1.figure.savefig(f"{PLOT_DIR}/comp_dist.png", dpi=110)
plt.close(ax1.figure)

print(f"\nSaved plots to {PLOT_DIR}/")

# --- 9. TMSOPPRS: Gibbs/Markov point processes (hard-core Ripley rejection sampler) ---
gibbs_win = Window((0, 1), (0, 1))
hc_pp = gen_markov_pp((0.05,), start_n=40, b=100, func_choice=1, window=gibbs_win, L=2000, rng=rng)
print(f"\nSimulated a hard-core pattern with {hc_pp.n} points (min separation 0.05).")

strauss_fit = fit_strauss(hc_pp, L=3000, rng=rng)
print("fit_strauss posterior:", strauss_fit.stats())

hardcore_fit = bayesian_markov_pp_mh(hc_pp, thetas=(0.05, 0), b=100, model_choice=1, L=3000, rng=rng)
print("bayesian_markov_pp_mh (improper prior) posterior:", hardcore_fit.stats())

# --- 10. TMSOPPRS: Hawkes process simulation and Bayesian fit ---
H = gen_hawkes(T_max=300, mu=2.0, alpha=1.0, beta=3.0, rng=rng)
print(f"\nSimulated a Hawkes process with {len(H.times)} events (true mu=2, alpha=1, beta=3).")

hawkes_fit = fit_hawkes(H.times, T_max=300, L=6000, hyperparams=(0.1, 0.1, 0.1), small_sig=0.12, rng=rng)
print("fit_hawkes posterior:", hawkes_fit.stats())

ax = plot_hawkes(H)
ax.figure.savefig(f"{PLOT_DIR}/hawkes.png", dpi=110)
plt.close(ax.figure)

# --- 11. TMSOPPRS: window/region shapes (ellipse, polygon) ---
ellipse = EllipseWindow(mu=[0, 0], A=np.array([[4.0, 1.0], [1.0, 1.0]]), r=1.5)
print(f"\nEllipse area={ellipse.area:.3f}, bounding box x={ellipse.xrange}, y={ellipse.yrange}")

# hppp/ippp generalize to any region, not just rectangles
region_pp = hppp(30, ellipse, rng=rng)
print(f"HPPP over the ellipse: {region_pp.n} points, all inside: {np.all(ellipse.contains(region_pp.x, region_pp.y))}")

result = ippp(lambda x, y: 50 * np.exp(-(x**2 + y**2)), ellipse, grid_size=40, rng=rng)
print(f"IPPP over the ellipse: {result['pp'].n} points after thinning")

# scatter random ellipses/polygons at point locations (rEllipseTMSO/rPolygonsTMSO demos)
scatter_win = Window((0, 10), (0, 10))
scatter_pp = hppp(0.15, scatter_win, rng=rng)  # lam is a rate per unit area, so ~15 points here
ellipses = random_ellipses(scatter_pp, A=np.eye(2) * 3, rng=rng)
polys = random_polygons(scatter_pp, num_vert=6, rng=rng)

ax = plot_regions(ellipses, pattern=scatter_pp, color="#1baf7a")
ax.figure.savefig(f"{PLOT_DIR}/random_ellipses.png", dpi=110)
plt.close(ax.figure)

ax = plot_regions(polys, pattern=scatter_pp, color="#eda100")
ax.figure.savefig(f"{PLOT_DIR}/random_polygons.png", dpi=110)
plt.close(ax.figure)

# --- 12. Marked IPPP: condition on mark (locations/marks independent) ---
mark_sim = rmippp_cond_mark(lam=400, params=[0.2, 0.5, 0.3], window=Window((-10, 10), (-10, 10)), rng=rng)
print(f"\nSimulated a MIPPP (cond. on mark) with {mark_sim.genMPP.n} points, true params=[0.2, 0.5, 0.3].")
m_true = [s.m for s in mark_sim.ground_surfs]
mark_fit = est_mippp_cond_mark(mark_sim.genMPP, m=m_true, L=1000, rng=rng)
print("Recovered mark distribution:", mark_fit.mark_dist)

# --- 13. Marked IPPP: condition on location (marks depend on neighbors' marks) ---
loc_sim = rmippp_cond_loc(gammas=[0.3, 1.0], r=0.4, window=Window((-3, 3), (-3, 3)), L=30000, rng=rng)
print(f"Simulated a MIPPP (cond. on location) with {loc_sim.genMPP.n} points, true gammas={loc_sim.gammas}.")
loc_fit = est_mippp_cond_loc(loc_sim.genMPP, r=loc_sim.r, hyper=0.15, L=4000, rng=rng)
print("Recovered posterior mean gammas:", loc_fit.mean_gammas())

fields = loc_fit.prob_fields(LL=100)
fig, axes = plt.subplots(1, len(fields), figsize=(5.5 * len(fields), 4.5))
for j, ax in enumerate(np.atleast_1d(axes)):
    plot_density_grid(fields[j], ax=ax)
    ax.set_title(f"P(mark={j + 1} | location)")
fig.savefig(f"{PLOT_DIR}/mippp_cond_loc_fields.png", dpi=110)
plt.close(fig)

if __name__ == "__main__":
    print("\nsandbox.py ran end-to-end without error.")