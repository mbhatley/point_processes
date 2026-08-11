# pysppmix

A Python library for modeling spatial point patterns - scattered `(x, y)`
locations in a 2D region - as realizations of a Poisson process whose
intensity is shaped by a finite mixture of bivariate normal components.
It provides the data model, Bayesian MCMC fitting (both a fixed and an
automatically-selected number of mixture components), simulation and
fitting for point-interaction (Gibbs/Markov) processes, self-exciting
(Hawkes) event streams, marked point patterns, and matplotlib-based
plotting for all of the above.

See [`docs/user_guide.md`](docs/user_guide.md) for a full description of
every function and class.

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies are NumPy, SciPy, pandas, and
matplotlib.

## Quick start

```python
import numpy as np
import matplotlib.pyplot as plt
from pysppmix import (
    Window, rnormmix, to_int_surf, rsppmix,
    est_mix_damcmc, fix_label_switching, get_pm_est, plotmix_2d,
)

rng = np.random.default_rng(0)
window = Window((0.0, 10.0), (0.0, 10.0))

# Build a random 3-component mixture intensity and simulate a point pattern from it.
true_mix = rnormmix(m=3, window=window, rng=rng)
true_surf = to_int_surf(true_mix, lam=200, win=window)
pattern = rsppmix(true_surf, rng=rng)

# Fit a 3-component mixture back to the simulated pattern via MCMC.
fit = est_mix_damcmc(pattern, m=3, L=5000, rng=rng)
fit = fix_label_switching(fit)
est_surf = get_pm_est(fit)

plotmix_2d(est_surf, pattern=pattern)
plt.show()
```

## What's included

- **Data model** (`Window`, `EllipseWindow`, `PolygonWindow`,
  `PointPattern`, `NormMix`, `IntensitySurface`) - validated on
  construction, so a malformed mixture or region raises immediately
  instead of propagating NaNs downstream.
- **Simulation** - `rsppmix` (mixture intensity surfaces), `hppp`/`ippp`
  (homogeneous/inhomogeneous Poisson processes for an arbitrary intensity
  function), Gibbs/Markov point processes with hard-core or Strauss
  interactions (`gen_markov_pp`, `gen_markov_pp_bd`, `markov_pp_bd`),
  Matérn hard-core thinning (`matern_hc_models`), self-exciting Hawkes
  processes (`gen_hawkes`), and marked point patterns under two different
  conditioning models (`rmippp_cond_mark`, `rmippp_cond_loc`).
- **Bayesian MCMC fitting** - Data Augmentation MCMC for a fixed number of
  mixture components (`est_mix_damcmc`) and Birth-Death MCMC for an
  automatically-selected number of components (`est_mix_bdmcmc`), plus
  label-switching correction (`fix_label_switching`), posterior-mean
  (`get_pm_est`) and MAP (`get_map_est`) summarization, and Bayesian
  fitting for Strauss/hard-core interaction models (`fit_strauss`,
  `bayesian_markov_pp_mh`), Hawkes processes (`fit_hawkes`), and marked
  point patterns (`est_mippp_cond_mark`, `est_mippp_cond_loc`).
- **Plotting** (`plotting.py`) - heatmaps, contour plots, 3D surfaces,
  point-pattern scatter plots, MCMC trace/diagnostic plots, and region
  outlines, all matplotlib-based. Density/intensity heatmaps use a single
  perceptually-uniform color ramp rather than a rainbow gradient, and
  per-component colors use a fixed, colorblind-validated palette;
  `grayscale=True` is available everywhere a heatmap is drawn.

## Known limitations

- Birth-Death MCMC (`est_mix_bdmcmc`) can occasionally get stuck with one
  extra spurious component for an entire run, since a component can only
  be removed once it happens to lose every assigned point - a known
  characteristic of this class of sampler. Always sanity-check
  `get_bd_table`'s mode against `plot_comp_dist` (and ideally more than
  one chain) rather than trusting a single run.
- The marks-depend-on-neighbors model (`est_mippp_cond_loc`) is fit by
  pseudo-likelihood, which under-covers the true parameter value once a
  typical point's neighborhood becomes large relative to the pattern -
  keep the neighborhood radius `r` modest.
- Only discrete marks are supported for marked point pattern fitting;
  continuous/random-field marks are not yet implemented.
- Non-rectangular regions (`EllipseWindow`, `PolygonWindow`) are supported
  for plain point-process simulation (`rsppmix`, `hppp`, `ippp`) but not
  yet for the Gibbs/Markov samplers or the MCMC fitters, which remain
  rectangular-`Window`-only.

## Development

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`sandbox.py` is a scratch script for exercising package functions
interactively - run it whole, run it with `python -i`, or step through it
in an IDE debugger.

## License

MIT - see [`LICENSE`](LICENSE).
