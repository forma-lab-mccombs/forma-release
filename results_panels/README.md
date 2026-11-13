# Committed aggregate result series

Aggregate-level (by-horizon / pooled) CSVs from the canonical camera-ready
(R14) paper run, build `r14_node_optionD_indfe_val8`. The pre-camera-ready (R13)
series are at the `r13` tag.

Unlike everything under `results/` (regenerated locally, gitignored), these
are committed: the paper points readers at them, they hold no firm-level
values — only metrics pooled over millions of grid cells per row — and so
fall outside the no-WRDS-derived-content rule.

## `full_sample_likelihood/`

Panel C (probabilistic quality) of the headline table, plus the calibration
claims made in the "Probabilistic quality" paragraphs of the results section.
Produced by the commands documented in `docs/ACCEPTANCE.md` §4:
`forma.scoring.evaluate` (seed-mixture track) and
`scripts/mixture_calibration.py`, over the released five-seed pools on the
Full sample (379,859,933 cells).

- `coverage_by_horizon.csv` — the per-horizon coverage series for the
  five-seed Gaussian Forma mixture: for each horizon h=1..20 plus a pooled
  row, the empirical coverage of the nominal 50/80/90/95% central intervals,
  mean PIT, mixture z̄², and mixture CRPS. The pooled row is the
  71.1/88.7/93.2/95.4% coverage quoted in the paper; the per-horizon rows
  back the "never under-covers at any horizon" claim (the tightest margin is
  +0.2 pp, at 95%) and the mixture z̄² drift from 1.14 at h=1 to 0.99 at
  h=20 (pooled 1.09: mildly overconfident at short horizons).
- `pit_hist_by_horizon.csv` — 20-bin PIT histograms per horizon.
- `<model>_calibration/` — the same two files for the Laplace Forma mixture
  (`forma_lap05_fgrid`), the two FFNN mixtures, and Chronos-2 (scored from
  its 21 native quantiles; includes `saturation_split.csv`, the degenerate
  zero-width-interval split). Chronos' Panel C CRPS is the quantile CRPS; its
  Gaussian-surrogate CRPS in `crps_scores__global.csv` (8.75) is dominated by
  one account, `txbcofq`, whose σ reaches 1.8e7 (0.38 without it), and is not
  a paper number. The Laplace pooled row is the one checked
  against `docs/ACCEPTANCE.md` §3.
- `*_scores__by_horizon.csv` / `*__global.csv` — per-model NLL / CRPS / MAE /
  R² / z̄² and observation-count aggregates from the same pass.
