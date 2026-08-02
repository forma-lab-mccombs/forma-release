# Committed aggregate result series

Aggregate-level (by-horizon / pooled) CSVs from the canonical paper run.
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
Full sample (327,244,429 cells).

- `coverage_by_horizon.csv` — the per-horizon coverage series for the
  five-seed Gaussian Forma mixture: for each horizon h=1..20 plus a pooled
  row, the empirical coverage of the nominal 50/80/90/95% central intervals,
  mean PIT, mixture z̄², and mixture CRPS. The pooled row is the
  72.6/89.6/93.7/95.8% coverage quoted in the paper; the per-horizon rows
  back the "never under-covers at any horizon" and "z̄² drifts 1.00 → 0.89"
  claims.
- `pit_hist_by_horizon.csv` — 20-bin PIT histograms per horizon.
- `<model>_calibration/` — the same two files for the Laplace Forma mixture
  (`forma_lap05_fgrid`), the two FFNN mixtures, and Chronos-2 (scored from
  its 21 native quantiles; includes `saturation_split.csv`, the degenerate
  zero-width-interval split). The Laplace pooled row is the one checked
  against `docs/ACCEPTANCE.md` §3.
- `*_scores__by_horizon.csv` / `*__global.csv` — per-model NLL / CRPS / MAE /
  R² / z̄² and observation-count aggregates from the same pass.
