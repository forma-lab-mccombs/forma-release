# Forma — model, competitors, and weights for ProForma-20Q

Code, configurations, and trained **Forma** checkpoints for the GNN-Transformer
pro-forma financial forecaster, plus code and configurations for the competitor
models, benchmarked with the companion package
[`proforma-20q`](https://github.com/ANONYMIZED/proforma-20q).

This repo is **inference-first**: a reviewer goes checkpoint → forecast parquet →
`proforma20q evaluate` → the paper's Table 1 numbers. It ships **no data**.
Real-data reproduction requires WRDS/Compustat credentials and the companion
package's dataset build.

**Verified:** the shipped checkpoints are bit-identical to the originals, and the
canonical artifacts reproduce Panel A — Forma **R² 0.289** (exact to 17
significant figures) and the FFNN rows **0.253 / 0.247** — plus Panel C
(**NLL 0.160 / CRPS 0.293**), all on the published 327,244,429-cell sample.
Full record: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).

## Division of labour with the benchmark package

`proforma-20q` is **authoritative** for the dataset build, the submission schema,
point metrics (Table 1 Panels A/B), and the reference baselines. This repo adds
only what the benchmark does not yet cover:

- the Forma model, its trained weights, and the competitor code/configs;
- the **exact 5-seed mixture** density scores (Panel C) — `scripts/mixture_calibration.py`
  plus the streaming slice in `forma.scoring`;
- the **Diebold-Mariano** significance machinery — `scripts/dm_tests.py`.

The mixture/PIT/DM pieces are extensions the benchmark package explicitly plans
to absorb (`proforma20q/evaluate.py`: *"port them here once they land, tracking
the main repo rather than forking"*); they will be upstreamed after submission,
at which point this repo drops its scoring slice entirely.

Scores are only comparable against a fixed evaluator version. **The benchmark
snapshot released alongside this repository *is* the pinned state** — if you
obtained the two together (the paired review mirrors, or the paired public
release), you already hold matching versions and nothing needs pinning. Only
mix-and-match installs (this repo against a later benchmark checkout) can
drift.

## Install

Both packages are source installs (neither is on PyPI):

```bash
pip install -e /path/to/proforma-20q          # benchmark: build + evaluator + reference baselines
pip install -e /path/to/forma-release          # this repo: Forma inference (core)
# optional model paths:
pip install -e '/path/to/forma-release[competitors]' # RF / GBM (+ FFNN uses core torch)
pip install -e '/path/to/forma-release[chronos]'     # Chronos-2 driver
pip install -e '/path/to/forma-release[llm]'         # LLM benchmark harness
```

The core install is **Forma inference only** (torch / lightning / torch-geometric);
the competitor, Chronos, and LLM paths pull their own stacks via the extras above.

### Verify the install

```bash
pip install -e '/path/to/forma-release[llm,dev]'   # pytest + the LLM-harness deps the suite imports
python -m pytest tests/ -q                         # expect: 77 passed
```

(`[llm]` matters: the harness tests import `httpx` from that extra. On a plain
`[dev]` install they are skipped rather than run, so the passed count comes up
short of 77.)

The suite needs no credentials, no GPU, and no downloads. It covers the scoring
math (CRPS, mixture CRPS, mixture NLL — checked against quadrature, Monte-Carlo
and closed-form reductions), the LLM harness, the id-map guards, and a
**synthetic end-to-end pipeline test** that builds a small synthetic panel with
the companion package, runs Forma inference on it, and validates the output
against the submission schema. If that one passes, the plumbing works.

### Reproducing the recorded environment

The published numbers in [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) were produced
with the versions pinned in [`requirements-lock.txt`](requirements-lock.txt):

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

`pyproject.toml` deliberately keeps runtime ranges loose (with upper bounds only
at the next major) so the package installs on a current scientific-Python stack —
CI exercises it on considerably newer releases than the lockfile. Use the
lockfile when you want the exact environment behind the recorded digits; use the
plain install for everything else.

## Data prerequisite

All real-data commands consume a canonical ProForma-20Q build. Build it once with
the companion package (WRDS credentials required), pinning the frozen canonical
regularization stats so the eval space matches the paper:

```bash
proforma20q build --wrds-user <you> --reg-stats canonical   # -> data/processed/...
```

This produces the `tuple_test` and `regularization_stats` the Forma inference
path reads. The checkpoint-coupled `account_id_map` / `industry_id_map` ship
**in this repo** (`src/forma/metadata/`) and are used at inference regardless of
the build, so the trained embeddings are never permuted.

> **One gap to know about.** The build also needs to hand you a `firm_id_map.csv`
> — the int→gvkey table that turns the tuple view's integer firm ids into the
> identifiers a submission requires — but `proforma20q build` currently computes
> it and discards it
> ([proforma-20q#13](https://github.com/ANONYMIZED/proforma-20q/issues/13)).
> Until that fix lands, rebuild it from the raw panel; `predict_forma.py` derives
> it with the build's own rule and verifies it against the tuple view before use:
>
> ```bash
> python scripts/predict_forma.py --family forma_fgrid --data-dir <build> \
>     --derive-firm-map-from-raw <build>/../raw/compustat_with_permno.parquet
> ```
>
> Pass `--firm-map <path>` instead if you already have one.

## Reproduction map

One row per paper exhibit → command. `<build>` is your `proforma20q build` output
dir (e.g. `data/processed`); `<mask>` is the Full-sample mask (`full_sample_mask_bits.npy`,
from the benchmark repo / archival release). Point metrics are scored by
`proforma20q evaluate`; the density track (Panel C) by `mixture_calibration.py`.

| Paper exhibit | Command |
|---|---|
| **Table 1 Panel A — Forma (Gaussian)** | `python scripts/predict_forma.py --family forma_fgrid --data-dir <build>` → `python scripts/group_seed_forecasts.py --src <per-seed dir> --dest <pool>/forecasts --arm forma_fgrid 'forma_fgrid__pf_full__6*__test__predictions.parquet' --materialize` → `proforma20q evaluate forma_fgrid__pf_full__test__predictions.parquet --against baselines --sample-mask <mask>` (Forma **0.289**) |
| **Table 1 Panel B — Forma (Laplace)** | `python scripts/predict_forma.py --family forma_lap05_fgrid --data-dir <build>` → materialize (as above) → `proforma20q evaluate … --sample-mask <mask>` |
| **Table 1 Panel C — mixture density** | `python scripts/mixture_calibration.py --exp_dir <lik_pool> --family forma_lap05_fgrid --seeds 60,61,62,63,64 --out results/panels` (Laplace **NLL 0.160 / CRPS 0.293**); Gaussian NLL 0.660 via `--family forma_fgrid`. Exact mixture NLL is emitted by `proforma20q`/`forma.scoring.evaluate` over the per-seed pool. |
| **Table 1 — FFNN (linear / large)** | `python scripts/predict_ffnn.py --variant linear` (and `large`) → materialize → `proforma20q evaluate … --sample-mask <mask>` (Full R² **0.253 / 0.247**). Weights are not shipped; forecasts regenerate from the configs (seeds 60–64). |
| **Table 1 — Random Forest** | `python scripts/regen_rf.py` → `proforma20q evaluate …` (GPU/cuML; see script header) |
| **Table 1 — Chained GBM (L1 / L2)** | `python scripts/regen_gbm.py` → `proforma20q evaluate …` (requires the `pf_full_glm` build; see script header) |
| **Table 1 — Chronos-2** | `python -m forma.train --config configs/chronos2_raw12q.yaml --forecast-splits test`; Panel C CRPS via `python scripts/chronos_quantile_calibration.py …` |
| **Table 1 — LLM column** | pin the sample (no spend): `python scripts/llm/llm_benchmark.py --build-origins-only --source tuples --present-in-q0 --sample-mode var-block --n-firms 133 --min-block 4 --max-block 20 --seed 20260615 --noise-subset-size 200 --data-dir <data-root> --feature-set pf_full --dataset r13_node_optionD_indfe_val8` (verify fingerprint `5715abbe3f9f5c3f`), then run each arm/model: `… --origins-file <origins.csv> --expect-origins-sha 5715abbe3f9f5c3f --prompt-version {unstructured,structured} --model {claude-opus-4-8,claude-sonnet-5,gpt-5.5} --batch` |
| **Table 1 — Naive / Fade / ElasticNet** | `proforma20q baselines --which naive,fade,elasticnet` (reference baselines shipped by the benchmark package) |
| **T1 / F1 Diebold-Mariano stars** | `python scripts/dm_tests.py --exp_dir <pool> --reference forma_fgrid__pf_full` (quarter-clustered, Newey-West h−1, HLN-corrected) |

> **Scoring a full-scale forecast.** `proforma20q evaluate` streams submissions
> by row-group and handles full scale on ordinary hardware — an earlier OOM
> ([proforma-20q#12](https://github.com/ANONYMIZED/proforma-20q/issues/12)) is
> fixed and closed, confirmed against the canonical Forma forecast
> (472,695,966 rows — the model's actual coverage, not the 550,620,720-row
> full grid): R² 0.289172 / MAE 0.408494 on the 327,244,429-cell Full sample.
> It is the authoritative evaluator for Panel A. The in-repo streaming
> implementation (`python -m forma.scoring.evaluate --exp_dir <pool> --splits
> test`) remains available, and the two agree (see
> [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) §6).

The two prompt arms ship **byte-exact**; verify them before trusting a rerun.
`.gitattributes` marks them `-text` so no checkout rewrites their bytes:

| arm | file | bytes | file md5 | SHA-256 of the loaded prompt (12/11/20 run config) |
|---|---|---|--:|---|
| unstructured | `scripts/llm/prompts/system_prompt_unstructured.txt` | 11,343 | `499ccefd14d2d5509397b502f925214f` | `6d2914b8fa9e0a56f0b197820dd93581a44e7dd1181d6b4cc0bdb897afdd03ec` |
| structured | `scripts/llm/prompts/system_prompt_structured.txt` | 13,450 | `1a0ac1edf6c99a73aa81410a2afb9f65` | `46f4d9d2b5bcfa27c6e94808577b1e08ece6287e86988e877cc8500fac06f69b` |

The two digests differ by design: the md5 covers the file on disk, while the
sha256 is taken over the *loaded* prompt — read as UTF-8 with universal
newlines, right-stripped, and with the `{lookback_qs}`/`{lookback_first}`/
`{max_horizon}` tokens substituted (`12`/`11`/`20` for the production run) —
so it covers the exact text sent to the models. The `system_prompt_sha` the
harness records in each run's metadata is the **first 16 hex characters** of
this digest (`6d2914b8fa9e0a56` / `46f4d9d2b5bcfa27`); match archived runs
against that prefix.

The LLM column uses **regeneration-pinned** origins: the exact 2,103-origin
sample is regenerated from seed `20260615` and verified against fingerprint
`5715abbe3f9f5c3f` (noise subset: `cec4a58adfcda085`). No firm identifiers are
distributed. Full LLM regeneration costs real API spend and is not required for
review — the harness, byte-exact prompt arms, and pinned-sample recipe are what
ship.

## What ships vs. what is regenerated

- **Forma** — trained checkpoints ship in-repo (`checkpoints/`): 5 seeds each for
  the Gaussian (`forma_fgrid`) and Laplace (`forma_lap05_fgrid`) families.
- **FFNN / Chained GBM** — training code, configs, and seeds ship; **weights do
  not** (the paper's "seeded regeneration scripts otherwise" clause). The
  original forecast parquets land in the on-publication archival release.
- **Random Forest** — seeded regeneration script only (1,560 forests are too
  large to bank).
- **Chronos-2** — zero-shot driver; weights pulled from the Hugging Face Hub
  (`amazon/chronos-2`, Apache-2.0) at runtime.
- **LLM benchmark** — harness, two byte-exact prompt arms
  (`scripts/llm/prompts/system_prompt_{unstructured,structured}.txt`), and the
  regeneration-pinned sample recipe ship; forecasts are regenerable from those.

## Checkpoint inventory

All ten Forma checkpoints: **942,275 parameters** each, ~11.38 MB. md5 (shipped,
post callback-strip):

| checkpoint | md5 |
|---|---|
| `forma_fgrid_seed60.ckpt` | `a3a82b078cbbc7078e1b948f30eddfbc` |
| `forma_fgrid_seed61.ckpt` | `b0c7969282a8dad303c3941d89810feb` |
| `forma_fgrid_seed62.ckpt` | `2dd994d0e519c02f38babe7822723410` |
| `forma_fgrid_seed63.ckpt` | `d10dea6baaa357b84b9b21c09fd4cd9e` |
| `forma_fgrid_seed64.ckpt` | `cb0800c14c822860ca7f8977133e16a6` |
| `forma_lap05_fgrid_seed60.ckpt` | `2d5bbd2e0d111cc5cd51b730d824a30f` |
| `forma_lap05_fgrid_seed61.ckpt` | `f51fcb3785f90062092eeb03d212bf95` |
| `forma_lap05_fgrid_seed62.ckpt` | `aff2481c93c762ccafec4cf6e256ecfa` |
| `forma_lap05_fgrid_seed63.ckpt` | `41651c5154c32b93432151539fd20130` |
| `forma_lap05_fgrid_seed64.ckpt` | `65b6442d24f0343563df9e5813d23a50` |

## Release documentation

[`docs/release_documentation.pdf`](docs/release_documentation.pdf) is the
detailed technical write-up behind the paper, in three parts:

| part | covers | relevant here |
|---|---|---|
| **A** | data pipeline — sample formation, splits, regularization, the 78-item universe, availability, accounting identities | context for the inputs |
| **B** | model training and implementation detail, plus every competitor's exact specification | **this repository** — Forma and the competitor scripts under `scripts/` |
| **C** | the LLM benchmark protocol, with both prompt arms reproduced in full | **this repository** — `scripts/llm/` |

> The **canonical copy lives in the benchmark repository** (`docs/` there); this
> is a byte-identical duplicate so that anyone holding only the model repository
> has Parts B and C, which describe *this* code. Both copies are
> `sha256 55fa6f76dffaca29b1027a8438d5c09e26dc182d89790f50ef9ec5803a46fa5e`
> — if that digest ever differs between the two repositories, the benchmark
> repository wins and this copy is stale. Regenerate both together.

Part C's prompt listings are ASCII transliterations; the byte-exact originals
ship separately with published digests (see *What ships vs. what is
regenerated*).

## Repository layout

```
src/forma/            Forma model, tuple dataset/collators, inference (forma.train),
                      and a torch-free density-scoring slice (forma.scoring)
src/forma/metadata/   checkpoint-coupled account/industry id maps (ship with the model)
configs/              Forma + competitor configs, accounting identities
checkpoints/          the 10 trained Forma checkpoints
docs/                 ACCEPTANCE.md + release_documentation.pdf (Parts A/B/C)
results_panels/       per-horizon calibration + metric series for the canonical run
scripts/              predict_forma / predict_ffnn / regen_* / density + DM scorers
scripts/llm/          LLM benchmark harness, prompt arms, cost/wobble tools
tests/                scoring-math tests + a WRDS-free synthetic pipeline smoke test
```

## License

Apache-2.0. NO DATA IS DISTRIBUTED — see `NOTICE`.
