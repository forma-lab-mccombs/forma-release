# Forma — model and competitor code for ProForma-20Q

Code and configurations for the tuple-set Transformer pro-forma financial
forecaster, plus code and configurations for the competitor models, benchmarked
with the companion package
[`proforma-20q`](https://github.com/forma-lab-mccombs/proforma-20q).

> **This repository contains source code and configuration only.** The trained
> Forma weights are **not** distributed here. They are published separately on
> the Hugging Face Hub under their own licence terms — see
> [Getting the weights](#getting-the-weights). This repository's
> [licence](#license) covers the code and configs, not the weights.

This repo is **inference-first**: a reviewer goes checkpoint → forecast parquet →
`proforma20q evaluate` → the paper's Table 1 numbers. It ships **no data**.
Real-data reproduction requires WRDS/Compustat credentials and the companion
package's dataset build.

> **This is the camera-ready (R14) release** of the ICAIF '26 paper. The
> camera-ready revision re-ran every model on a refreshed build,
> `r14_node_optionD_indfe_val8`, so every number, checkpoint and artifact
> changed from the pre-camera-ready (R13) release. What changed in the build:
> the asinh scale k now targets total kurtosis 3 (R13 used excess 3), the CRSP
> link-window trim is gone (WRDS Compustat only), and the no-CV train+val union
> is deduplicated on the tuple key. Splits, seeds, architectures and
> parameter counts are unchanged. The R13 state of this repository is tagged
> `r13`; the R13 weights and artifacts stay on the Hugging Face Hub at the `r13`
> revision of each repo. R13 and R14 scores live in different normalized spaces
> and on different samples, so never compare them as magnitudes.

**Verified:** the published checkpoints are the ones that produced the paper's
numbers (md5-identical to the training fleet's), and the canonical artifacts
reproduce Panel A — Forma **R² 0.315** (bit-identical across both evaluators) and the FFNN rows
**0.281 / 0.239** — plus Panel C (**NLL 0.211 / CRPS 0.295**), all on the
published 379,859,933-cell sample. Full record:
[`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).

## Division of labour with the benchmark package

`proforma-20q` is **authoritative** for the dataset build, the submission schema,
point metrics (Table 1 Panels A/B), and the reference baselines. This repo adds
only what the benchmark does not yet cover:

- the Forma model code and the competitor code/configs (weights are on the
  Hugging Face Hub, see [Getting the weights](#getting-the-weights));
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
python -m pytest tests/                            # expect: 302 passed, 4 skipped (Windows, no cuML)
```

(`[llm]` matters: the harness tests import `httpx` from that extra. On a plain
`[dev]` install they are skipped rather than run, so the passed count comes up
short. The counts are from Windows without cuML: 3 skips are cuML-only tests, and
1 is a symlink test that needs a Windows privilege and runs on Linux.)

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

## Getting the weights

The trained Forma checkpoints are **not in this repository**. They are published
on the Hugging Face Hub at
**[`forma-lab-mccombs/forma`](https://huggingface.co/forma-lab-mccombs/forma)**
under the licence terms presented on the model page, which govern those files;
this repository's licence does **not** extend to them. Commercial licensing
enquiries go to UT Discovery to Impact
(`discovery-to-impact-licensing@utexas.edu`).

The repo is gated, so you need a Hugging Face account and must accept the terms
on the model page first:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf download forma-lab-mccombs/forma --include "checkpoints/*" --local-dir .
```

`hf download` keeps the repo's own paths under `--local-dir`, so `--local-dir .`
lands the files in `./checkpoints/`, which is where `predict_forma.py` looks by
default. (Passing `--local-dir checkpoints` would nest them one level deeper;
`predict_forma.py` finds them there too.) Point `--checkpoints-dir` elsewhere if
you prefer. A local `checkpoints/` is git-ignored, so a download can't be
committed back into this repository.

The Hub repo also carries two files the weights are inert without:

| file | why |
|---|---|
| `reference/regularization_stats.parquet` | per-(account, quarter) μ / σ / k — **required** to encode inputs and decode outputs |
| `reference/account_id_map.csv` | the account embedding is indexed by this ordering |

The `account_id_map` / `industry_id_map` under `src/forma/metadata/` in this
repository are the same checkpoint-coupled maps and are what the inference path
actually reads; `predict_forma.py` validates them against the checkpoint's
embedding dimensions before running.

## Data prerequisite

All real-data commands consume a canonical ProForma-20Q build. Build it once with
the companion package (WRDS credentials required), pinning the frozen canonical
(R14) regularization stats so the eval space matches the paper:

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
> ([proforma-20q#13](https://github.com/forma-lab-mccombs/proforma-20q/issues/13)).
> Until that fix lands, rebuild it from the raw panel; `predict_forma.py` derives
> it with the build's own rule and verifies it against the tuple view before use:
>
> ```bash
> python scripts/predict_forma.py --family forma_fgrid --data-dir <build> \
>     --derive-firm-map-from-raw <build>/../raw/compustat_fundq.parquet
> ```
>
> Pass `--firm-map <path>` instead if you already have one.

## Reproduction map

One row per paper exhibit → command. `<build>` is your `proforma20q build` output
dir (e.g. `data/processed`); `<mask>` is the Full-sample mask
(`full_sample_mask_bits.npy`, [shipped in the benchmark repo](https://github.com/forma-lab-mccombs/proforma-20q/blob/main/artifacts/full_sample_mask_bits.npy);
a byte-identical copy is in the [artifacts dataset](#released-forecast-parquets)).
Point metrics are scored by `proforma20q evaluate`; the density track (Panel C)
by `mixture_calibration.py`.

| Paper exhibit | Command |
|---|---|
| **Table 1 Panel A — Forma (Gaussian)** | `python scripts/predict_forma.py --family forma_fgrid --data-dir <build>` → `python scripts/materialize_seed_mixture.py --src <per-seed dir> --dest <pool>/forecasts --arm 'forma_fgrid__pf_full__*__test__predictions.parquet' --expect-seeds 5 --expect-family gaussian` → `proforma20q evaluate forma_fgrid__pf_full__test__predictions.parquet --against baselines --sample-mask <mask>` (Forma **0.315**) |
| **Table 1 Panel B — Forma (Laplace)** | `python scripts/predict_forma.py --family forma_lap05_fgrid --data-dir <build>` → materialize (as above, `--expect-family laplace`; the `.nll.json` sidecar travels with the mixture) → `proforma20q evaluate … --sample-mask <mask>` (Forma MAE **0.374**) |
| **Table 1 Panel C — mixture density** | `python scripts/mixture_calibration.py --exp_dir <lik_pool> --family forma_lap05_fgrid --seeds 60,61,62,63,64 --out results/panels` (Laplace **NLL 0.211 / CRPS 0.295**); Gaussian NLL 0.675 via `--family forma_fgrid`. Exact mixture NLL is emitted by `proforma20q`/`forma.scoring.evaluate` over the per-seed pool. |
| **Table 1 — FFNN (linear / large)** | `python scripts/predict_ffnn.py --variant linear` (and `large`) → materialize → `proforma20q evaluate … --sample-mask <mask>` (Full R² **0.281 / 0.239**). Weights are not shipped; forecasts regenerate from the configs (seeds 60–64; CV-selected epochs: linear 8, large 2). |
| **Table 1 — Random Forest** | `python scripts/regen_rf.py` → `proforma20q evaluate …` (GPU/cuML; see script header). The camera-ready forests ran **unseeded**: cuML 25.12 dropped `random_state` through the wrapper, so a rerun reproduces the protocol, not the published forests bit for bit (the wrapper now passes the seed and warns if it is dropped). |
| **Table 1 — Chained GBM (L1 / L2)** | `python scripts/regen_gbm.py` → `proforma20q evaluate …` (requires the `pf_full_glm` build; see script header) |
| **Table 1 — Chronos-2** | `python -m forma.train --config configs/r14_chronos2_raw12q.yaml --forecast-splits test`; Panel C CRPS via `python scripts/chronos_quantile_calibration.py …` |
| **Table 1 — LLM column** | pin the sample (no spend) on the **R13** build it was drawn from: `python scripts/llm/llm_benchmark.py --build-origins-only --source tuples --present-in-q0 --sample-mode var-block --n-firms 133 --min-block 4 --max-block 20 --seed 20260615 --noise-subset-size 200 --data-dir <data-root> --feature-set pf_full --dataset r13_node_optionD_indfe_val8` (verify fingerprint `5715abbe3f9f5c3f`), then run each arm/model: `… --origins-file <origins.csv> --expect-origins-sha 5715abbe3f9f5c3f --prompt-version {unstructured,structured} --model {claude-opus-4-8,claude-sonnet-5,gpt-5.5} --batch`. The camera-ready column re-scores the archived responses on R14 without new API calls: `python scripts/llm/llm_renormalize.py --csv <archived>/*_predictions.csv --processed-dir <r14 build> --dataset-tag r14_node_optionD_indfe_val8 --out-dir <pool>/forecasts` (see [the LLM column on R14](#the-llm-column-on-r14)). |
| **Table 1 — Naive / Fade / ElasticNet** | `proforma20q baselines --which naive,fade,elasticnet` (reference baselines shipped by the benchmark package) |
| **T1 / F1 Diebold-Mariano stars** | `python scripts/dm_tests.py --exp_dir <pool> --reference forma_fgrid__pf_full` (quarter-clustered, Newey-West h−1, HLN-corrected) |

> **Scoring a full-scale forecast.** `proforma20q evaluate` streams submissions
> by row-group and handles full scale on ordinary hardware — an earlier OOM
> ([proforma-20q#12](https://github.com/forma-lab-mccombs/proforma-20q/issues/12)) is
> fixed and closed, confirmed against the canonical Forma forecast
> (468,443,876 rows — the model's actual coverage, not the full grid):
> R² 0.314885 / MAE 0.407668 on the 379,859,933-cell Full sample.
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

### The LLM column on R14

The camera-ready LLM column was **not** re-queried. The LLMs forecast raw USD, and
the standardized `prediction` the evaluator scores is computed afterwards from the
build's regularization statistics. So `scripts/llm/llm_renormalize.py` re-applies
that map, with the R14 build's statistics, to the archived R13-run `pred_raw`
values. All 2,103 origins and every row are kept. Two consequences:

- **The pin is drawn on the R13 build.** Run `--build-origins-only` with
  `--dataset r13_node_optionD_indfe_val8` to recover fingerprint
  `5715abbe3f9f5c3f`. Drawn on R14 the same seed gives a different sample,
  because the eligible pool grew from 321,892 origins / 12,230 firms to
  370,934 / 13,299.
- **87 of the 2,103 origins (4.1%) saw less history than the R14 build holds.**
  Their R13 prompts left blank 11,906 history cells that R14 fills: the
  quarters the dropped CRSP trim restores. A fresh run on R14 prompts would
  differ for those origins. On the 2,016 unchanged-prompt origins every model's
  R² moves by at most 0.005.

## What ships vs. what is regenerated

- **Forma** — trained checkpoints are published on the Hugging Face Hub
  ([`forma-lab-mccombs/forma`](https://huggingface.co/forma-lab-mccombs/forma),
  separate licence terms), **not** in this repository: 5 seeds each for the
  Gaussian (`forma_fgrid`) and Laplace (`forma_lap05_fgrid`) families. See
  [Getting the weights](#getting-the-weights).
- **FFNN** — training code, configs, and seeds ship; **weights do not** (the
  paper's "seeded regeneration scripts otherwise" clause). The original forecast
  parquets are in the [artifacts dataset](#released-forecast-parquets).
- **Chained GBM** — training code, configs, and seeds ship; weights and forecast
  parquets do not. `scripts/regen_gbm.py` regenerates them, but only from the
  `pf_full_glm` build, which `proforma20q build` does not produce — so outside
  the research pipeline these rows are neither distributed nor regenerable.
- **Random Forest** — seeded regeneration script only (1,560 forests are too
  large to bank).
- **Chronos-2** — zero-shot driver; weights pulled from the Hugging Face Hub
  (`amazon/chronos-2`, Apache-2.0) at runtime.
- **LLM benchmark** — harness, two byte-exact prompt arms
  (`scripts/llm/prompts/system_prompt_{unstructured,structured}.txt`), and the
  regeneration-pinned sample recipe ship; forecasts are regenerable from those.

### Released forecast parquets

The canonical Forma forecast — the exact file behind the paper's headline
numbers — is **`forma_fgrid__pf_full__test__predictions.parquet`** (3.7 GiB, md5
`2994e73f2706eb8aa4a7cfdcd018ee3f`). It is the 5-seed Gaussian mixture,
moment-matched to one row per forecasted cell by
`scripts/materialize_seed_mixture.py`, and it is what **Table 1 Panel A**
scores — its row count and the R²/MAE it reproduces are stated once, in the
scoring note under [Reproduction map](#reproduction-map). Pool your own model
against this file.

These forecasts are too large for git, so they are published on the Hugging Face
Hub at
**[`forma-lab-mccombs/proforma-20q-artifacts`](https://huggingface.co/datasets/forma-lab-mccombs/proforma-20q-artifacts)**
(~17.0 GiB total), under the licence terms presented on the dataset page, which
govern those files. The dataset is gated — accept the terms on the dataset page
first.

Nothing here requires these files — every command in the
[reproduction map](#reproduction-map) regenerates its forecast from the
checkpoints and configs — but they let you score against the paper's exact bytes
without a rebuild:

```bash
hf download forma-lab-mccombs/proforma-20q-artifacts --repo-type dataset \
    --include "forecasts/forma_fgrid__pf_full__test__predictions.parquet" \
    --local-dir data/artifacts
```

`hf download` verifies each file against the Hub's recorded digest and resumes
partial transfers. Pull a parquet's density sidecar explicitly — it is a separate
file. The companion files, all under `forecasts/`:

| artifact | size | md5 | for |
|---|---|---|---|
| `forma_lap05_fgrid__pf_full__test__predictions.parquet` | 3.7 GiB | `81c137a3f8ecaa306aabab9009c88d15` | the **Forma** Laplace mixture — **Table 1 Panel B**, the absolute-error track |
| `forma_lap05_fgrid__pf_full__test__predictions.nll.json` | 33 B | `a3d8659a201a2081dd693a8f0de051c3` | that file's **family sidecar**. Keep it next to the parquet — without it the evaluator **silently scores the file as Gaussian**. |
| `ffnn_linear_b50__pf_full__test__predictions.parquet` | 4.8 GiB | `a6b331f4cc4d65549b79290dc25bfcd3` | **FFNN (linear)** comparator row |
| `ffnn_large_b50__pf_full__test__predictions.parquet` | 4.8 GiB | `a7beb7846e5a0f8ce52f23d3c54b342e` | **FFNN (large)** comparator row |

The same dataset repo carries the Full-sample mask under `mask/` and the
per-horizon calibration series under `calibration/`.

A companion dataset,
**[`forma-lab-mccombs/forma-usd-forecasts`](https://huggingface.co/datasets/forma-lab-mccombs/forma-usd-forecasts)**
(12.3 GB, gated on the same footing), publishes the forecasts in USD levels rather than the
model's asinh z-score space. Nothing in this repository consumes it; see that
repo for its schema and provenance.

Every file carries the same seven columns — `firm_id`, `target`, `quarter`,
`forecast_horizon`, `prediction`, `sigma`, `model` — which `proforma20q` accepts
directly, mapping `firm_id` → `firm`, `quarter` → `origin`, and
`forecast_horizon` → `horizon` on read.

All four mixtures were written by the same tool
(`scripts/materialize_seed_mixture.py`: float32 `prediction`/`sigma`, rows
grouped by firm-hash bucket and key-sorted within a bucket, not in any seed's
row order). Join on the keys, never by position. The two Forma files hold
468,443,876 rows each and the two FFNN files 617,034,600: the FFNN emits a
wider grid, and the Full sample is the intersection.

See the
[benchmark repo's artifact tables](https://github.com/forma-lab-mccombs/proforma-20q#readme)
for the full manifest, including the mask and its canonical row index.

## Checkpoint inventory

All ten Forma checkpoints hold **942,275** tensor entries each (942,210 trainable
parameters + 65 buffers), 11,380,108 bytes. These files are **not in this
repository** — they are the ones hosted at
[`forma-lab-mccombs/forma`](https://huggingface.co/forma-lab-mccombs/forma). They are
uploaded unmodified from the training fleet, so the md5s below are those of the
Hub-hosted files **and** of the fleet's own outputs; use them to verify a download:

| checkpoint (on the Hub, under `checkpoints/`) | trained by config | md5 |
|---|---|---|
| `forma_gaussian_seed60.ckpt` | `r14_forma_fgrid_seed60.yaml` | `3a94aee01b5fbe6f5fe41d183edb667a` |
| `forma_gaussian_seed61.ckpt` | `r14_forma_fgrid_seed61.yaml` | `3771bb7b4249b57ca53c0c4de24130a4` |
| `forma_gaussian_seed62.ckpt` | `r14_forma_fgrid_seed62.yaml` | `67a8f3e6d66ef09f852d5d95b3807caa` |
| `forma_gaussian_seed63.ckpt` | `r14_forma_fgrid_seed63.yaml` | `670c21a29af45b675273c5285f80293a` |
| `forma_gaussian_seed64.ckpt` | `r14_forma_fgrid_seed64.yaml` | `fd2ab845ee0217a13e9bb257e0b64dae` |
| `forma_laplace_seed60.ckpt` | `r14_forma_lap05_fgrid_seed60.yaml` | `41b02696cb146adc7a3c48c71d7c1f26` |
| `forma_laplace_seed61.ckpt` | `r14_forma_lap05_fgrid_seed61.yaml` | `397f706e49f64cc1e2e34fb164133c4e` |
| `forma_laplace_seed62.ckpt` | `r14_forma_lap05_fgrid_seed62.yaml` | `90b6bc62b82a1aecbda5265e8f775a27` |
| `forma_laplace_seed63.ckpt` | `r14_forma_lap05_fgrid_seed63.yaml` | `d221086d1ed5bc7e3e09694ad30f37fe` |
| `forma_laplace_seed64.ckpt` | `r14_forma_lap05_fgrid_seed64.yaml` | `ae985ed69539df64cb00d754de0ab1d6` |

> **Public names vs config names.** The Hub publishes the Gaussian family
> (`forma_fgrid` in configs, forecasts and `--family`) as `forma_gaussian_*` and
> the Laplace family (`forma_lap05_fgrid`) as `forma_laplace_*`. Earlier versions
> of this table listed config-style names that were never the Hub's paths.
> `predict_forma.py` reads the Hub names. Superseded R13 checkpoints stay at the
> `r13` revision of the Hub repo; they are not valid against the R14 statistics.

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
> `sha256 23e8e565f19ab6d01c5698233f09e072984e9d292c9d63cce50381c6d19b3b7b`
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
docs/                 ACCEPTANCE.md + release_documentation.pdf (Parts A/B/C)
results_panels/       per-horizon calibration + metric series for the canonical run
scripts/              predict_forma / predict_ffnn / regen_* / density + DM scorers
scripts/llm/          LLM benchmark harness, prompt arms, cost/wobble tools
tests/                scoring-math tests + a WRDS-free synthetic pipeline smoke test
```

## License

Copyright 2026 The University of Texas at Austin. The source code and
configuration files in this repository are made available under
**The University of Texas at Austin Research License, Version 1.1** — see
[`LICENSE`](LICENSE). Commercial Distribution, as that License defines it,
requires a separate licence: contact UT Discovery to Impact at
`discovery-to-impact-licensing@utexas.edu`. NO DATA IS DISTRIBUTED — see
`NOTICE`.

Earlier versions of this repository included an Apache-2.0 license, which was
applied in error; the University has advised that it did not authorize that
license.

Trained model weights and released forecasts are **not** distributed here. They
are published on gated Hugging Face Hub repositories, and the terms presented on
each Hub page govern those files:

| artifact | where |
|---|---|
| trained Forma checkpoints | [`forma-lab-mccombs/forma`](https://huggingface.co/forma-lab-mccombs/forma) |
| released forecast parquets, mask, calibration | [`forma-lab-mccombs/proforma-20q-artifacts`](https://huggingface.co/datasets/forma-lab-mccombs/proforma-20q-artifacts) |
| USD-level forecasts | [`forma-lab-mccombs/forma-usd-forecasts`](https://huggingface.co/datasets/forma-lab-mccombs/forma-usd-forecasts) |
