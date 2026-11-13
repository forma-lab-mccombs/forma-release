# Canonical acceptance record

Verification that the artifacts this repository points at reproduce the paper's
published camera-ready (R14) numbers. Run 2026-11-13 against the canonical
build (`r14_node_optionD_indfe_val8`) and the canonical forecast store, with the
code of this repository at the commit that adds this record.

Scored companion: **`proforma-20q` R14 release** (branch `r14-camera-ready`; its merge commit is pinned here once that PR lands). Pin this commit when
reproducing — scores are only comparable against a fixed evaluator version.

**Summary: every number checked reproduces exactly.** The Hub checkpoints are byte-identical to the fleet's output. Panel A (Forma, both FFNNs) and Panel C (Laplace and Gaussian mixture NLL, CRPS, coverage) reproduce the research pipeline as exact doubles. The two evaluators agree bit for bit on Forma's R² and MAE. Every result below is on the published common sample
of **379,859,933** cells.

> **Superseded record.** The pre-camera-ready (R13) acceptance record — build
> `r13_node_optionD_indfe_val8`, 327,244,429 cells, Forma R² 0.289 — is at the
> `r13` tag of this repository. Its numbers are not comparable with the ones
> below: the R14 build re-fit the asinh scale k and grew the sample, so the
> two rounds score in different normalized spaces.

---

## 1. Checkpoint integrity

The ten checkpoints on the Hugging Face Hub
([`forma-lab-mccombs/forma`](https://huggingface.co/forma-lab-mccombs/forma),
`checkpoints/`) are the training fleet's own output files, uploaded
unmodified. Nothing was re-serialized, so integrity is a byte comparison rather
than the tensor-by-tensor comparison the R13 record needed.

Hub revisions at the time of this record: `forma` @ `8ee90ffb`,
`proforma-20q-artifacts` @ `f9e93e84`, `forma-usd-forecasts` @ `fdf22404`. The
R13 files are at the `r13` tag of each repo.

| File on the Hub | md5 (Hub download = fleet output) |
|---|---|
| `checkpoints/forma_gaussian_seed60.ckpt` | `3a94aee01b5fbe6f5fe41d183edb667a` |
| `checkpoints/forma_gaussian_seed61.ckpt` | `3771bb7b4249b57ca53c0c4de24130a4` |
| `checkpoints/forma_gaussian_seed62.ckpt` | `67a8f3e6d66ef09f852d5d95b3807caa` |
| `checkpoints/forma_gaussian_seed63.ckpt` | `670c21a29af45b675273c5285f80293a` |
| `checkpoints/forma_gaussian_seed64.ckpt` | `fd2ab845ee0217a13e9bb257e0b64dae` |
| `checkpoints/forma_laplace_seed60.ckpt` | `41b02696cb146adc7a3c48c71d7c1f26` |
| `checkpoints/forma_laplace_seed61.ckpt` | `397f706e49f64cc1e2e34fb164133c4e` |
| `checkpoints/forma_laplace_seed62.ckpt` | `90b6bc62b82a1aecbda5265e8f775a27` |
| `checkpoints/forma_laplace_seed63.ckpt` | `d221086d1ed5bc7e3e09694ad30f37fe` |
| `checkpoints/forma_laplace_seed64.ckpt` | `ae985ed69539df64cb00d754de0ab1d6` |
| `reference/regularization_stats.parquet` | `aa853a6a433ffe9d0de74180775507ea` |
| `reference/account_id_map.csv` | `bd0e2ba84544e65a95370d0c44df4c0e` |

Each Hub file was downloaded back after the upload, and its md5 matches the
fleet's own `models/*.ckpt` for that seed. The two `reference/` files match the
build's `regularization_stats__pf_full__r14_node_optionD_indfe_val8.parquet`
and `account_id_map.csv`. All ten checkpoints load through
`forma.models.forma.FormaModel.load_from_checkpoint` with this repository's
code: strict state-dict match, 942,210 parameters, epoch 12. They were written
by Lightning 2.6.6 and load under 2.5.2.

## 2. Table 1 Panel A — Full column (Gaussian, 5-seed mixture)

Scored with this repository's `forma.scoring.evaluate` over the three released
point-track forecasts (`forma_fgrid` mixture, `ffnn_linear_b50`,
`ffnn_large_b50`), common-sample mode. The intersection is the published
Full sample: **379,859,933** cells over 78 targets (the FFNNs alone cover
388,885,185).

| Model | R² (released code) | MAE (released code) | Paper R² |
|---|---|---|---|
| Forma (Gaussian, 5-seed) | 0.3148849598260647 | 0.4076683056514988 | 0.315 |
| FFNN-linear | 0.2807965194562326 | 0.4443536530908115 | 0.281 |
| FFNN-large | 0.2387551679167482 | 0.4804706454238374 | 0.239 |

All three match the research pipeline's `results/icaif26_panels` CSVs as
exact doubles. The Forma row's density scores from the same pass: NLL
0.7473453745967469, CRPS 0.316342853258982, z̄² 1.0917551182434817. The
NLL here is the moment-matched single Gaussian. The exact mixture NLL
(0.6752897045825152) is the one in §3.

## 3. Table 1 Panel C — density track (Laplace, exact 5-seed mixture)

Ten per-seed forecasts (`forma_fgrid` and `forma_lap05_fgrid`, seeds 60–64)
with their `.nll.json` sidecars, scored from cold caches with the commands in
§4.

| Quantity | Released code | Paper |
|---|---|---|
| Laplace mixture NLL | 0.2105174125098221 | 0.211 |
| Laplace mixture CRPS | 0.29522935224139835 | 0.295 |
| Coverage 50 / 80 / 90 / 95% | 0.6415 / 0.8539 / 0.9169 / 0.9476 | — |
| Mean PIT | 0.5169 | — |
| Gaussian mixture NLL (for reference) | 0.6752897045825152 | 0.675 |
| n | 379,859,933 | 379,859,933 |

`coverage_by_horizon.csv` and `pit_hist_by_horizon.csv` from this run are
**identical, line for line**, to the committed
`results_panels/full_sample_likelihood/forma_lap05_fgrid_calibration/` files (the repository stores them with LF line endings).
The Forma columns of `mixture_nll__global.csv` and `mixture_nll__by_horizon.csv`
are equal value for value to the committed ones. (The committed files also
carry the two FFNN mixture columns, which this pass did not include.)

## 4. Reproduction commands

Panel C, end to end from the per-seed forecasts (pool = a directory with
`forecasts/*.parquet` + each `.nll.json` sidecar + a `config.yaml` carrying
`data.dataset_tag` and `data.processed_dir`):

```bash
# exact 5-seed mixture NLL (also builds the per-forecast error caches)
python -m forma.scoring.evaluate --exp_dir <pool> --splits test --feature_set pf_full \
    --cache-dir <cache>
# -> <pool>/metrics/pf_full/test/mixture_nll__global.csv

# exact mixture CRPS + interval coverage + PIT
python scripts/mixture_calibration.py --exp_dir <pool> --family forma_lap05_fgrid \
    --feature_set pf_full --seeds 60,61,62,63,64 \
    --processed_dir <build> --cache-dir <cache> --out <out>
# -> <out>/coverage_by_horizon.csv  (pooled row)
```

Panel A, from the mixture forecast:

```bash
proforma20q evaluate <mixture>.parquet --truth <tabular_test>.parquet \
    --sample-mask <full_sample_mask_bits>.npy
```

## 5. Practical notes on scoring a full-scale submission

Either evaluator handles full scale on ordinary hardware: `proforma20q
evaluate` validates and scores by row-group, and `forma.scoring.evaluate`
streams through on-disk caches. `proforma20q evaluate` is the authoritative
one for Panel A.

A forecast parquet written **without pandas categorical metadata** (e.g. by a
bare `pyarrow` writer, which is how `materialize_seed_mixture.py` writes the
mixtures) materializes its string columns as `object` and roughly triples the
memory needed to read it with pandas. The evaluators stream it by row group,
so this matters only for your own whole-file reads.

## 6. Cross-implementation check

The Forma mixture parquet scored by both evaluators on the published mask:

| Evaluator | R² | MAE | n |
|---|---|---|---|
| `forma.scoring.evaluate` (this repository) | 0.3148849598260647 | 0.4076683056514988 | 379,859,933 |
| `proforma20q evaluate` (scored companion) | 0.31488495982606468 | 0.40766830565149881 | 379,859,933 |

The two are the same IEEE doubles; they differ only in how many digits each
evaluator prints. The `naive` and `fade` reference
rows from `proforma20q evaluate --against baselines` agree with the research
pipeline's values to the last bit, except fade, which differs by one ulp. The column
statistics of the released mixture match `checksums/canonical_column_stats.json`
on all 2,497 checked columns (0 drift).

## 7. Scope — what is *not* covered here

- **Chained-GBM Table-1 rows.** Not re-scored. Their weights are not shipped and
  the GBM column additionally needs the `pf_full_glm` build, which
  `proforma20q build` does not produce. The forecast parquets that produced
  these rows are not distributed either, and `scripts/regen_gbm.py` depends on
  that same absent build — so outside the research pipeline these rows are
  neither distributed nor regenerable.
- **Random Forest.** Not re-scored. The camera-ready forests ran **unseeded**
  (cuML 25.12 dropped `random_state` through the wrapper), so `regen_rf.py`
  reproduces the protocol, not the published forests bit for bit.
- **Regeneration of forecasts from the shipped checkpoints.** Not re-run here:
  inference needs the GPU environment the fleet ran on (R13 measured ~91 h per
  seed on a CPU-only machine). Checkpoint integrity is established by the byte
  comparison in §1 instead.
- **ElasticNet.** Not re-run on the R14 build (it exhausts memory with
  `n_jobs=-1` at full scale). The paper's 0.282 is quoted from the research run.
- **Chronos-2 and the LLM column.** Not re-scored. The camera-ready LLM column
  is the archived R13-prompt responses re-normalized onto the R14 build
  (`scripts/llm/llm_renormalize.py`); the responses themselves are not
  distributed. The 2,103-origin pin is drawn on the R13 build (fingerprint
  `5715abbe3f9f5c3f`).

## 8. Environment

Windows 11 Enterprise (10.0.22631), Intel Core i9-14900 (32 logical CPUs),
64 GB RAM, no GPU used. Python 3.12.4, numpy 2.3.2, pandas 2.2.3, pyarrow
24.0.0, scipy 1.15.1, torch 2.7.1+cu126. Panel A took about 40 minutes and
Panel C about 1 h 40 min (cold caches) on this machine.
