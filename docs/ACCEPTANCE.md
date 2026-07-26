# Canonical acceptance record

Verification that the artifacts in this repository reproduce the paper's
published numbers. Run 2026-07-24/25 against the canonical R13 build
(`r13_node_optionD_indfe_val8`) and the canonical forecast store.

Scored companion: **`proforma-20q` @ `4f5a1df`**. Pin this commit when
reproducing — scores are only comparable against a fixed evaluator version.

**Summary: PASS.** Both scored panels reproduce the published values, Panel A to
full float precision. Every result below is on the published common sample of
**327,244,429** cells.

---

## 1. Checkpoint integrity

The ten shipped checkpoints were exported from the originals by stripping the
Lightning `callbacks` blob (which carried a relative `results/<expdir>/…` path).
Re-serialization is a corruption risk, so it was checked directly rather than
inferred: for each checkpoint, every `state_dict` tensor was compared against
its original.

| check | result |
|---|---|
| tensors compared | 62 / 62 per checkpoint, all 10 checkpoints |
| `torch.equal` on every tensor | **True — bit-identical to the originals** |
| parameters per checkpoint | 942,275 (942,210 trainable + 65 buffer elements) |
| reload under the shipped code | `FormaModel.load_from_checkpoint` — strict match, 0 missing / 0 unexpected keys |

The shipped weights **are** the weights that produced the paper's numbers; the
strip changed only the metadata blob.

## 2. Table 1 Panel A — Full column (Gaussian, 5-seed mixture)

Artifact: canonical `forma_fgrid` run-2 mixture (`r13_forma_fgrid_mix5_run2`,
commit `fe38fca`), scored against `tabular_test__pf_full__r13_node_optionD_indfe_val8`
with the Full-sample mask.

| metric | reproduced | published | match |
|---|---|---|---|
| **R²** | **0.28917237380438254** | 0.28917237380438254 | exact (17 s.f.) |
| **MAE** | **0.4084936810118555** | 0.4084936810118555 | exact |
| z² | 0.9610393201296477 | 0.961 | ✓ |
| CRPS (moment-matched) | 0.3194120370653803 | — | — |
| NLL (moment-matched) | 0.7199038724557704 | 0.720 | ✓ |
| common sample | 327,244,429 | 327,244,429 | ✓ |

Paper Panel A headline **Forma 0.289** ✓.

### FFNN rows, scored in the same pool

The Full column is an **intersection across models**, and `forma_fgrid` is its
binding constraint — so the FFNN rows only reproduce when scored in a pool that
contains Forma. Pooling the two FFNN 5-seed mixtures (`r13_ffnn_{linear,large}_betanll_mix5`,
commit `194b67c`) with the Forma mixture:

| model | R² | published | MAE | published |
|---|---|---|---|---|
| `forma_fgrid` | 0.28917237380438254 | 0.289 | 0.4084936810118555 | 0.408 |
| **`ffnn_linear_b50`** | **0.2528736009805014** | 0.253 | **0.44228160832978064** | 0.4423 |
| **`ffnn_large_b50`** | **0.24705595722332863** | 0.247 | **0.4523003669712947** | 0.4523 |

All three report `n_complete_rows = 327,244,429`, and Forma's R² is **bit-identical**
to its single-model run above — an empirical confirmation of the coverage claim
that every other full-coverage model is a superset of Forma's footprint, so
adding models leaves the intersection (and every published number) unmoved.

## 3. Table 1 Panel C — density track (Laplace, exact 5-seed mixture)

Artifact: the five `forma_lap05_fgrid` per-seed forecasts (commit `25b3c16`)
with their `.nll.json` sidecars, each covering 327,244,429 rows. Family
dispatch resolved to `laplace` from the sidecars on all five seeds.

| metric | reproduced | paper | match |
|---|---|---|---|
| **exact mixture NLL** | **0.1599192500934292** | 0.160 | ✓ |
| **mixture CRPS** | **0.292664** | 0.293 | ✓ |
| Cov50 | 0.637479 | 0.637 | ✓ |
| Cov90 | 0.916450 | 0.916 | ✓ |
| Cov80 / Cov95 | 0.853008 / 0.947197 | — | — |
| mean PIT | 0.493918 | — | — |
| common sample | 327,244,429 | 327,244,429 | ✓ |

`z2_mixture = 6.760418` is emitted by the same pass but is **not** a
paper-displayed value for the Laplace row (Panel C reports NLL / CRPS / Cov50 /
Cov90), so there is nothing to compare it against.

Both Panel C numbers come from code shipped in this repository: the mixture NLL
from `forma.scoring.evaluate`'s seed-mixture track, and the CRPS/coverage from
`scripts/mixture_calibration.py`.

## 4. Reproduction commands

Panel C, end to end from the per-seed forecasts (pool = a directory with
`forecasts/*.parquet` + each `.nll.json` sidecar + a `config.yaml` carrying
`data.dataset_tag`):

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

⚠️ **See "Known limitation" below** — at full scale this command currently
requires the fix in `proforma-20q` issue #12. The Panel A numbers above were
produced with the streaming evaluator (`forma.scoring.evaluate`) instead; the
two implementations agree (see §6).

## 5. Known limitation — scoring a full-scale submission

`proforma20q evaluate` reads the whole submission into memory and its
`validate_forecast` materializes a fixed-width unicode array over every row. On
a complete `pf_full` forecast (472,695,966 rows) that needs tens of GB and fails
on ordinary hardware — reported upstream as
[proforma-20q#12](https://github.com/ANONYMIZED/proforma-20q/issues/12).
Until that lands, score full-sample forecasts with `forma.scoring.evaluate`,
which streams row-group-wise through on-disk caches.

Unrelated but worth knowing: a forecast parquet written **without pandas
categorical metadata** (e.g. by a bare `pyarrow` writer) materializes its string
columns as `object` and roughly triples the memory needed to read it. Write
forecasts with categorical string columns.

## 6. Cross-implementation check

`proforma-20q` re-implements the scoring math independently of the research
pipeline (its `truth_grid.py`: *"Ported from the Forma research repo's
`src/eval_cache.py`, minus the on-disk memmap machinery … the math … is
identical"*). Scoring the **same** artifact with both implementations agrees:

| implementation | R² | MAE | common sample |
|---|---|---|---|
| `proforma20q evaluate` | 0.287696 | 0.409168 | 327,244,429 |
| research pipeline (published, same run) | 0.2877 | 0.4092 | 327,244,429 |

(Both on the superseded run-1 mixture, which is what was in the store at the
time of that run — the agreement is the point, not the values.)

## 7. Scope — what is *not* covered here

- **Chained-GBM Table-1 rows.** Not re-scored. Their weights are not shipped and
  the GBM column additionally needs the `pf_full_glm` build, which
  `proforma20q build` does not produce. (The FFNN rows *were* verified — §2.)
  The original forecast parquets that produced these rows land in the
  on-publication archival release.
- **Regeneration of forecasts from the shipped checkpoints.** Inference was
  measured at ~27.5 s/batch × 11,896 batches ≈ 91 h per seed on the CPU-only
  machine used here (no CUDA), i.e. ~19 days for the 5-seed family — it needs
  the GPU environment the fleet was trained on. Checkpoint integrity was
  therefore established by the bit-identical comparison in §1, which is a
  stronger guarantee than a sampled numeric diff.
- **Chronos-2 and the LLM column.** The LLM sample pin *was* verified: the
  harness regenerates the paper's exact 2,103-origin sample with fingerprint
  `5715abbe3f9f5c3f` (noise subset `cec4a58adfcda085`).

## 8. Environment

Windows 11, Python 3.12.4, pandas 2.2.3, numpy 2.4.2, fastparquet 2025.12.0,
pyarrow 23.0.1, torch 2.10.0+cpu, 31.5 GB RAM, no CUDA.
