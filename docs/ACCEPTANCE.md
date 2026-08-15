# Canonical acceptance record

Verification that the artifacts in this repository reproduce the paper's
published numbers. Run 2026-07-24/25 against the canonical build
(`r13_node_optionD_indfe_val8`) and the canonical forecast store.

Scored companion: **`proforma-20q` @ `4f5a1df`**. Pin this commit when
reproducing — scores are only comparable against a fixed evaluator version.

**Summary: PASS.** Both scored panels reproduce the published values, Panel A to
full float precision. Every result below is on the published common sample of
**327,244,429** cells.

---

## 1. Checkpoint integrity

> **Note (post-release).** This section records the acceptance run performed when
> the checkpoints were distributed inside this repository. They no longer are:
> the weights are published on the Hugging Face Hub at
> [`forma-lab-mccombs/forma`](https://huggingface.co/forma-lab-mccombs/forma)
> under a non-commercial licence. The Hub copies **retain** the `callbacks` blob
> described below rather than having it stripped, so their file digests differ
> from the copies this section was run against; their tensors were re-verified as
> bit-identical (62/62 per checkpoint, all ten) and they load through
> `FormaModel.load_from_checkpoint` at 942,275 entries. The findings below stand
> as recorded.

The ten formerly shipped checkpoints were exported from the originals by stripping the
Lightning `callbacks` blob (which carried a relative `results/<expdir>/…` path).
Re-serialization is a corruption risk, so it was checked directly rather than
inferred: for each checkpoint, every `state_dict` tensor was compared against
its original.

| check | tree it was run against | result |
|---|---|---|
| tensors compared | working copy vs. originals | 62 / 62 per checkpoint, all 10 checkpoints |
| `torch.equal` on every tensor | working copy vs. originals | **True — bit-identical to the originals** |
| parameters per checkpoint | working copy | 942,275 (942,210 trainable + 65 buffer elements) |
| reload under the shipped code | **fresh `git clone` @ `bdc240c`** | `FormaModel.load_from_checkpoint` — 62 tensors, 942,275 params, 0 missing / 0 unexpected keys |

The shipped weights **are** the weights that produced the paper's numbers; the
strip changed only the metadata blob.

> **Correction (2026-07-26).** As first published, the reload row said "under the
> shipped code" but had only ever been run against the authors' working copy. It
> was false of the release: `.gitignore` was silently excluding `src/forma/data/`,
> so a clone could not import `FormaModel` at all, and every command in the
> reproduction map failed on `ModuleNotFoundError: No module named 'forma.data'`.
> A cold third-party replication attempt found it. The subpackage now ships
> (`bdc240c`), CI checks the import on every push from a fresh checkout, and the
> row above has been re-run against a clone rather than a working tree.
>
> The general lesson is recorded here deliberately: **every other claim in this
> document was verified against the authors' working copy too.** The numbers
> were checked carefully; the *artifact* was not. Where a row's tree is not
> stated, assume working copy — and treat the CI job, which runs from a clean
> checkout, as the authority on what the release can actually do.

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
`scripts/mixture_calibration.py`. The canonical-run outputs of both passes —
including the per-horizon coverage series the paper references — are committed
under [`results_panels/full_sample_likelihood/`](../results_panels/README.md),
so these tables can be checked without rerunning anything.

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

This works at full scale: `proforma20q evaluate` streams submissions by
row-group (an earlier whole-file-in-memory OOM,
[proforma-20q#12](https://github.com/forma-lab-mccombs/proforma-20q/issues/12), is
fixed and closed) and has been confirmed against the canonical Forma forecast —
472,695,966 rows, the model's actual coverage rather than the 550,620,720-row
full grid — reproducing R² 0.289172 / MAE 0.408494 on the 327,244,429-cell
Full sample. The Panel A numbers in this document were originally produced
with the streaming evaluator (`forma.scoring.evaluate`); the two
implementations agree (see §6).

## 5. Practical notes on scoring a full-scale submission

Either evaluator handles full scale on ordinary hardware: `proforma20q
evaluate` validates and scores by row-group, and `forma.scoring.evaluate`
streams through on-disk caches. `proforma20q evaluate` is the authoritative
one for Panel A.

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
  The original forecast parquets that produced these rows are not distributed
  either, and `scripts/regen_gbm.py` depends on that same absent build — so
  outside the research pipeline these rows are neither distributed nor
  regenerable.
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
