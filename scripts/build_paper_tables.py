#!/usr/bin/env python
"""Build the ICAIF'26 paper's data-derived LaTeX tables from the in-repo CSV
intermediates under ``results/icaif26_panels/``.

This is **layer 2** of the two-layer table-reproducibility policy (see
``docs/paper/TABLE_REPRODUCIBILITY.md``):

    Box forecasts  --(layer 1: evaluate.py / dm_tests.py / ...)-->  in-repo CSVs
    in-repo CSVs   --(layer 2: THIS script)-->                      LaTeX snippets

Every number, significance marker, and bold face in the generated snippets is
read from a committed CSV -- nothing is hand-typed. The snippets are written to
``tex/icaif26/tables/`` and are the **repo-side staging area**: a human copies
the numbers into the Overleaf source of truth
(``C:\\Dropbox\\Apps\\Overleaf\\Forma\\icaif26``). This script NEVER writes to
Overleaf -- Overleaf carries manual formatting / change-tracking that must not
be clobbered (see the memory note ``overleaf_source_of_truth``).

Conventions (chosen to reproduce the current Overleaf tables, verifiable with
``--diff-overleaf``):
  * Bold = the winning cell in each numeric column (max R^2; min MAE/NLL/CRPS;
    nearest nominal for Panel C's coverage columns, where neither extreme is
    "better"). This is deterministic, so a tie cell (GBM MAE) bolds Forma
    automatically.
  * Row *labels* (with their \\textsc{Forma}, \\cite, footnote superscripts) and
    the surrounding float/caption/minipage scaffolding stay hand-authored in the
    paper -- this script emits only the ``tabular`` blocks.

Snippets produced:
    t1_panelA.tex   T1 Panel A  -- squared-error track, change-space R^2
    t1_panelB.tex   T1 Panel B  -- absolute-error track, MAE
    t1_panelC.tex   T1 Panel C  -- density track, mixture NLL + CRPS + coverage
    t2_coherence.tex  T2        -- reconciliation ladder (R^2 / MAE / identity viol.)

The prior-work table (``tab:claims`` in related.tex) is qualitative, not
data-derived, so it is intentionally NOT generated here. The data-appendix
table has its own generator (``tex/icaif26/appendix_src/build_data_appendix.py``).

Usage:
    python scripts/build_paper_tables.py             # write snippets
    python scripts/build_paper_tables.py --stdout     # print instead of writing
    python scripts/build_paper_tables.py --check      # non-writing; exit 1 if stale
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PANELS = os.path.join(REPO_ROOT, "results", "icaif26_panels")
OUT_DIR = os.path.join(REPO_ROOT, "tex", "icaif26", "tables")


# --------------------------------------------------------------------------- #
# CSV readers
# --------------------------------------------------------------------------- #
def read_scores(rel_path):
    """Read a 2-row score CSV (header row of model keys + one data row).

    Returns {column_key: float}. The leading index cell (the metric name, e.g.
    'r2') is dropped. Handles the UTF-8 BOM some CSVs carry.
    """
    with open(os.path.join(PANELS, rel_path), newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1]
    return {k: float(v) for k, v in zip(header[1:], data[1:])}


def read_dm(rel_path):
    """Read a dm_tests.py output CSV (path relative to results/icaif26_panels/);
    return {model: (marker_sq, marker_abs)} from the pooled-horizon rows."""
    out = {}
    with open(os.path.join(PANELS, rel_path), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["horizon"] == "pooled":
                out[r["model"]] = (r["marker_sq"].strip(), r["marker_abs"].strip())
    return out


def read_coverage_pooled(rel_path,
                         wanted=("crps_mixture", "cov_50", "cov_80", "cov_90", "cov_95")):
    """Pooled row of a calibration coverage CSV, as {field: float}.

    Carries the CRPS *and* the empirical coverage of the nominal central
    intervals -- one streamed pass, one common sample (327.2M cells), so Panel
    C's CRPS and coverage columns are guaranteed to be the same cells. The
    default `wanted` matches mixture_calibration.py's schema; the chronos CSV
    (chronos_quantile_calibration.py, #246) passes its own -- quantile-based
    CRPS columns and no cov_95 (0.025/0.975 are off its native 21-level grid).

    Reads a fixed whitelist rather than float()-ing the whole row: a future
    non-numeric column in the calibration CSV (a build tag, a model name) is
    then a no-op here instead of a crash, while a *missing* wanted column still
    fails loud via the KeyError -- the same "loud on schema drift that matters,
    silent on drift that doesn't" split the other readers use.
    """
    with open(os.path.join(PANELS, "full_sample_likelihood", rel_path),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["horizon"] == "pooled":
                return {k: float(r[k]) for k in wanted}
    raise ValueError(f"no pooled row in {rel_path}")


def read_scenario_class(rel_path):
    """Read scenario_pilot.py's delta_by_class.csv; return
    {class_code: (d_mae, d_r2)} (BS/IS/CF/Der)."""
    out = {}
    with open(os.path.join(PANELS, "scenario", rel_path),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["class"]] = (float(r["d_mae"]), float(r["d_r2"]))
    return out


def read_violation_median(rel_path):
    """Pooled-over-horizons rel_median identity-violation share (the ALL-identity
    aggregate) from a coherence CSV. Match the producer's intent
    (reconcile_pass.py filters identity=='ALL' & horizon=='pooled') by identity,
    not row order -- these CSVs carry a pooled row per identity."""
    with open(os.path.join(PANELS, "coherence", rel_path),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("identity") == "ALL" and r.get("horizon") == "pooled":
                return float(r["rel_median"])
    raise ValueError(f"no ALL/pooled row in {rel_path}")


# --------------------------------------------------------------------------- #
# Cell model + formatting
#
# A cell is one of:
#   ("num", value_or_None, marker)   numeric; None => structural dash; eligible
#                                    for column-winner bolding
#   ("lit", latex_string)            passthrough (e.g. "$\approx 0$", "3.7\%",
#                                    "---$^{\P}$"); never bolded automatically
# --------------------------------------------------------------------------- #
def num(value, marker=""):
    return ("num", value, marker)


def lit(s):
    return ("lit", s)


def fmt_num(x):
    """3 decimals when |x|<1 (all our R^2/MAE/NLL/CRPS), 2 decimals otherwise
    (matches the paper's -5.56 OLS cell)."""
    return f"{x:.3f}" if abs(x) < 1 else f"{x:.2f}"


def markers_t1(marker):
    """T1 DM marker (*/**/*** or a unicode dagger run) -> LaTeX superscript body."""
    return marker.replace("†", r"\dagger") if marker else ""


def markers_t2(marker):
    """T2 marker semantics: reference is 'raw'. Asterisks (raw better) pass
    through with their count; a dagger run (variant better) collapses to a single
    \\dagger, per the T2 caption's legend (harmonized with Table 2: variant-better
    is \\dagger, matching T1's comparator-better convention)."""
    if not marker:
        return ""
    return r"\dagger" if "†" in marker else marker


def render_cell(c, is_winner, render_marker):
    kind = c[0]
    if kind == "lit":
        return c[1]
    _, value, marker = c
    if value is None:
        return "---"
    m = render_marker(marker)
    if value < 0:
        body = f"-{fmt_num(abs(value))}"
        if is_winner:
            # bold the number, marker (if any) OUTSIDE the bold -- mirrors the
            # positive branch so a marker is never bolded.
            return rf"\textbf{{${body}$}}" + (f"$^{{{m}}}$" if m else "")
        # non-winner: marker stays inside the math ($-0.041^{***}$, per the paper)
        return f"${body}^{{{m}}}$" if m else f"${body}$"
    tok = rf"\textbf{{{fmt_num(value)}}}" if is_winner else fmt_num(value)
    return tok + f"$^{{{m}}}$" if m else tok


def render_table(colspec, preamble, header, groups, better, render_marker=markers_t1):
    """Assemble a full tabular block.

    colspec   : e.g. "lccc"
    preamble  : list of literal lines emitted before \\toprule (multicolumn title)
    header    : the column-header row (without trailing \\\\)
    groups    : list of groups; each group is a list of (label, [cell, ...]) with
                cells aligned to the numeric/literal columns (label excluded)
    better    : per-column direction over the numeric columns:
                'max' | 'min' | ('near', target) | None. ('near', t) bolds the
                cell closest to t in absolute deviation -- the calibration
                convention, where neither extreme is "better" (an interval that
                over-covers is as miscalibrated as one that under-covers).
    """
    n_cols = len(better)

    # Validate directions UP FRONT, not inside beats(): beats() short-circuits on
    # the first candidate (cur is None), so a typo'd direction in a column with a
    # single numeric row would never reach the dispatch and would pass silently.
    for ci, d in enumerate(better):
        if d is None or d in ("max", "min"):
            continue
        if isinstance(d, tuple) and len(d) == 2 and d[0] == "near":
            continue
        raise ValueError(
            f"column {ci}: unknown direction {d!r}. Expected 'max', 'min', "
            "('near', target), or None.")

    def beats(direction, v, cur):
        if cur is None:
            return True
        if direction == "max":
            return v > cur
        if direction == "min":
            return v < cur
        return abs(v - direction[1]) < abs(cur - direction[1])  # ('near', target)

    # find the winning row-cell per column across all groups
    winners = [None] * n_cols
    best = [None] * n_cols
    for gi, group in enumerate(groups):
        for ri, (_, cells) in enumerate(group):
            for ci in range(n_cols):
                c = cells[ci]
                if c[0] != "num" or c[1] is None or better[ci] is None:
                    continue
                v = c[1]
                if beats(better[ci], v, best[ci]):
                    best[ci] = v
                    winners[ci] = (gi, ri)

    lines = list(preamble)
    lines.append(r"\toprule")
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for gi, group in enumerate(groups):
        if gi > 0:
            lines.append(r"\midrule")
        for ri, (label, cells) in enumerate(group):
            rendered = [render_cell(cells[ci], winners[ci] == (gi, ri), render_marker)
                        for ci in range(n_cols)]
            lines.append(label + " & " + " & ".join(rendered) + r" \\")
    lines.append(r"\bottomrule")
    return "\\begin{tabular}{%s}\n%s\n\\end{tabular}" % (colspec, "\n".join(lines))


# --------------------------------------------------------------------------- #
# T1 Panel A -- squared-error track (R^2), conditional-mean forecasts
# --------------------------------------------------------------------------- #
def build_panel_a():
    full = read_scores("full_sample_point/r2_scores__global.csv")
    gbm = read_scores("gbm_sample/r2_scores__global.csv")
    llm = read_scores("llm_sample/r2_scores__global.csv")
    dm_full = read_dm("dm/dm_full_sample__vs_forma_fgrid.csv")
    dm_gbm = read_dm("dm/dm_gbm_sample__vs_forma_fgrid.csv")
    dm_llm = read_dm("dm/dm_llm_sample__vs_forma_fgrid.csv")

    def cells(fk, gk, lk):
        out = []
        for scores, dm, key in ((full, dm_full, fk), (gbm, dm_gbm, gk), (llm, dm_llm, lk)):
            if key is None:
                out.append(num(None))
            else:
                out.append(num(scores[key], dm.get(key, ("", ""))[0]))  # marker_sq
        return out

    full_cov = [
        (r"\textbf{\textsc{Forma} (Gaussian, 5-seed)}",
         cells("forma_fgrid__pf_full", "forma_fgrid__pf_full_glm", "forma_fgrid__pf_full")),
        ("Random Forest", cells("sklearn_random_forest__pf_full",
         "sklearn_random_forest__pf_full_glm", "sklearn_random_forest__pf_full")),
        ("Elastic Net", cells("sklearn_elasticnet__pf_full",
         "sklearn_elasticnet__pf_full_glm", "sklearn_elasticnet__pf_full")),
        ("FFNN (linear, 5-seed)", cells("ffnn_linear_b50__pf_full",
         "ffnn_linear_b50__pf_full_glm", "ffnn_linear_b50__pf_full")),
        ("FFNN (large, 5-seed)", cells("ffnn_large_b50__pf_full",
         "ffnn_large_b50__pf_full_glm", "ffnn_large_b50__pf_full")),
        ("Fade / AR(1)", cells("fade__pf_full", "fade__pf_full_glm", "fade__pf_full")),
        (r"Chronos-2 (mean)",
         cells("chronos_raw__pf_full", "chronos_raw__pf_full_glm", "chronos_raw__pf_full")),
        ("Naive (no change)", cells("naive__pf_full", "naive__pf_full_glm", "naive__pf_full")),
    ]
    restricted = [
        (r"Chained GBM (MSE)~\cite{geertsema2026chained}",
         cells(None, "chained_gbm_mse_v2__pf_full_glm", None)),
        ("Claude Opus 4.8",
         cells(None, None, "llm_unstructured_q0__claude_opus_4_8__pf_full")),
        ("GPT-5.5", cells(None, None, "llm_unstructured_q0__gpt_5_5__pf_full")),
        ("Claude Sonnet 5",
         cells(None, None, "llm_unstructured_q0__claude_sonnet_5__pf_full")),
    ]
    preamble = [r"\multicolumn{4}{l}{\emph{Panel A: squared-error track --- "
                r"change-space $R^2\!\uparrow$ (conditional-mean forecasts)}}\\"]
    header = r"Model & Full (327.2M) & GLM (109.1M) & LLM (2.15M)"
    return render_table("lccc", preamble, header, [full_cov, restricted], ["max", "max", "max"])


# --------------------------------------------------------------------------- #
# T1 Panel B -- absolute-error track (MAE), median / as-emitted forecasts
# --------------------------------------------------------------------------- #
def build_panel_b():
    full = read_scores("full_sample_point/mae_scores__global.csv")
    gbm = read_scores("gbm_sample/mae_scores__global.csv")
    llm = read_scores("llm_sample/mae_scores__global.csv")
    dm_full = read_dm("dm/dm_full_sample__vs_forma_lap05_fgrid.csv")
    dm_gbm = read_dm("dm/dm_gbm_sample__vs_forma_lap05_fgrid.csv")
    dm_llm = read_dm("dm/dm_llm_sample__vs_forma_lap05_fgrid.csv")

    def cells(fk, gk, lk):
        out = []
        for scores, dm, key in ((full, dm_full, fk), (gbm, dm_gbm, gk), (llm, dm_llm, lk)):
            if key is None:
                out.append(num(None))
            else:
                out.append(num(scores[key], dm.get(key, ("", ""))[1]))  # marker_abs
        return out

    full_cov = [
        (r"\textsc{Forma} (Laplace, 5-seed)",
         cells("forma_lap05_fgrid__pf_full", "forma_lap05_fgrid__pf_full_glm",
               "forma_lap05_fgrid__pf_full")),
        (r"Chronos-2 (median)",
         cells("chronos_raw_med__pf_full", "chronos_raw_med__pf_full_glm",
               "chronos_raw_med__pf_full")),
        ("Naive (no change)", cells("naive__pf_full", "naive__pf_full_glm", "naive__pf_full")),
    ]
    restricted = [
        (r"Chained GBM (L1)~\cite{geertsema2026chained}",
         cells(None, "chained_gbm_v2__pf_full_glm", None)),
        ("Claude Opus 4.8",
         cells(None, None, "llm_unstructured_q0__claude_opus_4_8__pf_full")),
        ("GPT-5.5", cells(None, None, "llm_unstructured_q0__gpt_5_5__pf_full")),
        ("Claude Sonnet 5",
         cells(None, None, "llm_unstructured_q0__claude_sonnet_5__pf_full")),
    ]
    preamble = [r"\multicolumn{4}{l}{\emph{Panel B: absolute error --- "
                r"MAE$\downarrow$ (median/as-emitted)}}\\"]
    header = r"Model & Full & GLM & LLM"
    return render_table("lccc", preamble, header, [full_cov, restricted], ["min", "min", "min"])


# --------------------------------------------------------------------------- #
# T1 Panel C -- density track (mixture NLL + exact mixture CRPS), full sample
# --------------------------------------------------------------------------- #
def build_panel_c():
    nll = read_scores("full_sample_likelihood/mixture_nll__global.csv")
    cov = {
        "forma_fgrid": read_coverage_pooled("coverage_by_horizon.csv"),
        "forma_lap05_fgrid": read_coverage_pooled(
            "forma_lap05_fgrid_calibration/coverage_by_horizon.csv"),
        "ffnn_large_b50": read_coverage_pooled(
            "ffnn_large_b50_calibration/coverage_by_horizon.csv"),
        "ffnn_linear_b50": read_coverage_pooled(
            "ffnn_linear_b50_calibration/coverage_by_horizon.csv"),
        # chronos_quantile_calibration.py output (#246): CRPS + coverage from
        # the model's own 21 native quantile levels, same common sample. The
        # crps_gaussian_surrogate column (the pre-#246 published methodology,
        # kept for the record) is read so schema drift fails loud, not used.
        "chronos_raw": read_coverage_pooled(
            "chronos_raw_calibration/coverage_by_horizon.csv",
            wanted=("crps_quantile", "crps_gaussian_surrogate",
                    "cov_50", "cov_80", "cov_90")),
    }

    def cells(key):
        """NLL, exact-mixture CRPS, then mixture-PIT coverage at nominal 50/90."""
        c = cov[key]
        return [num(nll[key + "__pf_full"]), num(c["crps_mixture"]),
                num(c["cov_50"]), num(c["cov_90"])]

    # Labels are shortened vs. Panels A/B ("Forma (Laplace)", not "(Laplace,
    # exact 5-seed mixture)"): with two coverage columns added, the long form
    # overfills the 0.44\textwidth minipage by 28.9pt. "exact 5-seed mixture"
    # moves into the panel title, which is where it scopes anyway -- and every
    # row here is a mixture, so ", mixture" on the FFNN rows said nothing the
    # title does not.
    group = [
        (r"\textbf{\textsc{Forma} (Laplace)}", cells("forma_lap05_fgrid")),
        (r"\textsc{Forma} (Gaussian)", cells("forma_fgrid")),
        ("FFNN (large)", cells("ffnn_large_b50")),
        ("FFNN (linear)", cells("ffnn_linear_b50")),
        # Chronos NLL stays a structural dash (#246 keeps it out of scope): 21
        # knots give only a crude piecewise-constant density, and the
        # degenerate zero-width cells still break the log score; CRPS and
        # coverage are floor-insensitive. Both now come from the model's OWN
        # quantiles (chronos_quantile_calibration.py) rather than the Gaussian
        # surrogate that crps_scores__global.csv carries.
        (r"Chronos-2",
         [num(None), num(cov["chronos_raw"]["crps_quantile"]),
          num(cov["chronos_raw"]["cov_50"]), num(cov["chronos_raw"]["cov_90"])]),
    ]
    # Title kept short (the long "(exact mixtures; Chronos-2 from its
    # quantiles)" form overflowed the minipage). The scoping detail -- exact
    # mixture CRPS for the Forma/FFNN rows, Chronos-2 scored from its own 21
    # native quantile levels (#246) -- lives in the T1 caption and the density
    # paragraph of the results prose instead.
    preamble = [r"\multicolumn{5}{l}{\emph{Panel C: density track (full sample)}}\\"]
    # The nominal level rides in the header, so the bolding rule is legible from
    # the table alone: NLL/CRPS advertise direction with an arrow, and coverage
    # advertises its target the same way rather than relying on the caption.
    header = (r"Model & NLL$\downarrow$ & CRPS$\downarrow$ & "
              r"Cov$_{50}$ (0.50) & Cov$_{90}$ (0.90)")
    # Coverage columns bold the cell nearest nominal, not the largest: the
    # panel's story is that every density over-covers, graded by how much.
    return render_table("lcccc", preamble, header, [group],
                        ["min", "min", ("near", 0.50), ("near", 0.90)])


# --------------------------------------------------------------------------- #
# T2 -- reconciliation ladder (coherence.tex)
# --------------------------------------------------------------------------- #
def build_t2():
    r2 = read_scores("coherence/t2_r2_scores__global.csv")
    mae = read_scores("coherence/t2_mae_scores__global.csv")
    gnn_r2 = read_scores("coherence/t2_gnn_r2_scores__global.csv")
    gnn_mae = read_scores("coherence/t2_gnn_mae_scores__global.csv")
    dm = read_dm("coherence/t2_dm_vs_raw.csv")
    dm_gnn = read_dm("coherence/t2_gnn_dm_vs_raw.csv")
    viol_raw = read_violation_median("identity_violation_raw.csv")
    viol_gnn = read_violation_median("identity_violation_gnn_r11corr.csv")

    def pct(x):
        return lit(f"{x * 100:.1f}\\%")

    raw, ols, wls = "forma_fgrid__pf_full", "forma_fgrid_ols__pf_full", "forma_fgrid_wls__pf_full"
    gnn = "forma_r11corr__pf_full"

    group = [
        ("Transformer-only (raw)",
         [num(r2[raw]), num(mae[raw]), pct(viol_raw)]),
        (r"\quad + OLS reconciliation",
         [num(r2[ols], dm[ols][0]), num(mae[ols], dm[ols][1]), lit(r"$\approx 0$")]),
        (r"\quad + WLS reconciliation",
         [num(r2[wls], dm[wls][0]), num(mae[wls], dm[wls][1]), lit(r"$\approx 0$")]),
        (r"Constraint layer (GNN)$^{a}$",
         [num(gnn_r2[gnn], dm_gnn[gnn][0]), num(gnn_mae[gnn], dm_gnn[gnn][1]), pct(viol_gnn)]),
    ]
    preamble = []
    header = r"Variant & $R^2\!\uparrow$ & MAE$\downarrow$ & Identity viol.$\downarrow$"
    return render_table("lccc", preamble, header, [group], ["max", "min", None],
                        render_marker=markers_t2)


# --------------------------------------------------------------------------- #
# T-scenario -- oracle revenue-conditioning gain by statement class (scenario.tex)
#
# Companion to Figure 2 (make_f2_figure.py): the within-model gain from pinning
# the TRUE realized revenue path and forecasting the rest of the statement,
# pooled over horizons and split by statement class. Delta MAE < 0 and Delta R^2
# > 0 both mean conditioning helps; the column winner is the biggest gain
# (most-negative Delta MAE / most-positive Delta R^2). Source CSV:
# results/icaif26_panels/scenario/delta_by_class.csv (scripts/scenario_pilot.py).
# --------------------------------------------------------------------------- #
def build_scenario():
    d = read_scenario_class("delta_by_class.csv")
    order = [("IS", "Income statement"), ("BS", "Balance sheet"),
             ("CF", "Cash flow"), ("Der", "Derived")]
    group = [(label, [num(d[k][0]), num(d[k][1])]) for k, label in order if k in d]
    preamble = [r"\multicolumn{3}{l}{\emph{Scenario conditioning: within-model gain "
                r"from pinning the realized revenue path (oracle)}}\\"]
    header = r"Statement class & $\Delta$MAE$\downarrow$ & $\Delta R^2\!\uparrow$"
    return render_table("lcc", preamble, header, [group], ["min", "max"])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
TABLES = {
    "t1_panelA.tex": build_panel_a,
    "t1_panelB.tex": build_panel_b,
    "t1_panelC.tex": build_panel_c,
    "t2_coherence.tex": build_t2,
    "t_scenario.tex": build_scenario,
}

HEADER = ("%% AUTO-GENERATED by scripts/build_paper_tables.py -- do not edit by hand.\n"
          "%% Source CSVs: results/icaif26_panels/. Migrate numbers into Overleaf\n"
          "%% manually; never let a generator overwrite the Overleaf source of truth.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true",
                    help="print snippets instead of writing files")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any on-disk snippet is stale. "
                         "NOTE: this verifies CSV->snippet only. It cannot catch "
                         "upstream CSV *schema* drift -- if a DM CSV renamed a key "
                         "or its 'pooled' label, a marker silently vanishes and "
                         "--check still passes (both sides drop it). A missing "
                         "score column, by contrast, fails loud via scores[key].")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    changed = []
    for name, fn in TABLES.items():
        body = HEADER + fn() + "\n"
        if args.stdout:
            print(f"%% ===== {name} =====")
            print(body)
            continue
        out_path = os.path.join(OUT_DIR, name)
        old = None
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                old = f.read()
        if old != body:
            changed.append(name)
        if not args.check:
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
            print(f"wrote {os.path.relpath(out_path, REPO_ROOT)}")

    if args.check:
        if changed:
            print("STALE (run without --check to update): " + ", ".join(changed))
            sys.exit(1)
        print("OK: all generated snippets match on-disk files.")


if __name__ == "__main__":
    main()
