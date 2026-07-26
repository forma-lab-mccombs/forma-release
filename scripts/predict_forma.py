#!/usr/bin/env python
"""Forma inference: checkpoint -> per-seed forecast parquet (ProForma-20Q schema).

Reuses the exact, paper-validated inference path (``forma.train.generate_predictions``)
so the forecasts reproduce the published Table 1 numbers bit-for-bit. On top of it
this script enforces the checkpoint<->id-map contract that silent-corruption depends
on, and takes its tuple/reg-stats inputs from a ProForma-20Q build directory.

Typical use (one seed):

    python scripts/predict_forma.py \
        --config configs/r13_forma_fgrid_seed60.yaml \
        --checkpoint checkpoints/forma_fgrid_seed60.ckpt \
        --data-dir /path/to/proforma20q/data/processed \
        --out results/forecasts

Whole family (5 seeds -> per-seed parquets; combine with group_seed_forecasts.py):

    python scripts/predict_forma.py --family forma_fgrid \
        --data-dir /path/to/proforma20q/data/processed --out results/forecasts

The ProForma-20Q build directory must be the canonical R13 build
(``proforma20q build`` against the frozen canonical regularization stats): it
supplies ``tuple_test`` and ``regularization_stats__*``.

It does NOT supply ``firm_id_map.csv`` -- the build computes that map and
discards it (upstream proforma-20q issue #13) -- so pass either ``--firm-map``
or ``--derive-firm-map-from-raw``; without one of them this script stops and
says so. The account/industry id maps are taken from THIS repo (they are
coupled to the shipped checkpoints, see below), never from the build.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _shipped_metadata_dir() -> Path:
    """Directory holding the checkpoint-coupled account/industry id maps.

    Prefers the installed package data (``forma/metadata``); falls back to the
    source tree so the script also runs from a plain checkout.
    """
    try:
        from importlib.resources import files
        cand = Path(str(files("forma"))) / "metadata"
        if (cand / "account_id_map.csv").exists():
            return cand
    except Exception:
        pass
    return REPO_ROOT / "src" / "forma" / "metadata"


def _read_id_map(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _assert_id_maps(checkpoint: Path, data_dir: Path) -> tuple[Path, Path]:
    """Enforce the checkpoint<->id-map contract (the most dangerous failure mode).

    The account/industry categorical->embedding-index maps are baked into the
    trained embeddings. If inference uses a *different* ordering (e.g. one a
    fresh build derived from data ordering), every forecast is silently wrong --
    no error, plausible-looking numbers. So we:

      (a) verify the shipped maps' sizes equal the checkpoint's embedding dims;
      (b) if the build directory carries its own maps, verify they are IDENTICAL
          to the shipped ones (row for row) and abort otherwise.

    Returns the (account_map, industry_map) paths to USE for inference: always
    the shipped ones.
    """
    meta = _shipped_metadata_dir()
    acc_path = meta / "account_id_map.csv"
    ind_path = meta / "industry_id_map.csv"
    for p in (acc_path, ind_path):
        if not p.exists():
            sys.exit(f"ERROR: shipped id map missing: {p}")

    acc = _read_id_map(acc_path)
    ind = _read_id_map(ind_path)

    hp = torch.load(checkpoint, map_location="cpu", weights_only=False).get("hyper_parameters", {})
    n_acc_ckpt = int(hp.get("num_account_types", -1))
    n_ind_ckpt = int(hp.get("num_industries", -1))

    if n_acc_ckpt != -1 and len(acc) != n_acc_ckpt:
        sys.exit(
            f"ERROR: account map has {len(acc)} entries but checkpoint expects "
            f"num_account_types={n_acc_ckpt}. The shipped map does not match this "
            f"checkpoint's embedding dimension."
        )
    if n_ind_ckpt != -1 and len(ind) != n_ind_ckpt:
        sys.exit(
            f"ERROR: industry map has {len(ind)} entries but checkpoint expects "
            f"num_industries={n_ind_ckpt}."
        )

    # (b) A build that derived a different ordering would corrupt every forecast.
    for name, shipped in (("account_id_map.csv", acc), ("industry_id_map.csv", ind)):
        build_copy = data_dir / name
        if build_copy.exists():
            built = _read_id_map(build_copy)
            if not built.reset_index(drop=True).equals(shipped.reset_index(drop=True)):
                sys.exit(
                    f"ERROR: {build_copy} differs from the shipped {name}. The build "
                    f"produced a different categorical->index ordering than the one the "
                    f"checkpoints were trained under; forecasts would be silently corrupted. "
                    f"Rebuild against the canonical maps, or remove the build copy so the "
                    f"shipped map is used."
                )

    print(f"  id-map contract OK: {len(acc)} accounts, {len(ind)} industries "
          f"(match checkpoint: acc={n_acc_ckpt}, ind={n_ind_ckpt}).")
    return acc_path, ind_path


def _derive_firm_map(raw_path: Path, data_dir: Path) -> Path:
    """Reconstruct ``firm_id_map.csv`` using the build's own rule, then verify it.

    ``proforma20q build`` computes this map and discards it (upstream issue #13),
    which leaves the tuple view's integer ``firm_id`` unmappable to the gvkey
    strings the submission schema requires. The rule is deterministic --
    ``{gvkey: i for i, gvkey in enumerate(sorted(unique gvkeys))}`` over the raw
    panel -- so it can be rebuilt exactly.

    Deriving identifiers is exactly the kind of step that fails silently, so the
    result is checked against the tuple view before it is used: every integer id
    present in ``tuple_test`` must fall inside the derived map. A mismatch means
    the raw panel is not the one this build came from, and we refuse rather than
    emit forecasts under wrong firm identifiers.
    """
    import pandas as pd

    if not raw_path.exists():
        sys.exit(f"ERROR: raw panel not found: {raw_path}")
    print(f"  deriving firm_id_map from {raw_path.name} (upstream issue #13) ...")
    raw = pd.read_parquet(raw_path, columns=["gvkey"], engine="fastparquet")
    gvkeys = sorted(raw["gvkey"].astype(str).unique())
    out = pd.DataFrame({"firm_id_int": range(len(gvkeys)), "firm_id": gvkeys})

    tuples = sorted(data_dir.glob("tuple_test__*.parquet"))
    if tuples:
        from fastparquet import ParquetFile
        pf = ParquetFile(str(tuples[0]))
        seen_max = -1
        for chunk in pf.iter_row_groups(columns=["firm_id"]):
            seen_max = max(seen_max, int(chunk["firm_id"].max()))
        if seen_max >= len(out):
            sys.exit(
                f"ERROR: derived firm map has {len(out):,} entries but "
                f"{tuples[0].name} references id {seen_max:,}. The raw panel does "
                f"not match this build, so the mapping would assign wrong gvkeys. "
                f"Point --derive-firm-map-from-raw at the panel this build used."
            )
        print(f"  verified: {len(out):,} firms cover tuple ids 0..{seen_max:,}")
    else:
        print("  WARNING: no tuple_test found to verify the derived map against.")

    dest = data_dir / "firm_id_map.csv"
    try:
        out.to_csv(dest, index=False)
    except OSError as e:
        sys.exit(f"ERROR: could not write {dest} ({e}). Supply --firm-map instead.")
    print(f"  wrote {dest}")
    return dest


def _resolve_firm_map(data_dir: Path, explicit: Path | None,
                      derive_from_raw: Path | None) -> Path:
    """Locate the int->gvkey map, with an actionable message when it is absent."""
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"ERROR: --firm-map not found: {explicit}")
        return explicit
    existing = data_dir / "firm_id_map.csv"
    if existing.exists():
        return existing
    if derive_from_raw is not None:
        return _derive_firm_map(derive_from_raw, data_dir)
    sys.exit(
        f"ERROR: {data_dir / 'firm_id_map.csv'} is missing.\n"
        f"`proforma20q build` computes this map but does not write it out "
        f"(upstream issue #13), and without it the tuple view's integer firm ids "
        f"cannot be turned into the gvkey strings a submission needs.\n"
        f"Either:\n"
        f"  * pass --firm-map <path> if you already have one, or\n"
        f"  * pass --derive-firm-map-from-raw <compustat panel>.parquet to "
        f"rebuild it (verified against the tuple view before use)."
    )


def predict_one(config_path: Path, checkpoint: Path, data_dir: Path, out_dir: Path,
                validate: bool, firm_map: Path | None = None,
                derive_firm_map_from_raw: Path | None = None) -> Path:
    from forma.train import create_data_module, generate_predictions, load_config
    from forma.models.forma import FormaModel

    if not checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found: {checkpoint}")
    if not data_dir.exists():
        sys.exit(f"ERROR: --data-dir not found: {data_dir}. Run `proforma20q build` first.")

    acc_path, ind_path = _assert_id_maps(checkpoint, data_dir)

    config = load_config(str(config_path))
    dcfg = config.setdefault("data", {})
    dcfg["processed_dir"] = str(data_dir)
    # Force the checkpoint-coupled maps; take firm map + identities from repo/build.
    dcfg["account_map_path"] = str(acc_path)
    dcfg["industry_id_map_path"] = str(ind_path)
    dcfg["firm_id_map_path"] = str(
        _resolve_firm_map(data_dir, firm_map, derive_firm_map_from_raw))
    dcfg["identities_path"] = str(REPO_ROOT / "configs" / "identities.json")
    config["forecast_splits"] = ["test"]
    config.setdefault("results_dir", str(out_dir))

    # Build the data module directly so account_map_path is the SHIPPED map
    # (create_data_module would instead derive it from the build dir).
    dm = create_data_module("forma", dcfg)
    dm.account_map_path = acc_path  # belt-and-suspenders: pin the shipped account map

    model = FormaModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    forecast_name = config.get("models", {}).get("forma", {}).get("forecast_name", "forma")
    model.model_name = str(forecast_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  inference: {forecast_name} seed={config.get('seed')} "
          f"fp32={config.get('inference', {}).get('fp32')} -> {out_dir}")
    generate_predictions(model, dm, config, exp_dir=out_dir)

    # Locate the written forecast (seed-aware 5-token name when configured).
    produced = sorted(out_dir.glob(f"{forecast_name}__*__test__predictions.parquet"))
    if not produced:
        sys.exit(f"ERROR: no forecast parquet produced under {out_dir}")
    fpath = produced[-1]
    print(f"  wrote {fpath.name}")
    side = fpath.with_name(f"{fpath.stem}.nll.json")
    if side.exists():
        print(f"  density sidecar: {side.name} ({side.read_text().strip()})")

    if validate:
        _validate(fpath)
    return fpath


def _validate(fpath: Path) -> None:
    """Run `proforma20q validate` on a produced forecast (acceptance gate 14.3a)."""
    import subprocess
    from forma._deps import require_proforma20q
    require_proforma20q("--validate")
    print(f"  validating {fpath.name} against the ProForma-20Q schema ...")
    r = subprocess.run([sys.executable, "-m", "proforma20q.cli", "validate", str(fpath)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(f"ERROR: `proforma20q validate` failed for {fpath.name}")


FAMILIES = {
    "forma_fgrid": "r13_forma_fgrid_seed{seed}.yaml",
    "forma_lap05_fgrid": "r13_forma_lap05_fgrid_seed{seed}.yaml",
}
SEEDS = [60, 61, 62, 63, 64]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="Forma seed config yaml (single-seed mode).")
    ap.add_argument("--checkpoint", type=Path, help="Checkpoint .ckpt (single-seed mode).")
    ap.add_argument("--family", choices=sorted(FAMILIES), help="Run all 5 seeds of a family.")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="ProForma-20Q build directory (canonical R13): tuple_test + reg-stats + firm map.")
    ap.add_argument("--out", type=Path, default=Path("results/forecasts"),
                    help="Output directory for per-seed forecast parquets.")
    ap.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    ap.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    ap.add_argument("--validate", action="store_true",
                    help="Run `proforma20q validate` on each produced forecast.")
    ap.add_argument("--firm-map", type=Path, default=None,
                    help="Explicit int->gvkey firm_id_map.csv. Default: the one in "
                         "--data-dir.")
    ap.add_argument("--derive-firm-map-from-raw", type=Path, default=None,
                    metavar="RAW.PARQUET",
                    help="If the build has no firm_id_map.csv (upstream issue #13), "
                         "rebuild it from the raw Compustat panel using the build's "
                         "own rule. Verified against the tuple view before use.")
    args = ap.parse_args()

    # Inference itself does not import the benchmark package, but the forecasts
    # this produces are only useful once scored with it -- so say so up front
    # rather than letting the reviewer discover it after a long run.
    from forma._deps import require_proforma20q, warn_if_missing
    if args.validate:
        require_proforma20q("--validate")
    else:
        warn_if_missing("scoring the forecasts this produces "
                        "(`proforma20q evaluate`)")

    if args.family:
        produced = []
        for seed in SEEDS:
            cfg = args.configs_dir / FAMILIES[args.family].format(seed=seed)
            ckpt = args.checkpoints_dir / f"{args.family}_seed{seed}.ckpt"
            print(f"[{args.family} seed {seed}]")
            produced.append(predict_one(cfg, ckpt, args.data_dir, args.out, args.validate,
                                        firm_map=args.firm_map,
                                        derive_firm_map_from_raw=args.derive_firm_map_from_raw))
        print(f"\nProduced {len(produced)} per-seed forecasts for {args.family}.")
        print("Combine into the 5-seed mixture with:")
        print(f"  python scripts/group_seed_forecasts.py --materialize "
              f"--forecasts {args.out} --arm {args.family}")
        return 0

    if not args.config or not args.checkpoint:
        ap.error("provide --family, or both --config and --checkpoint")
    predict_one(args.config, args.checkpoint, args.data_dir, args.out, args.validate,
                firm_map=args.firm_map,
                derive_firm_map_from_raw=args.derive_firm_map_from_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
