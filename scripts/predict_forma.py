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
supplies ``tuple_test``, ``regularization_stats__*``, and ``firm_id_map.csv``.
The account/industry id maps are taken from THIS repo (they are coupled to the
shipped checkpoints — see below), never from the build.
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
    fresh build derived from data ordering), every forecast is silently wrong —
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


def predict_one(config_path: Path, checkpoint: Path, data_dir: Path, out_dir: Path,
                validate: bool) -> Path:
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
    dcfg["firm_id_map_path"] = str(data_dir / "firm_id_map.csv")
    dcfg["identities_path"] = str(REPO_ROOT / "configs" / "identities.json")
    config["forecast_splits"] = ["test"]
    config.setdefault("results_dir", str(out_dir))

    # Pre-flight: the build dir must carry the inputs FormaDataModule.setup() needs.
    for needed in ("firm_id_map.csv",):
        if not (data_dir / needed).exists():
            sys.exit(f"ERROR: {data_dir/needed} missing (from `proforma20q build`).")

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
            produced.append(predict_one(cfg, ckpt, args.data_dir, args.out, args.validate))
        print(f"\nProduced {len(produced)} per-seed forecasts for {args.family}.")
        print("Combine into the 5-seed mixture with:")
        print(f"  python scripts/group_seed_forecasts.py --materialize "
              f"--forecasts {args.out} --arm {args.family}")
        return 0

    if not args.config or not args.checkpoint:
        ap.error("provide --family, or both --config and --checkpoint")
    predict_one(args.config, args.checkpoint, args.data_dir, args.out, args.validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
