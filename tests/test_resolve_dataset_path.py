# Copyright 2026 The University of Texas at Austin.
#
# Licensed under The University of Texas at Austin Research License, Version
# 1.1 ("License"). You MAY NOT use this file except in compliance with the
# License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, AND THE
# UNIVERSITY OF TEXAS AT AUSTIN EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
# CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND
# WARRANTIES OF PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM
# COURSE OF DEALING OR USAGE OF TRADE. NO WARRANTY IS EITHER EXPRESS OR IMPLIED
# WITH RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no
# circumstances shall The University of Texas at Austin be liable for
# incidental, special, indirect, direct or consequential damages or loss of
# profits, interruption of business, or related expenses which may arise from
# use of Software or Documentation, including but not limited to those
# resulting from defects in Software and/or Documentation, or loss or
# inaccuracy of data of any kind.
#
# See the License for the specific language governing permissions and
# limitations under this License.

"""No silent tagged->untagged fallback when resolving processed-data paths.

resolve_dataset_path used to return the untagged ``{kind}__{fs}.parquet`` when
the tagged ``{kind}__{fs}__{tag}.parquet`` was missing -- so a config pointing at
a new build on a machine that only had an older, differently-standardized build
trained/scored on the wrong data without a word (issue #236). These tests pin
the fail-loud contract for resolve_dataset_path and evaluate's
resolve_ground_truth_path.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from forma.data.config_utils import (
    UnspecifiedDatasetTagError,
    UntaggedDatasetFallbackError,
    resolve_dataset_path,
    resolve_id_map_path,
    tagged_siblings,
)
from forma.scoring.evaluate import (
    add_dataset_tag_argument,
    add_processed_dir_argument,
    resolve_eval_dataset_tag,
    resolve_eval_processed_dir,
    resolve_ground_truth_path,
)

FS = "pf_full"
TAG = "newbuild"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_first_party(top: str) -> bool:
    """True if ``top`` is a repo package/module (forma, scripts, or a module
    directly under src/ or scripts/)."""
    if top in {"forma", "scripts"}:
        return True
    return any((base / top).is_dir() or (base / f"{top}.py").is_file()
               for base in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "scripts"))


def _import_entry_point(module_name: str):
    """Import an entry point, skipping only when a third-party dependency is absent.
    A missing repo module or a renamed name (``cannot import name ...``, a plain
    ImportError) is a real break in the script under test, so it is re-raised."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        top = (e.name or "").split(".")[0]
        if not top or _is_first_party(top):
            raise
        pytest.skip(f"{module_name} needs third-party module {e.name!r}: {e}")


def test_import_entry_point_skips_only_for_third_party(monkeypatch):
    import importlib
    with pytest.raises(pytest.skip.Exception):
        _import_entry_point("no_such_thirdparty_pkg_xyz")
    with pytest.raises(ModuleNotFoundError):
        _import_entry_point("scripts.no_such_script_xyz")

    def renamed(name):
        raise ImportError("cannot import name 'discover_feature_setz' from 'forma.scoring.evaluate'")

    monkeypatch.setattr(importlib, "import_module", renamed)
    with pytest.raises(ImportError, match="cannot import name"):
        _import_entry_point("scripts.dm_tests")


def _touch(p: Path) -> Path:
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------- resolve_dataset_path


def test_tag_set_and_tagged_present_returns_tagged(tmp_path):
    tagged = _touch(tmp_path / f"tuple_train__{FS}__{TAG}.parquet")
    _touch(tmp_path / f"tuple_train__{FS}.parquet")
    assert resolve_dataset_path(tmp_path, "tuple_train", FS, TAG) == tagged


def test_tag_set_only_untagged_present_raises(tmp_path):
    untagged = _touch(tmp_path / f"tuple_train__{FS}.parquet")
    with pytest.raises(UntaggedDatasetFallbackError) as ei:
        resolve_dataset_path(tmp_path, "tuple_train", FS, TAG)
    msg = str(ei.value)
    assert f"tuple_train__{FS}__{TAG}.parquet" in msg
    assert str(untagged) in msg
    assert isinstance(ei.value, FileNotFoundError)


def test_tag_set_neither_present_returns_missing_tagged(tmp_path):
    p = resolve_dataset_path(tmp_path, "tuple_train", FS, TAG)
    assert p == tmp_path / f"tuple_train__{FS}__{TAG}.parquet"
    assert not p.exists()


@pytest.mark.parametrize("tag", [None, ""])
def test_no_tag_returns_untagged(tmp_path, tag):
    untagged = _touch(tmp_path / f"tuple_train__{FS}.parquet")
    _touch(tmp_path / f"tuple_train__{FS}__{TAG}.parquet")
    assert resolve_dataset_path(tmp_path, "tuple_train", FS, tag) == untagged


def test_no_tag_missing_returns_untagged_path(tmp_path):
    p = resolve_dataset_path(tmp_path, "tuple_train", FS, None)
    assert p == tmp_path / f"tuple_train__{FS}.parquet"


def test_ext_and_str_data_dir(tmp_path):
    _touch(tmp_path / f"regularization_stats__{FS}.csv")
    with pytest.raises(UntaggedDatasetFallbackError):
        resolve_dataset_path(str(tmp_path), "regularization_stats", FS, TAG, ext="csv")
    tagged = _touch(tmp_path / f"regularization_stats__{FS}__{TAG}.csv")
    assert resolve_dataset_path(str(tmp_path), "regularization_stats", FS, TAG,
                                ext="csv") == tagged


def test_other_tag_present_is_not_used(tmp_path):
    """A different tag's file must never satisfy the configured tag."""
    _touch(tmp_path / f"tuple_train__{FS}__oldbuild.parquet")
    p = resolve_dataset_path(tmp_path, "tuple_train", FS, TAG)
    assert p.name == f"tuple_train__{FS}__{TAG}.parquet"
    assert not p.exists()


# ------------------------------------------------------- resolve_ground_truth_path


def test_truth_tagged_present(tmp_path):
    tagged = _touch(tmp_path / f"tabular_test__{FS}__{TAG}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}.parquet")
    assert resolve_ground_truth_path(tmp_path, FS, "test", dataset_tag=TAG) == tagged


def test_truth_tag_set_only_untagged_raises(tmp_path):
    """The exact #236 shape: tagged glm truth absent, untagged Jun-16 build present."""
    _touch(tmp_path / f"tabular_test__{FS}.parquet")
    with pytest.raises(UntaggedDatasetFallbackError):
        resolve_ground_truth_path(tmp_path, FS, "test", dataset_tag=TAG)


def test_truth_tag_set_only_legacy_split_name_raises(tmp_path):
    legacy = _touch(tmp_path / f"test__{FS}.parquet")
    with pytest.raises(UntaggedDatasetFallbackError) as ei:
        resolve_ground_truth_path(tmp_path, FS, "test", dataset_tag=TAG)
    assert str(legacy) in str(ei.value)


def test_truth_tag_set_nothing_present_returns_missing_tagged(tmp_path):
    p = resolve_ground_truth_path(tmp_path, FS, "val", dataset_tag=TAG)
    assert p == tmp_path / f"tabular_val__{FS}__{TAG}.parquet"
    assert not p.exists()


def test_truth_no_tag_untagged_and_legacy(tmp_path):
    legacy = _touch(tmp_path / f"test__{FS}.parquet")
    assert resolve_ground_truth_path(tmp_path, FS, "test") == legacy
    untagged = _touch(tmp_path / f"tabular_test__{FS}.parquet")
    assert resolve_ground_truth_path(tmp_path, FS, "test") == untagged


# ------------------------------------------------------------ datamodule callers


def test_tabular_datamodule_refuses_untagged_substitute(tmp_path):
    """TabularDataModule.prepare_data must not accept untagged files for a tag."""
    from forma.data.tabular_data import TabularDataModule

    for kind in ("tabular_train", "tabular_val", "tabular_test", "regularization_stats"):
        _touch(tmp_path / f"{kind}__{FS}.parquet")
    _touch(tmp_path / f"column_metadata__{FS}.json")
    dm = TabularDataModule.__new__(TabularDataModule)
    dm.data_dir = tmp_path
    dm.feature_set = FS
    dm.dataset_tag = TAG
    with pytest.raises(UntaggedDatasetFallbackError):
        dm.prepare_data()


# ------------------------------------ unspecified tag (hand-assembled eval pools)


def test_truth_unspecified_tag_with_tagged_siblings_raises(tmp_path):
    """No config.yaml / no --dataset_tag, tagged truths present -> refuse to guess."""
    _touch(tmp_path / f"tabular_test__{FS}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}__{TAG}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}__other.parquet")
    with pytest.raises(UnspecifiedDatasetTagError) as ei:
        resolve_ground_truth_path(tmp_path, FS, "test", dataset_tag=None)
    msg = str(ei.value)
    assert f"tabular_test__{FS}__{TAG}.parquet" in msg
    assert f"tabular_test__{FS}__other.parquet" in msg
    assert "--dataset_tag" in msg


def test_truth_explicit_untagged_bypasses_sibling_guard(tmp_path):
    untagged = _touch(tmp_path / f"tabular_test__{FS}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}__{TAG}.parquet")
    assert resolve_ground_truth_path(tmp_path, FS, "test", dataset_tag="") == untagged


def test_truth_unspecified_tag_without_siblings_is_untagged(tmp_path):
    untagged = _touch(tmp_path / f"tabular_test__{FS}.parquet")
    _touch(tmp_path / f"tabular_val__{FS}__{TAG}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}_glm__{TAG}.parquet")
    assert resolve_ground_truth_path(tmp_path, FS, "test", dataset_tag=None) == untagged


def test_tagged_siblings_scoped_to_feature_set(tmp_path):
    a = _touch(tmp_path / f"tabular_test__{FS}__{TAG}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}_glm__{TAG}.parquet")
    _touch(tmp_path / f"tabular_test__{FS}.parquet")
    assert tagged_siblings(tmp_path, "tabular_test", FS) == [a]
    assert tagged_siblings(tmp_path / "missing", "tabular_test", FS) == []


def _cfg(exp_dir: Path, text: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(text, encoding="utf-8")


def test_resolve_eval_dataset_tag_precedence(tmp_path):
    exp = tmp_path / "exp"
    assert resolve_eval_dataset_tag(None, None) is None
    assert resolve_eval_dataset_tag(exp, None) is None
    _cfg(exp, "data:\n  feature_set: pf_full\n")
    assert resolve_eval_dataset_tag(exp, None) is None
    _cfg(exp, f"data:\n  dataset_tag: {TAG}\n")
    assert resolve_eval_dataset_tag(exp, None) == TAG
    assert resolve_eval_dataset_tag(exp, "other") == "other"
    assert resolve_eval_dataset_tag(exp, "") == ""
    assert resolve_eval_dataset_tag(None, "cli") == "cli"


def test_add_dataset_tag_argument():
    ap = argparse.ArgumentParser()
    add_dataset_tag_argument(ap)
    assert ap.parse_args([]).dataset_tag is None
    assert ap.parse_args(["--dataset_tag", ""]).dataset_tag == ""
    assert ap.parse_args(["--dataset-tag", TAG]).dataset_tag == TAG


def test_dataset_tag_equals_empty_parses_to_untagged():
    """The documented deliberate-untagged spelling. `--dataset_tag ""` is NOT
    usable from Windows PowerShell 5.1 (the empty argument is dropped)."""
    ap = argparse.ArgumentParser()
    add_dataset_tag_argument(ap)
    assert ap.parse_args(["--dataset_tag="]).dataset_tag == ""
    assert ap.parse_args(["--dataset-tag="]).dataset_tag == ""
    with pytest.raises(SystemExit):
        ap.parse_args(["--dataset_tag"])


@pytest.mark.parametrize("cfg_line", ['  dataset_tag: ""\n', "  dataset_tag: null\n",
                                      "  feature_set: pf_full\n"])
def test_config_empty_or_null_tag_is_unspecified_not_untagged(tmp_path, cfg_line):
    """Only the CLI can choose 'untagged on purpose'; a blank config tag is unspecified."""
    exp = tmp_path / "exp"
    _cfg(exp, "data:\n" + cfg_line)
    assert resolve_eval_dataset_tag(exp, None) is None


@pytest.mark.parametrize("module_name", [
    "scripts.dm_tests",
    "scripts.mixture_calibration", "scripts.chronos_quantile_calibration",
])
def test_eval_entry_points_share_the_tag_helpers(module_name):
    """Every truth-resolving entry point takes --dataset_tag via the shared helper."""
    mod = _import_entry_point(module_name)
    assert mod.add_dataset_tag_argument is add_dataset_tag_argument
    assert mod.resolve_eval_dataset_tag is resolve_eval_dataset_tag


# --------------------------------------------------------- resolve_eval_processed_dir


def test_resolve_eval_processed_dir_precedence(tmp_path):
    exp = tmp_path / "exp"
    assert resolve_eval_processed_dir(None, None) == Path("data/processed")
    assert resolve_eval_processed_dir(exp, None) == Path("data/processed")
    _cfg(exp, f"data:\n  dataset_tag: {TAG}\n")
    assert resolve_eval_processed_dir(exp, None) == Path("data/processed")
    _cfg(exp, f"data:\n  dataset_tag: {TAG}\n  processed_dir: data/processed_r14\n")
    assert resolve_eval_processed_dir(exp, None) == Path("data/processed_r14")
    assert resolve_eval_processed_dir(exp, "D:/elsewhere") == Path("D:/elsewhere")
    assert resolve_eval_processed_dir(None, "cli_dir") == Path("cli_dir")
    _cfg(exp, "data:\n  processed_dir: null\n")
    assert resolve_eval_processed_dir(exp, None) == Path("data/processed")


def test_add_processed_dir_argument():
    ap = argparse.ArgumentParser()
    add_processed_dir_argument(ap)
    assert ap.parse_args([]).processed_dir is None
    assert ap.parse_args(["--processed_dir", "a"]).processed_dir == "a"
    assert ap.parse_args(["--processed-dir", "b"]).processed_dir == "b"


@pytest.mark.parametrize("module_name", [
    "scripts.dm_tests",
    "scripts.mixture_calibration", "scripts.chronos_quantile_calibration",
])
def test_eval_entry_points_share_the_processed_dir_helpers(module_name):
    """Every truth-resolving entry point reads the truth dir via the shared helper,
    so a pool config.yaml with data.processed_dir is enough everywhere."""
    mod = _import_entry_point(module_name)
    assert mod.add_processed_dir_argument is add_processed_dir_argument
    assert mod.resolve_eval_processed_dir is resolve_eval_processed_dir


# ------------------------------------------------------ evaluate.main end-to-end


def _eval_world(tmp_path, monkeypatch, *, config_text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    fdir = tmp_path / "exp" / "forecasts"
    fdir.mkdir(parents=True)
    _touch(fdir / f"forma__{FS}__test__predictions.parquet")
    if config_text is not None:
        (tmp_path / "exp" / "config.yaml").write_text(config_text, encoding="utf-8")
    return tmp_path / "data" / "processed"


def _run_eval_main(monkeypatch, *extra):
    import forma.scoring.evaluate as ev
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--exp_dir", "exp",
                                      "--splits", "test", *extra])
    return ev.main()


def test_evaluate_main_raises_on_untagged_substitute(tmp_path, monkeypatch):
    proc = _eval_world(tmp_path, monkeypatch, config_text=f"data:\n  dataset_tag: {TAG}\n")
    _touch(proc / f"tabular_test__{FS}.parquet")
    with pytest.raises(UntaggedDatasetFallbackError):
        _run_eval_main(monkeypatch)


def test_evaluate_main_raises_without_tag_when_tagged_truth_exists(tmp_path, monkeypatch):
    proc = _eval_world(tmp_path, monkeypatch, config_text=None)
    _touch(proc / f"tabular_test__{FS}.parquet")
    _touch(proc / f"tabular_test__{FS}__{TAG}.parquet")
    with pytest.raises(UnspecifiedDatasetTagError):
        _run_eval_main(monkeypatch)


def test_evaluate_main_nothing_scored_exits_nonzero(tmp_path, monkeypatch):
    _eval_world(tmp_path, monkeypatch, config_text=f"data:\n  dataset_tag: {TAG}\n")
    # The release's evaluate.main() returns its exit status (the CLI entry point
    # is `raise SystemExit(main())`); a later in-loop failure may sys.exit(1).
    try:
        rc = _run_eval_main(monkeypatch)
    except SystemExit as e:
        rc = e.code
    assert rc == 1


@pytest.mark.parametrize("module_name,prog", [
    ("scripts.dm_tests", "dm_tests.py"),
])
def test_dm_tests_nothing_scored_exit_nonzero(tmp_path, monkeypatch, module_name, prog):
    """A tagged pool whose truth is missing (e.g. a wrong processed_dir) scores
    nothing; like evaluate.main, dm_tests must not exit 0 after only printing 'skipping'."""
    mod = _import_entry_point(module_name)
    _eval_world(tmp_path, monkeypatch, config_text=f"data:\n  dataset_tag: {TAG}\n")
    monkeypatch.setattr(sys, "argv", [prog, "--exp_dir", "exp", "--splits", "test"])
    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert ei.value.code == 1


class _Resolved(Exception):
    pass


@pytest.mark.parametrize("config_extra,cli,expected", [
    ("", (), "data/processed"),
    ("  processed_dir: data/processed_r14\n", (), "data/processed_r14"),
    ("  processed_dir: data/processed_r14\n", ("--processed_dir", "other"), "other"),
])
def test_evaluate_main_reads_truth_from_resolved_processed_dir(tmp_path, monkeypatch,
                                                                config_extra, cli, expected):
    """evaluate.main passes the resolved dir (CLI > config.yaml > default) to the
    truth resolver; it used to hardcode data/processed."""
    import forma.scoring.evaluate as ev
    _eval_world(tmp_path, monkeypatch, config_text=f"data:\n  dataset_tag: {TAG}\n" + config_extra)
    seen = []

    def fake_resolve(processed_dir, *a, **k):
        seen.append(Path(processed_dir))
        raise _Resolved()

    monkeypatch.setattr(ev, "resolve_ground_truth_path", fake_resolve)
    with pytest.raises(_Resolved):
        _run_eval_main(monkeypatch, *cli)
    assert seen == [Path(expected)]


_TRUTH_ENTRY_POINTS = {
    "scripts.dm_tests": ["dm_tests.py", "--exp_dir", "exp", "--splits", "test"],
    "scripts.mixture_calibration": ["mixture_calibration.py", "--exp_dir", "exp",
                                    "--out", "out"],
    "scripts.chronos_quantile_calibration": ["chronos_quantile_calibration.py",
                                             "--quantiles", "q.parquet",
                                             "--exp_dir", "exp", "--out", "out"],
}


@pytest.mark.parametrize("module_name", sorted(_TRUTH_ENTRY_POINTS))
@pytest.mark.parametrize("config_extra,cli,expected", [
    ("", (), "data/processed"),
    ("  processed_dir: data/processed_r14\n", (), "data/processed_r14"),
    ("  processed_dir: data/processed_r14\n", ("--processed_dir", "other"), "other"),
])
def test_entry_point_main_reads_truth_from_resolved_processed_dir(
        tmp_path, monkeypatch, module_name, config_extra, cli, expected):
    """Like evaluate.main: every other truth-resolving main() hands the resolved dir
    (CLI > config.yaml > default) to the truth resolver. Reverting a call site to a
    hardcoded data/processed, or to the CLI flag alone, fails the config.yaml case."""
    mod = _import_entry_point(module_name)
    _eval_world(tmp_path, monkeypatch, config_text=f"data:\n  dataset_tag: {TAG}\n" + config_extra)
    seen = []

    def fake_resolve(processed_dir, *a, **k):
        seen.append(Path(processed_dir))
        raise _Resolved()

    monkeypatch.setattr(mod, "resolve_ground_truth_path", fake_resolve)
    monkeypatch.setattr(sys, "argv", [*_TRUTH_ENTRY_POINTS[module_name], *cli])
    with pytest.raises(_Resolved):
        mod.main()
    assert seen == [Path(expected)]


def test_evaluate_main_cli_tag_overrides(tmp_path, monkeypatch):
    """--dataset_tag with no config.yaml selects the tag: the untagged truth on disk
    is then refused as a substitute (it is not silently scored)."""
    proc = _eval_world(tmp_path, monkeypatch, config_text=None)
    _touch(proc / f"tabular_test__{FS}.parquet")
    with pytest.raises(UntaggedDatasetFallbackError):
        _run_eval_main(monkeypatch, "--dataset_tag", TAG)


# ------------------------------------------------ NaiveDrift no_cv optional val


def _naive_world(tmp_path, monkeypatch, *, val):
    from forma.data.tabular_data import NaiveDriftDataModule

    (tmp_path / f"column_metadata__{FS}.json").write_text(
        json.dumps({"feature_columns": ["niq_level_0"]}), encoding="utf-8")
    train = pd.DataFrame({"firm_id": ["000001"], "niq_level_0": [1.0]})
    train.to_parquet(tmp_path / f"tabular_train__{FS}__{TAG}.parquet",
                     engine="fastparquet", index=False)
    if val == "untagged":
        train.to_parquet(tmp_path / f"tabular_val__{FS}.parquet",
                         engine="fastparquet", index=False)
    seen = {}

    def fake_create(self, df, split_name="test"):
        seen[split_name] = len(df)
        return df

    monkeypatch.setattr(NaiveDriftDataModule, "_create_naive_prediction_data", fake_create)
    dm = NaiveDriftDataModule(data_dir=str(tmp_path), data_config={
        "feature_set": FS, "dataset_tag": TAG, "no_cv": True,
        "targets": {"variables": ["niq"]},
    })
    return dm, seen


def test_naive_no_cv_refuses_untagged_val(tmp_path, monkeypatch):
    dm, _ = _naive_world(tmp_path, monkeypatch, val="untagged")
    with pytest.raises(UntaggedDatasetFallbackError):
        dm.setup(stage="fit")


def test_naive_no_cv_missing_val_warns_and_trains_on_train(tmp_path, monkeypatch, capsys):
    dm, seen = _naive_world(tmp_path, monkeypatch, val=None)
    dm.setup(stage="fit")
    assert seen == {"train": 1}
    out = capsys.readouterr().out
    assert "WARNING: no_cv=True but the validation split is missing" in out
    assert f"tabular_val__{FS}__{TAG}.parquet" in out


# ---------------------------------------------------------------- id-map paths


def test_id_map_defaults_to_processed_dir(tmp_path):
    dc = {"processed_dir": str(tmp_path / "processed_r14")}
    for which, fname in (("firm", "firm_id_map.csv"), ("account", "account_id_map.csv"),
                         ("industry", "industry_id_map.csv")):
        assert resolve_id_map_path(dc, which) == tmp_path / "processed_r14" / fname
    assert resolve_id_map_path(None, "firm") == Path("data/processed/firm_id_map.csv")
    assert resolve_id_map_path({}, "firm", processed_dir=tmp_path) == tmp_path / "firm_id_map.csv"


def test_id_map_explicit_inside_processed_dir_kept_verbatim():
    dc = {"processed_dir": "data/processed",
          "account_id_map_path": "data/processed/account_id_map.csv"}
    assert resolve_id_map_path(dc, "account").as_posix() == "data/processed/account_id_map.csv"


def test_id_map_explicit_outside_processed_dir_raises():
    """A config copied from r13 into a data/processed_r14 build still names the
    data/processed maps -- that must fail, not mix builds."""
    dc = {"processed_dir": "data/processed_r14",
          "firm_id_map_path": "data/processed/firm_id_map.csv"}
    with pytest.raises(ValueError, match="not inside data.processed_dir"):
        resolve_id_map_path(dc, "firm")


def test_id_map_in_subdirectory_of_processed_dir_raises(tmp_path):
    dc = {"processed_dir": str(tmp_path),
          "firm_id_map_path": str(tmp_path / "sub" / "firm_id_map.csv")}
    with pytest.raises(ValueError, match="not inside data.processed_dir"):
        resolve_id_map_path(dc, "firm")


def test_id_map_symlinked_file_inside_processed_dir_accepted(tmp_path):
    """A map that lives in processed_dir as a symlink into another store is fine:
    only the directory is compared, the file itself is not resolved."""
    store = tmp_path / "store"
    store.mkdir()
    real = store / "firm_id_map.csv"
    real.write_text("firm_id,firm_id_int\n", encoding="utf-8")
    proc = tmp_path / "processed_r14"
    proc.mkdir()
    link = proc / "firm_id_map.csv"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlinks unavailable here: {e}")
    dc = {"processed_dir": str(proc), "firm_id_map_path": str(link)}
    assert resolve_id_map_path(dc, "firm") == link


def test_id_map_unknown_kind():
    with pytest.raises(ValueError):
        resolve_id_map_path({}, "gvkey")
