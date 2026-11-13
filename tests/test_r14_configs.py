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

"""Pin the shipped camera-ready (R14) config set.

The R14 configs re-run every paper exhibit with (1) the corrected asinh-k target,
TOTAL kurtosis 3 (#285), and (2) the untrimmed raw panel (#286), in their own
processed dir. The third fix, the no_cv train+val dedup (#279), lives in code and
is covered by test_no_cv_union_dedup.py. A config that quietly kept an R13 value
(the legacy 6.0, the trimmed raw file, data/processed's id maps) would train on
the old data. Adapted from the research repo's test_r14_configs.py, without its
config-generator and dataset-builder checks (neither ships here). Data-free.
"""
import glob
import importlib
import sys
from pathlib import Path

import pytest
import yaml

from forma.data.config_utils import resolve_id_map_path

REPO = Path(__file__).resolve().parents[1]
TAG = "r14_node_optionD_indfe_val8"
SEEDS = [60, 61, 62, 63, 64]
EXPECTED_FORMA = {  # config stem infix -> (forecast_name, loss_type)
    "fgrid": ("forma_fgrid", "gaussian"),
    "lap05_fgrid": ("forma_lap05_fgrid", "laplace"),
}
EXPECTED_FILES = sorted(
    [f"r14_forma_{infix}_seed{s}.yaml" for infix in EXPECTED_FORMA for s in SEEDS]
    + [f"r14_ffnn_{net}_betanll_seed{s}.yaml" for net in ("large", "linear") for s in SEEDS]
    + ["r14_ffnn_betanll_cv.yaml", "r14_random_forest.yaml", "r14_glm.yaml",
       "r14_glm_mse.yaml", "r14_chronos2_raw12q.yaml"]
)
R14_PATHS = sorted(glob.glob(str(REPO / "configs" / "r14_*.yaml")))


def _load(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def test_shipped_r14_config_set_is_complete_and_r13_is_gone():
    assert sorted(Path(p).name for p in R14_PATHS) == EXPECTED_FILES
    assert not list((REPO / "configs").glob("r13_*.yaml"))


@pytest.mark.parametrize("path", R14_PATHS, ids=lambda p: Path(p).stem)
def test_r14_config_carries_the_data_fixes(path):
    data = _load(path)["data"]
    assert data["dataset_tag"] == TAG
    # #286: the untrimmed panel, never the CRSP-trimmed legacy file.
    assert data["raw_path"] == "data/raw/compustat_fundq.parquet"
    # #285: the corrected target, pinned explicitly.
    assert data["feature_engineering"]["k_target_total_kurtosis"] == 3.0
    # Own build dir: the id maps resolve inside it, not into data/processed.
    assert data["processed_dir"] == "data/processed"
    for which in ("firm", "account", "industry"):
        assert f"{which}_id_map_path" not in data
        p = resolve_id_map_path(data, which)
        assert p.parent == Path("data/processed"), (which, p)


def test_forma_families_and_no_cv():
    for infix, (forecast_name, loss_type) in EXPECTED_FORMA.items():
        for s in SEEDS:
            cfg = _load(REPO / "configs" / f"r14_forma_{infix}_seed{s}.yaml")
            assert cfg["seed"] == s
            assert cfg["data"]["no_cv"] is True          # exercises the #279 dedup path
            assert cfg["data"]["use_future_grid"] is True
            assert cfg["data"]["future_grid_sample_overhang"] is False
            forma = cfg["models"]["forma"]
            assert forma["forecast_name"] == forecast_name
            assert forma["parameters"]["loss_type"] == loss_type
            assert cfg["inference"]["seed_in_forecast_name"] is True


@pytest.mark.parametrize("net", ["large", "linear"])
def test_ffnn_benchmark_epochs_are_set_and_the_sentinel_is_refused(net):
    """Every benchmark seed carries the CV-selected epochs (same value across
    seeds), and train.py refuses the -1 "set me" sentinel: FeedForwardModel on its
    own would silently train 0 epochs."""
    from forma.train import check_ffnn_epochs, create_model

    epochs = set()
    for s in SEEDS:
        cfg = _load(REPO / "configs" / f"r14_ffnn_{net}_betanll_seed{s}.yaml")
        (name, model_cfg), = cfg["models"].items()
        assert model_cfg["parameters"]["seed"] == cfg["seed"] == s
        epochs.add(model_cfg["parameters"]["epochs"])
    assert len(epochs) == 1 and min(epochs) >= 1

    model = create_model(model_cfg, cfg["data"])
    check_ffnn_epochs(name, model_cfg["type"], model)  # the shipped value passes
    model.epochs = -1
    with pytest.raises(ValueError, match="not a valid training length"):
        check_ffnn_epochs(name, model_cfg["type"], model)


def test_train_main_refuses_unset_ffnn_epochs(tmp_path, monkeypatch):
    """Behavioural: train.main() on an R14 FFNN benchmark config with epochs -1 must
    stop with the guard's ValueError before it builds a data module."""
    from forma import train

    cfg = _load(REPO / "configs" / "r14_ffnn_large_betanll_seed60.yaml")
    cfg["models"]["ffnn_large_b50"]["parameters"]["epochs"] = -1
    assert cfg["stop_on_error"] is True   # so main() re-raises instead of logging
    cfg_path = tmp_path / "ffnn_unset_epochs.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def reached_data_module(*args, **kwargs):
        raise RuntimeError("main() reached create_data_module without the epochs guard")

    monkeypatch.setattr(train, "create_data_module", reached_data_module)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--debug"])
    with pytest.raises(ValueError, match="not a valid training length"):
        train.main()


def _script(name):
    return importlib.import_module(f"scripts.{name}")


def test_scripts_point_at_shipped_configs():
    """The regeneration / inference drivers name configs that exist."""
    pf = _script("predict_forma")
    for family, pattern in pf.FAMILIES.items():
        for s in pf.SEEDS:
            assert (REPO / "configs" / pattern.format(seed=s)).exists(), (family, s)
    gbm = _script("regen_gbm")
    for cfg in gbm.VARIANTS.values():
        assert (REPO / "configs" / cfg).exists(), cfg
    for net in ("large", "linear"):
        for s in _script("predict_ffnn").SEEDS:
            assert (REPO / "configs" / f"r14_ffnn_{net}_betanll_seed{s}.yaml").exists()
    assert "r14_random_forest.yaml" in (REPO / "scripts" / "regen_rf.py").read_text(encoding="utf-8")
