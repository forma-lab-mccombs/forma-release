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

"""Laplace / Student-t Forma forecasts must carry their .nll.json density sidecar.

evaluate.py scores a forecast with no `{stem}.nll.json` as Gaussian. train.py's
chunked streaming save is the only forecast writer in use for Forma (the old
in-memory _save_forecasts has no caller), and it wrote no sidecar, so R13's
Laplace seeds had to be labelled by hand. The repair tool
scripts/write_nll_sidecars.py missed them too (it only recognised the substring
'laplace', not the `lap05` arm token).
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from forma.data.forma_data import FormaData
from forma.models.forma import FormaModel
from scripts import write_nll_sidecars as wns

REPO = Path(__file__).resolve().parents[1]


# A one-graph Forma batch and a data-module stub that yields it on the test split
# (from the research repo's test_forma_mse.py).
def _model(loss_type, **kw):
    return FormaModel(num_account_types=5, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, dropout=0.0, constraint_mode='none',
                      loss_type=loss_type, **kw)


def _one_graph_batch():
    from torch_geometric.data import Batch
    d1 = FormaData(
        x_id=torch.tensor([0, 1, 2], dtype=torch.long),
        x_time=torch.tensor([0, 0, 1], dtype=torch.long),
        x_val=torch.tensor([0.5, -0.3, 0.0]),
        mask=torch.tensor([False, False, True]),
        data_avail=torch.tensor([True, True, True]),
        y=torch.tensor([0.5, -0.3, 0.1]),
        reg_k=torch.ones(3), reg_mu=torch.zeros(3), reg_sigma=torch.ones(3),
        constraint_src=torch.tensor([0, 0, 0], dtype=torch.long),
        constraint_dst=torch.tensor([0, 1, 2], dtype=torch.long),
        constraint_sign=torch.tensor([1.0, -1.0, -1.0]),
        num_constraints=torch.tensor([1], dtype=torch.long),
        num_constraint_edges=torch.tensor([3], dtype=torch.long),
        firm_id=torch.tensor([0], dtype=torch.long),
        center_q=torch.tensor([100], dtype=torch.long),
    )
    return Batch.from_data_list([d1])


class _OneBatchDM:
    """Minimal data-module stub: yields a single Forma batch on the test split."""
    feature_set = "pf"
    batch_size = 4

    def __init__(self, batch):
        self._batch = batch

    def setup(self, stage=None):
        pass

    def test_dataloader(self):
        return [self._batch]

LAPLACE = {"family": "laplace", "df": None}


def _predict(model, tmp_path):
    from forma.train import generate_predictions
    latest = tmp_path / "latest"
    exp = tmp_path / "exp"
    config = {
        "inference": {"reconcile": False},
        "results_dir": str(latest),
        "forecast_splits": ["test"],
        "data": {"predict_batch_size": 4, "processed_dir": str(tmp_path)},
    }
    generate_predictions(model, _OneBatchDM(_one_graph_batch()), config, exp_dir=exp)
    return [exp / "forecasts", latest]


@pytest.mark.parametrize("loss_type, expect", [("laplace", LAPLACE), ("gaussian", None)])
def test_real_predict_writes_the_sidecar_next_to_both_copies(tmp_path, loss_type, expect):
    torch.manual_seed(0)
    for d in _predict(_model(loss_type), tmp_path):
        assert (d / "forma__pf__test__predictions.parquet").exists()
        side = d / "forma__pf__test__predictions.nll.json"
        if expect is None:
            assert not side.exists()
        else:
            assert json.loads(side.read_text(encoding="utf-8")) == expect


def test_gaussian_predict_removes_a_stale_laplace_sidecar(tmp_path):
    for d in (tmp_path / "exp" / "forecasts", tmp_path / "latest"):
        d.mkdir(parents=True)
        (d / "forma__pf__test__predictions.nll.json").write_text(json.dumps(LAPLACE), encoding="utf-8")
    torch.manual_seed(0)
    for d in _predict(_model("gaussian"), tmp_path):
        assert (d / "forma__pf__test__predictions.parquet").exists()
        assert not (d / "forma__pf__test__predictions.nll.json").exists()


def test_stream_save_writes_the_sidecar_after_the_atomic_write():
    """Ordering only (presence is covered behaviourally above): the sidecar must not
    be written inside the atomic-writer block, before the parquet is in place."""
    tree = ast.parse((REPO / "src" / "forma" / "train.py").read_text(encoding="utf-8"))
    fn = next(sub for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "generate_predictions"
              for sub in ast.walk(node)
              if isinstance(sub, ast.FunctionDef) and sub.name == "_stream_save_forecasts")
    calls = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_write_nll_sidecar"]
    with_end = max(n.end_lineno for n in ast.walk(fn) if isinstance(n, ast.With))
    assert calls and all(line > with_end for line in calls)


@pytest.mark.parametrize("stem, fam", [
    ("forma_lap05_fgrid__pf_full__60__test__predictions", "laplace"),
    ("forma_lap10_fgrid__pf_full__test__predictions", "laplace"),
    ("forma_laplace__pf_full__test__predictions", "laplace"),
    ("forma_fgrid__pf_full__61__test__predictions", "gaussian"),
    ("ffnn_large_b50__pf_full__60__test__predictions", "gaussian"),
    ("sklearn_elasticnet__pf_full__test__predictions", "gaussian"),
    ("forma_overlap2__pf_full__test__predictions", "gaussian"),
    ("forma_lap05x__pf_full__test__predictions", "gaussian"),
    ("forma_student_t__pf_full__test__predictions", "student_t"),
])
def test_infer_family(stem, fam):
    assert wns.infer_family(stem) == fam


def _run(*args):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "write_nll_sidecars.py"), *args],
                          capture_output=True, text=True, timeout=60)


def test_lap05_seeds_get_a_laplace_sidecar(tmp_path):
    for s in (60, 61):
        (tmp_path / f"forma_lap05_fgrid__pf_full__{s}__test__predictions.parquet").write_bytes(b"x")
    (tmp_path / "forma_fgrid__pf_full__60__test__predictions.parquet").write_bytes(b"x")
    p = _run(str(tmp_path))
    assert p.returncode == 0, p.stderr
    for s in (60, 61):
        side = tmp_path / f"forma_lap05_fgrid__pf_full__{s}__test__predictions.nll.json"
        assert json.loads(side.read_text()) == LAPLACE
    assert not (tmp_path / "forma_fgrid__pf_full__60__test__predictions.nll.json").exists()


def test_explicit_family_and_glob(tmp_path):
    (tmp_path / "arm_x__pf_full__60__test__predictions.parquet").write_bytes(b"x")
    (tmp_path / "arm_x__pf_full__60__test__predictions.parquet.tmp").write_bytes(b"x")
    (tmp_path / "arm_y__pf_full__60__test__predictions.parquet").write_bytes(b"x")
    for _ in range(2):
        p = _run(str(tmp_path), "--glob", "arm_x__*", "--family", "laplace")
        assert p.returncode == 0, p.stderr
    sides = sorted(f.name for f in tmp_path.iterdir() if f.name.endswith(".json"))
    assert sides == ["arm_x__pf_full__60__test__predictions.nll.json"]
    p = _run(str(tmp_path), "--glob", "nothing__*", "--family", "laplace")
    assert p.returncode != 0 and "no files match" in (p.stderr + p.stdout)
    p = _run(str(tmp_path), "--glob", str(tmp_path / "*.parquet"), "--family", "laplace")
    assert p.returncode != 0 and "relative to" in (p.stderr + p.stdout)
