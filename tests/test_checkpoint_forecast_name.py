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

"""A predict-only (--checkpoint) run must keep the config's forecast_name.

train.py replaces the config-built model with ``load_from_checkpoint``, which
builds a new instance from the saved hparams; ``model_name`` is set on the
instance, not saved as an hparam. Before load_lightning_checkpoint, every
predict-only Forma run therefore wrote ``forma__...`` forecasts, colliding the
Gaussian and Laplace arms (R13 renamed them by hand).
"""
from pathlib import Path

import lightning as L
import torch

from forma import train

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Tiny(L.LightningModule):
    def __init__(self, width: int = 3):
        super().__init__()
        self.save_hyperparameters()
        self.lin = torch.nn.Linear(width, 1)


def _save_ckpt(model, path):
    torch.save({"state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": L.__version__}, path)


def test_reload_keeps_forecast_name_and_weights(tmp_path):
    trained = _Tiny(width=4)
    ckpt = tmp_path / "forma_lap05_fgrid.ckpt"
    _save_ckpt(trained, ckpt)

    fresh = _Tiny(width=4)
    fresh.model_name = "forma_lap05_fgrid"   # what the forecast_name override sets
    loaded = train.load_lightning_checkpoint(fresh, ckpt)

    assert loaded is not fresh
    assert loaded.model_name == "forma_lap05_fgrid"
    assert loaded.hparams.width == 4
    for k, v in trained.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)
    assert train._forecast_filename(loaded.model_name, "pf_full", "test", seed=61) == \
        "forma_lap05_fgrid__pf_full__61__test__predictions.parquet"


def test_plain_reload_loses_the_name(tmp_path):
    """Pins the failure mode the helper exists for."""
    m = _Tiny()
    ckpt = tmp_path / "m.ckpt"
    _save_ckpt(m, ckpt)
    m.model_name = "forma_fgrid"
    assert getattr(_Tiny.load_from_checkpoint(str(ckpt)), "model_name", "forma") == "forma"


def test_reload_without_a_name_leaves_it_unset(tmp_path):
    m = _Tiny()
    ckpt = tmp_path / "m.ckpt"
    _save_ckpt(m, ckpt)
    assert not hasattr(train.load_lightning_checkpoint(_Tiny(), ckpt), "model_name")


def test_main_uses_the_helper():
    src = (REPO_ROOT / "src" / "forma" / "train.py").read_text(encoding="utf-8")
    assert "model = load_lightning_checkpoint(model, checkpoint_path)" in src
    assert "model = model.__class__.load_from_checkpoint(" not in src
