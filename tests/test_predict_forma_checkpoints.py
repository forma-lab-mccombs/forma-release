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

"""`predict_forma.py --family` must find the checkpoints under their Hub names.

The Hugging Face repo publishes ``checkpoints/forma_gaussian_seed{60..64}.ckpt``
(family forma_fgrid) and ``checkpoints/forma_laplace_seed{60..64}.ckpt`` (family
forma_lap05_fgrid); the script used to look only for ``{family}_seed{seed}.ckpt``,
so a reviewer who downloaded the weights as documented could not run it. Data-free:
the checkpoint files are empty placeholders, nothing is loaded.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "predict_forma", Path(__file__).resolve().parents[1] / "scripts" / "predict_forma.py")
pf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pf)


def test_every_family_has_a_hub_name_and_a_density():
    assert set(pf.HUB_CHECKPOINTS) == set(pf.FAMILIES) == set(pf.FAMILY_DENSITY)
    assert pf.HUB_CHECKPOINTS["forma_fgrid"].format(seed=60) == "forma_gaussian_seed60.ckpt"
    assert pf.HUB_CHECKPOINTS["forma_lap05_fgrid"].format(seed=64) == "forma_laplace_seed64.ckpt"


@pytest.mark.parametrize("family,hub", [("forma_fgrid", "forma_gaussian_seed61.ckpt"),
                                        ("forma_lap05_fgrid", "forma_laplace_seed61.ckpt")])
def test_hub_name_is_found(tmp_path, family, hub):
    (tmp_path / hub).write_bytes(b"")
    assert pf.resolve_checkpoint(tmp_path, family, 61) == tmp_path / hub


def test_hub_name_wins_over_the_legacy_name(tmp_path):
    (tmp_path / "forma_gaussian_seed60.ckpt").write_bytes(b"")
    (tmp_path / "forma_fgrid_seed60.ckpt").write_bytes(b"")
    assert pf.resolve_checkpoint(tmp_path, "forma_fgrid", 60) == tmp_path / "forma_gaussian_seed60.ckpt"


def test_hub_layout_nested_one_level_is_found(tmp_path):
    """`hf download --include "checkpoints/*" --local-dir checkpoints` nests the files."""
    nested = tmp_path / "checkpoints"
    nested.mkdir()
    (nested / "forma_laplace_seed62.ckpt").write_bytes(b"")
    assert pf.resolve_checkpoint(tmp_path, "forma_lap05_fgrid", 62) == nested / "forma_laplace_seed62.ckpt"


def test_legacy_name_is_the_fallback(tmp_path):
    (tmp_path / "forma_lap05_fgrid_seed63.ckpt").write_bytes(b"")
    assert pf.resolve_checkpoint(tmp_path, "forma_lap05_fgrid", 63) == \
        tmp_path / "forma_lap05_fgrid_seed63.ckpt"


def test_missing_checkpoint_lists_every_name_tried(tmp_path):
    with pytest.raises(SystemExit) as ei:
        pf.resolve_checkpoint(tmp_path, "forma_fgrid", 64)
    msg = str(ei.value.code)
    assert "forma_gaussian_seed64.ckpt" in msg
    assert "forma_fgrid_seed64.ckpt" in msg
    assert "hf download" in msg
