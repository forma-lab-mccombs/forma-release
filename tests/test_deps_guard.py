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

"""The companion-package guard must fail loudly and helpfully, not obscurely."""
import builtins
import importlib

import pytest

from forma import _deps


def _hide_proforma20q(monkeypatch):
    real_import = importlib.import_module

    def fake(name, *a, **k):
        if name == "proforma20q" or name.startswith("proforma20q."):
            raise ModuleNotFoundError("No module named 'proforma20q'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)
    monkeypatch.setattr(builtins, "__import__", builtins.__import__)


def test_available_reports_true_when_installed():
    # The benchmark package is a workflow dependency; CI installs it.
    assert isinstance(_deps.proforma20q_available(), bool)


def test_require_exits_with_install_hint_when_missing(monkeypatch, capsys):
    _hide_proforma20q(monkeypatch)
    assert _deps.proforma20q_available() is False
    with pytest.raises(SystemExit) as ei:
        _deps.require_proforma20q("Panel A scoring")
    msg = str(ei.value)
    assert "Panel A scoring" in msg
    assert "pip install -e" in msg
    assert "proforma-20q" in msg


def test_warn_if_missing_is_non_fatal(monkeypatch, capsys):
    _hide_proforma20q(monkeypatch)
    assert _deps.warn_if_missing("scoring") is False
    err = capsys.readouterr().err
    assert "not installed" in err
    assert "pip install -e" in err
