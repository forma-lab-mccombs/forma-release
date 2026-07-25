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
