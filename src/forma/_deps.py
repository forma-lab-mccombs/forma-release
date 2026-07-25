"""Runtime checks for the companion benchmark package.

``proforma-20q`` is a hard dependency of the *workflow* (it builds the dataset,
defines the submission schema, and scores forecasts) but not of Forma inference
itself, and it is a source install rather than a PyPI package -- so it is not
declared in ``pyproject.toml``. These helpers turn a missing install into an
actionable message instead of a bare ``ModuleNotFoundError`` three frames deep.
"""
from __future__ import annotations

import importlib
import sys

_INSTALL_HINT = (
    "The companion benchmark package `proforma-20q` is required for this step.\n"
    "It is a source install (not on PyPI):\n"
    "    pip install -e /path/to/proforma-20q\n"
    "See this repo's README (\"Install\") for the full quickstart."
)


def proforma20q_available() -> bool:
    """True if the companion package can be imported."""
    try:
        importlib.import_module("proforma20q")
        return True
    except Exception:
        return False


def require_proforma20q(feature: str = "this step"):
    """Import and return the companion package, or exit with a clear message.

    Args:
        feature: short description of what needs it, used in the error text.
    """
    try:
        return importlib.import_module("proforma20q")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: {feature} requires `proforma20q` ({e.__class__.__name__}: {e}).\n"
                 f"{_INSTALL_HINT}")


def warn_if_missing(feature: str = "scoring") -> bool:
    """Print a non-fatal note if the companion package is absent. Returns availability."""
    if proforma20q_available():
        return True
    print(f"NOTE: `proforma20q` is not installed, so {feature} is unavailable.\n"
          f"{_INSTALL_HINT}", file=sys.stderr)
    return False
