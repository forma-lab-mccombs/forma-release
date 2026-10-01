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
