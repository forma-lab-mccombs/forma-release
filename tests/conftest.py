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

"""Make ``forma.*`` (the package) and ``scripts.*`` (the standalone scripts)
importable during tests whether or not the package is pip-installed.

CI installs the package (``pip install -e .``) so ``forma`` resolves anyway; this
belt-and-suspenders keeps `python -m pytest` green from a plain checkout too, and
puts the repo root on the path so tests can import ``scripts.mixture_calibration``.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
