"""Import shared modules from colocate_single_demo."""

from __future__ import annotations

import os
import sys

_SINGLE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "colocate_single_demo"))
if _SINGLE not in sys.path:
    sys.path.insert(0, _SINGLE)

SINGLE_DEMO_ROOT = _SINGLE
DEFAULT_REPO_ROOT = os.environ.get("REPO_ROOT", "/data/nfs/kaiyuan")
