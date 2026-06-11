"""Add sibling demos to sys.path."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SINGLE = os.path.normpath(os.path.join(_ROOT, "..", "colocate_single_demo"))

for _p in (_SINGLE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SINGLE_DEMO_ROOT = _SINGLE
DEFAULT_SLIME_ROOT = os.environ.get("SLIME_ROOT") or os.environ.get("VIME_ROOT", "/data/nfs/kaiyuan/slime")
DEFAULT_REPO_ROOT = os.environ.get("REPO_ROOT", "/data/nfs/kaiyuan")
