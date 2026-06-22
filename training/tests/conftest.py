"""Make `rlm_train` and `rlm_eval_suite` importable when running the tests.

These two packages live in sibling source trees rather than being installed, so
plain `pytest` from the repo root (or anywhere) would fail to import them. The
validators under `tools/` do the same sys.path insertion; this keeps the test
suite runnable with a bare `pytest` invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
for p in (TRAINING / "src", TRAINING / "environments" / "rlm_eval_suite"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
