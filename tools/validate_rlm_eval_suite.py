#!/usr/bin/env python3
"""Validate the source-traced RLM eval-suite local environments.

Each environment is its own standalone verifiers package under
training/environments/ (mirroring training/environments/oolong/): a package dir
with pyproject.toml, a thin __init__ re-exporting `load_environment`, a
self-contained env.py, and a README. This validator checks that structure plus
the traced user prologues and config wiring, without GPUs or network.
"""
from __future__ import annotations

import inspect
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "training/environments"

# One standalone package per HF-picture eval environment (env id == package name,
# mirroring the upstream oolong/ layout).
PACKAGES = {
    "oolong": "oolong",
    "oolong_pairs": "oolong_pairs",
    "browsecomp_plus": "browsecomp_plus",
    "longbench_codeqa": "longbench_codeqa",
}

sys.path[:0] = [str(ENVS / pkg) for pkg in PACKAGES.values()]
sys.path[:0] = [str(ROOT / "training/src"), str(ROOT)]

from browsecomp_plus.env import BROWSECOMP_PLUS_USER_PROLOGUE  # noqa: E402
from browsecomp_plus.env import load_environment as load_bcp  # noqa: E402
from longbench_codeqa.env import LONGBENCH_CODEQA_USER_PROLOGUE  # noqa: E402
from longbench_codeqa.env import load_environment as load_lbv2  # noqa: E402
from oolong.env import OOLONG_USER_PROLOGUE  # noqa: E402
from oolong.env import load_environment as load_oolong  # noqa: E402
from oolong_pairs.env import OOLONG_PAIRS_USER_PROLOGUE  # noqa: E402
from oolong_pairs.env import load_environment as load_pairs  # noqa: E402

CONFIG = ROOT / "training/configs/rlm-qwen3-30b-efficient-eval-suite.toml"


def _structure_checks() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for env_id, pkg in PACKAGES.items():
        base = ENVS / pkg
        pyproject = base / "pyproject.toml"
        checks[f"{pkg}_dir_exists"] = base.is_dir()
        checks[f"{pkg}_env_module_exists"] = (base / pkg / "env.py").exists()
        checks[f"{pkg}_init_exists"] = (base / pkg / "__init__.py").exists()
        checks[f"{pkg}_readme_exists"] = (base / "README.md").exists()
        if pyproject.exists():
            cfg = tomllib.loads(pyproject.read_text())
            eps = cfg.get("project", {}).get("entry-points", {}).get("verifiers.environments", {})
            checks[f"{pkg}_entrypoint_is_load_environment"] = (
                eps.get(env_id) == f"{pkg}:load_environment"
            )
        else:
            checks[f"{pkg}_entrypoint_is_load_environment"] = False
    return checks


def main() -> int:
    cfg = tomllib.loads(CONFIG.read_text())
    ids = [e["id"] for e in cfg["orchestrator"]["train"]["env"]]
    ids += [e["id"] for e in cfg["orchestrator"]["eval"]["env"]]
    missing = set(PACKAGES) - set(ids)

    checks = _structure_checks()
    checks.update(
        {
            "oolong_has_plan_hint": "Plan before you act" in OOLONG_USER_PROLOGUE,
            "oolong_mentions_context_var": "REPL variable `context`" in OOLONG_USER_PROLOGUE,
            "pairs_has_plan_hint": "Plan before you act" in OOLONG_PAIRS_USER_PROLOGUE,
            "pairs_mentions_pair_format": "(user_id_1, user_id_2)" in OOLONG_PAIRS_USER_PROLOGUE,
            "bcp_has_plan_hint": "Plan before you act" in BROWSECOMP_PLUS_USER_PROLOGUE,
            "bcp_mentions_exact_answer": "Exact Answer" in BROWSECOMP_PLUS_USER_PROLOGUE,
            "lbv2_has_plan_hint": "Plan before you act" in LONGBENCH_CODEQA_USER_PROLOGUE,
            "lbv2_mentions_letter_format": "A, B, C, or D" in LONGBENCH_CODEQA_USER_PROLOGUE,
            "config_mentions_all_four_env_ids": not missing,
            "oolong_signature_has_user_prologue": "user_prologue"
            in inspect.signature(load_oolong).parameters,
            "pairs_signature_has_user_prologue": "user_prologue"
            in inspect.signature(load_pairs).parameters,
            "bcp_signature_has_user_prologue": "user_prologue"
            in inspect.signature(load_bcp).parameters,
            "lbv2_signature_has_user_prologue": "user_prologue"
            in inspect.signature(load_lbv2).parameters,
        }
    )
    for k, v in checks.items():
        print(f"{k}: {v}")
    if not all(checks.values()):
        print(f"FAILED (missing config env ids: {sorted(missing)})")
        return 1
    print("PASSED: per-env eval-suite packages expose traced user_prologues and config references them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
