#!/usr/bin/env python3
"""Validate the source-traced RLM eval-suite local environments."""
from __future__ import annotations

import inspect
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "training/environments/rlm_eval_suite"),
    str(ROOT / "training/src"),
    str(ROOT),
]

from rlm_eval_suite import (  # noqa: E402
    load_browsecomp_plus_environment,
    load_longbench_codeqa_environment,
    load_oolong_environment,
    load_oolong_pairs_environment,
)
from rlm_eval_suite.envs import (  # noqa: E402
    BROWSECOMP_PLUS_USER_PROLOGUE,
    LONGBENCH_CODEQA_USER_PROLOGUE,
    OOLONG_PAIRS_USER_PROLOGUE,
    OOLONG_USER_PROLOGUE,
)

CONFIG = ROOT / "training/configs/rlm-qwen3-30b-efficient-eval-suite.toml"
PKG = ROOT / "training/environments/rlm_eval_suite"
PYPROJECT = PKG / "pyproject.toml"


def main() -> int:
    cfg = tomllib.loads(CONFIG.read_text())
    pkg_cfg = tomllib.loads(PYPROJECT.read_text())
    ids = [e["id"] for e in cfg["orchestrator"]["train"]["env"]]
    ids += [e["id"] for e in cfg["orchestrator"]["eval"]["env"]]
    # All four HF-picture eval environments must be wired into the config.
    expected = {
        "rlm-oolong-local",
        "rlm-oolong-pairs-local",
        "rlm-browsecomp-plus-local",
        "rlm-longbench-codeqa-local",
    }
    missing = expected - set(ids)
    entry_points = pkg_cfg["project"]["entry-points"]["verifiers.environments"]

    checks = {
        "split_module_common_exists": (PKG / "rlm_eval_suite/common.py").exists(),
        "split_module_oolong_exists": (PKG / "rlm_eval_suite/oolong.py").exists(),
        "split_module_bcp_exists": (PKG / "rlm_eval_suite/browsecomp_plus.py").exists(),
        "split_module_lbv2_exists": (PKG / "rlm_eval_suite/longbench_codeqa.py").exists(),
        "entrypoint_oolong_direct": entry_points.get("rlm-oolong-local")
        == "rlm_eval_suite.oolong:load_oolong_environment",
        "entrypoint_pairs_direct": entry_points.get("rlm-oolong-pairs-local")
        == "rlm_eval_suite.oolong:load_oolong_pairs_environment",
        "entrypoint_bcp_direct": entry_points.get("rlm-browsecomp-plus-local")
        == "rlm_eval_suite.browsecomp_plus:load_browsecomp_plus_environment",
        "entrypoint_lbv2_direct": entry_points.get("rlm-longbench-codeqa-local")
        == "rlm_eval_suite.longbench_codeqa:load_longbench_codeqa_environment",
        "oolong_has_plan_hint": "Plan before you act" in OOLONG_USER_PROLOGUE,
        "oolong_mentions_context_var": "REPL variable `context`" in OOLONG_USER_PROLOGUE,
        "pairs_has_plan_hint": "Plan before you act" in OOLONG_PAIRS_USER_PROLOGUE,
        "pairs_mentions_pair_format": "(user_id_1, user_id_2)" in OOLONG_PAIRS_USER_PROLOGUE,
        "bcp_has_plan_hint": "Plan before you act" in BROWSECOMP_PLUS_USER_PROLOGUE,
        "bcp_mentions_exact_answer": "Exact Answer" in BROWSECOMP_PLUS_USER_PROLOGUE,
        "lbv2_has_plan_hint": "Plan before you act" in LONGBENCH_CODEQA_USER_PROLOGUE,
        "lbv2_mentions_letter_format": "A, B, C, or D" in LONGBENCH_CODEQA_USER_PROLOGUE,
        "config_mentions_all_four_env_ids": not missing,
        "oolong_signature_has_user_prologue": "user_prologue" in inspect.signature(load_oolong_environment).parameters,
        "pairs_signature_has_user_prologue": "user_prologue" in inspect.signature(load_oolong_pairs_environment).parameters,
        "bcp_signature_has_user_prologue": "user_prologue" in inspect.signature(load_browsecomp_plus_environment).parameters,
        "lbv2_signature_has_user_prologue": "user_prologue" in inspect.signature(load_longbench_codeqa_environment).parameters,
    }
    for k, v in checks.items():
        print(f"{k}: {v}")
    if not all(checks.values()):
        print(f"FAILED (missing config env ids: {sorted(missing)})")
        return 1
    print("PASSED: eval-suite envs expose traced user_prologues and config references them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
