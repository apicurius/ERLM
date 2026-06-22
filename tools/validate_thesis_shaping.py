#!/usr/bin/env python3
"""Validate the stronger-thesis efficiency-shaping layer (THESIS.md).

Proves, without GPUs or network, that:
- THESIS.md exists and states R = c.(1+lambda.e) and the parity claim.
- The thesis config exists, trains on OOLONG-Spam + BrowseComp-Plus as a
  multi-env split, sets ratios on all train envs, leaves evals unshaped, and
  toggles the shaping reward via `shaping_coef`.
- `EfficiencyGatedRubric` is exported and reproduces R = c when shaping_coef=0,
  applies a bonus only to correct rollouts, and never lets a wrong rollout beat
  a correct one (correctness dominance).
- The eval-suite loaders expose the opt-in shaping kwargs.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "training/environments"
sys.path[:0] = [
    str(ENVS / "oolong"),
    str(ENVS / "oolong_pairs"),
    str(ENVS / "browsecomp_plus"),
    str(ENVS / "longbench_codeqa"),
    str(ROOT / "training/src"),
    str(ROOT),
]

import browsecomp_plus.env as _bcp  # noqa: E402
import longbench_codeqa.env as _lbv2  # noqa: E402
import oolong.env as _ool  # noqa: E402
import oolong_pairs.env as _pairs  # noqa: E402
from rlm_train import EfficiencyGatedRubric, RLMTrainRubric  # noqa: E402

load_oolong_environment = _ool.load_environment
load_oolong_pairs_environment = _pairs.load_environment
load_browsecomp_plus_environment = _bcp.load_environment
load_longbench_codeqa_environment = _lbv2.load_environment

THESIS = ROOT / "THESIS.md"
CONFIG = ROOT / "training/configs/rlm-qwen3-30b-thesis.toml"


def _correct(value: float):
    async def c(**kwargs):
        return value

    c.__name__ = "correctness"
    return c


def _main(rubric):
    return rubric.funcs[0]


async def _call(func, state):
    return float(await func(state=state, info={}))


def _state(iters=4, sub_calls=8, tokens=10_000, final="ans"):
    return {
        "rlm_iterations": iters,
        "rlm_sub_llm_calls": sub_calls,
        "rlm_sub_llm_tokens": tokens,
        "rlm_final_answer": final,
    }


def main() -> int:
    checks: dict[str, bool] = {}

    thesis_text = THESIS.read_text() if THESIS.exists() else ""
    checks["thesis_file_exists"] = THESIS.exists()
    checks["thesis_states_objective"] = "R = c" in thesis_text and "(1 + λ·e)" in thesis_text
    checks["thesis_states_parity"] = "byte-identical" in thesis_text
    checks["thesis_has_falsification"] = "falsified" in thesis_text.lower()

    cfg = tomllib.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    train_envs = cfg.get("orchestrator", {}).get("train", {}).get("env", [])
    eval_envs = cfg.get("orchestrator", {}).get("eval", {}).get("env", [])
    train_ids = {e.get("id") for e in train_envs}
    eval_ids = {e.get("id") for e in eval_envs}
    checks["config_exists"] = CONFIG.exists()
    checks["config_uses_eval_suite_env"] = bool(
        train_ids
        & {
            "oolong",
            "oolong_pairs",
            "browsecomp_plus",
            "longbench_codeqa",
        }
    )
    checks["config_trains_oolong_and_bcp"] = train_ids == {
        "oolong",
        "browsecomp_plus",
    }
    checks["config_train_ratios_all_set"] = bool(train_envs) and all(
        float(e.get("ratio", 0.0)) > 0.0 for e in train_envs
    )
    checks["config_train_ratio_equal_split"] = sorted(float(e.get("ratio", 0.0)) for e in train_envs) == [
        0.5,
        0.5,
    ]
    checks["config_sets_shaping_coef_on_all_train_envs"] = bool(train_envs) and all(
        float(e.get("args", {}).get("shaping_coef", 0.0)) > 0.0 for e in train_envs
    )
    checks["config_eval_includes_all_four_envs"] = eval_ids == {
        "oolong",
        "oolong_pairs",
        "browsecomp_plus",
        "longbench_codeqa",
    }
    checks["config_eval_unshaped"] = bool(eval_envs) and not any(
        "shaping_coef" in e.get("args", {}) for e in eval_envs
    )

    # Loaders expose the opt-in knob.
    for name, fn in [
        ("oolong", load_oolong_environment),
        ("oolong_pairs", load_oolong_pairs_environment),
        ("browsecomp_plus", load_browsecomp_plus_environment),
        ("longbench_codeqa", load_longbench_codeqa_environment),
    ]:
        params = inspect.signature(fn).parameters
        checks[f"loader_{name}_has_shaping_coef"] = "shaping_coef" in params

    # Parity: shaping_coef=0 == stock R=c.
    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    off = EfficiencyGatedRubric(
        correctness=_correct(1.0), min_iterations=2, shaping_coef=0.0, subcall_budget=64.0
    )
    st = _state()
    r_stock = asyncio.run(_call(_main(stock), st))
    r_off = asyncio.run(_call(_main(off), st))
    checks["parity_coef_zero_equals_stock"] = r_stock == r_off == 1.0
    checks["stock_exposes_token_metric"] = "rlm_sub_llm_tokens" in {
        getattr(f, "__name__", "") for f in stock.funcs
    }

    # Bonus only for correct rollouts.
    on_correct = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        max_iterations=20,
        subcall_budget=64.0,
    )
    on_wrong = EfficiencyGatedRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        shaping_coef=0.5,
        max_iterations=20,
        subcall_budget=64.0,
    )
    r_correct = asyncio.run(_call(_main(on_correct), _state(iters=2, sub_calls=2, tokens=500)))
    r_wrong_cheap = asyncio.run(_call(_main(on_wrong), _state(iters=1, sub_calls=0, tokens=0)))
    checks["bonus_only_when_correct"] = r_correct > 1.0 and r_wrong_cheap == 0.0
    checks["correctness_dominates"] = r_correct >= r_wrong_cheap

    # Monotonic: cheaper correct rollout scores higher.
    cheap = asyncio.run(_call(_main(on_correct), _state(iters=2, sub_calls=2, tokens=500)))
    pricey = asyncio.run(_call(_main(on_correct), _state(iters=18, sub_calls=60, tokens=190_000)))
    checks["cheaper_scores_higher"] = cheap >= pricey >= 1.0

    for k, v in checks.items():
        print(f"{k}: {v}")
    if not all(checks.values()):
        print("FAILED: thesis shaping layer did not satisfy all invariants.")
        return 1
    print("PASSED: stronger-thesis efficiency shaping is opt-in, gated, and parity-safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
