"""Shared helpers for source-traced RLM eval-suite environments."""

from __future__ import annotations

from typing import Any

import verifiers as vf

import rlm_train

PLAN_HINT = "Plan before you act."


def build_rubric(
    correctness: Any,
    *,
    min_iterations: int,
    min_subcall: int,
    max_iterations: int,
    shaping_coef: float = 0.0,
    correct_threshold: float = 1.0,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
) -> vf.Rubric:
    """Return stock correctness-only rubric unless efficiency shaping is enabled."""

    if shaping_coef and shaping_coef > 0.0:
        return rlm_train.EfficiencyGatedRubric(
            correctness=correctness,
            weight=1.0,
            min_iterations=min_iterations,
            min_subcall=min_subcall,
            shaping_coef=shaping_coef,
            correct_threshold=correct_threshold,
            max_iterations=max_iterations,
            subcall_budget=subcall_budget,
            token_budget=token_budget,
            iteration_weight=iteration_weight,
            subcall_weight=subcall_weight,
            token_weight=token_weight,
        )
    return rlm_train.RLMTrainRubric(
        correctness=correctness,
        weight=1.0,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
    )
