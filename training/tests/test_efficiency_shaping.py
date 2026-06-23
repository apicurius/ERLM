"""Invariant tests for the opt-in, correctness-gated efficiency-shaping reward.

These tests pin the THESIS.md guarantees:
- Parity:        shaping_coef=0 reproduces the upstream correctness-only reward.
- Gating:        the bonus only applies to correct, gate-passing rollouts.
- Dominance:     a correct rollout always scores >= an incorrect rollout.
- Monotonicity:  holding correctness fixed, more usage never pays.
- Boundedness:   R in [0, c*(1+lambda)].
"""

from __future__ import annotations

import asyncio

import pytest

from rlm_train import EfficiencyGatedRubric, RLMTrainRubric
from rlm_train.env import _record_sub_call
from rlm_train.shaping import EfficiencyAxis, default_axes, efficiency_score


def _state(*, iters=4, sub_calls=8, tokens=10_000, final="ans"):
    return {
        "rlm_iterations": iters,
        "rlm_sub_llm_calls": sub_calls,
        "rlm_sub_llm_tokens": tokens,
        "rlm_final_answer": final,
    }


def _main_func(rubric):
    # The first registered reward func is the main (correctness/shaped) reward.
    return rubric.funcs[0]


def _metric(rubric, name):
    for f in rubric.funcs:
        if getattr(f, "__name__", "") == name:
            return f
    raise KeyError(name)


async def _call(func, state):
    return float(await func(state=state, info={}))


def _correct(value):
    async def c(**kwargs):
        return value

    c.__name__ = "correctness"
    return c


# --- Parity -----------------------------------------------------------------


def test_parity_with_stock_rubric_when_coef_zero():
    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.0,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    st = _state()
    r_stock = asyncio.run(_call(_main_func(stock), st))
    r_shaped = asyncio.run(_call(_main_func(shaped), st))
    assert r_stock == r_shaped == 1.0


def test_parity_holds_for_partial_correctness():
    stock = RLMTrainRubric(correctness=_correct(0.5), min_iterations=2)
    shaped = EfficiencyGatedRubric(
        correctness=_correct(0.5),
        min_iterations=2,
        shaping_coef=0.5,
        subcall_budget=64.0,
    )
    st = _state()
    # 0.5 < correct_threshold(1.0) -> no bonus even with coef>0.
    assert asyncio.run(_call(_main_func(stock), st)) == 0.5
    assert asyncio.run(_call(_main_func(shaped), st)) == 0.5


# --- Gating -----------------------------------------------------------------


def test_no_bonus_for_incorrect_rollout():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        shaping_coef=0.5,
        subcall_budget=64.0,
    )
    # Cheap-but-wrong: zero usage would maximize e, but base=0 -> reward stays 0.
    st = _state(iters=1, sub_calls=0, tokens=0, final="wrong")
    assert asyncio.run(_call(_main_func(shaped), st)) == 0.0


def test_bonus_applied_for_correct_rollout():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        max_iterations=20,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    st = _state(iters=2, sub_calls=2, tokens=1000)
    r = asyncio.run(_call(_main_func(shaped), st))
    assert r > 1.0


def test_structural_caps_leave_correctness_only_rollouts_tied():
    """Caps stop runaway rollouts; they do not credit-assign thrift.

    Both states satisfy the configured ceilings used in the thesis configs:
    max_iterations=20, subcall_budget=64, token_budget=200k. Under the stock
    correctness-only reward they are indistinguishable even though the second
    rollout spends 6x turns, 18x calls, and >4x tokens.
    """

    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    cheap = _state(iters=3, sub_calls=3, tokens=35_000)
    expensive = _state(iters=18, sub_calls=54, tokens=160_000)
    assert asyncio.run(_call(_main_func(stock), cheap)) == 1.0
    assert asyncio.run(_call(_main_func(stock), expensive)) == 1.0


def test_efficiency_reward_separates_cap_satisfying_correct_rollouts():
    """The reward has operating room inside the structural caps."""

    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.2,
        max_iterations=20,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    cheap = _state(iters=3, sub_calls=3, tokens=35_000)
    expensive = _state(iters=18, sub_calls=54, tokens=160_000)
    cheap_reward = asyncio.run(_call(_main_func(shaped), cheap))
    expensive_reward = asyncio.run(_call(_main_func(shaped), expensive))
    assert cheap_reward > expensive_reward > 1.0
    assert cheap_reward == pytest.approx(1.1752083333)
    assert expensive_reward == pytest.approx(1.0304166667)


# --- Dominance --------------------------------------------------------------


def test_correct_always_beats_incorrect():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.9,
        subcall_budget=64.0,
    )
    wrong = EfficiencyGatedRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        shaping_coef=0.9,
        subcall_budget=64.0,
    )
    # Correct but maximally expensive vs. wrong but maximally cheap.
    r_correct = asyncio.run(
        _call(_main_func(shaped), _state(iters=20, sub_calls=64, tokens=200_000))
    )
    r_wrong = asyncio.run(_call(_main_func(wrong), _state(iters=1, sub_calls=0, tokens=0)))
    assert r_correct >= r_wrong


# --- Monotonicity -----------------------------------------------------------


def test_more_usage_never_pays():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        max_iterations=20,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    cheap = asyncio.run(_call(_main_func(shaped), _state(iters=2, sub_calls=2, tokens=1000)))
    mid = asyncio.run(_call(_main_func(shaped), _state(iters=8, sub_calls=16, tokens=50_000)))
    pricey = asyncio.run(_call(_main_func(shaped), _state(iters=18, sub_calls=60, tokens=190_000)))
    assert cheap >= mid >= pricey >= 1.0


# --- Boundedness ------------------------------------------------------------


def test_reward_is_bounded():
    coef = 0.5
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=coef,
        iteration_weight=0.0,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    # Iteration axis disabled; pass the gate (iters>=2) with zero sub-call/token
    # usage -> max efficiency on enabled axes -> reward == c*(1+coef).
    r = asyncio.run(_call(_main_func(shaped), _state(iters=2, sub_calls=0, tokens=0)))
    assert r == pytest.approx(1.0 * (1.0 + coef))


def test_reward_never_exceeds_bound_under_any_usage():
    coef = 0.5
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=coef,
        max_iterations=20,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    for iters, sub_calls, tokens in [(2, 0, 0), (5, 10, 5000), (20, 64, 200_000)]:
        r = asyncio.run(
            _call(_main_func(shaped), _state(iters=iters, sub_calls=sub_calls, tokens=tokens))
        )
        assert 1.0 <= r <= 1.0 * (1.0 + coef) + 1e-9


# --- efficiency_score helper ------------------------------------------------


def test_efficiency_score_disabled_when_no_axes():
    # No positive budgets -> all axes disabled -> efficiency 0 (safe no-op).
    axes = default_axes(max_iterations=0, subcall_budget=0.0, token_budget=0.0)
    assert efficiency_score(_state(), axes) == 0.0


def test_efficiency_axis_clamped_to_zero():
    axis = EfficiencyAxis("rlm_sub_llm_calls", budget=10.0)
    # Over-budget usage clamps to 0, not negative.
    assert axis.efficiency({"rlm_sub_llm_calls": 25}) == 0.0
    assert axis.efficiency({"rlm_sub_llm_calls": 0}) == 1.0


def test_negative_coef_rejected():
    with pytest.raises(ValueError):
        EfficiencyGatedRubric(correctness=_correct(1.0), shaping_coef=-0.1)


# --- Gate interaction (min_iterations) --------------------------------------


def test_bonus_requires_passing_iteration_gate():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=3,
        shaping_coef=0.5,
        subcall_budget=64.0,
    )
    # iters=1 < min_iterations=3 -> gate fails -> no bonus, base correctness kept.
    st = _state(iters=1, sub_calls=2, tokens=100)
    assert asyncio.run(_call(_main_func(shaped), st)) == 1.0


# --- Metric surface ---------------------------------------------------------


def test_efficiency_metrics_exposed():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        subcall_budget=64.0,
        token_budget=200_000.0,
    )
    names = {getattr(f, "__name__", "") for f in shaped.funcs}
    assert {
        "efficiency_bonus",
        "rlm_efficiency_score",
        "rlm_sub_llm_tokens",
        "rlm_sub_llm_usage_missing",
    } <= names
    bonus = _metric(shaped, "efficiency_bonus")
    st = _state(iters=2, sub_calls=2, tokens=1000)
    assert asyncio.run(_call(bonus, st)) > 0.0


def test_stock_rubric_exposes_token_telemetry_without_shaping():
    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    names = {getattr(f, "__name__", "") for f in stock.funcs}
    assert "rlm_sub_llm_tokens" in names
    assert "rlm_sub_llm_usage_missing" in names
    metric = _metric(stock, "rlm_sub_llm_tokens")
    assert asyncio.run(metric(state=_state(tokens=1234))) == 1234


def test_record_sub_call_accumulates_calls_and_usage_tokens():
    state = {"rlm_sub_llm_calls": 0, "rlm_sub_llm_tokens": 0, "rlm_sub_llm_usage_missing": 0}
    _record_sub_call(state, {"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    _record_sub_call(state, {"usage": {"total_tokens": 7}})
    _record_sub_call(state, {})
    assert state["rlm_sub_llm_calls"] == 3
    assert state["rlm_sub_llm_tokens"] == 22
    assert state["rlm_sub_llm_usage_missing"] == 1


def test_record_sub_call_counts_malformed_usage_as_missing():
    state = {"rlm_sub_llm_calls": 0, "rlm_sub_llm_tokens": 0, "rlm_sub_llm_usage_missing": 0}
    _record_sub_call(state, {"usage": {"total_tokens": "not-an-int"}})
    assert state["rlm_sub_llm_calls"] == 1
    assert state["rlm_sub_llm_tokens"] == 0
    assert state["rlm_sub_llm_usage_missing"] == 1
