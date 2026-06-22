"""Opt-in, correctness-gated efficiency shaping for RLM scaffold training.

Why this exists
---------------
The upstream / released RLM scaffold was trained with a *correctness-only*
reward (``R = c``). That answers "can the model get the right answer?" but is
silent on "did it orchestrate the scaffold well?". The stronger thesis this
repo argues (see ``THESIS.md``) is:

    Scaffold *quality* is a separable, learnable objective. On top of a
    correctness-only policy, a correctness-gated process-efficiency reward can
    shape *how* an RLM uses its REPL and sub-LLM budget -- fewer wasted turns,
    fewer redundant sub-calls, fewer tokens -- WITHOUT trading away correctness.

``EfficiencyGatedRubric`` operationalizes that thesis as a strict superset of
the stock :class:`~rlm_train.rubric.RLMTrainRubric`:

* With ``shaping_coef == 0.0`` (the default), the reward is *byte-identical* to
  the upstream correctness-only reward. This is the control arm.
* With ``shaping_coef > 0.0``, a bounded efficiency bonus is added *only* to
  rollouts that are already correct (``value >= correct_threshold``) and that
  pass the same usage gates as the stock rubric. Efficiency therefore acts as a
  tie-breaker among correct rollouts; it can never make a wrong-but-cheap
  rollout outscore a right-but-expensive one.

The shaped reward is ``R = c * (1 + shaping_coef * e)`` where ``e in [0, 1]`` is
a normalized efficiency score over the enabled budget axes (turns, sub-LLM
calls, sub-LLM tokens). ``c`` always dominates; efficiency only reshapes the
ranking *within* the correct set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verifiers.types import State

from rlm_train.rubric import RLMTrainRubric


@dataclass(frozen=True)
class EfficiencyAxis:
    """One normalized-efficiency budget axis (fewer-is-better).

    ``state_key`` is read from rollout state; ``budget`` is the usage at which
    the axis contributes zero efficiency. Axis efficiency is
    ``max(0, 1 - used / budget)`` and is 1.0 at zero usage. A non-positive
    ``budget`` or ``weight`` disables the axis.
    """

    state_key: str
    budget: float
    weight: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.budget > 0.0 and self.weight > 0.0

    def efficiency(self, state: State) -> float:
        used = float(state.get(self.state_key) or 0.0)
        return max(0.0, 1.0 - used / self.budget)


def default_axes(
    *,
    max_iterations: int = 20,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
) -> list[EfficiencyAxis]:
    """Build the standard turns / sub-call / token efficiency axes.

    Only axes with a positive budget are enabled, so callers opt in per axis.
    The turns axis uses ``max_iterations`` as its budget by default because that
    bound always exists in the RLM harness.
    """

    return [
        EfficiencyAxis("rlm_iterations", float(max_iterations), iteration_weight),
        EfficiencyAxis("rlm_sub_llm_calls", float(subcall_budget), subcall_weight),
        EfficiencyAxis("rlm_sub_llm_tokens", float(token_budget), token_weight),
    ]


def efficiency_score(state: State, axes: list[EfficiencyAxis]) -> float:
    """Weighted-mean efficiency in ``[0, 1]`` over the enabled axes.

    Returns 0.0 when no axis is enabled, which makes the efficiency bonus a
    safe no-op (``R = c``) rather than an accidental free reward.
    """

    enabled = [a for a in axes if a.enabled]
    if not enabled:
        return 0.0
    total_weight = sum(a.weight for a in enabled)
    if total_weight <= 0.0:
        return 0.0
    score = sum(a.weight * a.efficiency(state) for a in enabled) / total_weight
    return max(0.0, min(1.0, score))


class EfficiencyGatedRubric(RLMTrainRubric):
    """Correctness-gated efficiency-shaping rubric.

    A strict superset of :class:`RLMTrainRubric`. With ``shaping_coef == 0.0``
    it reproduces the upstream correctness-only reward exactly; with
    ``shaping_coef > 0.0`` it adds a bounded efficiency bonus to correct,
    gate-passing rollouts only.
    """

    def __init__(
        self,
        correctness: Callable[..., float] | None = None,
        weight: float = 1.0,
        *,
        shaping_coef: float = 0.0,
        correct_threshold: float = 1.0,
        axes: list[EfficiencyAxis] | None = None,
        max_iterations: int = 20,
        subcall_budget: float = 0.0,
        token_budget: float = 0.0,
        iteration_weight: float = 1.0,
        subcall_weight: float = 1.0,
        token_weight: float = 1.0,
        **kwargs: Any,
    ):
        if shaping_coef < 0.0:
            raise ValueError("shaping_coef must be >= 0.0")
        self._shaping_coef = float(shaping_coef)
        self._correct_threshold = float(correct_threshold)
        self._axes = (
            axes
            if axes is not None
            else default_axes(
                max_iterations=max_iterations,
                subcall_budget=subcall_budget,
                token_budget=token_budget,
                iteration_weight=iteration_weight,
                subcall_weight=subcall_weight,
                token_weight=token_weight,
            )
        )
        super().__init__(correctness=correctness, weight=weight, **kwargs)
        if correctness is not None:
            self.add_metric(self.rlm_efficiency_score)
            self.add_metric(self._make_efficiency_bonus(correctness))

    @property
    def shaping_enabled(self) -> bool:
        return self._shaping_coef > 0.0

    def _base_value(self, state: dict, value: float) -> float:
        """Replicate the stock rubric's gated value (the upstream ``R``)."""

        if not self._gate_reward:
            return value
        if not self._passes_gates(state):
            return 0.0
        if value < self._min_reward:
            return 0.0
        return value

    def _shaped_value(self, state: dict, base: float) -> float:
        """Apply the correctness-gated efficiency bonus on top of ``base``."""

        if not self.shaping_enabled:
            return base
        if base < self._correct_threshold:
            return base
        if not self._passes_gates(state):
            return base
        e = efficiency_score(state, self._axes)
        return base * (1.0 + self._shaping_coef * e)

    def _make_main_correctness(self, correctness: Callable[..., float]) -> Callable[..., Any]:
        async def main(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            value = await self._call_correctness(correctness, kwargs)
            base = self._base_value(state, value)
            return self._shaped_value(state, base)

        main.__name__ = getattr(correctness, "__name__", "correctness")
        return main

    def _make_efficiency_bonus(self, correctness: Callable[..., float]) -> Callable[..., Any]:
        async def efficiency_bonus(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            value = await self._call_correctness(correctness, kwargs)
            base = self._base_value(state, value)
            shaped = self._shaped_value(state, base)
            return shaped - base

        efficiency_bonus.__name__ = "efficiency_bonus"
        return efficiency_bonus

    async def rlm_efficiency_score(self, state: State) -> float:
        return efficiency_score(state, self._axes)
