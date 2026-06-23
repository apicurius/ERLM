"""OOLONG synth long-context QA (eval-suite port), wired through RLMTrainEnv.

Source-traced from PrimeIntellect-ai/research-environments `rlm_oolong` and the
LMxLM OOLONG task description. The long context is exposed as the REPL variable
`context`; the model finalizes via answer["content"]/answer["ready"]. Scoring is
byte-faithful to the upstream alexzhang13/rlm OOLONG-synth scorer (the same
yardstick that produced the HF eval-picture numbers), NOT the stricter
abertsch72 "official" metric.
"""

from __future__ import annotations

import ast
import json
import random
from datetime import datetime
from typing import Any

import dateutil.parser
import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

PLAN_HINT = "Plan before you act."

# Suffixed forms, byte-faithful to the upstream alexzhang13/rlm OOLONG scorer
# (training/environments/oolong/oolong/env.py:16). The released HF eval-picture
# numbers were produced by this upstream training-env scorer (the repo reuses
# the env for eval), so matching it exactly keeps our OOLONG accuracy on the
# same yardstick as the model card and avoids over-grading bare comparison
# phrases (which the shaped reward could otherwise exploit).
COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")
OOLONG_LABELS = (
    "numeric value",
    "entity",
    "location",
    "description and abstract concept",
    "abbreviation",
    "human being",
)

OOLONG_USER_PROLOGUE = f"""{PLAN_HINT}

CONTEXT BUDGET (critical — your model window is only ~16k tokens): the full `context` lives in the REPL and is far larger than your window. NEVER print, paste, or echo raw context, lines, or chunks into REPL output or your messages — REPL outputs are fed back to you and accumulate, so big prints overflow the window and waste the rollout. Print ONLY tiny results (counts, the few values you need); keep every printed output under ~400 characters and every ```repl``` block short. Pass chunks as ARGUMENTS to `llm_query_batched` (never print them). Finish in as few turns as possible: plan -> one or two batched sub-LLM passes -> aggregate in Python -> finalize.

Task-specific OOLONG guidance (adapted from PrimeIntellect-ai/research-environments rlm_oolong and LMxLM OOLONG):
- The long OOLONG context is available as the REPL variable `context`; do not print or paste it all at once.
- The context contains thousands of general-knowledge questions, one per line. Each line has a User ID and a question. Each question's answer belongs to one of these six labels: {', '.join(repr(x) for x in OOLONG_LABELS)}. The labels are not explicitly present; infer them from semantics.
- For long-context retrieval/aggregation, inspect a small sample, split the context into chunky windows, build focused prompts, and use `llm_query_batched` to scan chunks in parallel. Avoid one-subcall-per-line loops.
- Aggregate the chunk findings in Python. Keep final answers short: a single token / word / date / label / comparison phrase / integer as required by the question.
- When ready, set `answer["content"]` to ONLY the final answer and then `answer["ready"] = True`.
"""


def _build_rubric(
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
    """Stock correctness-only rubric unless opt-in efficiency shaping is enabled."""

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


def _find_comparison_phrase(output: str) -> str | None:
    out_low = output.lower()
    hits = [(out_low.rfind(p), p) for p in COMPARISON_PHRASES if p in out_low]
    return max(hits)[1] if hits else None


def _synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    # Byte-faithful to upstream alexzhang13/rlm `_attempt_answer_parse`: comparison
    # phrase first, then colon/length heuristics. No quote-stripping or second
    # comparison check (those were ERLM additions that over-grade vs upstream).
    cmp = _find_comparison_phrase(answer)
    if cmp is not None:
        return cmp, "high"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, "low"
        return answer.split()[-1], "low"
    candidate = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    if len(candidate) < 20:
        return candidate, "vhigh"
    return candidate, "med"


def _oolong_synth_score(datapoint: dict[str, Any], output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    trimmed, _ = _synth_attempt_answer_parse(output)
    gold_s = str(gold)

    if str(trimmed) == gold_s or str(trimmed).lower() == gold_s.lower():
        return 1.0

    atype = str(datapoint.get("answer_type", ""))
    if atype == "ANSWER_TYPE.NUMERIC":
        try:
            return float(0.75 ** abs(int(gold) - int(trimmed)))
        except Exception:
            return 0.0
    if atype == "ANSWER_TYPE.DATE":
        try:
            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except Exception:
            return 0.0

    # Last-resort official-ish substring fallback for non-comparison exact strings.
    if gold_s and gold_s.lower() not in [p.lower() for p in COMPARISON_PHRASES]:
        if gold_s.lower() in output.lower():
            return 1.0
    return 0.0


async def _score_oolong(info, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    meta = json.loads(info) if isinstance(info, str) else info
    return _oolong_synth_score(meta, str(final))


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return [x]
    return list(x)


def _build_oolong_dataset(
    *,
    dataset_name: str | list[str] = "trec_coarse",
    context_len: int | list[int] = 131072,
    num_examples: int = -1,
    seed: int = 42,
    filter_numerical: bool = True,
    split: str = "validation",
    shuffle: bool = False,
) -> Dataset:
    names = {str(x) for x in _as_list(dataset_name)}
    lens = {int(x) for x in _as_list(context_len)}
    ds = load_dataset("oolongbench/oolong-synth", split=split)
    rows: list[dict[str, Any]] = []
    for ex in ds:
        e = dict(ex)
        if names and str(e.get("dataset", "")) not in names:
            continue
        if lens and int(e.get("context_len", 0)) not in lens:
            continue
        if filter_numerical and e.get("answer_type") == "ANSWER_TYPE.NUMERIC":
            continue
        rows.append(e)
    if shuffle:
        random.Random(seed).shuffle(rows)
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]

    out: list[dict[str, Any]] = []
    for i, s in enumerate(rows):
        question = str(s["question"])
        context = str(s.get("context_window_text", s.get("context", "")))
        root_prompt = (
            "Answer the following OOLONG aggregate question over the long context. "
            "Do not guess, estimate, or approximate; compute from the context.\n\n"
            f"Question: {question}"
        )
        meta = {
            "id": s.get("id", f"oolong_{i}"),
            "dataset": s.get("dataset", ""),
            "answer_type": s.get("answer_type", ""),
            "answer": str(s.get("answer", "")),
            "context_len": s.get("context_len", 0),
            "context": context,
            "root_prompt": root_prompt,
            "source_env": "PrimeIntellect-ai/research-environments:rlm_oolong + LMxLM:oolong",
        }
        out.append(
            {
                "example_id": i,
                "prompt": [{"role": "user", "content": question}],
                "answer": str(s.get("answer", "")),
                "info": json.dumps(meta),
            }
        )
    return Dataset.from_list(out)


def load_environment(
    *,
    dataset_name: str | list[str] = "trec_coarse",
    context_len: int | list[int] = 131072,
    num_examples: int = -1,
    seed: int = 42,
    filter_numerical: bool = True,
    split: str = "validation",
    shuffle: bool = False,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    user_prologue: str | None = None,
    shaping_coef: float = 0.0,
    correct_threshold: float = 1.0,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_oolong_dataset(
        dataset_name=dataset_name,
        context_len=context_len,
        num_examples=num_examples,
        seed=seed,
        filter_numerical=filter_numerical,
        split=split,
        shuffle=shuffle,
    )
    rubric = _build_rubric(
        _score_oolong,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
        max_iterations=max_iterations,
        shaping_coef=shaping_coef,
        correct_threshold=correct_threshold,
        subcall_budget=subcall_budget,
        token_budget=token_budget,
        iteration_weight=iteration_weight,
        subcall_weight=subcall_weight,
        token_weight=token_weight,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        user_prologue=user_prologue or OOLONG_USER_PROLOGUE,
        **kwargs,
    )


__all__ = ["load_environment", "OOLONG_USER_PROLOGUE"]
