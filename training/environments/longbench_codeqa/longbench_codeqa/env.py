"""LongBench-v2 Code Repository Understanding QA (eval-suite port).

Source: THUDM/LongBench-v2 filtered to domain == "Code Repository
Understanding" (exactly 50 examples). 4-choice MCQ (gold = letter A/B/C/D), so
scoring is deterministic (no judge). The long repo context is exposed as the
REPL variable `context`; the model finalizes via answer["content"]/answer["ready"].
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

PLAN_HINT = "Plan before you act."

LONGBENCH_CODE_DOMAIN = "Code Repository Understanding"

LONGBENCH_CODEQA_USER_PROLOGUE = f"""{PLAN_HINT}

CONTEXT BUDGET (critical — your model window is only ~16k tokens): the `context` repository lives in the REPL and is far larger than your window. NEVER print, paste, or echo raw source code, files, or chunks into REPL output or your messages — REPL outputs are fed back to you and accumulate, so big prints overflow the window and waste the rollout. Print ONLY tiny results (a filename, a line count, the few snippets you need); keep every printed output under ~400 characters and every ```repl``` block short. Pass code regions as ARGUMENTS to `llm_query_batched` (never print them). Finish in as few turns as possible: plan -> grep/locate in Python -> one or two batched sub-LLM passes over candidate regions -> decide -> finalize.

Task-specific LongBench-v2 Code-Repository-QA guidance:
- The REPL variable `context` holds a long source-code repository (potentially many files concatenated). Do not paste it all into the REPL at once.
- This is a 4-choice multiple-choice question. Read the question and the four options (A, B, C, D) in the prompt.
- Strategy: search/grep the repo in Python for the relevant symbols/files, and use chunky `llm_query_batched` calls to analyze the candidate regions. Avoid one-subcall-per-file loops.
- Decide which single option is correct based on the code evidence.
- Final answer format: output ONLY the single letter of the correct choice: A, B, C, or D.
- When ready, set `answer["content"]` to that single letter and then `answer["ready"] = True`.
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


def _extract_choice_letter(output: str) -> str:
    text = str(output).strip()
    m = re.search(r"answer\s*[:=]?\s*\(?([ABCD])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([ABCD])\b", text.upper())
    return m2.group(1) if m2 else ""


async def _score_longbench_codeqa(info, state: vf.State, **_kw: Any) -> float:
    final = str(state.get("rlm_final_answer") or state.get("final_answer") or "")
    meta = json.loads(info) if isinstance(info, str) else info
    gold = str(meta.get("answer", "")).strip().upper()
    return 1.0 if gold in {"A", "B", "C", "D"} and _extract_choice_letter(final) == gold else 0.0


def _build_longbench_codeqa_dataset(
    *,
    num_examples: int = 50,
    seed: int = 42,
    shuffle: bool = False,
) -> Dataset:
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    rows = [dict(e) for e in ds if str(e.get("domain", "")) == LONGBENCH_CODE_DOMAIN]
    if shuffle:
        random.Random(seed).shuffle(rows)
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]

    out: list[dict[str, Any]] = []
    for i, s in enumerate(rows):
        question = str(s.get("question", ""))
        choices = (
            f"A. {s.get('choice_A', '')}\n"
            f"B. {s.get('choice_B', '')}\n"
            f"C. {s.get('choice_C', '')}\n"
            f"D. {s.get('choice_D', '')}"
        )
        context = str(s.get("context", ""))
        root_prompt = (
            "Answer the following multiple-choice question about the code repository in `context`. "
            "Respond with ONLY the letter (A, B, C, or D).\n\n"
            f"Question: {question}\n\n{choices}"
        )
        meta = {
            "id": str(s.get("_id", f"lbv2_code_{i}")),
            "answer": str(s.get("answer", "")),
            "sub_domain": s.get("sub_domain", ""),
            "difficulty": s.get("difficulty", ""),
            "context": context,
            "root_prompt": root_prompt,
            "source_env": "THUDM/LongBench-v2:Code Repository Understanding",
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
    num_examples: int = 50,
    seed: int = 42,
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
    dataset = _build_longbench_codeqa_dataset(num_examples=num_examples, seed=seed, shuffle=shuffle)
    rubric = _build_rubric(
        _score_longbench_codeqa,
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
        user_prologue=user_prologue or LONGBENCH_CODEQA_USER_PROLOGUE,
        **kwargs,
    )


__all__ = ["load_environment", "LONGBENCH_CODE_DOMAIN", "LONGBENCH_CODEQA_USER_PROLOGUE"]
