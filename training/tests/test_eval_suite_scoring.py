"""Unit tests for the deterministic scoring/parsing logic of the four
source-traced RLM eval-suite environments.

These cover the network-free helpers only (no dataset download): choice
extraction, OOLONG-Pairs F1, OOLONG-synth scoring, BrowseComp-Plus answer
extraction / canary decryption / judge parsing. They pin the behaviour the
async `_score_*` rubric functions depend on.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from rlm_eval_suite.browsecomp_plus import (
    _BCP_CANARY,
    _bcp_decrypt_string,
    _bcp_derive_key,
    _extract_exact_answer,
    _normalize_text,
    _parse_bcp_judge_correct,
    _score_browsecomp_plus,
)
from rlm_eval_suite.longbench_codeqa import _extract_choice_letter, _score_longbench_codeqa
from rlm_eval_suite.oolong import (
    COMPARISON_PHRASES,
    _find_comparison_phrase,
    _oolong_synth_score,
    _parse_pairs,
    _score_oolong,
    _score_oolong_pairs,
    _score_pairs,
)


def _run(coro):
    return asyncio.run(coro)


def _state(final):
    return {"rlm_final_answer": final}


# --- LongBench-v2 CodeQA: 4-choice MCQ letter extraction --------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is (C).", "C"),
        ("Answer: B", "B"),
        ("answer=D", "D"),
        ("I think it's A", "A"),
        ("C", "C"),
        ("nothing here", ""),
    ],
)
def test_extract_choice_letter(text, expected):
    assert _extract_choice_letter(text) == expected


def test_longbench_score_correct_and_wrong():
    info = json.dumps({"answer": "B"})
    assert _run(_score_longbench_codeqa(info, _state("Answer: B"))) == 1.0
    assert _run(_score_longbench_codeqa(info, _state("Answer: C"))) == 0.0


def test_longbench_score_rejects_non_letter_gold():
    # A malformed gold (not A-D) can never score 1.0.
    info = json.dumps({"answer": "E"})
    assert _run(_score_longbench_codeqa(info, _state("Answer: E"))) == 0.0


def test_longbench_score_accepts_dict_info():
    # info may arrive as a dict rather than a JSON string.
    assert _run(_score_longbench_codeqa({"answer": "A"}, _state("A"))) == 1.0


# --- OOLONG-Pairs: pair parsing + F1 ----------------------------------------


def test_parse_pairs_orders_low_first_and_dedupes():
    assert _parse_pairs("(3,1)\n(1, 3)\n(2,5)") == {(1, 3), (2, 5)}


def test_parse_pairs_handles_list_input():
    assert _parse_pairs(["(7,2)", "(4,4)"]) == {(2, 7), (4, 4)}


def test_score_pairs_exact_match():
    assert _score_pairs("(1,3)\n(2,5)", [[1, 3], [2, 5]]) == (1.0, 1.0, 1.0)


def test_score_pairs_empty_prediction_and_gold():
    assert _score_pairs("[]", []) == (1.0, 1.0, 1.0)


def test_score_pairs_partial_overlap_f1():
    # predicted {(1,3),(2,5)} vs gold {(1,3)}: precision .5, recall 1, f1 = 2/3.
    precision, recall, f1 = _score_pairs("(1,3)\n(2,5)", [[1, 3]])
    assert precision == 0.5
    assert recall == 1.0
    assert f1 == pytest.approx(2 / 3)


def test_score_pairs_strips_think_block():
    # <think> content must not leak phantom pairs into the prediction.
    p, r, f1 = _score_pairs("<think>(9,9)</think>(1,3)", [[1, 3]])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_score_oolong_pairs_async_f1():
    info = json.dumps({"answer": [[1, 3]]})
    assert _run(_score_oolong_pairs(info, _state("(1,3)"))) == 1.0
    assert _run(_score_oolong_pairs(info, _state("(2,4)"))) == 0.0


# --- OOLONG-synth scoring ---------------------------------------------------


def test_find_comparison_phrase():
    assert _find_comparison_phrase("they are more common now") == "more common"
    assert _find_comparison_phrase("no phrase here") is None
    assert set(COMPARISON_PHRASES) == {"more common", "less common", "same frequency"}


def test_oolong_exact_string_match():
    assert _oolong_synth_score({"answer": "['entity']"}, "Answer: entity") == 1.0


def test_oolong_comparison_phrase_match():
    assert _oolong_synth_score({"answer": "['more common']"}, "result is more common") == 1.0


def test_oolong_numeric_partial_credit_decays():
    meta = {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"}
    exact = _oolong_synth_score(meta, "Answer: 10")
    off_by_two = _oolong_synth_score(meta, "Answer: 12")
    assert exact == 1.0
    assert 0.0 < off_by_two < 1.0
    assert off_by_two == pytest.approx(0.75**2)


def test_oolong_wrong_answer_scores_zero():
    assert _oolong_synth_score({"answer": "['location']"}, "Answer: entity") == 0.0


def test_score_oolong_async():
    info = json.dumps({"answer": "['entity']"})
    assert _run(_score_oolong(info, _state("Answer: entity"))) == 1.0


# --- BrowseComp-Plus: answer extraction, decryption, judge parsing ----------


def test_extract_exact_answer():
    block = "Explanation: because reasons\nExact Answer: Marie Curie\nConfidence: 85%"
    assert _extract_exact_answer(block) == "Marie Curie"


def test_extract_exact_answer_falls_back_to_full_output():
    assert _extract_exact_answer("just a bare answer") == "just a bare answer"


def test_normalize_text_strips_punctuation_and_case():
    assert _normalize_text("New York, NY!") == "new york ny"


def test_bcp_decrypt_roundtrip():
    plain = "the quick brown fox"
    key = _bcp_derive_key(_BCP_CANARY, len(plain.encode()))
    enc = base64.b64encode(bytes(a ^ b for a, b in zip(plain.encode(), key, strict=True))).decode()
    assert _bcp_decrypt_string(enc) == plain


def test_bcp_decrypt_non_base64_returns_input():
    # Non-base64 strings (e.g. already-plaintext fields) pass through unchanged.
    assert _bcp_decrypt_string("not base64 !!!") == "not base64 !!!"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("reasoning: ok\ncorrect: yes\nconfidence: 90", True),
        ("correct: no", False),
        ('{"correct": "yes"}', True),
        ('{"correct": false}', False),
        ("no verdict line", False),
    ],
)
def test_parse_bcp_judge_correct(raw, expected):
    assert _parse_bcp_judge_correct(raw) is expected


def test_bcp_deterministic_score_containment():
    info = json.dumps({"answer": "Marie Curie"})
    good = "Explanation: ...\nExact Answer: Marie Curie\nConfidence: 90%"
    bad = "Explanation: ...\nExact Answer: Albert Einstein\nConfidence: 90%"
    assert _run(_score_browsecomp_plus(info, _state(good))) == 1.0
    assert _run(_score_browsecomp_plus(info, _state(bad))) == 0.0


def test_bcp_deterministic_score_empty_gold_is_zero():
    info = json.dumps({"answer": ""})
    assert _run(_score_browsecomp_plus(info, _state("Exact Answer: anything"))) == 0.0
