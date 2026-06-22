"""Compatibility re-exports for the split RLM eval-suite environment modules."""

from rlm_eval_suite.browsecomp_plus import (
    BROWSECOMP_ANSWER_FORMAT,
    BROWSECOMP_GRADER_TEMPLATE,
    BROWSECOMP_PLUS_USER_PROLOGUE,
    load_browsecomp_plus_environment,
)
from rlm_eval_suite.longbench_codeqa import (
    LONGBENCH_CODE_DOMAIN,
    LONGBENCH_CODEQA_USER_PROLOGUE,
    load_longbench_codeqa_environment,
)
from rlm_eval_suite.oolong import (
    OOLONG_PAIRS_USER_PROLOGUE,
    OOLONG_USER_PROLOGUE,
    load_oolong_environment,
    load_oolong_pairs_environment,
)

__all__ = [
    "OOLONG_USER_PROLOGUE",
    "OOLONG_PAIRS_USER_PROLOGUE",
    "BROWSECOMP_ANSWER_FORMAT",
    "BROWSECOMP_GRADER_TEMPLATE",
    "BROWSECOMP_PLUS_USER_PROLOGUE",
    "LONGBENCH_CODE_DOMAIN",
    "LONGBENCH_CODEQA_USER_PROLOGUE",
    "load_oolong_environment",
    "load_oolong_pairs_environment",
    "load_browsecomp_plus_environment",
    "load_longbench_codeqa_environment",
]
