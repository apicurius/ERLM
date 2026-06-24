# A Correct-First, Efficiency-Second Reward for Recursive Language Models

> **Proposed title:** *A Correct-First, Efficiency-Second Reward for Recursive Language Models: Operationalizing $R = c\,(1 + \lambda e)$ for Efficient RLMs (ERLM)*

This thesis proposes, implements, and designs an experiment for a stronger training objective for Recursive Language Models (RLMs) [Zhang et al., 2026]: a strictly correctness-gated efficiency bonus $R = c\,(1 + \lambda e)$, in which task correctness $c \in [0,1]$ always dominates and a normalized process-efficiency term $e \in [0,1]$ acts only as a post-correctness tie-breaker. It contributes the formal objective and its invariants, a faithful drop-in implementation (`EfficiencyGatedRubric`), and a confound-free A/B experimental design (control $\lambda=0$ vs. treatment $\lambda=0.2$) at fixed model, data, and compute. The work is a **proposal + implementation + experimental design**; the A/B study is in-flight and has produced no completed result artifacts.

## Table of Contents (reading order)

| # | File | Contents |
|---|------|----------|
| — | [`00-abstract.md`](00-abstract.md) | Front matter, title page, and Abstract |
| 1 | [`01-introduction.md`](01-introduction.md) | Introduction and contributions |
| 2 | [`02-background.md`](02-background.md) | Background and related work (RLMs, CodeAct, GRPO/RLVR, reward shaping, reward hacking) |
| 3 | [`03-objective.md`](03-objective.md) | A correct-first, efficiency-second objective (formal: parity, dominance, monotonicity, boundedness) |
| 4 | [`04-method.md`](04-method.md) | Method and implementation (`EfficiencyGatedRubric`, telemetry, training stack) |
| 5 | [`05-experimental-design.md`](05-experimental-design.md) | Experimental design, falsifiable predictions, anti-reward-hacking protocol |
| 6 | [`06-results.md`](06-results.md) | Results — **PENDING**, placeholder tables only |
| 7 | [`07-discussion.md`](07-discussion.md) | Discussion |
| 8 | [`08-conclusion.md`](08-conclusion.md) | Conclusion and future work |
| — | [`references.md`](references.md) | References |

## Status

- **Results are placeholders.** Chapter 6 (`06-results.md`) contains only GFM table skeletons with em-dash / TBD cells and a `> **[RESULTS PENDING]**` callout. The A/B pilot (`pilot-l0.toml` control vs. `pilot-l02.toml` treatment) is an in-flight feasibility study with no completed result artifacts; no empirical accuracy, cost, or significance numbers appear anywhere.
- **Title-page fields must be filled.** The placeholders `{{AUTHOR}}`, `{{INSTITUTION}}`, `{{DEPARTMENT}}`, `{{ADVISOR}}`, `{{DEGREE}}`, and `{{DATE}}` in `00-abstract.md` are unset and must be completed by the author.
- The only concrete numbers permitted are config/hyperparameter values (e.g. $\lambda=0.2$, `subcall_budget=32`, `token_budget=200000`, `max_iterations=20`) and the labelled *illustrative* reward arithmetic in Chapter 3.

## Build

Concatenate the files in reading order into a PDF via `pandoc` (LaTeX/MathJax math, generated table of contents):

```bash
pandoc --toc --number-sections --mathjax \
  -V geometry:margin=1in \
  --pdf-engine=xelatex \
  00-abstract.md \
  01-introduction.md \
  02-background.md \
  03-objective.md \
  04-method.md \
  05-experimental-design.md \
  06-results.md \
  07-discussion.md \
  08-conclusion.md \
  references.md \
  -o thesis.pdf
```

## How this maps to the repo

Every technical claim is grounded in the ERLM fork (`github.com/apicurius/ERLM`, forked from `alexzhang13/rlm`). Key artifacts:

| Thesis element | Repo file |
|----------------|-----------|
| The objective $R = c\,(1+\lambda e)$ and `EfficiencyGatedRubric` | `training/src/rlm_train/shaping.py` |
| Treatment arm config ($\lambda=0.2$) | `training/configs/rlm-qwen3-30b-thesis.toml` |
| Control arm config ($\lambda=0.0$, faithful twin) | `training/configs/rlm-qwen3-30b-thesis-control.toml` |
| A/B parity + invariant validator (parity, gating, dominance, monotonicity, boundedness) | `tools/validate_thesis_shaping.py` |
| Eval environments (unshaped) | `training/environments/{oolong,oolong_pairs,browsecomp_plus,longbench_codeqa}/` |
| Pilot configs (A40-tuned feasibility study) | `training/configs/pilot-l0.toml`, `training/configs/pilot-l02.toml` |

See Chapter 4 for the training stack (`RLMTrainEnv`, `prime-rl` GRPO, LoRA hot-loading) and Chapter 5 for the launch and analysis tooling (`scripts/valar_pilot.sh`, `tools/analyze_pilot.py`).
