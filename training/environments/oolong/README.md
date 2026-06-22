# oolong

OOLONG-synth long-context aggregate QA wired through `RLMTrainEnv`.

- **Env id:** `oolong`
- **Data:** `oolongbench/oolong-synth` (HF eval picture: `trec_coarse @132k (n=50)`;
  trained on the `spam` split, multi-env with `browsecomp_plus`).
- **Scoring:** official OOLONG-synth port (exact / comparison-phrase / numeric
  decay / date), deterministic — no judge.
- **Source trace:** PrimeIntellect-ai/research-environments `rlm_oolong` +
  LMxLM OOLONG task description; the "Plan before you act" hint from the HF card.

The long context is exposed as the REPL variable `context`; the model finalizes
by setting `answer["content"]` and `answer["ready"] = True`.

Opt-in efficiency shaping: `load_environment` accepts `shaping_coef` /
`correct_threshold` / `subcall_budget` / `token_budget` / `*_weight`. At
`shaping_coef=0` (default) the env uses the stock correctness-only
`RLMTrainRubric`; at `shaping_coef>0` it uses `EfficiencyGatedRubric`
(`R = c·(1 + λ·e)`). See repo-root `THESIS.md`.

> Replaces the earlier simpler upstream `oolong` environment; this is the fuller
> eval-suite port (official scoring + source-traced prologue + opt-in shaping).
