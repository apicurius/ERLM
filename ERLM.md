# ERLM additions

This branch/repository is based on `alexzhang13/rlm` and adds public-source-traced RLM training/evaluation artifacts for efficient RLM work.

Key additions:

- `training/environments/rlm_eval_suite/` — local `RLMTrainEnv` ports of the four HF model-card eval environments with evidence-derived `user_prologue`s:
  - `rlm-oolong-local`
  - `rlm-oolong-pairs-local`
  - `rlm-browsecomp-plus-local`
  - `rlm-longbench-codeqa-local`
- `training/configs/rlm-qwen3-30b-efficient.toml` — upstream-only efficient config.
- `training/configs/rlm-qwen3-30b-efficient-eval-suite.toml` — train/eval config wired to all four HF-picture eval environments.
- `notes/` — source and paper trace notes, including HF README line-by-line findings, Alex/Prime research-environments commits, BrowseComp-Plus paper details, and claim audit.
- `tools/validate_*.py` — validators for upstream-only config and eval-suite environment wiring.

Important fidelity notes:

- BrowseComp-Plus for the small-model/HF-card setup is consistently `k=50` with judge scoring. The RLM paper's `k=1000` setting is a separate GPT-5 stress benchmark.
- HF README says exact per-env prologues are TBD; the prologues here are evidence-derived from PrimeIntellect research-environments, LMxLM, and the HF "Plan before you act" hint.
- Local experimental cost-shaping files in this worktree are intentionally not part of this public commit.
