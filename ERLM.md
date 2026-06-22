# ERLM additions

This branch/repository is based on `alexzhang13/rlm` and adds public-source-traced RLM training/evaluation artifacts for efficient RLM work.

Key additions:

- `training/environments/` — local `RLMTrainEnv` ports of the four HF model-card
  eval environments with evidence-derived `user_prologue`s. Each environment is
  its own standalone verifiers package (one dir per env, mirroring the upstream
  `oolong/` layout: `pyproject.toml` + `<pkg>/__init__.py` + `<pkg>/env.py` +
  `README.md`, with a single `load_environment` entry point):
  - `oolong/` (env id `oolong`) — OOLONG synth (replaces the earlier simpler
    upstream `oolong` env; this is the fuller eval-suite port)
  - `oolong_pairs/` (env id `oolong_pairs`) — OOLONG-Pairs
  - `browsecomp_plus/` (env id `browsecomp_plus`) — BrowseComp-Plus
  - `longbench_codeqa/` (env id `longbench_codeqa`) — LongBench-v2 CodeQA
- `training/configs/rlm-qwen3-30b-efficient.toml` — upstream-only efficient config.
- `training/configs/rlm-qwen3-30b-efficient-eval-suite.toml` — train/eval config wired to all four HF-picture eval environments.
- `notes/` — source and paper trace notes, including HF README line-by-line findings, Alex/Prime research-environments commits, BrowseComp-Plus paper details, and claim audit.
- `tools/validate_*.py` — validators for upstream-only config and eval-suite environment wiring.

Important fidelity notes:

- BrowseComp-Plus for the small-model/HF-card setup is consistently `k=50` with judge scoring. The RLM paper's `k=1000` setting is a separate GPT-5 stress benchmark.
- HF README says exact per-env prologues are TBD; the prologues here are evidence-derived from PrimeIntellect research-environments, LMxLM, and the HF "Plan before you act" hint.
- Local experimental cost-shaping files in this worktree are intentionally not part of this public commit.

## Stronger thesis: correct-first, efficiency-second scaffold training

The released RLM scaffold is trained with a correctness-only reward (`R = c`),
so the objective never says whether the scaffold was operated *well*. `THESIS.md`
states a stronger, falsifiable claim — scaffold efficiency is a separable,
learnable objective that can be optimized on top of correctness without trading
it away — and this branch operationalizes it:

- `THESIS.md` — the thesis, the formal objective `R = c·(1 + λ·e)`, invariants,
  and the P1–P4 falsifiable predictions (A/B of control vs. treatment).
- `training/src/rlm_train/shaping.py` — `EfficiencyGatedRubric`, a strict
  superset of `RLMTrainRubric`. At `shaping_coef=0` it is byte-identical to
  `R = c`; at `shaping_coef>0` it adds a bounded, correctness-gated efficiency
  bonus over turns / sub-LLM calls / sub-LLM tokens.
- `training/src/rlm_train/env.py` — accumulates `rlm_sub_llm_tokens` as
  telemetry only (no effect on the default reward); the stock rubric exposes it
  as a zero-weight metric so unshaped control/eval runs can still report token
  cost.
- The four per-env loaders take opt-in `shaping_coef` / budget / weight
  kwargs; default behavior is unchanged (stock correctness-only rubric).
- `training/configs/rlm-qwen3-30b-thesis.toml` — **treatment** arm: equal
  prime-rl ratio split between OOLONG-Spam and BrowseComp-Plus, with
  `shaping_coef>0`.
- `training/configs/rlm-qwen3-30b-thesis-control.toml` — **control** arm: a
  faithful λ=0 twin of the treatment (identical model/LoRA/optim/caps/sampling/
  filters, the same OOLONG-Spam + BrowseComp-Plus 0.5/0.5 split, and the same
  unshaped four-env eval suite) with both `shaping_coef = 0.0`. This is the clean
  A/B counterpart — do **not** use the single-env `efficient-eval-suite.toml` as
  the control (it changes the training distribution). The
  "control == treatment except the shaping knobs" invariant is enforced by
  `tools/validate_thesis_shaping.py`.
  Verified against local prime-rl: `TrainSource` treats `ratio` as relative
  sampling weights, so `[0.5, 0.5]` and `[1.0, 1.0]` are equivalent equal
  splits when all train envs set `ratio`.
- Tests: `training/tests/test_efficiency_shaping.py`; validator:
  `tools/validate_thesis_shaping.py` (parity, gating, dominance, monotonicity,
  boundedness).

## Tests

- `training/tests/test_efficiency_shaping.py` — invariants of the opt-in
  correctness-gated efficiency reward (parity / gating / dominance /
  monotonicity / boundedness).
- `training/tests/test_eval_suite_scoring.py` — deterministic, network-free
  scoring/parsing of all four eval environments (LongBench-v2 letter
  extraction, OOLONG-Pairs F1, OOLONG-synth scoring, BrowseComp-Plus answer
  extraction / canary decryption / judge parsing).
- `training/tests/conftest.py` puts `rlm_train` and the four per-env eval-suite
  packages on `sys.path`, so the suite runs with a bare `pytest training/tests/`
  from the repo root (no manual `PYTHONPATH`).
