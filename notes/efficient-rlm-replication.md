# Efficient RLM — RL-training replication (upstream-only)

Scope: replicate RL-training configs for an **efficient RLM** using **only the
upstream `alexzhang13/rlm` code** (this clone is at `HEAD = 156fd72`). The
failure cases come from the RLM paper, arXiv:2512.24601 (v3), §5, §7, and
Appendix B ("Negative Results"). See `notes/source-claim-audit.md` for the
full claim-by-claim audit across X, GitHub, Hugging Face, and the paper.

## Why "upstream-only" matters here

This working tree contains **untracked local additions** that are *not* part of
upstream and are easy to mistake for the real harness:

- `configs/` (top-level `rl/`, `eval/`, `gepa/`) — local, not in upstream `rlm`.
- `training/configs/rlm-qwen3-30b-cost-v1.toml`, `...-cost-v2*.toml`,
  `...-stock-verify-a40*.toml` — local cost-shaping experiments.
- Local edits to `training/src/rlm_train/rubric.py`, `.../env.py`, `.../proxy.py`
  and `training/environments/oolong/oolong/env.py` adding `subcall_penalty_rate`,
  `free_context_reads`, `per_call_tokens`, round-trip metering, etc.
- `training/scripts/`, `training/tests/test_cost_accounting.py`,
  `training/REWARD-LFD.md`, `outputs/` — local campaign artifacts.

Verified against `git show HEAD:...`, **upstream ships exactly one RL config**
(`training/configs/rlm-qwen3-30b-example.toml`) and an **unshaped, correctness-
only** `RLMTrainRubric` (`R = c`, plus monitoring metrics; gating only via
`min_iterations` / `min_subcall` / `min_reward`). There is **no cost/efficiency
reward term upstream.** The replicated config therefore does not invent one.

## What "efficient RLM" actually means upstream

Important: this is a **public-repo-compatible** config, not a proven exact reconstruction of the released adapter run; X/HF mention a mixed suite including BC+/BrowseComp-Plus, but upstream currently publishes only the OOLONG environment/config.

The paper's efficiency gains from *training* (Obs. 6, Fig. 6: post-trained
RLM-Qwen3-8B is 3.2x–9.6x faster with 68–90% less runtime) come from the model
learning better decomposition + fewer errors — **not** from a cost penalty in
the reward. The upstream levers that produce/preserve efficiency are:

1. **Orchestrator addendum** (`rlm/utils/prompts.py:ORCHESTRATOR_ADDENDUM`,
   ON by default via `RLMTrainEnv(orchestrator=True)`). It explicitly forbids
   the paper's main inefficiency — "tiny-prompt mega-batches (hundreds or
   thousands of single-item prompts)" — and mandates "fat-prompt small batches"
   (~20-wide, ~100K chars each), plus "prefer batched over sequential loops".
2. **Non-thinking sampling + output cap** (`enable_thinking=false`,
   `max_completion_tokens=4096`, `sub_max_tokens=4096`).
3. **Bounded iterations** (`max_iterations=20`) + **per-REPL-block wall-clock
   guard** (`RLM_TRAIN_EXEC_TIMEOUT_S`, read in
   `training/src/rlm_train/worker.py`, default 600s; we set 180s).
4. **Premature-finalization telemetry** (`min_iterations=2`). Important: upstream `oolong.load_environment()` does not expose or set `gate_reward=True`, so this does not change the training reward; it only affects `gated_reward` / `rlm_below_min_iterations` metrics.

## Failure case (paper) -> upstream lever (this config)

| # | Paper failure case (loc) | Upstream lever used | Where |
|---|--------------------------|---------------------|-------|
| 1 | "Exploding sub-call costs" side-effect of the extra RLM layer (§7) | bounded `max_iterations` + per-block exec timeout + orchestrator batch-budget rules | `max_iterations=20`; `RLM_TRAIN_EXEC_TIMEOUT_S=180`; `ORCHESTRATOR_ADDENDUM` |
| 2 | "RLMs without asynchronous LM calls are slow" — blocking/sequential sub-calls dominate p95 runtime (Appx B; §F.2) | prefer-batched discipline + timeout early-stop on serial loops | `ORCHESTRATOR_ADDENDUM` ("prefer batched"); `RLM_TRAIN_EXEC_TIMEOUT_S` |
| 3 | Qwen3-Coder "uses thousands of recursive sub-calls" / a sub-LM call per line (Appx C, E.3) | addendum bans tiny-prompt mega-batches; filter-in-Python-first guidance | `ORCHESTRATOR_ADDENDUM` fan-out budget (~20 prompts, ~100K chars) |
| 4 | "Thinking models without sufficient output tokens struggle" — run out of output budget on thinking tokens (Appx B) | disable thinking + hard output cap | `enable_thinking=false`, `max_completion_tokens=4096` |
| 5 | "Distinguishing a final answer from a thought is brittle" — model returns its plan as the answer / premature FINAL (Appx B, E.2) | upstream-only config can only monitor this with gated metrics; actually changing the reward would require upstream code/config support for `gate_reward=True` | `min_iterations=2` telemetry (`gated_reward`, `rlm_below_min_iterations`) |
| 6 | "Models without sufficient coding capabilities struggle as RLMs" (Appx B) | replicate the 30B-A3B base (not 8B), matching the released adapter | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| 7 | RLM cost/runtime is long-tailed & high-variance; tail trajectories re-verify/re-generate many times (§F.2, E.2) | zero-advantage + repetition rollout filters; bounded turns | `[[orchestrator.filters]]` + `max_iterations` |

## Reward = upstream stock (R = c)

`RLMTrainRubric` (upstream `HEAD`) computes correctness only unless it is
constructed with `gate_reward=True`. The upstream OOLONG loader does **not** pass
`gate_reward=True`, nor does it expose that parameter in its config surface, so
`min_iterations` / `min_subcall` are monitoring thresholds rather than reward
gates for this config. The rubric exposes `rlm_iterations`, `rlm_repl_calls`,
`rlm_sub_llm_calls`, `rlm_has_final_answer`, `gated_reward`, and below-threshold
metrics as telemetry. The efficient behavior is therefore shaped by the prompt +
structural caps above, consistent with the paper, which never adds a cost term
to the training objective. (If you later want an explicit efficiency penalty or
true reward gating, that is a *new* design beyond upstream — out of scope here.)

## Hyperparameters anchored to upstream / the released model

- LoRA r=32, alpha=64, attention-only `q,k,v,o` projections — matches the
  released `mit-oasys/rlm-qwen3-30b-a3b-v0.1` adapter_config. (Upstream's
  *example* config additionally adapts MLP/expert projections; we follow the
  released-adapter parity instead, since that is the published efficient RLM.)
- `lr=5e-5`, `seq_len=8192`, `max_model_len=16384` — upstream example values.
- `batch_size=32`, `rollouts_per_example=8` — group-relative (GRPO-style) RL,
  matching the paper's RL setup (MRCRv2: batch 128, 4 rollouts/example, 150
  steps, max 20 iters, 4096 out-tok/turn — Appx A).
- Train = OOLONG `spam` 32k–65k (exclude numeric); eval = OOLONG `trec_coarse`
  @131k — held-out task family + larger context (length generalization, Obs. 6).

## Validation

```bash
python3 tools/validate_upstream_efficient_config.py
```

This script reads upstream files through `git show HEAD:...`, not through Python imports, so local working-tree experiments cannot contaminate the check. It verifies that every environment arg in the efficient config exists in upstream `oolong.load_environment`, that no local-only cost-shaping knobs appear, and that the config does not use schema sections absent from the upstream example config.

## How to run

```bash
# 1) Install prime-rl and wire this harness in (see training/README.md):
#    uv pip install -e . -e training -e training/environments/oolong
# 2) Launch with the runaway-loop guard set:
RLM_TRAIN_EXEC_TIMEOUT_S=180 uv run rl @ training/configs/rlm-qwen3-30b-efficient.toml
# (or use scripts/launch_efficient.sh)
```

## Sources

- Paper: arXiv:2512.24601 v3 — `papers/2512.24601-rlm.pdf` (local), §5/§7/Appx A,B,C,E,F.
- Tweet thread: `.firecrawl/a1zhang-rlm-thread.md` (training harness announcement).
- Released model card: `.firecrawl/hf-rlm-qwen3-30b-a3b-v0.1.md`.
- Upstream code: `git show HEAD:training/...` and `git show HEAD:rlm/utils/prompts.py`.
