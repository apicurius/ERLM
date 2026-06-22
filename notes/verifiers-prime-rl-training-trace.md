# Verifiers + prime-rl training behavior trace

Date: 2026-06-22. Purpose: pin the package behavior that matters for ERLM's
multi-env scaffold-training thesis config.

## Sources inspected

- Installed `verifiers` in `.venv/lib/python3.11/site-packages/verifiers`.
- Local `prime-rl` checkout at `/Users/oerdogan/rlm-scratch/prime-rl`:
  - runtime: `src/prime_rl/orchestrator/*`
  - config schemas: `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py`

## Verifiers environment loading

`verifiers.utils.env_utils.load_environment(env_id, **env_args)`:

- Converts env id to an import module name by replacing hyphens with underscores
  and taking the last path segment: `module_name = env_id.replace("-", "_").split("/")[-1]`.
- Imports that module and requires it to expose `load_environment(...)`.
- Calls `load_environment(**env_args)` and records `env_instance.env_id/env_args`.

Implication for ERLM:

- We preserve verifiers env IDs (`rlm-oolong-local`, `rlm-browsecomp-plus-local`,
  etc.) via package entry points, but the package now points those IDs directly
  to separated modules:
  - `rlm_eval_suite.oolong:load_oolong_environment`
  - `rlm_eval_suite.oolong:load_oolong_pairs_environment`
  - `rlm_eval_suite.browsecomp_plus:load_browsecomp_plus_environment`
  - `rlm_eval_suite.longbench_codeqa:load_longbench_codeqa_environment`
- `rlm_eval_suite/envs.py` remains as a compatibility re-export layer for older
  imports and validators.

## Verifiers reward / advantage behavior

`verifiers.rubrics.rubric.Rubric.score_group(states)`:

- Calls each reward function for each rollout.
- Aggregates weighted reward scores into `aggregated_rewards`.
- Computes group-relative advantage as `state["advantage"] = reward - avg_reward`.
- Writes reward/advantage into trajectory steps.

Implication for the thesis:

- Correctness-only `R=c` provides no advantage among equally-correct rollouts.
- `EfficiencyGatedRubric` gives lower-cost correct rollouts a slightly higher
  reward, creating a GRPO-compatible advantage signal among correct trajectories.
- Because the bonus is gated on correctness, cheap wrong rollouts still receive
  no efficiency advantage.

## prime-rl multi-env sampling

`prime_rl.orchestrator.train_source.TrainSource`:

- Builds a row buffer for every train env.
- Adds `env_name` to each example.
- If every env config has `ratio`, uses those values as `random.choices(..., weights=...)`.
- If ratios are not all set, falls back to per-env dataset sizes.
- Cursor exhaustion reshuffles an env's rows and continues indefinitely.

`prime_rl.configs.orchestrator.EnvConfig.ratio`:

- `ratio: float | None = Field(None, gt=0)`.
- Documentation says values are relative weights normalized to probabilities;
  `[1, 1]` and `[0.5, 0.5]` are equivalent equal splits.
- When set, it should be set on all envs.

Implication for ERLM:

- `training/configs/rlm-qwen3-30b-thesis.toml` uses two train envs:
  - `rlm-oolong-local`, `ratio = 0.5`
  - `rlm-browsecomp-plus-local`, `ratio = 0.5`
- This is an equal OOLONG-Spam / BrowseComp-Plus split, matching the public
  description of Alex's released run better than OOLONG-only.
- The exact private BC+ split/sampling weights remain unknown; this config is a
  public-source-compatible split, not an exact private-run reconstruction.

## prime-rl env wrapper behavior

`prime_rl.orchestrator.envs.Env`:

- Wraps `vf.Environment` loaded by `vf.load_environment(config.stripped_id, **config.args)`.
- Spawns a ZMQ env server per env unless `address` is provided.
- `TrainEnv.get_dataset(seed)` delegates to the underlying verifiers env.
- `EvalEnv` samples from `env.get_eval_dataset(n=config.num_examples)`.

Implication:

- The separate env modules are loaded independently by verifiers/prime-rl.
- Train/eval env names must be unique; ERLM configs set explicit `name` values.

## Verified ERLM checks

- `tools/validate_rlm_eval_suite.py` checks split modules exist and pyproject
  entry points target them directly.
- `tools/validate_thesis_shaping.py` checks:
  - thesis config trains exactly OOLONG + BrowseComp+;
  - all train envs set positive ratios;
  - ratios are equal `[0.5, 0.5]`;
  - all train envs enable `shaping_coef`;
  - eval includes all four envs and is unshaped;
  - reward invariants still hold (`λ=0` parity, correctness gating, dominance,
    cheaper-correct monotonicity).
