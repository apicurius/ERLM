# 5. Experimental Design

This chapter specifies the empirical apparatus that turns the correct-first, efficiency-second objective of Chapter 3 and its implementation in Chapter 4 into a falsifiable scientific test. The central design commitment is that the experiment must isolate the *training reward* as the single independent variable while holding every other factor — base model, adapter, optimizer, sampling, structural caps, training distribution, and evaluation protocol — fixed. The objective under test is

$$R = c\,(1 + \lambda e),$$

where $c \in [0,1]$ is gated task correctness, $e \in [0,1]$ is normalized process-efficiency, and $\lambda$ (the config key `shaping_coef`) is the shaping coefficient. The experiment is a paired A/B comparison between a **treatment** arm at $\lambda = 0.2$ and a **control** arm at $\lambda = 0$, the latter being byte-identical in reward to the upstream correctness-only objective $R = c$. Because the A/B run is in-flight and has produced no completed result artifacts, this chapter describes the *design and instrumentation*; all empirical tables in Chapter 6 are skeletons. No outcome is reported here as fact.

The design proceeds from hypotheses (Section 5.1) to the paired protocol that makes them testable (Section 5.2), the model and training configuration (Section 5.3), the training distribution (Section 5.4), the unshaped evaluation suite (Section 5.5), the metrics and their mapping to the analysis tooling (Section 5.6), the anti-reward-hacking protocol and $\lambda$ sweep (Section 5.7), the feasibility pilot (Section 5.8), reproduction instructions (Section 5.9), and threats to validity (Section 5.10).

## 5.1 Hypotheses and predictions

The thesis advances four falsifiable predictions, each tied to a quantity the evaluation suite measures directly. They are stated as one-sided claims about the *difference* between the treatment and control arms, under fixed model, data, compute, and evaluation.

**P1 — No correctness regression.** The shaping term is a strict post-correctness tie-breaker, so adding it must not degrade task accuracy. Formally, letting $A_{\text{trt}}$ and $A_{\text{ctrl}}$ be mean accuracy on the held-out eval suite, P1 predicts

$$A_{\text{trt}} \ge A_{\text{ctrl}} - \varepsilon,$$

for a small tolerance $\varepsilon$ absorbing sampling noise. P1 is the *guardrail*: a method that buys cheaper trajectories at the cost of correctness has failed, regardless of any efficiency gain. The measurable quantity is per-environment and pooled accuracy (Section 5.6).

**P2 — Cheaper trajectories at equal accuracy (a Pareto improvement).** At accuracy no worse than control, the treatment is predicted to use *strictly fewer* resources on at least one cost axis: mean turns (`rlm_iterations`), mean sub-LLM calls (`rlm_sub_llm_calls`), or mean sub-LLM tokens (`rlm_sub_llm_tokens`). The measurable quantities are the per-axis arithmetic means; the claim is a movement of the accuracy-vs-cost operating point toward the lower-cost frontier.

**P3 — Lower tail cost.** Because RLM cost is reported as long-tailed and high-variance — tail trajectories re-verify and re-generate many times [Zhang et al., 2026] (paper §F.2) — P3 targets the upper tail specifically: the treatment is predicted to shrink the $p_{95}$ of turns and of sub-LLM tokens. The measurable quantities are the 95th percentiles of the per-axis cost distributions over the eval rollouts.

**P4 — Generalization.** The efficiency gain is predicted to transfer beyond the training conditioning to a held-out task family and to a larger context window, mirroring the paper's length-generalization finding (Observation 6, where a post-trained RLM is reported 3.2$\times$–9.6$\times$ faster) [Zhang et al., 2026]. The measurable quantity is the cost reduction on evaluation environments whose distribution differs from the training mix — in particular OOLONG `trec_coarse` at 131072 tokens (the training OOLONG draws context in $[32768, 65536]$) and the `longbench_codeqa` environment, which is absent from training entirely.

The conjunction is falsifiable. The thesis is **falsified** if, across a reasonable range of $\lambda$, the treatment cannot beat control on P2/P3 *without* violating P1 — that is, if every operating point that reduces cost also reduces accuracy beyond tolerance, so that no genuine Pareto improvement exists.

## 5.2 The paired A/B protocol

The experiment is a paired comparison between two training runs that differ in exactly one knob. Treatment uses `rlm-qwen3-30b-thesis.toml` ($\lambda = 0.2$); control uses `rlm-qwen3-30b-thesis-control.toml` ($\lambda = 0$). The control is a *faithful twin*: it shares the same base model, LoRA adapter shape, optimizer, token and iteration caps, sampling settings, pre-batch filters, training-split definition, and evaluation suite. At $\lambda = 0$ the `EfficiencyGatedRubric` collapses to the stock `RLMTrainRubric`, so the control's reward is byte-identical to the upstream correctness-only $R = c$.

**Validator-enforced single-knob difference.** That the two configs differ in only the reward knob is not left to manual inspection; it is mechanically enforced by `tools/validate_thesis_shaping.py`. The validator flattens both TOML files and asserts that the set of differing keys is a *subset* of an explicitly enumerated allow-list, `_CONTROL_ALLOWED_DIFFS`: `output_dir`, `wandb.name`, the LoRA adapter name (`orchestrator.model.lora.name`), the two train-environment names (`orchestrator.train.env[0-1].name`), and the two `shaping_coef` values (`orchestrator.train.env[0-1].args.shaping_coef`). It further asserts that `shaping_coef` flips from a positive value in the treatment to $0$ in the control. The first five differences are *cosmetic or bookkeeping* — they rename output directories and adapters so the two runs do not clobber each other's artifacts — and cannot affect the learned policy's behavior; only the sixth (the reward knob) is causally relevant. Among its 31 named checks, the same validator also proves the analytic properties on which the design rests: parity (`parity_coef_zero_equals_stock`), correctness-gating, correctness dominance, and monotonicity (`cheaper_scores_higher`). Crucially, the single-environment `rlm-qwen3-30b-efficient-eval-suite.toml` is **not** used as the control, because it changes the training distribution and would confound the reward manipulation with a distribution shift.

**Identical rollouts-per-prompt.** Both arms use the same GRPO group size, so each training prompt is answered by the same number of sampled rollouts in both arms. This matters because GRPO computes a per-group advantage $s_i - \text{mean}(s)$ over a group of `group_size` rollouts [Shao et al., 2024]; holding the group size fixed ensures the advantage estimator's variance and the credit-assignment granularity are identical across arms, so any behavioral divergence is attributable to how the reward *re-ranks within* each group (the shaped value's monotone re-ordering of the fully-correct subset), not to a different sampling budget. The reward shaping changes only the scalar reward attached to each rollout; the rollout-generation machinery is shared.

## 5.3 Model and training configuration

Both arms instantiate the configuration in Table 5.1, drawn verbatim from `rlm-qwen3-30b-thesis.toml` and its control twin. The only intended divergence between arms is the `shaping_coef` row.

**Table 5.1 — Full-scale training configuration (treatment and control; the only difference is `shaping_coef`).**

| Group | Key | Value |
|---|---|---|
| Base model | `model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` (30B-A3B MoE instruct) |
| Sampling | `enable_thinking` | `false` (non-thinking) |
| LoRA | rank / alpha / dropout | `32` / `64` / `0.0` |
| LoRA | `target_modules` | `q_proj, k_proj, v_proj, o_proj` |
| Optimizer | `lr` | `5e-5` |
| Trainer | `max_steps` | `200` |
| Trainer | `seq_len` | `8192` |
| Trainer | `batch_size` | `32` |
| GRPO | `group_size` | `8` |
| Trainer | ckpt interval / eval interval | `20` / `20` |
| Inference | `max_model_len` | `16384` |
| Inference | `max_completion_tokens` | `4096` (train and eval) |
| Inference | `gpu_memory_utilization` | `0.80` |
| Inference | `tp` / `dp` | `2` / `2` |
| Deployment | GPUs / node | `8` (4 train + 4 infer) |
| Inference | `enforce_eager` | `false` |
| Filters | repetition | window `3000`, prob_threshold `0.99` |
| Filters | zero-advantage | enabled |
| Reward | `shaping_coef` ($\lambda$) | **`0.2` (treatment) / `0.0` (control)** |
| Reward | `correct_threshold` | `1.0` |
| Reward | `subcall_budget` ($B_{\text{calls}}$) | `32.0` |
| Reward | `token_budget` ($B_{\text{tok}}$) | `200000.0` |
| Reward | axis weights (iter/subcall/token) | `1.0` / `1.0` / `1.0` |
| Reward | `max_iterations` ($B_{\text{turns}}$) | `20` |
| Reward | `sub_max_tokens` | `4096` |
| Reward | `min_iterations` / `min_subcall` | `2` / `0` |

The three efficiency budgets are therefore $B_{\text{turns}} = 20$ (the turns axis budget equals `max_iterations`, which always exists), $B_{\text{calls}} = 32$, and $B_{\text{tok}} = 200000$, with per-axis efficiency $\max\!\left(0, 1 - u/B\right)$ for usage $u$. Training uses prime-rl's GRPO with a DPPO loss that clips the importance ratio and adds a KL regularizer [prime-rl]; LoRA adapters are hot-loaded into the vLLM inference server between steps via its `/load_lora_adapter` endpoint, with runtime LoRA updating enabled. The deployment is held fixed across arms so that compute is not a confound — consistent with the thesis's explicit rejection of "train longer/bigger" as the mechanism: the claim concerns the *objective*, testable at fixed model and compute.

## 5.4 Training environments and split

Both arms train on a two-environment mixture at equal sampling weight, $0.5 / 0.5$. prime-rl treats the per-environment `ratio` as a *relative sampling weight*, so the equal ratios mean each batch draws, in expectation, equally from the two sources.

- **`oolong`** (`dataset_name = "spam"`): the OOLONG-Spam long-context QA environment, with context length sampled in $[32768, 65536]$, `filter_numerical = true`, and `num_examples = -1` (the full available set). Correctness is computed by `_oolong_synth_score` (exact match, then numeric $0.75^{|\Delta|}$ decay, date parsing, then a case-insensitive substring containment fallback).
- **`browsecomp_plus`**: the BrowseComp-Plus evidence-document retrieval-QA environment with `num_examples = 150`, `k = 50` documents per query, `reward_mode = "judge"`, and `judge_model = "openai/gpt-4.1"`. Judge mode is wired by default precisely so that training credits the strict (LLM-judged) notion of correctness rather than the lenient containment proxy.

Only these two training environments carry a `shaping_coef`; the choice of which two environments and at what mixture is identical across arms, again so that the reward knob is the sole manipulated variable.

## 5.5 The unshaped evaluation suite

Evaluation uses four environments, listed in Table 5.2. The defining property of the suite is that it is **unshaped**: no `shaping_coef` appears on any eval environment. Evaluation therefore measures the *publication metric* — stock correctness plus cost — and never the training reward. This separation is deliberate and load-bearing: the training reward is what we optimize, but the headline result is the accuracy-vs-cost frontier under the untouched correctness scorer. Reporting the shaped reward as an outcome would conflate the objective with its effect and could mask reward hacking. Evaluation runs at `eval interval = 20` with `skip_first_step = false`, so step 0 records a base-model baseline against which each arm's learning curve is measured, and each rollout is bounded by `timeout_seconds = 1200`.

**Table 5.2 — Unshaped evaluation suite (identical for both arms).**

| Env | Setting | $n$ | Context / docs | Scoring |
|---|---|---|---|---|
| `oolong` | `dataset_name = "trec_coarse"` | 50 | `context_len = 131072` | `_oolong_synth_score` |
| `oolong_pairs` | pair classification | 20 | `context_len = 32768` | F1 over unordered ID pairs |
| `browsecomp_plus` | `reward_mode = "judge"` | 150 | `k = 50` | judge `openai/gpt-4.1` |
| `longbench_codeqa` | 4-choice code MCQ | 50 | repo context | exact letter A–D |

This suite probes generalization (P4) by construction: `oolong @ 131072` exceeds the training context range $[32768, 65536]$, and `longbench_codeqa` is a task family absent from training. `oolong_pairs` is F1-scored over `(id1, id2)` tuples (stripping `<think>` blocks before parsing), and `longbench_codeqa` extracts a single letter via regex and exact-matches the gold choice. The benchmark provenance for these environments is sourced from PrimeIntellect research-environments and LMxLM traces and is not byte-identical to private eval conditioning (details to verify).

## 5.6 Metrics and mapping to the analysis tooling

Each prediction maps to a concrete statistic computed from the evaluation rollouts and the per-rollout telemetry that `RLMTrainEnv` records in state — `rlm_iterations`, `rlm_sub_llm_calls`, `rlm_sub_llm_tokens`, and `rlm_sub_llm_usage_missing` (the last incremented whenever a sub-call returns no usable usage metadata).

- **Accuracy (P1).** Per-environment mean stock correctness, and a pooled accuracy across the suite. Reported as a base$\to$trained delta per environment.
- **Mean cost (P2).** Arithmetic means of `rlm_iterations`, `rlm_sub_llm_calls`, and `rlm_sub_llm_tokens` per environment, treatment vs control.
- **Tail cost (P3).** The $p_{95}$ of turns and of tokens per environment.
- **Accuracy-vs-cost Pareto frontier.** The headline figure: each arm (and each $\lambda$ from the sweep) is a point in (accuracy, cost) space; the frontier is the set of non-dominated points. A treatment point that lies below-and-not-left of control (lower cost, accuracy within $\varepsilon$) confirms P2.
- **Fallback-hit rate.** The fraction of "correct" rollouts whose correctness was awarded by a lenient fallback (OOLONG substring containment, or BrowseComp-Plus mutual-containment) rather than the strict scorer. A *rising* fallback-hit rate under treatment signals reward hacking (Section 5.7), not progress.
- **Trajectory-audit criteria.** Beyond scalars, a sample of trajectories is inspected for genuine work: evidence the context was actually inspected (REPL reads of the offloaded context, sub-calls passed real chunks), not a `min_iterations`-satisfying shell that emits a guessed answer.

These map onto the existing analysis tool `tools/analyze_pilot.py`, which parses offline W&B runs and emits three labelled sections. **`## EVAL ACCURACY`** prints, per environment, the step-0 base value, the final trained value, and their delta $D = \text{trained} - \text{base}$, for control and treatment side by side (matched by `EVAL_PAT` $\wedge$ `ACC_PAT`). **`## TRAJECTORY COST`** matches metrics via `COST_PAT` — names containing `iter`, `sub_llm`, `subcall`, `sub_call`, or `token` — and prints the control and treatment last values with a percent delta $D\% = (\text{treat} - \text{ctrl}) / \text{ctrl} \times 100$; a negative $D\%$ on a cost axis is the P2/P3 signal. **`## SHORTCUT / HEALTH DIAGNOSTICS`** matches `DIAG_PAT` (`usage_missing`, `efficiency`, `shaped`, `gibberish`, `repetition`, `truncat`, `zero_advantage`) and prints control vs treatment last values; these surface reward-hacking and telemetry-coverage problems. The tool closes with a heuristic decision-rule read: efficiency is "working" when treatment cost is below control, "not shortcutting" when treatment accuracy is at least control within noise, and the token axis is "live" only when `rlm_sub_llm_usage_missing` is near zero — otherwise the token efficiency term is a silent no-op. The full-scale runs reuse this tooling unchanged; the metric *definitions* are identical to the pilot's, only the eval $n$ and step budget differ.

## 5.7 Anti-reward-hacking protocol and the $\lambda$ sweep

Adding an efficiency term amplifies every weakness in the correctness scorer $c$: the policy is rewarded for the *cheapest* trajectory the scorer still calls correct, which surfaces scorer loopholes. Two such loopholes are known — the OOLONG substring fallback (`_oolong_synth_score` returns $1.0$ when the gold string is a case-insensitive substring of the raw output) and the BrowseComp-Plus containment fallback (`_score_browsecomp_plus` returns $1.0$ on mutual containment when `reward_mode != "judge"` or the judge call fails). The experiment therefore embeds a seven-point protocol, stated here in full:

1. **Gate all efficiency by correctness.** With `correct_threshold = 1.0`, the efficiency bonus applies only to fully-correct, gate-passing rollouts; a wrong-but-cheap rollout can never outscore a right-but-expensive one [Skalse et al., 2022; Amodei et al., 2016].
2. **Keep evaluation unshaped.** The eval suite (Section 5.5) measures stock correctness and cost, never the shaped reward.
3. **Paired A/B with identical rollouts per prompt.** The single-knob, validator-enforced design of Section 5.2, with matched `group_size`.
4. **Sweep $\lambda$ and keep only Pareto-frontier points.** Run $\lambda \in \{0.05, 0.1, 0.2\}$ (with $\lambda = 0$ as control), and report only non-dominated operating points, so the headline cannot be cherry-picked from a single coefficient.
5. **Audit trajectories, not just scalar rewards.** Inspect sampled trajectories for genuine context inspection.
6. **Stress with adversarial canaries.** Use prompts whose obvious short answer is *wrong* unless the context is actually inspected, so a shortcut policy is penalized.
7. **Separate the train reward from the publication metric.** The headline is the accuracy-vs-cost frontier plot, computed under the untouched scorer — distinct from the shaped reward that drives learning [Ng et al., 1999].

The operational red flag is a cost reduction that *coincides* with a rising fallback-hit rate: that is reward hacking, not efficiency. Mitigations, applied to the shaped arm, are to gate on the strictest available $c$ (BrowseComp-Plus judge rather than containment), to tighten or disable lenient fallbacks, optionally to gate on a second independent check, and to require minimum genuine work beyond `min_iterations >= 2`.

The **$\lambda$ sweep** also serves the falsification logic of Section 5.1: P-falsification requires showing that no value of $\lambda$ in a reasonable range yields a P2/P3 gain without a P1 violation. Sweeping $\{0.05, 0.1, 0.2\}$ samples that range; the frontier across these points, against the $\lambda = 0$ control, is the evidence base for accepting or rejecting the headline claim.

## 5.8 The pilot as a feasibility study

Before committing the full 200-step, 8-GPU budget, an in-flight pilot establishes that the pipeline runs end-to-end and that the instrumentation produces interpretable signal. The pilot is tuned for a single 8$\times$A40 allocation (the "Valar" cluster) and trades scale for turnaround while keeping the comparison structurally faithful. Its A40-specific settings are `max_steps = 10`, inference `tp = 4 / dp = 1`, `gpu_memory_utilization = 0.90`, `enforce_eager = true`, `group_size = 2`, and eval $n = 20$ per environment (versus the full-scale $50/20/150/50$). Critically, the pilot keeps the *same* datasets, context lengths, $k$, judge model, budgets ($B_{\text{calls}} = 32$, $B_{\text{tok}} = 200000$), and the $\lambda \in \{0, 0.2\}$ contrast, so its numbers remain on the same yardstick as the full run. The two arms are `pilot-l0.toml` (control, `shaping_coef = 0.0` on both train environments) and `pilot-l02.toml` (treatment, `shaping_coef = 0.2` on both), launched sequentially by `scripts/valar_pilot.sh` and analyzed by `tools/analyze_pilot.py`.

> **[RESULTS PENDING]** The pilot is in-flight and has produced **no** completed result artifacts. No accuracy, cost, or shortcut numbers are reported. All result tables in Chapter 6 are skeletons.

The pilot's role is strictly feasibility: it confirms that telemetry axes populate (e.g. `rlm_sub_llm_usage_missing` near zero, so the token axis is live), that the offline W&B runs parse, and that the three report sections render. It is *underpowered* for the hypotheses — at $n = 20$ per environment and 10 steps, it cannot adjudicate P1–P4 — and its results, when available, are reported only as a smoke test, never as confirmation of the thesis.

## 5.9 Reproduction

The experiment is reproducible from the repository with the following steps, drawn from `AGENTS.md` and the run scripts.

1. **Install.** `uv init && uv venv --python 3.12`, then `uv pip install -e .`; for the test and dev groups, `uv sync --group dev --group test`.
2. **Validate the A/B configs.** `python tools/validate_thesis_shaping.py` runs the 27 checks that enforce parity, the single-knob control/treatment diff, the $0.5/0.5$ OOLONG-Spam + BrowseComp-Plus training split, the four unshaped eval environments, and the analytic invariants.
3. **Run the invariant tests.** `uv run pytest` (discovery rooted at `tests/`) executes `test_efficiency_shaping.py`, which pins parity, gating, dominance, monotonicity, boundedness, and the metric-telemetry exposure.
4. **Launch the full-scale arms.** Treatment: `RLM_TRAIN_EXEC_TIMEOUT_S=180 uv run rl @ training/configs/rlm-qwen3-30b-thesis.toml`; control: the same command with `rlm-qwen3-30b-thesis-control.toml`. The environment variables `RLM_TRAIN_EXEC_TIMEOUT_S=180` and `RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=180` bound the per-REPL-block wall-clock and tolerate cold worker imports.
5. **Launch the pilot.** `scripts/valar_pilot.sh` runs the two arms (`pilot-l0.toml` then `pilot-l02.toml`) sequentially via `launch_efficient.sh`, which executes `uv run --no-sync rl @ $CFG`.
6. **Analyze.** `python tools/analyze_pilot.py` over the offline W&B output directories produces the three report sections of Section 5.6.

## 5.10 Threats to validity

Five threats temper the strength of any conclusion this design can support.

**Verifier faithfulness.** The entire correct-first guarantee rests on $c$ being a faithful measure of correctness. The known OOLONG substring and BrowseComp-Plus containment fallbacks mean a syntactically lucky output can be scored correct; under shaping, the policy is incentivized to find exactly such cheap-but-lucky outputs. The protocol of Section 5.7 mitigates but does not eliminate this — the judge-mode default and fallback tightening shrink the loophole surface, yet residual scorer leniency remains a confound between "efficiency" and "scorer exploitation."

**Token-telemetry coverage.** The token efficiency axis is only meaningful when `rlm_sub_llm_tokens` is accurately accumulated. `_record_sub_call` increments `rlm_sub_llm_usage_missing` whenever a sub-call returns no usable usage metadata; if this counter is non-trivial, the token axis silently degrades toward a no-op and the token-based components of P2/P3 lose power. This must be monitored (it is one of the diagnostic keys) and reported alongside any token-cost claim.

**Judge dependence.** BrowseComp-Plus correctness, on half of the training mix and 150 of the eval examples, depends on an external LLM judge (`openai/gpt-4.1`). The judge is a closed model whose behavior may drift, may be non-deterministic, and may itself be gameable; both arms inherit identical judge dependence, which preserves the *paired* comparison, but absolute accuracy levels and any judge drift across the run window are outside our control (details to verify).

**Single model and single compute budget.** All claims are established at one base model (Qwen3-30B-A3B-Instruct-2507), one adapter shape, and one fixed compute allocation. This is by design — the thesis isolates the objective from scale — but it means external validity to other model families, sizes, or longer training is unestablished and must be claimed only as a hypothesis.

**Pilot scale.** The in-flight evidence is, at the time of writing, only the feasibility pilot at $n = 20$ per environment and 10 steps. This is far below the sample size needed to resolve the small accuracy and cost differences the hypotheses concern, especially given the high-variance, long-tailed cost distribution that P3 explicitly targets. Any pilot reading is a smoke test; adjudication of P1–P4 requires the full-scale $50/20/150/50$ suite at $200$ steps, whose results are pending.

Taken together, these threats define the evidentiary ceiling of the design: it can demonstrate, at fixed model and compute, whether a correctness-gated efficiency reward moves the accuracy-vs-cost operating point without sacrificing correctness, *conditional on* a faithful scorer and adequate telemetry — and it is explicitly instrumented to detect when those conditions fail. The results that fill the skeletons of Chapter 6 must be read against exactly these caveats.
