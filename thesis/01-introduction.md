# 1. Introduction

## 1.1 Motivation: programmatic context and the recursive-language-model paradigm

A persistent obstacle to deploying large language models on real tasks is the fixed, finite context window. Documents, code repositories, and search corpora routinely exceed any single model's window by orders of magnitude, and naive truncation or retrieval discards information the model may need. The Recursive Language Model (RLM) paradigm [Zhang et al., 2026] reframes this problem. Rather than feeding the full context into one forward pass, an RLM treats the context as *data* — a variable living inside a read-eval-print loop (REPL) — and gives the language model the ability to write code that programmatically inspects, slices, and reasons over that variable. Critically, the orchestrating model can issue *recursive sub-LLM calls* as first-class functions in the code it writes, dispatching bounded chunks of context to fresh model instances and aggregating their answers. The canonical primitive `llm.completion(prompt, model)` is replaced by `rlm.completion(prompt, model)`, which offloads the context to the REPL and lets the model decide, in code, how to decompose the work [Zhang et al., 2026]. In principle this yields task-agnostic inference over *near-infinite* contexts: the model never has to hold the whole input in its window, only the small program and the intermediate results it chooses to surface.

This design is a direct descendant of the CodeAct lineage [Wang et al., 2024], which argues that executable code actions — rather than JSON-schema tool calls — are a more expressive and composable action space for LLM agents. In the RLM instantiation, context is an object in code, sub-LLM calls are functions, and the orchestrator's "plan" is literally a Python program. The released reference scaffold studied in this thesis, the LoRA adapter `mit-oasys/rlm-qwen3-30b-a3b-v0.1` over the `Qwen/Qwen3-30B-A3B-Instruct-2507` base [Zhang et al., 2026][Qwen Team, 2025], was trained inside exactly such a harness: the `RLMTrainEnv` wrapper (`training/src/rlm_train/env.py`) mirrors a depth-1 `rlm.completion` call, executes the model's emitted code blocks in a subprocess-isolated REPL, and logs per-rollout telemetry — turns (`rlm_iterations`), sub-LLM calls (`rlm_sub_llm_calls`), and sub-LLM tokens (`rlm_sub_llm_tokens`) — while reinforcement learning optimizes a scalar reward [verifiers][prime-rl].

The empirical promise of this paradigm is substantial. The foundational paper reports not only strong long-context accuracy but a *length-generalization* result (their Observation 6): a post-trained RLM-Qwen3-8B is reported to be 3.2x–9.6x faster with 68–90% less runtime than its starting point on longer contexts than it was trained on [Zhang et al., 2026]. That speedup, however, is the very thing this thesis interrogates. It is achieved without ever making efficiency an explicit objective — a fact that defines the gap below.

## 1.2 The gap: a correctness-only reward says nothing about how the scaffold was operated

The released scaffold is trained on a *correctness-only* reward, $R = c$, where $c \in [0,1]$ is task correctness. The stock rubric (`training/src/rlm_train/rubric.py`) computes a gated correctness value and exposes the cost telemetry only as zero-weight metrics — they are recorded but contribute nothing to the gradient. The reinforcement-learning signal is therefore completely silent on *how* the scaffold was operated: two fully-correct rollouts that differ by an order of magnitude in turns, sub-calls, or tokens receive *identical* reward.

Where, then, does the paper's efficiency come from? It is *emergent*, the product of four mechanisms none of which is an optimization target [Zhang et al., 2026]:

1. an orchestrator system-prompt addendum forbidding tiny-prompt mega-batches;
2. non-thinking sampling (`enable_thinking=false`) with output caps (`max_completion_tokens=4096`);
3. bounded iterations (`max_iterations=20`) plus a per-REPL-block wall-clock guard (`RLM_TRAIN_EXEC_TIMEOUT_S=180`);
4. structural pre-batch filters (repetition and zero-advantage).

These are prompts, samplers, ceilings, and filters — *exhortations and constraints*, not preferences encoded in the reward. The consequence is a credit-assignment gap. The policy receives no gradient that distinguishes a lean correct trajectory from a wasteful correct one, because both score exactly $R = c = 1$. Any efficiency the model exhibits is incidental to what it was actually rewarded for. The paper itself documents the cost of leaving this to chance: RLM cost is *long-tailed and high-variance*, with tail trajectories re-verifying and re-generating many times (their Section F.2) [Zhang et al., 2026]. A correctness-only objective has no lever against that tail, because every member of the tail that eventually answers correctly is, to the reward, indistinguishable from a frugal first-try success.

It is worth stating precisely why the structural caps do *not* close this gap. Caps such as `max_iterations=20`, `seq_len=8192`, `max_model_len=16384`, `max_completion_tokens=4096`, `sub_max_tokens=4096`, and the REPL timeout are *ceilings, not preferences*. Two fully-correct, cap-satisfying rollouts can have wildly different cost and still tie at $R = c$. A ceiling tells the model what it may not exceed; it never expresses that, among permitted trajectories, cheaper is better. That distinction — that the reward should credit-assign cost *within* the correct, cap-satisfying set — is the conceptual core of this work, and it is pinned by the regression tests `test_structural_caps_leave_correctness_only_rollouts_tied` and `test_efficiency_reward_separates_cap_satisfying_correct_rollouts`.

## 1.3 Thesis statement

This thesis advances and operationalizes a single claim:

> **Scaffold quality is a separable, learnable objective.** On top of a correctness-only policy, a *correctness-gated* process-efficiency reward can shape *how* an RLM uses its REPL and sub-LLM budget — fewer wasted turns, fewer redundant sub-calls, fewer tokens — without trading away correctness.

Concretely, we replace $R = c$ with the *correct-first, efficiency-second* objective

$$R = c\,(1 + \lambda\, e),$$

where $e \in [0,1]$ is a normalized process-efficiency score over the budget axes (turns, sub-LLM calls, sub-LLM tokens), and $\lambda$ (the config key `shaping_coef`) is a small shaping coefficient. The efficiency term is *strictly subordinate* to correctness, enforced by gating: the bonus is applied only to rollouts whose gated correctness value already meets the threshold ($c \ge$ `correct_threshold`, with `correct_threshold=1.0`) and that pass the stock usage gates. Otherwise $R$ collapses to the base correctness value. A wrong-but-cheap rollout can therefore *never* outscore a right-but-expensive one; efficiency only re-ranks *within* the fully-correct set. Each axis contributes $\max(0, 1 - u/B)$ for usage $u$ against budget $B$, so spending more never pays and the bonus is bounded by $\lambda$.

From this objective we derive falsifiable predictions, the headline being equal-correctness *cost reductions* — in particular a contraction of the high-variance upper tail. We specifically predict a smaller **p95** of turns and tokens, directly targeting the long-tailed cost documented in the paper's Section F.2 [Zhang et al., 2026].

A short, *illustrative* arithmetic makes the mechanism concrete (parameters: $\lambda=0.2$, equal-weighted budgets $B_{\text{turns}}=20$, $B_{\text{calls}}=32$, $B_{\text{tok}}=200000$, and $c=1$). A lean correct rollout of 3 turns, 3 sub-calls, and 35,000 sub-LLM tokens has axis efficiencies $0.85$, $0.90625$, $0.825$, hence $e = 0.86042$ and $R = 1.1721$. A wasteful correct rollout of 18 turns, 30 sub-calls, and 160,000 tokens has $e = 0.12083$ and $R = 1.0242$. Both score *exactly* $1.0$ under the correctness-only control. The shaped reward is what separates them; this separation is the gradient the control arm lacks.

## 1.4 Contributions

This thesis makes the following contributions.

1. **The objective and its invariants.** We formalize $R = c\,(1+\lambda e)$ as a correctness-gated reward and prove four invariants (Chapter 3): *parity* ($\lambda=0$ is byte-identical to upstream $R=c$), *dominance* (any fully-correct rollout scores $\ge$ any incorrect one), *monotonicity* (holding $c$ fixed and correct, $R$ is non-increasing in each axis usage), and *boundedness* ($R \in [0, c(1+\lambda)]$, bonus capped at $\lambda$).

2. **A parity-safe implementation.** We implement `EfficiencyGatedRubric` (`training/src/rlm_train/shaping.py`) as a *strict superset* of the stock `RLMTrainRubric`. At `shaping_coef=0.0` (its default) the reward is byte-identical to the upstream correctness-only reward; at `shaping_coef>0` it adds the gated bonus and registers two telemetry metrics, `rlm_efficiency_score` and `efficiency_bonus`. This makes the control arm a *configuration* of the treatment code, not a separate codebase.

3. **A verifier-hardening analysis.** Because gating presupposes that $c$ is faithful, an efficiency term *amplifies* every weakness in the scorer: the policy is rewarded for the cheapest trajectory the verifier still calls correct, surfacing scorer loopholes (Chapter 4). We analyze two concrete fallbacks — the OOLONG substring-containment fallback in `_oolong_synth_score` and the BrowseComp-Plus mutual-containment fallback in `_score_browsecomp_plus` — and specify mitigations (gate on the strictest available $c$, e.g. the LLM judge for BrowseComp-Plus; tighten or disable lenient fallbacks for the shaped arm).

4. **A faithful, validator-enforced A/B design.** We build the treatment (`rlm-qwen3-30b-thesis.toml`, $\lambda=0.2$) and control (`rlm-qwen3-30b-thesis-control.toml`, $\lambda=0.0$) as faithful twins: identical model, LoRA, optimizer, caps, sampling, filters, training split, and eval suite. `tools/validate_thesis_shaping.py` flattens both TOMLs and asserts the diff set is a subset of a whitelist (output directory, run names, adapter and train-env names, and the two `shaping_coef` values). From this design we state four falsifiable predictions **P1–P4** (Section 1.3, Chapter 5).

5. **An in-flight pilot.** We provide an A40-tuned feasibility pilot (`pilot-l0.toml` / `pilot-l02.toml`) launched via `scripts/valar_pilot.sh` and analyzed by `tools/analyze_pilot.py`, comparing $\lambda=0$ against $\lambda=0.2$ on accuracy, cost, and shortcut diagnostics. The pilot is *in-flight* and has produced no completed result artifacts; Chapter 6 contains only table skeletons.

## 1.5 Scope and non-claims

To keep the contribution honest, we delimit it sharply.

- **We do not reproduce the authors' private run.** The released scaffold's exact mixed training suite, orchestrator hints, and rollout logs are not public; our environments are source-derived from PrimeIntellect research-environments and LMxLM traces, not byte-identical to the private eval conditioning (details to verify) [Zhang et al., 2026]. Our claim concerns the *objective*, not exact replication.
- **We do not change the default reward.** `EfficiencyGatedRubric` defaults to `shaping_coef=0.0`, so absent an opt-in the behavior is exactly upstream $R=c$. The treatment is strictly additive.
- **We do not assert a best $\lambda$.** $\lambda=0.2$ is the pilot/treatment setting; the protocol sweeps $\lambda \in \{0.05, 0.1, 0.2\}$ and reports only Pareto-frontier points. We make no claim of optimality.

We also explicitly *reject* three weaker framings as the headline (developed in Chapters 2 and 7): (i) an ungated token-cost penalty $R = c - \mu\,\text{cost}$, which can let wrong-but-cheap beat right-but-expensive — the "plan-as-answer" failure mode; (ii) "efficiency is just prompting," which cannot be credit-assigned and gives the model no gradient distinguishing lean from wasteful correct trajectories; and (iii) "train longer/bigger," which is orthogonal to the objective and testable at fixed model and compute.

Finally, because this is a proposal-plus-design with an *in-flight* pilot, the thesis reports **no completed empirical outcomes**. Following the results policy, every results table is a skeleton with TBD cells under a `[RESULTS PENDING]` callout; the only concrete numbers are configuration values and the labelled illustrative arithmetic above.

## 1.6 Thesis outline

The remainder of the thesis maps to chapters as follows.

| Chapter | Title | Role |
| --- | --- | --- |
| 2 | Background and Related Work | RLM/CodeAct paradigm, RLVR and GRPO [Shao et al., 2024][Lambert et al., 2024], reward shaping [Ng et al., 1999], reward hacking and Goodhart's law [Skalse et al., 2022][Amodei et al., 2016] |
| 3 | A Correct-First, Efficiency-Second Objective | Formal statement; derivations of parity, dominance, monotonicity, boundedness; why caps are not preferences |
| 4 | Method and Implementation | `EfficiencyGatedRubric`, the efficiency axes and budgets ($B_{\text{turns}}=20$, $B_{\text{calls}}=32$, $B_{\text{tok}}=200000$), telemetry, LoRA/GRPO training stack, and verifier hardening |
| 5 | Experimental Design | The validator-enforced A/B, the unshaped eval suite, falsifiable predictions P1–P4, and the seven-point anti-reward-hacking protocol |
| 6 | Results | **Pending** — table skeletons only |
| 7 | Discussion | Interpretation, threats to validity, the three rejected framings |
| 8 | Conclusion and Future Work | Summary and the $\lambda$-sweep / generalization roadmap |

Chapter 2 next situates the work within recursive inference, verifiable-reward RL, and the reward-shaping and reward-hacking literatures, establishing the conceptual tools used to argue that scaffold quality is a separable, learnable objective.
