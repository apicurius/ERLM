# 8. Conclusion and Future Work

## 8.1 Thesis and Contributions

This thesis advances a single claim: in Recursive Language Models (RLMs), inference-time efficiency is a *separable, learnable* property of the policy rather than an emergent by-product of structural caps and prompting, and it can be trained for without compromising correctness by adopting a correct-first, efficiency-second reward,
$$R = c\,(1 + \lambda e),$$
where $c \in [0,1]$ is task correctness, $e \in [0,1]$ is a normalized process-efficiency score, and $\lambda$ (the `shaping_coef`) is a small shaping coefficient. The released RLM scaffold trains on a correctness-only objective $R = c$ [Zhang et al., 2026], obtaining efficiency only indirectly through an orchestrator prompt addendum, non-thinking sampling, output caps, bounded iterations, and structural filters. We argued (Chapter 3) and implemented (Chapter 4) that this leaves cost *unassigned*: two fully-correct, cap-satisfying rollouts can differ greatly in turns, sub-LLM calls, and tokens yet tie at $R = c$, so the optimizer receives no gradient distinguishing the lean trajectory from the wasteful one.

The concrete contributions are: (i) a *formal objective* with four invariants — parity ($\lambda=0 \Rightarrow R = c$, byte-identical to upstream), correctness dominance, per-axis monotonicity, and boundedness $R \in [0, c(1+\lambda)]$ — derived in Chapter 3; (ii) an *implementation*, `EfficiencyGatedRubric` in `training/src/rlm_train/shaping.py`, a strict superset of the stock `RLMTrainRubric` whose efficiency bonus is gated on correctness (`correct_threshold=1.0`) and the stock usage gates, with `e` a weighted mean over enabled axes of $\max(0, 1 - u/B)$ for budgets $B_\text{turns}=20$, $B_\text{calls}=32$, $B_\text{tok}=200000$; (iii) a *confound-free experimental design* (Chapter 5) — a paired A/B between a treatment config (`rlm-qwen3-30b-thesis.toml`, $\lambda=0.2$) and a faithful twin control (`rlm-qwen3-30b-thesis-control.toml`, $\lambda=0$), mechanically enforced by `tools/validate_thesis_shaping.py` to differ in nothing but the reward knob; and (iv) an *anti-reward-hacking protocol* with verifier-hardening recommendations, recognizing that an efficiency term amplifies every weakness in $c$ (Chapter 5, Chapter 7).

Crucially, this thesis is a proposal, implementation, and experimental design. The A/B pilot (`pilot-l0.toml` vs `pilot-l02.toml`) is in-flight and has produced no completed result artifacts, so Chapter 6 contains only table skeletons.

## 8.2 What Success Looks Like

The empirical case rests on four falsifiable predictions over the unshaped held-out eval suite (OOLONG TREC@131072, OOLONG-Pairs@32768, BrowseComp-Plus $k=50$ judged by `openai/gpt-4.1`, LongBench-v2 CodeQA), at fixed model, data, compute, and evaluation:

- **P1 (no correctness regression):** treatment accuracy $\ge$ control $- \epsilon$.
- **P2 (cheaper trajectories):** strictly fewer mean turns and/or sub-LLM calls and/or sub-LLM tokens at equal accuracy — a Pareto improvement.
- **P3 (lower tail cost):** a smaller p95 of turns/tokens, targeting the long-tailed, high-variance RLM cost reported in the paper's §F.2 [Zhang et al., 2026].
- **P4 (generalization):** the efficiency gain transfers to a held-out task family or larger context, mirroring the paper's length-generalization result (Observation 6).

The intended outcome can be stated geometrically. On the headline accuracy-versus-cost frontier — the publication metric, deliberately separated from the training reward — success means moving treatment points *left* (cheaper) without moving them *down* (less accurate). The dominance and monotonicity invariants make this the only behavior the reward can incentivize: because the bonus is strictly subordinate to correctness, a wrong-but-cheap rollout can never outscore a right-but-expensive one, so the gradient can only re-rank *within* the fully-correct set toward cheaper trajectories. The hypothesis is falsified if, across a reasonable sweep of $\lambda$, the treatment cannot beat the control on P2/P3 without violating P1 — i.e., if every cost reduction comes at an accuracy cost, or coincides with a rising scorer-fallback-hit rate (which we treat as reward hacking, not progress; Chapter 5).

> **[RESULTS PENDING]** No completed run artifacts exist. The frontier plot, the P1–P4 table, and the pilot diagnostics emitted by `tools/analyze_pilot.py` (eval accuracy, trajectory cost, shortcut/health diagnostics) are populated only when the A/B completes.

## 8.3 Future Work

### 8.3.1 Completing the $\lambda$ sweep

The immediate next step is the planned sweep over $\lambda \in \{0.05, 0.1, 0.2\}$ (Chapter 5), retaining only Pareto-frontier points. The single in-flight comparison ($\lambda=0$ vs $\lambda=0.2$) fixes one operating point; the sweep maps how the accuracy–cost trade-off responds to shaping strength and locates the largest cost reduction that still satisfies P1. Because the design is paired and confound-free, each sweep point is a drop-in change to `shaping_coef`, requiring no other configuration edits and preserving the validator's parity guarantees.

### 8.3.2 Additional efficiency axes and learned budgets

The current efficiency score weights three axes — turns (`rlm_iterations`), sub-LLM calls (`rlm_sub_llm_calls`), and sub-LLM tokens (`rlm_sub_llm_tokens`) — equally (`iteration_weight = subcall_weight = token_weight = 1.0`). Future work can (i) reweight axes to reflect the deployment cost model (e.g., wall-clock latency or monetary token cost dominating), since `EfficiencyAxis` already supports per-axis weights and an axis is enabled iff `budget > 0` and `weight > 0`; and (ii) add axes such as REPL execution time or peak context occupancy. A second, more open direction replaces the *fixed* budgets $B_\text{turns}=20$, $B_\text{calls}=32$, $B_\text{tok}=200000$ with *learned or instance-adaptive* budgets — for example, normalizing each axis against the empirical cost distribution of correct rollouts for a given task difficulty, so that the efficiency target scales with the genuine work a problem requires rather than a static ceiling.

### 8.3.3 Other base models and larger scale

All experiments fix the base model `Qwen/Qwen3-30B-A3B-Instruct-2507` and LoRA adapter (rank 32, $\alpha=64$, targeting `q_proj`/`k_proj`/`v_proj`/`o_proj`), training for `max_steps=200` under GRPO [Shao et al., 2024]. The objective is model-agnostic; replicating the A/B across model families and sizes would test whether the efficiency-second signal generalizes beyond a single MoE instruct model. Because the claim is explicitly about the *objective* — not about training longer or bigger, which we reject as orthogonal (Chapter 1) — scaling experiments are valuable for external validity rather than for the core hypothesis, which is testable at fixed model and compute.

### 8.3.4 Transfer to held-out task families and longer contexts (P4)

Prediction P4 is the most consequential and the least exercised by the current eval suite. The paper's Observation 6 reports that a post-trained RLM generalizes its efficiency to longer contexts than seen in training [Zhang et al., 2026]. Future work should evaluate treatment adapters on (i) task families absent from the OOLONG-Spam + BrowseComp-Plus training mix, and (ii) contexts substantially longer than the training range (`context_len` sampled in $[32768, 65536]$ for the OOLONG training env). A clean P4 result — efficiency learned on short-to-moderate contexts transferring to much longer ones — would establish the objective as teaching a *transferable disposition* toward parsimony rather than overfitting a budget to the training distribution.

### 8.3.5 Potential-based formulations

A theoretically attractive alternative is to express the efficiency signal as a *potential-based* shaping term, which is guaranteed not to change the set of optimal policies [Ng et al., 1999]. Our reward is deliberately *not* potential-based: by multiplying correctness by $(1 + \lambda e)$ it is designed to alter the policy's preferences *within* the optimal-correctness set, breaking the tie toward cheaper trajectories. A potential-based variant would instead preserve the optimal policy exactly while reshaping the learning dynamics, and comparing the two would sharpen the distinction between *changing what is optimal* (our intent) and *changing how fast it is reached*. Relatedly, future work should study learned versus fixed budgets jointly with the formulation choice, since a potential function and an adaptive budget interact in how they distribute credit across a trajectory.

### 8.3.6 Verifier hardening

Because gating assumes $c$ is faithful, an efficiency term surfaces scorer loopholes — the policy is rewarded for the cheapest trajectory the scorer still calls correct. Two known fallbacks invite hacking: the OOLONG substring-containment fallback in `_oolong_synth_score`, and the BrowseComp-Plus mutual-containment fallback in `_score_browsecomp_plus` when judging is unavailable. Hardening the shaped arm — gating on the strictest available $c$ (LLM judge, not containment), tightening or disabling lenient fallbacks, and requiring evidence the context was genuinely inspected rather than merely satisfying `min_iterations` — is a prerequisite for any larger-scale or longer-horizon run, and a worthwhile contribution in its own right.

## 8.4 Closing

The structural levers of the RLM scaffold — `max_iterations=20`, `seq_len=8192`, `max_model_len=16384`, output caps, and REPL timeouts — are *ceilings*, not *preferences*: they bound the worst case but say nothing about which of two correct, cap-satisfying trajectories to prefer. This thesis fills that gap with a reward that credit-assigns cost *strictly within* the correct set. The objective is conservative by construction — at $\lambda=0$ it is byte-identical to the upstream correctness-only reward, so adopting it can never silently degrade correctness — and it is falsifiable by construction, via the paired A/B and the P1–P4 predictions. The pilot and the full-scale A/B are in-flight; their completion will turn the frontier skeleton of Chapter 6 into evidence for, or against, the central claim that efficiency in recursive language models can be *trained for* without ever being allowed to outrank being *right*.
