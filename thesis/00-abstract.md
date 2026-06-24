# Correct-First, Efficiency-Second: Learning to Operate the Recursive-Language-Model Scaffold

| | |
|---|---|
| **Author** | {{AUTHOR}} |
| **Degree** | {{DEGREE}} |
| **Department** | {{DEPARTMENT}} |
| **Institution** | {{INSTITUTION}} |
| **Advisor** | {{ADVISOR}} |
| **Date** | {{DATE}} |

---

## Note on status

This thesis presents an *objective*, an *implementation*, and an *experimental design*; it does **not** yet present empirical results. The central contribution — a correct-first, efficiency-second reward $R = c\,(1 + \lambda e)$ for training Recursive Language Models (RLMs) — is fully specified and operationalized in code (`training/src/rlm_train/shaping.py`), and the paired A/B protocol (control $\lambda=0$ vs. treatment $\lambda=0.2$) is wired into faithful-twin configurations (`training/configs/rlm-qwen3-30b-thesis.toml` and `-control.toml`). The pilot feasibility study (`pilot-l0.toml` / `pilot-l02.toml`) is **in-flight** and has produced no completed result artifacts. Accordingly, Chapter 6 (Results) is intentionally a set of placeholders and table skeletons marked **[RESULTS PENDING]**; every concrete number appearing elsewhere in this document is either a configuration value or a clearly labelled *illustrative* reward computation. Empirical validation of the falsifiable predictions in Chapter 5 (P1–P4) is the subject of ongoing and future work (Chapter 8).

## Abstract

Recursive Language Models [Zhang et al., 2026] are a task-agnostic inference paradigm in which a language model programmatically inspects and recursively calls itself over near-infinite contexts, offloading the context into a REPL variable and treating sub-LLM calls as first-class functions — a CodeAct-style harness [Wang et al., 2024]. The released scaffold (`mit-oasys/rlm-qwen3-30b-a3b-v0.1`) is trained with prime-rl [prime-rl] under a **correctness-only** reward $R = c$, where $c \in [0,1]$ measures task success. This thesis identifies a gap in that objective: when reward depends on correctness alone, the *operation* of the scaffold — how many orchestrator turns, sub-LLM calls, and sub-LLM tokens a trajectory spends — is left entirely unsupervised. Two fully-correct rollouts that differ greatly in cost tie at $R = c$, so the policy receives no gradient distinguishing a lean correct trajectory from a wasteful one. This matters precisely because RLM cost is long-tailed and high-variance (paper Section F.2).

We propose and operationalize a **correct-first, efficiency-second** objective, $R = c\,(1 + \lambda e)$, where $e \in [0,1]$ is a normalized process-efficiency score and $\lambda$ (config key `shaping_coef`) is a shaping coefficient. The efficiency bonus is strictly *gated*: it is applied only when the gated correctness value reaches `correct_threshold=1.0` and the rollout passes the stock usage gates; otherwise $R = \mathrm{base}(c)$. Efficiency is a weighted mean of per-axis scores $\max(0, 1 - u/B)$ over three budget axes (turns $B=20$, sub-calls $B=32$, sub-LLM tokens $B=200000$). By construction a wrong-but-cheap rollout can never outscore a right-but-expensive one; efficiency only re-ranks *within* the fully-correct set. We prove four invariants — parity (at $\lambda=0$, byte-identical to upstream $R=c$), correctness-dominance, monotonicity in cost, and boundedness $R \in [0, c(1+\lambda)]$ — and explicitly reject three weaker alternatives (ungated cost penalties, prompting-only efficiency, and "train longer/bigger").

The method is an `EfficiencyGatedRubric` that is a strict superset of the stock `RLMTrainRubric`; we also harden the verifier, since gating assumes $c$ is faithful and any efficiency term amplifies scorer loopholes (e.g., OOLONG substring and BrowseComp-Plus containment fallbacks). The experimental design is a confound-free paired A/B — control and treatment differing only in `shaping_coef` (enforced by `tools/validate_thesis_shaping.py`) — at fixed model (`Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA $r=32$), data, compute, and an **unshaped** four-environment eval suite (OOLONG TREC, OOLONG-Pairs, BrowseComp-Plus, LongBench-v2 CodeQA). We state falsifiable predictions P1–P4 (no correctness regression; cheaper trajectories at equal accuracy; lower tail cost; generalization). Contributions: the gated objective and its invariant proofs; a drop-in, parity-preserving implementation; a verifier-hardening and anti-reward-hacking protocol; and a reproducible experimental harness. Empirical validation remains pending.

**Keywords:** recursive language models; reward shaping; process efficiency; reinforcement learning with verifiable rewards; long-context; reward hacking; GRPO; LoRA.

## Acknowledgements

The author thanks {{ADVISOR}} for supervision and guidance, {{INSTITUTION}} and the {{DEPARTMENT}} for computational resources, and {{ACKNOWLEDGEMENTS}} for discussions and support throughout this work.
