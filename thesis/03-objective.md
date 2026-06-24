# 3. A Correct-First, Efficiency-Second Objective

This chapter develops the formal core of the thesis: a training objective for Recursive Language Models (RLMs) that treats *correctness* as a strict precondition and *process efficiency* as a subordinate tie-breaker. We state the objective precisely, prove the invariants that make it safe to optimize, work a fully labelled illustrative example at the canonical budgets, and argue against weaker alternatives. The objective is implemented in `training/src/rlm_train/shaping.py` as the class `EfficiencyGatedRubric`; throughout, we ground every claim, constant, and design choice in that implementation and its companion configuration files, so that the mathematics and the code are demonstrably the same artifact.

## 3.1 Problem setup: a policy over a scaffold

An RLM does not answer a long-context query in a single forward pass. Following the recursive paradigm of [Zhang et al., 2026], it answers by *orchestrating a scaffold*: the full task context is offloaded into a Python REPL as a variable, and the model emits code that reads, slices, and recursively dispatches sub-LLM calls over that context — a CodeAct-style harness [Wang et al., 2024] in which sub-calls are first-class functions rather than JSON tool invocations. In the training harness this is realized by `RLMTrainEnv` (`training/src/rlm_train/env.py`), a `verifiers` `MultiTurnEnv` that mirrors a depth-1 `RLM.completion` call: it stands up a subprocess-isolated REPL backend and a per-rollout `SubLLMProxy`, then advances the trajectory by extracting code blocks and executing them turn by turn.

A **rollout** (equivalently, a trajectory) $\tau$ is therefore a sequence of REPL turns produced by the policy $\pi_\theta$ on a single prompt. Two kinds of quantity are observable at the end of a rollout.

First, a scalar **correctness** $c(\tau) \in [0,1]$, computed by the task's scorer. For the training environments these are: OOLONG-synth's `_oolong_synth_score` (exact / numeric-decay / date / substring), OOLONG-Pairs' F1 over user-id pairs, BrowseComp-Plus's LLM judge (`reward_mode="judge"`, `judge_model="openai/gpt-4.1"`) with a deterministic containment fallback, and LongBench-v2 CodeQA's exact-letter match. We treat $c$ as a black box here and return to its fragility in Chapter 5; in this chapter $c$ is simply the correctness signal that the reward is built on top of.

Second, a vector of **trajectory cost** along three axes that the environment accumulates into rollout state during execution:

- `rlm_iterations` — the number of REPL turns the policy consumed, incremented after each executed block in `get_prompt_messages`;
- `rlm_sub_llm_calls` — the number of recursive sub-LLM calls, tallied by `_record_sub_call`;
- `rlm_sub_llm_tokens` — the total tokens consumed by those sub-calls, summed from each call's usage metadata (with `rlm_sub_llm_usage_missing` incremented when a usage payload is absent or malformed).

These three counters are exactly the cost axes the objective will price. Crucially, the *stock* rubric `RLMTrainRubric` already records all three but exposes them as **zero-weight metrics** (`training/src/rlm_train/rubric.py`): the upstream scaffold *measures* cost but never *optimizes* it, training instead on a correctness-only reward $R = c$.

The view we adopt is that the policy is a **policy over a scaffold**. Within a rollout, $\pi_\theta$ makes a recurring sequence of structural decisions — *read* a slice of context, *chunk* it, *delegate* a sub-problem to a sub-LLM, *batch* several sub-queries into one call, *verify* an intermediate result, and *stop* by setting `answer["content"]` with `answer["ready"]=True`. Each decision moves the trajectory along the cost axes. Correctness scores the *destination*; the cost axes score the *route*. The objective of this chapter is to price the route without ever letting price override the destination.

## 3.2 The objective

Let $c \in [0,1]$ be correctness and $e \in [0,1]$ be a normalized process-efficiency score (defined below). With shaping coefficient $\lambda \geq 0$ (the config key `shaping_coef`), the shaped reward is

$$
R \;=\; \mathrm{base}(c)\,\bigl(1 + \lambda\, e\bigr)
\qquad\text{applied only under the correctness gate.}
$$

Here $\mathrm{base}(c)$ is the **stock gated correctness value**: the value the upstream rubric would assign, after the usage gates and minimum-reward floor. Concretely, `_base_value` returns the raw correctness if reward-gating is off, and otherwise returns $0$ if the rollout fails `_passes_gates` (i.e. `rlm_iterations < min_iterations` or `rlm_sub_llm_calls < min_subcall`) or if the value is below `min_reward`, else the value itself. The bonus factor $(1+\lambda e)$ is applied by `_shaped_value` **only** when three conditions hold simultaneously: shaping is enabled ($\lambda > 0$), the base value clears the correctness threshold ($\mathrm{base}(c) \geq$ `correct_threshold`, with `correct_threshold = 1.0`), and the rollout passes the same usage gates. Otherwise `_shaped_value` returns $\mathrm{base}(c)$ unchanged. This is the formal sense in which efficiency is a **strict post-correctness tie-breaker**: a wrong-but-cheap rollout cannot receive any bonus, because the multiplicative factor is never reached unless the rollout is already (fully) correct.

In the house notation we write the objective compactly as $R = c\,(1+\lambda e)$, understanding $c$ to mean the gated correctness value and the bonus to be live only inside the correct, gate-passing set.

**The efficiency score $e$.** Efficiency is a *weighted mean over enabled axes* of a per-axis efficiency

$$
\mathrm{eff}_a \;=\; \max\!\Bigl(0,\; 1 - \tfrac{u_a}{B_a}\Bigr),
$$

where $u_a$ is the axis usage read from state and $B_a$ is its budget (`EfficiencyAxis.efficiency`). Each axis is $1$ at zero usage, decreases linearly, and is clamped at $0$ once usage reaches or exceeds the budget — overspending past a budget is no worse than spending exactly the budget, but it is never rewarded. The aggregate is

$$
e \;=\; \frac{\sum_{a\in\mathcal{A}} w_a\,\mathrm{eff}_a}{\sum_{a\in\mathcal{A}} w_a},
\qquad e \in [0,1],
$$

clamped to $[0,1]$ in `efficiency_score`, where $\mathcal{A}$ is the set of **enabled** axes. If $\mathcal{A}$ is empty, `efficiency_score` returns $0.0$ by construction.

**The three axes and their budgets.** `default_axes` constructs three axes keyed to the cost counters of Section 3.1:

| Axis | State key | Budget $B$ | Canonical value | Weight |
|---|---|---|---|---|
| Turns | `rlm_iterations` | `max_iterations` | $B_{\text{turns}} = 20$ | `iteration_weight = 1.0` |
| Sub-calls | `rlm_sub_llm_calls` | `subcall_budget` | $B_{\text{calls}} = 32$ | `subcall_weight = 1.0` |
| Tokens | `rlm_sub_llm_tokens` | `token_budget` | $B_{\text{tok}} = 200000$ | `token_weight = 1.0` |

These constants come from `training/configs/rlm-qwen3-30b-thesis.toml` and its faithful control twin.

**Enabled-axis semantics and the $e=0$ no-op.** An axis is *enabled* iff `budget > 0 and weight > 0` (the `EfficiencyAxis.enabled` property). This makes axis participation strictly opt-in per caller: a budget of zero silently disables an axis rather than dividing by zero. The turns axis always has a budget because `max_iterations` always exists in the harness, so it is enabled by default; the sub-call and token axes are enabled only when the config supplies positive budgets, which the thesis configs do. If a caller enables *no* axis (or all weights vanish), `efficiency_score` returns $0.0$ by construction. This is a deliberate safety property: an unconfigured shaping term collapses to $e=0$, hence $R = c\,(1+\lambda\cdot 0) = c$, a no-op rather than an accidental free reward. The control arm exploits the same mechanism from the other direction by setting $\lambda = 0$ (see Section 3.3, Parity).

## 3.3 Invariants

The objective is engineered so that four invariants hold for *all* rollouts. Each is stated precisely, derived from the definitions above, and pinned by a test in `training/tests/test_efficiency_shaping.py`. The validator `tools/validate_thesis_shaping.py` additionally proves parity, gating, dominance, and monotonicity at config level.

### Parity ($\lambda = 0 \Rightarrow R = c$)

**Claim.** At `shaping_coef = 0.0`, the shaped reward is byte-identical to the upstream correctness-only reward $R = \mathrm{base}(c)$.

**Derivation.** When $\lambda = 0$, shaping is disabled, so `_shaped_value` returns `base` on its very first branch, never reaching `efficiency_score`. Thus $R = \mathrm{base}(c)$, which is exactly what the stock `RLMTrainRubric` computes. Because `EfficiencyGatedRubric` is a *strict superset* of the stock rubric and shares `_base_value` with it, the equality is structural rather than numerical coincidence: the same gate, the same floor, the same value. This is what makes the control arm a confound-free twin — the only knob that moves between treatment and control is $\lambda$. Pinned by `test_parity_with_stock_rubric_when_coef_zero`.

### Dominance (within-correct-set re-ranking)

**Claim.** With `correct_threshold = 1.0`, any fully-correct, gate-passing rollout scores at least as high as *any* incorrect rollout. Efficiency re-ranks only *within* the fully-correct set; it never promotes a wrong rollout over a right one.

**Derivation.** Partition rollouts into the correct, gate-passing set $\mathcal{C}$ (those with $\mathrm{base}(c) \geq 1$) and its complement. For $\tau \in \mathcal{C}$, $R(\tau) = \mathrm{base}(c)\,(1+\lambda e) \geq \mathrm{base}(c) \geq 1$ because $\lambda, e \geq 0$. For $\tau \notin \mathcal{C}$, the bonus branch is never taken, so $R(\tau) = \mathrm{base}(c) \leq 1$ (correctness is in $[0,1]$, and a gate failure floors it to $0$). Hence $R(\tau \in \mathcal{C}) \geq 1 \geq R(\tau \notin \mathcal{C})$. The bonus, bounded by $\lambda$ (next invariant), only *re-orders elements already inside* $\mathcal{C}$ relative to one another. Pinned by `test_correct_always_beats_incorrect`, which compares a maximally expensive correct rollout against a maximally cheap wrong one and asserts the correct one wins.

### Monotonicity (spending more never pays)

**Claim.** Holding $c$ fixed and correct, $R$ is non-increasing in each axis usage $u_a$: increasing any one of turns, sub-calls, or tokens can only lower or preserve the reward, never raise it.

**Derivation.** For a fixed correct, gate-passing rollout, $R = \mathrm{base}(c)\,(1+\lambda e)$ with $\mathrm{base}(c)$ and $\lambda$ constant. Each per-axis term $\mathrm{eff}_a = \max(0, 1 - u_a/B_a)$ is non-increasing in $u_a$ (it is linearly decreasing until it saturates at $0$). The aggregate $e$ is a non-negative weighted mean of these terms, hence non-increasing in every $u_a$; and $R$ is an increasing affine function of $e$ for $\lambda > 0$. Therefore $\partial R / \partial u_a \leq 0$ wherever defined. The economic reading is direct: there is no usage profile for which spending an extra turn, sub-call, or token strictly increases reward. Pinned by `test_more_usage_never_pays`, which checks $R(\text{cheap}) \geq R(\text{mid}) \geq R(\text{pricey}) \geq 1$ across a usage gradient.

### Boundedness ($R \in [0,\, c(1+\lambda)]$)

**Claim.** The shaped reward satisfies $0 \leq R \leq c\,(1+\lambda)$; the bonus is capped at $\lambda$, attained only when $e = 1$.

**Derivation.** Since $e \in [0,1]$ (clamped in `efficiency_score`) and $\mathrm{base}(c) \in [0, c]$, we have $1 \leq 1 + \lambda e \leq 1 + \lambda$ in the shaped branch, so $R = \mathrm{base}(c)(1+\lambda e) \leq c(1+\lambda)$. The lower bound is immediate: $\mathrm{base}(c) \geq 0$ and the factor is positive. The upper bound is reached only at $e = 1$, i.e. zero usage on every enabled axis — an idealization the gates ($\texttt{min\_iterations} = 2$) make unreachable in practice, which is precisely why the bound is a ceiling and not a typical value. Pinned by `test_reward_is_bounded`, which disables the turns axis so that $e=1$ is attainable, yielding $R = c(1+\lambda)$ exactly.

Together these four invariants make the objective *safe to optimize*: parity guarantees a clean baseline, dominance guarantees the correctness ordering is preserved, monotonicity guarantees the gradient points toward thrift, and boundedness guarantees the bonus cannot run away.

## 3.4 Worked illustrative example

The following arithmetic is **illustrative**. It is computed at the canonical parameters — $\lambda = 0.2$, budgets $B_{\text{turns}}=20$, $B_{\text{calls}}=32$, $B_{\text{tok}}=200000$, equal weights $1.0$, and a fully-correct rollout $c = 1$ — to show how the objective separates two trajectories that the correctness-only control cannot tell apart.

**Lean correct rollout.** Suppose a rollout reaches the right answer in $3$ turns, $3$ sub-calls, and $35{,}000$ sub-LLM tokens:

$$
\mathrm{eff}_{\text{turns}} = 1 - \tfrac{3}{20} = 0.85, \quad
\mathrm{eff}_{\text{calls}} = 1 - \tfrac{3}{32} = 0.90625, \quad
\mathrm{eff}_{\text{tok}} = 1 - \tfrac{35000}{200000} = 0.825.
$$

$$
e = \tfrac{0.85 + 0.90625 + 0.825}{3} = 0.86042,
\qquad
R = 1 \cdot (1 + 0.2 \cdot 0.86042) = 1.1721.
$$

**Wasteful correct rollout.** Suppose another rollout reaches the *same* right answer but spends $18$ turns, $30$ sub-calls, and $160{,}000$ tokens:

$$
\mathrm{eff}_{\text{turns}} = 0.10, \quad
\mathrm{eff}_{\text{calls}} = 1 - \tfrac{30}{32} = 0.0625, \quad
\mathrm{eff}_{\text{tok}} = 0.20,
$$

$$
e = \tfrac{0.10 + 0.0625 + 0.20}{3} = 0.12083,
\qquad
R = 1 \cdot (1 + 0.2 \cdot 0.12083) = 1.0242.
$$

| Rollout | turns | sub-calls | tokens | $e$ | $R$ (treatment, $\lambda=0.2$) | $R$ (control, $\lambda=0$) |
|---|---|---|---|---|---|---|
| Lean correct | 3 | 3 | 35,000 | 0.86042 | **1.1721** | 1.0 |
| Wasteful correct | 18 | 30 | 160,000 | 0.12083 | **1.0242** | 1.0 |

Under the correctness-only control both rollouts score exactly $1.0$ — they are indistinguishable, and GRPO's per-group advantage $s_i - \mathrm{mean}(s)$ assigns them identical baselines, providing *no* gradient to prefer the lean route. Under the treatment they score $1.1721$ vs. $1.0242$, a gap of $\approx 0.148$ that GRPO converts into a positive advantage for the lean rollout within its group.

**Contrast with structural caps.** The harness already imposes *ceilings*: `max_iterations = 20`, `seq_len = 8192`, `max_model_len = 16384`, `max_completion_tokens = 4096`, `sub_max_tokens = 4096`, and a per-REPL wall-clock guard (`RLM_TRAIN_EXEC_TIMEOUT_S`). These are levers the released scaffold relied on to keep cost bounded — never an optimization objective [Zhang et al., 2026]. But a cap is a *constraint*, not a *preference*: both rollouts in the table satisfy every cap, and under correctness-only both tie at $R = c$. Caps *tie* cost; they cannot express that the lean route is better. The efficiency reward *separates* cost by credit-assigning it **within** the correct, cap-satisfying set — exactly the regime caps leave flat. This complementarity is pinned by the tests `test_structural_caps_leave_correctness_only_rollouts_tied` and `test_efficiency_reward_separates_cap_satisfying_correct_rollouts`. Specifically, `test_efficiency_reward_separates_cap_satisfying_correct_rollouts` asserts exactly the table's $R = 1.1721$ versus $1.0242$ at the canonical `subcall_budget = 32`. The reward is thus not redundant with the caps; it does the work the caps structurally cannot.

## 3.5 Why not weaker objectives

The thesis explicitly *rejects* three superficially simpler alternatives as the headline objective.

**(1) An ungated cost penalty, $R = c - \mu\,\mathrm{cost}$.** Subtracting a cost term *additively* and *ungated* breaks dominance. There exist a cheap-but-wrong rollout and an expensive-but-right rollout with $c_{\text{wrong}} = 0$, $\mathrm{cost}_{\text{wrong}} \approx 0$ and $c_{\text{right}} = 1$, $\mathrm{cost}_{\text{right}}$ large, such that $0 - \mu\cdot 0 > 1 - \mu\cdot\mathrm{cost}_{\text{right}}$ once $\mu\,\mathrm{cost}_{\text{right}} > 1$. The policy then learns the pathological shortcut [Zhang et al., 2026] of emitting a premature `FINAL` / plan-as-answer to bank the cost saving — a wrong-but-cheap rollout *outscoring* a right-but-expensive one. The correct-first design forecloses this structurally: because the bonus is *multiplicative* on a *gated* base and is unreachable unless $\mathrm{base}(c) \geq 1$, no value of usage can lift a wrong rollout above a right one (Dominance, Section 3.3). Efficiency must remain strictly subordinate to correctness; an additive ungated penalty cannot guarantee that ordering.

**(2) "Efficiency is just prompting."** The prologues already instruct the model to *"Plan before you act,"* to respect a $\sim$16k context budget, and to pass chunks as arguments to `llm_query_batched` rather than print them. One might argue that careful prompting suffices and no reward term is needed. This conflates *instruction* with *credit assignment*. Under correctness-only training, a lean correct trajectory and a wasteful correct trajectory receive identical reward, so the GRPO advantage that flows to their tokens is identical: the optimizer gets *no gradient* distinguishing them. Prompting changes the prior over trajectories; it cannot teach the policy *which* of two correct trajectories to prefer, because the learning signal is flat across them. The objective supplies precisely the missing signal. The signal, not just the instruction, is what makes efficiency learnable.

**(3) "Just train longer or bigger."** Scaling steps or model size is *orthogonal* to the question this thesis asks. The claim is about the *objective* — whether a correctness-gated efficiency term changes *how* a fixed policy orchestrates the scaffold — and it is testable at fixed model (Qwen3-30B-A3B-Instruct-2507), fixed data (the OOLONG-Spam / BrowseComp-Plus 0.5/0.5 split), fixed compute, and fixed eval suite. The A/B design holds all of these constant and varies only $\lambda$, so any difference is attributable to the objective and not to scale. Training longer or bigger might raise $c$, but it does not address the cost-separation question and would confound the comparison if allowed to vary.

## 3.6 Relationship to potential-based shaping

Classical reward shaping warns that adding a term to a reward can change the optimal policy [Ng et al., 1999]. The safe family is *potential-based* shaping, $F(s,s') = \gamma\,\Phi(s') - \Phi(s)$, which is provably *policy-invariant*: it can speed learning but cannot alter the set of optimal policies. Our efficiency bonus is **not** potential-based — it is a terminal multiplicative factor on the episode return, not a telescoping per-transition potential — so policy invariance does not follow automatically, and we must establish the property we actually need by other means.

That property is **correctness dominance**, and Section 3.3 establishes it directly from the gate rather than from a telescoping argument. The decisive structural choices are: (i) the bonus is *multiplicative* on the gated base, so it scales correctness rather than competing with it additively; (ii) it is *gated* at `correct_threshold = 1.0`, so it is identically zero outside the fully-correct set; and (iii) it is *bounded* by $\lambda$, so even at maximal efficiency a correct rollout's reward stays in $[1, 1+\lambda]$ while every incorrect rollout stays in $[0,1]$. The optimal *correctness* ordering — every right answer ranked above every wrong one — is therefore preserved exactly, for all $\lambda \geq 0$. What the bonus *does* change, intentionally, is the ordering *within* the optimal-correctness set: among trajectories that are all maximally correct, it prefers the cheap ones. This is the one place we *want* to break invariance, and the gate confines the break to exactly that place.

In the language of reward hacking [Skalse et al., 2022; Amodei et al., 2016], the design is a deliberate hedge against Goodhart's law: by making the efficiency proxy *strictly dominated* by correctness, we ensure that any pressure the proxy exerts can only act after the true objective is satisfied. The residual risk is not that efficiency overrides correctness — the invariants forbid that — but that the *correctness signal itself* is gameable, since the bonus rewards the cheapest trajectory the scorer still calls correct and thereby surfaces any scorer loophole (for example the OOLONG substring fallback or the BrowseComp-Plus containment fallback). Hardening the verifier $c$ is therefore the necessary complement to gating, and we take it up in Chapter 5. The objective of this chapter is correct-first by construction; making it *robustly* correct-first is an empirical and engineering obligation addressed there.
