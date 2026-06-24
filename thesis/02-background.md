# 2. Background and Related Work

This chapter situates the contribution of this thesis within several overlapping
bodies of work: the management of long contexts in language models
(Section 2.1); the Recursive Language Model (RLM) paradigm and the code-action
agents from which it descends (Section 2.2); reinforcement learning for large
language models, with particular attention to the verifiers + prime-rl training
stack that this work uses (Section 2.3); the theory of reward shaping and the
distinction between process and outcome supervision (Section 2.4); the failure
mode of reward hacking under Goodhart's law (Section 2.5); the evaluation
benchmarks and base model that define the empirical envelope (Section 2.6); and
finally the small but growing literature on cost-awareness in agentic systems,
against which the precise delta of this work is positioned (Section 2.7).

The through-line is a single claim that the remaining chapters formalize and
operationalize: the *efficiency* of an RLM trajectory — how many turns, sub-LLM
calls, and sub-LLM tokens it consumes to reach a correct answer — is a
separable, *learnable* objective rather than an emergent side effect of
structural caps. The released RLM scaffold trains on a correctness-only reward
$R = c$ and obtains efficiency only indirectly, through prompt hints and
hard ceilings [Zhang et al., 2026]. This thesis instead proposes the
correct-first, efficiency-second objective $R = c\,(1 + \lambda e)$ and asks
whether the efficiency that the scaffold leaves to emergence can instead be
credit-assigned by the reward. Everything below is the scaffolding required to
make that question precise and falsifiable.

## 2.1 Long-context language models and the context-management problem

The dominant trajectory of recent language-model research has been the
extension of the usable context window — from a few thousand tokens to hundreds
of thousands or millions. Two structural limits, however, persist regardless of
the nominal window. First, the *effective* window is far smaller than the
advertised one: models attend unevenly across long inputs, and accuracy
degrades sharply when the relevant evidence is buried in a large distractor
field. Second, even where attention is adequate, the *cost* of a long context is
quadratic-or-worse in attention and linear in memory, so simply enlarging the
window is neither free nor reliable.

This gap motivates *context management* as a first-class concern. Rather than
forcing the entire context through a single forward pass, a context-management
system arranges for a model to inspect, summarize, filter, and selectively
re-read a large external object. The base model used in this work,
`Qwen/Qwen3-30B-A3B-Instruct-2507`, advertises a long native window, but the
RLM scaffold deliberately constrains the *orchestrator's* window to roughly
$16\mathrm{k}$ tokens (`max_model_len=16384`) and instructs the policy never to
print or echo raw context: every per-environment user prologue carries the
warning that "the full context lives in the REPL and is far larger than your
window. NEVER print, paste, or echo raw context" and that chunks must be passed
"as ARGUMENTS to `llm_query_batched` (never print them)"
(`training/environments/oolong/oolong/env.py:45`). The context is therefore not
a thing the model reads in one pass; it is a thing the model *manages*. The
remainder of this chapter is about a system that makes this management
programmatic, and about how to teach that system to manage *cheaply*.

The benchmarks in Section 2.6 are constructed precisely to stress this gap:
OOLONG places a single answer inside $131\mathrm{k}$–$262\mathrm{k}$ tokens of
TREC-derived text; BrowseComp-Plus places a few gold and evidence documents
amid dozens of near-duplicate negatives; LongBench-v2 CodeQA places a multiple-
choice answer inside a multi-megabyte code repository. In all three the naive
strategy — paste everything into the window — is either impossible or
ruinously expensive, which is what makes them suitable testbeds for both the
RLM paradigm and for an *efficiency* objective layered on top of it.

## 2.2 Recursive Language Models and code-action agents

### 2.2.1 The RLM paradigm

A Recursive Language Model is, in the words of the foundational paper, a
*task-agnostic inference strategy* that lets a language model "programmatically
examine and recursively call" itself over near-infinite-length contexts
[Zhang et al., 2026]. Concretely, the canonical call
`llm.completion(prompt, model)` is replaced by `rlm.completion(prompt, model)`.
The difference is that `rlm.completion` does not feed the prompt to the model
directly. Instead it offloads the (possibly enormous) context into a variable
inside a Python read-eval-print loop (REPL), and lets an *orchestrator* language
model interact with that variable through code: slicing it, grepping it,
chunking it, and — crucially — issuing *sub-LLM calls* on selected pieces. Those
sub-calls are first-class functions in the code the orchestrator writes; a
sub-call may itself be another RLM call, which is where the recursion in the
name originates [Zhang et al., 2026]. This thesis, like the released training
environment, operates at *depth 1*: a single orchestrator that issues flat
(non-recursive) sub-LLM calls, which is the regime the training harness mirrors
(see Section 2.3).

Three properties of this design matter for what follows. (i) The context is a
*variable*, not a prompt: it never enters the orchestrator's attention in bulk,
which decouples the problem size from the orchestrator's window. (ii) The
sub-LLM call is a *function*: the orchestrator composes calls, batches them, and
consumes their returned strings, exactly as it would any other library call.
(iii) The whole thing is *code-action*: the unit of action is a Python block,
not a JSON tool invocation. These properties make an RLM trajectory naturally
*measurable* — turns, sub-calls, and sub-call tokens are observable counters —
which is precisely what an efficiency objective needs.

### 2.2.2 CodeAct lineage

The decision to make the action space *code* rather than structured JSON tool
calls is inherited from CodeAct [Wang et al., 2024]. CodeAct argues that
executable Python actions elicit better agent behavior than the prevailing
JSON-tool-calling interface, because code composes: an agent can branch, loop,
store intermediate results in variables, and call functions on prior outputs,
all within a single action. The RLM repository cites this lineage explicitly as
a design bet — a code environment with function-like sub-LLM calls in place of
JSON tools (`README.md:33`). For this thesis the CodeAct framing is more than
heritage: because actions are code, the *cost* of a trajectory is exposed
through the same channel as the actions themselves. The orchestrator that writes
a lean three-line script and the one that writes an eighteen-turn re-verifying
loop differ in observable, creditable ways, which is what makes efficiency a
trainable target rather than a stylistic preference.

### 2.2.3 The released scaffold and emergent efficiency

The released RLM scaffold, `mit-oasys/rlm-qwen3-30b-a3b-v0.1`, is a LoRA adapter
($r=32$, $\alpha=64$, on the $q/k/v/o$ projections) over
`Qwen/Qwen3-30B-A3B-Instruct-2507`, trained on a mixed long-context suite
(an OOLONG-Spam + BrowseComp-Plus split) with `RLMTrainEnv` and `prime-rl`
under a *correctness-only* reward $R = c$ [Zhang et al., 2026]
[zhang2026rlm_hf]. Critically for this thesis, the scaffold obtains its
efficiency not from any optimization objective but from four structural levers
[Zhang et al., 2026]:

1. an orchestrator system-prompt addendum that forbids tiny-prompt mega-batches
   (the "Plan before you act" hint, propagated into the per-environment
   prologues, e.g. `training/environments/oolong/oolong/env.py:43`);
2. non-thinking sampling with output caps (`enable_thinking=false`,
   `max_completion_tokens=4096`);
3. bounded iterations (`max_iterations=20`) plus a per-REPL-block wall-clock
   guard (`RLM_TRAIN_EXEC_TIMEOUT_S=180`s in `training/src/rlm_train/worker.py`);
   and
4. pre-batch structural filters (repetition and zero-advantage).

The paper reports two observations that this thesis directly leverages. First,
RLM cost is *long-tailed and high-variance*: a minority of trajectories re-verify
or re-generate many times, dominating mean cost (paper §F.2). Second, the
paradigm *generalizes in length*: a post-trained RLM-Qwen3-8B is reported as
$3.2\times$–$9.6\times$ faster with $68$–$90\%$ less runtime (Observation 6).
Both observations describe efficiency that *emerged* from training on a
correctness-only signal. The central wager of this thesis is that what emerged
indirectly can be obtained *directly* and more reliably by making efficiency a
gated, subordinate term in the reward — testable at fixed model, data, and
compute. The structural caps remain in place; Section 2.4 and Chapter 3 argue at
length why a *reward* over efficiency is not redundant with those caps.

It must be stated plainly, as the source-claim audit does
(`notes/source-claim-audit.md`), that the exact private training recipe is not
fully public: upstream publishes the OOLONG environment but not the
BrowseComp-Plus training environment, and the exact per-environment eval-time
prologues are *evidence-derived* from PrimeIntellect research-environments and
LMxLM traces rather than byte-identical to the private conditioning. Where this
thesis depends on a reconstructed detail rather than a verified one, that
dependency is flagged.

## 2.3 Reinforcement learning for language models and the training stack

### 2.3.1 From RLHF to verifiable rewards

The modern recipe for aligning and improving language models with reinforcement
learning was crystallized by InstructGPT, which fit a reward model to human
preference comparisons and then optimized the policy against that learned reward
with PPO [Ouyang et al., 2022]. A learned preference model is, however, both
expensive and itself hackable. A complementary line of work replaces the learned
reward with a *verifiable* one: where a task admits a programmatic check of
correctness — a unit test, an exact-match grader, a math answer — the reward is
that check, applied directly. Tulu 3 systematized this as *RL with verifiable
rewards* (RLVR) and demonstrated its effectiveness across reasoning and
instruction-following [Lambert et al., 2024]. The RLM training setup is squarely
in the RLVR family: every one of its four evaluation environments computes a
programmatic correctness scalar $c \in [0,1]$ (Section 2.6), and the orchestrator
is optimized against that scalar. This thesis inherits the RLVR stance and adds
exactly one term — a correctness-*gated* efficiency bonus — leaving the verifier
as the dominant signal.

### 2.3.2 GRPO

The policy-optimization algorithm used here is Group Relative Policy
Optimization (GRPO), introduced with DeepSeekMath [Shao et al., 2024]. GRPO
dispenses with a separately trained value/critic network. For each prompt it
samples a *group* of `group_size` rollouts and computes a per-rollout baseline as
the group mean reward; every token in rollout $i$ then receives the *relative*
advantage $s_i - \mathrm{mean}(s)$ over the group, which prime-rl implements as
the default DR-GRPO form (per-group reward minus per-group baseline, without
standard-deviation normalization) [prime-rl]. This group-relative structure is
exactly why a *bounded, gated* reward is attractive: advantages are computed
*within* a group of rollouts for the *same* prompt, so a reward that re-ranks
trajectories only inside the fully-correct set (Section 2.4, Chapter 3) directly
shapes the gradient *between* a lean correct rollout and a wasteful correct one,
provided both appear in the same group. A reward shaped this way nudges the
group baseline and the relative advantages in favor of the cheaper correct
trajectory without ever rewarding an incorrect one.

### 2.3.3 The verifiers + prime-rl stack

The concrete training apparatus is the verifiers environment library
[verifiers] driven by the prime-rl trainer [prime-rl]. The RLM is wrapped as a
verifiers `MultiTurnEnv` subclass, `RLMTrainEnv`
(`training/src/rlm_train/env.py:32`), which mirrors a depth-1 `RLM.completion`
call. At `setup_state`
(`training/src/rlm_train/env.py:108-169`) the environment registers a
`SubLLMProxy` (an aiohttp server) per rollout, stands up a subprocess-isolated
`SubprocessReplBackend`, and initializes the telemetry state keys that the
efficiency objective will later read: `rlm_iterations`, `rlm_sub_llm_calls`,
`rlm_sub_llm_tokens`, and `rlm_sub_llm_usage_missing`
(`training/src/rlm_train/env.py:153-157`). As the trajectory proceeds, the
environment extracts code blocks and executes them in the REPL, incrementing
`rlm_iterations` after each block
(`training/src/rlm_train/env.py:180-227`), while `_record_sub_call`
accumulates sub-call token counts from the proxy's usage metadata, incrementing
`rlm_sub_llm_usage_missing` whenever usage is absent or malformed
(`training/src/rlm_train/env.py:254-277`).

Reward computation lives in `RLMTrainRubric`
(`training/src/rlm_train/rubric.py:13-126`). It computes the correctness scalar
$c$ via a per-environment scoring function and then *gates* it through
`_passes_gates` (`training/src/rlm_train/rubric.py:44-51`), which enforces the
usage-floor thresholds `min_iterations` and `min_subcall`; in gated mode a
rollout that fails the gates, or whose value falls below `min_reward`, is zeroed
(`training/src/rlm_train/rubric.py:61-77`). The stock rubric already *exposes*
`rlm_sub_llm_tokens` as a zero-weight metric
(`training/src/rlm_train/rubric.py:103-104`) — it is observed but not rewarded.
The contribution of this thesis, `EfficiencyGatedRubric`
(`training/src/rlm_train/shaping.py:106-201`), is a strict superset of that stock
rubric: at `shaping_coef == 0.0` its reward is byte-identical to the upstream
$R = c$, and at `shaping_coef > 0.0` it adds the bounded, correctness-gated
bonus, registering two new metrics, `rlm_efficiency_score` and
`efficiency_bonus`. The mechanics are the subject of Chapters 3 and 4; what
matters here is that the entire objective is implemented *inside* the existing
verifiers rubric abstraction, with no change to the trainer.

On the systems side, prime-rl trains the LoRA adapter and *hot-loads* the updated
weights into the inference server between steps. The vLLM inference server
exposes a `/load_lora_adapter` endpoint
(`prime-rl/src/prime_rl/inference/vllm/server.py:98-105`), and prime-rl
monkey-patches vLLM's `LoadLoRAAdapter` to permit reloading the *same* adapter
name repeatedly (`.../server.py:35-37`), with runtime LoRA updating enabled via
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
(`prime-rl/src/prime_rl/inference/server.py:18-19`). This hot-load mechanism is
what makes the paired A/B economical: control ($\lambda = 0$) and treatment
($\lambda = 0.2$) differ only in a LoRA adapter and a single reward knob, so the
same infrastructure runs both arms with identical caps and sampling. The loss
itself is prime-rl's DPPO, which clips the importance ratio against a `delta`
threshold and adds a KL regularizer with temperature `tau_KL`
(`prime-rl/docs/algorithms.md:40-62`); these are left at their stock values and
are identical across arms, so they are confounds neither introduces.

## 2.4 Reward shaping: process versus outcome, additive versus multiplicative

### 2.4.1 Potential-based shaping and policy invariance

The classical theory of reward shaping is due to Ng, Harada, and Russell, who
proved that augmenting a reward with a *potential-based* shaping term
$F(s,s') = \gamma \Phi(s') - \Phi(s)$ leaves the set of optimal policies
unchanged: such shaping can accelerate learning without altering what is
optimal [Ng et al., 1999]. The result is a cautionary baseline. Most ad-hoc
reward additions are *not* potential-based and therefore *do* change the optimal
policy — sometimes catastrophically, by making a behavior optimal that the
designer never intended (Section 2.5). The objective in this thesis is, by
construction, *not* potential-based; it deliberately changes the optimal policy,
but in a constrained way: it re-ranks only *within* the set of fully-correct
trajectories and never promotes an incorrect one above a correct one. The formal
invariants that make this change safe — parity, dominance, monotonicity, and
boundedness — are derived in Chapter 3. The point to register here is that the
design is a conscious departure from policy-invariant shaping, justified by an
explicit argument about *which* policies it is permissible to re-rank.

### 2.4.2 Process versus outcome supervision

A second axis is *what* the reward supervises. Outcome supervision rewards the
final answer; process supervision rewards the trajectory that produced it. The
correctness scalar $c$ is pure outcome supervision — it grades `answer["content"]`
and ignores how the orchestrator got there. The efficiency term $e$ is a form of
*process* supervision: it grades the trajectory's resource consumption — turns,
sub-calls, tokens — rather than its final answer. The two are combined so that
process supervision is strictly subordinate to outcome supervision. This is a
deliberate inversion of the usual worry about process supervision, namely that
rewarding the *form* of reasoning can reward plausible-looking but wrong
processes. Here process is rewarded *only after* the outcome is verified correct,
so the failure mode of "rewarding a nice-looking wrong process" is structurally
excluded.

### 2.4.3 Why a gated multiplicative bonus, not an additive cost penalty

This subsection states the central design decision that distinguishes this work
from the obvious alternative. The tempting baseline is an *additive cost
penalty*:
$$R = c - \mu\,\mathrm{cost},$$
which simply subtracts a scaled cost from correctness. This thesis explicitly
*rejects* that form as the headline objective. The reason is decisive: for any
positive $\mu$, a sufficiently cheap *wrong* rollout ($c = 0$, tiny cost) can
out-score an expensive *correct* one ($c = 1$, large cost), because the penalty
is ungated and applies to correct and incorrect trajectories alike. This is
exactly the paper's premature-`FINAL` / plan-as-answer failure: the policy learns
to emit a short, cheap, confident answer that the grader sometimes accepts,
trading correctness for cheapness. Efficiency must remain *strictly subordinate*
to correctness, and an additive penalty cannot enforce that.

The objective adopted instead is the *gated multiplicative* form
$$R = c\,(1 + \lambda e),$$
applied only when the gated correctness value meets `correct_threshold`
($= 1.0$) and the rollout passes the stock usage gates; otherwise $R$ collapses
to the base correctness value (`training/src/rlm_train/shaping.py:167-177`). The
multiplicative coupling makes the bonus a *fraction of correctness*: a wrong
rollout has $c$ small or zero, so the bonus — even at maximal efficiency — is
correspondingly small or zero and can never lift it above a correct rollout. With
`correct_threshold=1.0` the bonus is moreover applied *only* to fully-correct
rollouts; efficiency becomes a pure tie-breaker *within* the correct set. The
formal statement is that with $c \in [0,1]$ and the gate at $1.0$, any
fully-correct rollout scores in $[1, 1+\lambda]$ while any incorrect rollout
scores at most its (sub-unit) correctness value, so correctness dominates
unconditionally; monotonicity then guarantees that, holding correctness fixed,
spending more on any budgeted axis never pays. These claims are proved in
Chapter 3 and pinned by the test suite (Chapters 4 and 5).

The efficiency score itself is a weighted mean over enabled axes of the per-axis
efficiency $\max(0,\,1 - u/B)$, where $u$ is the usage on that axis and $B$ its
budget (`training/src/rlm_train/shaping.py:43-103`). The three axes and their
canonical budgets are turns ($B_{\text{turns}} = $ `max_iterations` $= 20$),
sub-LLM calls ($B_{\text{calls}} = $ `subcall_budget` $= 32$), and sub-LLM
tokens ($B_{\text{tok}} = $ `token_budget` $= 200000$), with equal unit weights.
If no axis is enabled the score is $0.0$, making the bonus a safe no-op. A short
*illustrative* worked example (parameters: $\lambda = 0.2$, budgets
$20/32/200000$, equal weights, $c = 1$) makes the re-ranking concrete. A lean
correct rollout using $3$ turns, $3$ sub-calls, and $35{,}000$ sub-LLM tokens has
per-axis efficiencies $0.85$, $0.90625$, and $0.825$, mean $e \approx 0.86042$,
giving $R = 1 + 0.2 \times 0.86042 \approx 1.1721$. A wasteful correct rollout
using $18$ turns, $30$ sub-calls, and $160{,}000$ tokens has efficiencies
$0.10$, $0.0625$, and $0.20$, mean $e \approx 0.12083$, giving
$R \approx 1.0242$. Both score *exactly* $1.0$ under the correctness-only
control. The example is illustrative only; no empirical claim is
attached to it.

## 2.5 Reward hacking, Goodhart's law, and verifier weaknesses

Optimizing a proxy for a true objective is the recurring failure mode of
reinforcement learning. *Concrete Problems in AI Safety* catalogued it under
"reward hacking" and "negative side effects" [Amodei et al., 2016], and Skalse
et al. later gave a formal account, defining a reward as *hackable* when there
exist policies that increase proxy return while decreasing true return, and
characterizing when such hacking is possible [Skalse et al., 2022]. Both are
restatements, for learned agents, of Goodhart's law: once a measure becomes a
target it ceases to be a good measure. *Specification gaming* is the same
phenomenon viewed from the designer's side — the agent satisfies the letter of
the specification while violating its intent.

This literature is not background decoration for the present work; it is the
sharpest objection to it. Adding *any* efficiency term amplifies *every* weakness
in the correctness scalar $c$, because the policy is now rewarded for finding the
*cheapest* trajectory that the scorer still calls correct — which surfaces exactly
the scorer's loopholes. Two such loopholes exist in the current scoring code and
are explicitly in scope. First, the OOLONG synthetic scorer
`_oolong_synth_score` returns $1.0$ on a case-insensitive *substring* fallback
when the gold string appears anywhere in the output
(`training/environments/oolong/oolong/env.py:146-150`); under an efficiency
incentive the policy could learn to dump a short string that happens to contain
the gold token. Second, the BrowseComp-Plus deterministic scorer
`_score_browsecomp_plus` returns $1.0$ on *mutual containment* of normalized
answer and gold (`training/environments/browsecomp_plus/browsecomp_plus/env.py:197-210`),
which fires when `reward_mode != "judge"` or when the judge call fails. An
efficiency term that drives cost down while a fallback-hit rate quietly rises is
reward hacking masquerading as progress.

The mitigations, developed fully in Chapter 5, follow directly: gate the shaped
arm on the *strictest available* correctness signal — for BrowseComp-Plus the
LLM judge (`reward_mode="judge"`, `judge_model="openai/gpt-4.1"`), not the
containment fallback; tighten or disable the lenient substring/containment
fallbacks for the shaped arm; optionally require a second independent check; and
demand a minimum of *genuine* work rather than the bare `min_iterations >= 2`
floor — evidence that the context was actually inspected, since a canary whose
obvious short answer is wrong unless the context is read is the cleanest test of
whether the policy is shortcutting. The governing rule is that any cost reduction
that *coincides* with a rising fallback-hit rate is treated as reward hacking,
not as an efficiency gain.

## 2.6 Evaluation benchmarks, the base model, and LoRA

The empirical envelope of this thesis is fixed by a four-environment evaluation
suite, evaluated *unshaped* (no `shaping_coef` on any eval environment) so that
the published metric is stock correctness and cost, never the training reward.
The training distribution is a separate two-environment mix (OOLONG-Spam +
BrowseComp-Plus at a $0.5/0.5$ ratio); the evaluation suite below is held
constant across arms.

| Eval environment | Source key | Context / setting | $n$ | Scoring |
|---|---|---|---|---|
| OOLONG `trec_coarse` | [OOLONG] | `context_len=131072` | 50 | `_oolong_synth_score`: exact, numeric $0.75^{\lvert\Delta\rvert}$ decay, date parse, substring fallback |
| OOLONG-Pairs | [OOLONG], [oolong_hf] | `context_len=32768` | 20 | F1 over unordered `(id1,id2)` pairs |
| BrowseComp-Plus | [BrowseComp-Plus], [browsecomp_plus_corpus] | $k=50$ docs, judge | 150 | LLM judge `gpt-4.1`; deterministic containment fallback |
| LongBench-v2 CodeQA | [LongBench-v2] | repo context | 50 | exact-match letter A–D |

A few details warrant emphasis. *OOLONG* is a long-context QA task built on the
`trec_coarse` subset; the eval uses a $131\mathrm{k}$-token context to match the
Hugging Face eval picture, with comparison answers drawn from exactly three
suffixed phrases — `more common than`, `less common than`, `same frequency as`
(`training/environments/oolong/oolong/env.py:33`) — and the substring fallback
explicitly excludes those phrases to avoid trivial matches
(`.../env.py:146-150`). *OOLONG-Pairs* (the `mit-oasys/oolong-pairs` dataset)
scores F1 over unordered user-ID pairs after stripping `<think>...</think>`
blocks (`training/environments/oolong_pairs/oolong_pairs/env.py:115`).
*BrowseComp-Plus* in the small-model setting used here is the $k=50$
document configuration with judge scoring; this is *not* the paper's
$k=1000$-document stress test, which loads $6$M–$11$M tokens per task for a
GPT-5-class model (`notes/browsecomp-plus-paper-rlm-details.md:35-48`). The two
settings — $k=50$ judge here versus $k=1000$ stress in the paper — must not be
conflated; numbers from one do not transfer to the other (details to verify).
*LongBench-v2 CodeQA* is the THUDM/LongBench-v2 benchmark filtered to the Code
Repository Understanding domain, a four-choice MCQ with deterministic
exact-letter scoring (`training/environments/longbench_codeqa/longbench_codeqa/env.py:87-91`).
All four environments finalize through `answer["content"]` and
`answer["ready"] = True` rather than file writes, and all carry the uniform
$16\mathrm{k}$ context-budget warning discussed in Section 2.1.

The *base model* is `Qwen/Qwen3-30B-A3B-Instruct-2507`, a 30-billion-parameter
mixture-of-experts instruct model with roughly 3B active parameters per token,
sampled in non-thinking mode (`enable_thinking=false`) [Qwen Team, 2025]
[qwen3_30b]. It is chosen over the paper's 8B model because the paper reports
(Appendix B) that small models lacking coding ability struggle with the
code-action harness (details to verify). The policy is adapted with *LoRA*
[Hu et al., 2021] — low-rank adapters that freeze the base weights and learn a
small additive low-rank update — at $r = 32$, $\alpha = 64$, dropout $0.0$, on
the `q_proj`, `k_proj`, `v_proj`, `o_proj` projections, matching the released
scaffold's adapter geometry. LoRA is not incidental: it is what makes the
hot-load A/B of Section 2.3 cheap, since each arm is a small adapter swapped into
a shared base.

## 2.7 Efficiency and cost-awareness in agentic systems: positioning

Cost-awareness in tool-using and agentic systems has, to date, been pursued
predominantly through two channels, neither of which is what this thesis
proposes. The first is *structural*: hard ceilings on iterations, tool calls,
tokens, and wall-clock time — exactly the caps the RLM scaffold uses
(`max_iterations=20`, `seq_len=8192`, `max_model_len=16384`,
`max_completion_tokens=4096`, `sub_max_tokens=4096`, and the REPL wall-clock
guard). Caps are *ceilings, not preferences*. Two fully-correct, cap-satisfying
rollouts can have very different cost and yet tie at $R = c$ under a
correctness-only reward; the cap says only "do not exceed", never "prefer
cheaper". This thesis credit-assigns cost *within* the correct, cap-satisfying
set — a discrimination the caps cannot make, and one pinned by dedicated tests
(`test_structural_caps_leave_correctness_only_rollouts_tied` and
`test_efficiency_reward_separates_cap_satisfying_correct_rollouts`).

The second channel is *prompting*: instructing the model to be terse, to plan
before acting, to batch its sub-calls (the "Plan before you act" addendum). This
thesis rejects the claim that *efficiency is just prompting*. A prompt cannot be
credit-assigned: it conditions every rollout identically and leaves the policy no
gradient distinguishing a lean correct trajectory from a wasteful correct one. A
*reward* over efficiency does provide that gradient. The signal, not merely the
instruction, is what teaches the model to be cheap.

Against this landscape the delta of the present work is specific and narrow. It
is *not* a claim that training longer or with a bigger model yields efficiency —
that axis is orthogonal, and the present claim is deliberately testable at fixed
model and fixed compute. It is the claim that efficiency is a *separable,
learnable objective* expressible as a correctness-gated, bounded, multiplicative
reward $R = c\,(1 + \lambda e)$, distinct from the structural caps it coexists
with and from the prompt hints it complements. The thesis operationalizes this
as a paired A/B — control $\lambda = 0$ (byte-identical to the upstream
correctness-only reward) versus treatment $\lambda = 0.2$ — under four
falsifiable predictions: no correctness regression (P1), strictly cheaper
trajectories at equal accuracy (P2, a Pareto improvement), a shrunken cost tail
(P3, targeting the paper's §F.2 long tail), and transfer of the efficiency gain
to a held-out task family or larger context (P4, mirroring Observation 6). The
formal objective is the subject of Chapter 3; its implementation in
`EfficiencyGatedRubric` and the validated control/treatment twin configs are
the subject of Chapter 4; and the experimental design — including the
seven-point anti-reward-hacking protocol motivated by Section 2.5 — is the
subject of Chapter 5. Results are reported in Chapter 6.

> **[RESULTS PENDING]** No empirical outcomes from the A/B pilot or full-scale
> run are reported in this chapter. The only concrete numbers above are
> configuration constants and the explicitly labelled *illustrative* reward
> arithmetic of Section 2.4.3.
