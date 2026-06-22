# A stronger thesis for RLM scaffold training

## Where the authors stopped

The released RLM scaffold (`mit-oasys/rlm-qwen3-30b-a3b-v0.1`) and the upstream
training harness optimize a **correctness-only** reward:

```
R = c        # c in [0, 1], a verifiable task-correctness score
```

Concretely, upstream `RLMTrainRubric` returns correctness and exposes
`rlm_iterations`, `rlm_repl_calls`, `rlm_sub_llm_calls`, `gated_reward`, etc.
*only as monitoring metrics*. The OOLONG loader does not even enable reward
gating. Efficiency in the paper is therefore an **emergent side effect** of (a)
the orchestrator system-prompt addendum, (b) non-thinking sampling + output
caps, and (c) bounded iterations + an exec timeout — never an optimization
target. (See `notes/efficient-rlm-replication.md` and
`notes/source-claim-audit.md`.)

This is a real result, but it leaves the central question of scaffold training
unanswered: **the reward says nothing about whether the model used the scaffold
well, only whether it eventually got the answer.**

## The stronger thesis

> **Scaffold quality is a separable, learnable objective.**
> A Recursive Language Model is not just a model that emits a correct final
> token; it is a *policy over a scaffold* — it decides when to read context,
> when to chunk, when to delegate to a sub-LLM, how wide to batch, when to
> verify, and when to stop. Correctness-only RL (`R = c`) trains *whether* the
> answer is right but leaves *how the scaffold is operated* unsupervised and
> high-variance. We claim that scaffold-process quality can be optimized
> **on top of** correctness, as a strictly dominated bonus, so that the policy
> learns to reach the same correct answers with measurably fewer turns,
> sub-calls, and tokens — **without** sacrificing correctness.

Said as a single sentence:

> **RLM training should optimize a correct-first, efficiency-second objective —
> `R = c · (1 + λ·e)` — where correctness `c` always dominates and a bounded,
> correctness-gated efficiency term `e` shapes the trajectory the model takes to
> get there.**

This is intentionally a *superset* of the authors' objective. At `λ = 0` it is
exactly `R = c`. The thesis is the claim that some `λ > 0` Pareto-improves
trajectory cost at equal correctness.

## Second deep-dive readiness verdict

This is strong enough to be the headline for a second deep dive because it is:

- **A real delta over the authors' result:** the first result shows that
  correctness-only RL can make the scaffold useful; this thesis asks whether
  scaffold-operation quality is itself trainable.
- **Mechanistic, not just empirical:** it names the policy being learned
  (read/chunk/delegate/batch/verify/stop), the failure modes it targets
  (premature finalization, one-call-per-item loops, long-tail cost), and the
  training signal that should change those behaviors.
- **Strictly controlled:** the `λ = 0` arm is the authors' `R = c`; the treatment
  differs by one reward knob, so a win is attributable to process shaping rather
  than model/config drift.
- **Falsifiable:** it predicts equal-correctness cost reductions, especially in
  p95 trajectory cost; if those reductions do not appear without an accuracy
  hit, the thesis fails.
- **Actionable in this repo:** the reward, telemetry, config, and validators are
  already present, so the second deep dive can be run as an experiment rather
  than argued as prose.

## Why this is the right "stronger" claim (and not a weaker one)

Three weaker claims we explicitly reject as the headline:

1. *"Add a token-cost penalty to the reward."* — Rejected as the primary
   framing. An ungated `R = c − μ·cost` term can make a wrong-but-cheap rollout
   outscore a right-but-expensive one, which is exactly the paper's documented
   failure mode (premature `FINAL`, plan-as-answer). The authors were right to
   avoid it. Our thesis keeps efficiency **strictly subordinate** to
   correctness.
2. *"Efficiency is just prompting."* — The paper already shows prompting helps,
   but prompting cannot be *credit-assigned*: the model gets no gradient signal
   distinguishing a lean correct trajectory from a wasteful correct one. The
   thesis is that the *signal*, not just the instruction, matters.
3. *"Train longer / bigger."* — Orthogonal. Our claim is about the *objective*,
   and is testable at fixed model and compute.

## Formal statement

For a rollout with correctness `c ∈ [0, 1]` and normalized efficiency
`e ∈ [0, 1]` (fewer turns/sub-calls/tokens → higher `e`):

```
R = base(c)                       if c < correct_threshold   (no bonus)
R = base(c) · (1 + λ · e)         if c >= correct_threshold  (gated bonus)
```

where `base(c)` is the upstream (optionally gated) correctness value, `λ ≥ 0` is
`shaping_coef`, and `e` is a weighted mean of per-axis efficiencies
`max(0, 1 − used/budget)` over enabled axes (turns, sub-LLM calls, sub-LLM
tokens). Key invariants:

- **Parity:** `λ = 0` ⇒ `R = base(c)` ⇒ byte-identical to upstream.
- **Dominance:** in the main binary/full-correct setup (`correct_threshold =
  1.0`, `c ∈ [0, 1]`), any fully-correct rollout still scores ≥ any incorrect
  rollout, so efficiency can only re-rank *within* the fully-correct set. If an
  environment exposes continuous partial credit and sets `correct_threshold <
  1.0`, choose `λ` below the minimum correctness gap you want to preserve, or
  binarize the shaping gate, because efficiency should never be allowed to
  convert "less correct but cheap" into the preferred policy.
- **Monotonicity:** holding `c` fixed and correct, `R` is non-increasing in each
  budget axis usage. Spending more never pays.
- **Boundedness:** `R ∈ [0, c·(1+λ)]`; the bonus is capped.

## Falsifiable predictions

Train two arms at fixed base model, data, compute, and train-environment mix
(same OOLONG-Spam + BrowseComp-Plus split):

- **Control:** `shaping_coef = 0` (the authors' `R = c`).
- **Treatment:** `shaping_coef = λ > 0` with turn/sub-call/token axes enabled.

Then:

- **P1 (no correctness regression):** treatment task accuracy ≥ control − ε on
  the held-out eval suite (OOLONG @132k, OOLONG-Pairs @32k, BrowseComp-Plus
  n=150/k=50, LongBench-v2 CodeQA n=50).
- **P2 (cheaper trajectories):** treatment uses strictly fewer mean turns and/or
  sub-LLM calls and/or sub-LLM tokens at equal accuracy (a Pareto improvement).
- **P3 (lower tail cost):** treatment shrinks the p95 of turns/tokens — directly
  targeting the paper's "long-tailed, high-variance cost" failure mode (§F.2).
- **P4 (generalization):** the efficiency gain transfers to a held-out task
  family / larger context, mirroring the paper's length-generalization result
  (Obs. 6).

The thesis is **falsified** if, across reasonable `λ`, treatment cannot beat
control on P2/P3 without violating P1 — i.e. if scaffold cost and correctness
turn out to be inseparable under this objective.

## Uncertainties the deep dive must resolve

- **Best `λ` and budgets:** `λ=0.2`, `subcall_budget=64`, and
  `token_budget=200000` are starting values, not established optima.
- **Token telemetry coverage:** sub-LLM token cost depends on provider/client
  usage metadata. When token usage is missing, turns and sub-call counts remain
  reliable, but token-cost claims should be reported only where usage is
  populated.
- **Metric alignment:** OOLONG and LongBench CodeQA have deterministic scoring;
  BrowseComp-Plus depends on judge behavior, so it should be reported with the
  judge model/version and fallback behavior.
- **Reward hacking:** the treatment should be inspected for degenerate
  shortcuts such as doing less work by finalizing early. P1 and trajectory
  audits are required, not optional.

## Overcoming the main caveat: reward hacking

The caveat is real: an efficiency term can accidentally teach "do less" instead
of "operate the scaffold better." The way around it is to make efficiency a
**post-correctness discriminator**, not an independent reward. The treatment
should be considered valid only under this anti-hacking protocol:

1. **Gate all efficiency by correctness.** The bonus is only available after the
   rollout is correct (`correct_threshold = 1.0` for binary tasks). Cheap wrong
   answers, empty answers, premature finals, and scorer exploits get no
   efficiency reward. This is already the role of `EfficiencyGatedRubric`.
2. **Keep evaluation unshaped.** Train may use `R = c·(1+λ·e)`, but eval must
   report plain task correctness plus trajectory cost. The thesis config does
   this: all eval envs omit `shaping_coef`.
3. **Use paired A/B with identical rollouts per prompt.** Compare `λ=0` control
   vs. `λ>0` treatment at the same base model, data, sampling, caps, and eval
   suite. The only intended behavioral difference is the reward knob.
4. **Sweep `λ`, do not trust one value.** Start small (`λ=0.05, 0.1, 0.2`) and
   select only points on the Pareto frontier: no correctness loss, lower mean
   and p95 cost. If larger `λ` improves cost by degrading accuracy, reject it.
5. **Audit trajectories, not just scalar rewards.** For each eval env, sample
   correct treatment rollouts with unusually low cost and manually/automatically
   inspect whether they still (a) read enough context, (b) cite/extract evidence
   where required, (c) use plausible chunking/delegation, and (d) avoid
   "answer-first, verify-never" behavior.
6. **Stress with adversarial canaries.** Add or reserve cases where the obvious
   short answer is wrong unless the model actually inspects the context. The
   treatment must not gain efficiency by skipping the discriminating evidence.
7. **Separate train reward from publication metric.** The headline result should
   be a frontier plot: accuracy on the y-axis, cost on the x-axis (turns,
   sub-calls, tokens, p95). Claim success only for treatment points that move
   left without moving down.

In short: **never reward cheapness directly; reward cheapness only among
verified-correct, audit-passing trajectories.** If this protocol holds, the
efficiency term becomes a tie-breaker over good scaffold policies rather than a
shortcut to under-thinking.

### Why gating is necessary but not sufficient: harden `c` first

The protocol above gates efficiency on the correctness signal `c`. That is the
right structure, but it quietly assumes `c` is faithful. It is not automatically
so, and **adding an efficiency term amplifies every weakness in the verifier**:
the policy is now explicitly rewarded for finding the *cheapest* trajectory the
scorer will still call correct, which is exactly the behavior that surfaces
scorer loopholes.

Two concrete loopholes already exist in this repo's scorers:

- **OOLONG substring fallback** (`_oolong_synth_score`): a last-resort branch
  returns `1.0` when `gold.lower()` is a substring of the raw output. A verbose,
  low-effort answer that happens to contain the gold token can score correct.
- **BrowseComp-Plus containment fallback** (`_score_browsecomp_plus`, used when
  `reward_mode != "judge"` or the judge call fails): returns `1.0` when the
  normalized gold and answer contain each other. Containment is weaker than the
  paper's LLM-judge accuracy.

Under correctness-only `R = c`, these fallbacks mostly add a little reward noise.
Under `R = c·(1 + λ·e)`, they become **targets**: the cheapest correct-looking
trajectory is the global optimum, so the policy is pushed toward "emit something
that trips the fallback, then stop." Gating does not fix this, because the
exploit is scored as genuinely correct.

So the first mitigation is not a reward knob, it is the **verifier**:

1. **Make the gate use the strictest available `c`.** For BrowseComp-Plus, gate
   the efficiency bonus on the LLM judge (`reward_mode="judge"`), never on the
   containment proxy. Treat the proxy as offline smoke-test only.
2. **Remove or tighten lenient fallbacks for the shaped arm.** For OOLONG, the
   shaped run should disable the substring fallback (require parsed-answer
   equality), so "answer contains gold" is not enough to unlock the bonus.
3. **Optionally gate on a stricter signal than the trained reward.** The bonus
   may require a second, independent correctness check (e.g. exact-match *and*
   judge agreement) so a single scorer loophole cannot release efficiency
   reward.
4. **Require minimum genuine work, not just `min_iterations≥2`.** Two iterations
   is a weak floor; for context-heavy envs, also require evidence the context
   was actually inspected (a sub-LLM call or a context read) before the bonus is
   available, so "guess on turn 2" cannot be both correct-by-fallback and
   maximally efficient.

The test: if you can write a trivial policy that scores `c=1` cheaply *without*
doing the task, the efficiency term will find it. Close that hole in `c` before
turning up `λ`. Treat any cost reduction that coincides with a rising
fallback-hit rate as reward hacking, not progress.

## How it is operationalized in this repo

- `training/src/rlm_train/shaping.py` — `EfficiencyGatedRubric`, a strict
  superset of `RLMTrainRubric`. `shaping_coef=0.0` reproduces `R = c` exactly;
  `shaping_coef>0` adds the correctness-gated bonus. Emits `efficiency_bonus`
  and `rlm_efficiency_score` as shaping metrics.
- `training/src/rlm_train/env.py` — accumulates `rlm_sub_llm_tokens` from proxy
  usage as **telemetry only** (no effect on `R = c`). `RLMTrainRubric` exposes
  that telemetry as a zero-weight metric even for unshaped control/eval runs.
- `training/environments/rlm_*_local/` — the four standalone per-env packages;
  each `env.py` `load_environment` accepts opt-in `shaping_coef` / budget /
  weight kwargs via `_build_rubric`; default is the stock correctness-only rubric.
- `training/configs/rlm-qwen3-30b-thesis.toml` — paired control/treatment
  config: equal-weight prime-rl multi-env training over OOLONG-Spam and
  BrowseComp-Plus, with `shaping_coef` enabled for both train envs. Set both
  `shaping_coef` values to `0.0` for the correctness-only control.
- `tools/validate_thesis_shaping.py` — proves the invariants above
  (parity, dominance, monotonicity, boundedness, gating).

## What this does NOT claim

- It does not claim to reproduce the authors' private mixed-suite run.
- It does not change the default reward anywhere; correctness-only remains the
  out-of-the-box behavior.
- It does not assert a specific best `λ`; finding it is the experiment P1–P4
  define.
