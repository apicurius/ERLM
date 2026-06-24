# 4. Method and Implementation

This chapter describes how the correct-first, efficiency-second objective formalized in Chapter 3 is realized as running code, and how that code is wired into a controlled A/B training pilot. The central design commitment is that the new objective must be a *strict superset* of the upstream correctness-only reward: at the shaping coefficient $\lambda = 0$ the system must reproduce the released RLM training pipeline byte-for-byte, so that any difference between the control and treatment arms is attributable to the reward knob alone and nothing else. Every component below is built to preserve that invariant.

We proceed bottom-up. Section 4.1 fixes the system architecture: how `rlm.RLM` is wrapped as a verifiers `Environment`, isolated behind a subprocess REPL, and plugged into the prime-rl GRPO trainer. Section 4.2 documents the stock rubric and its usage gates, which the shaped rubric inherits unchanged. Section 4.3 presents `EfficiencyGatedRubric`, the core contribution, grounded line-by-line in `training/src/rlm_train/shaping.py`. Section 4.4 covers the telemetry that makes cost observable even on unshaped runs. Section 4.5 describes the four evaluation environments, their scorers, and the verifier-hardening measures that a cost-bearing reward makes necessary. Section 4.6 closes with the config wiring and the *faithful-twin* invariant that the validator and tests enforce.

## 4.1 System architecture: RLM as a verifiers environment inside prime-rl GRPO

The Recursive Language Model paradigm [Zhang et al., 2026] replaces a flat `llm.completion(prompt, model)` call with `rlm.completion(prompt, model)`: the long context is offloaded into a Python REPL as a variable, and the orchestrator model interacts with it programmatically, issuing sub-LLM calls as first-class functions in code (a CodeAct-style harness [Wang et al., 2024]). Training such a scaffold with reinforcement learning requires turning one depth-1 `rlm.completion` call into an object the RL trainer can roll out, score, and back-propagate through.

That object is `RLMTrainEnv`, a subclass of the verifiers `MultiTurnEnv` defined in `training/src/rlm_train/env.py:32` [verifiers]. It mirrors a depth-1 `RLM.completion`: the orchestrator emits an assistant turn containing one or more fenced code blocks; the environment extracts those blocks and executes them, returns the (truncated) REPL output as the next user turn, and repeats until the policy writes a final answer or the iteration cap is reached. The trajectory-processing loop that extracts and runs code spans `env.py:180-227`, and the per-step turn counter `rlm_iterations` is advanced there as `state["rlm_iterations"] = n_processed + 1` after each successful `backend.execute(code)`.

Two isolation decisions are load-bearing for training stability:

- **Subprocess-isolated REPL.** Code is not executed in-process. The backend factory instantiates a `SubprocessReplBackend` with a startup timeout (`env.py:60-62`), so a rollout that writes an infinite loop, allocates unbounded memory, or imports something slow cannot corrupt the trainer process. A per-REPL-block wall-clock guard, configured through the environment variable `RLM_TRAIN_EXEC_TIMEOUT_S` (180 s in the pilot), kills runaway sequential sub-call loops before they consume the trajectory budget. This guard is one of the structural levers the foundational paper identified for taming RLM's long-tailed cost (Section F.2 of [Zhang et al., 2026]).
- **Per-rollout sub-LLM proxy.** At `setup_state` (`env.py:108-169`) each rollout registers a `SubLLMProxy` (an `aiohttp` server) and a `ClientHandle`, and starts the REPL backend with `proxy_url` and `rollout_id` at `depth=1`. Sub-LLM calls made inside the REPL (e.g. `llm_query_batched(...)`) route through this proxy, which lets the environment both serve those calls from the same inference server and *record their usage* (Section 4.4). The same `setup_state` initializes four telemetry keys to zero: `rlm_iterations`, `rlm_sub_llm_calls`, `rlm_sub_llm_tokens`, and `rlm_sub_llm_usage_missing` (`env.py:153-157`).

The trainer is prime-rl [prime-rl], which implements Group Relative Policy Optimization (GRPO) [Shao et al., 2024]. For each prompt it samples a group of `group_size` rollouts, scores each with the rubric, and assigns every token in rollout $i$ the per-group advantage $s_i - \mathrm{mean}(s)$ — a DR-GRPO-style baseline without standard-deviation normalization (`prime-rl/docs/algorithms.md:135-145`). The policy-gradient update is a decoupled-clip PPO (DPPO) loss that clips the importance ratio against a threshold $\delta$ and adds a KL regularizer with temperature $\tau_{\mathrm{KL}}$ (`algorithms.md:40-62`). This is the RL with verifiable rewards (RLVR) regime [Lambert et al., 2024; Ouyang et al., 2022], in which the reward is a programmatic verifier rather than a learned preference model.

Because the policy is trained with LoRA [Hu et al., 2021], the trainer and the inference server must exchange adapter weights between steps. prime-rl's vLLM server exposes `POST /load_lora_adapter` (`prime-rl/src/prime_rl/inference/vllm/server.py:98-105`) and monkey-patches vLLM's `LoadLoRAAdapter` so the *same* adapter name can be reloaded after each weight update (`server.py:35-37`); runtime adapter swapping is enabled by setting `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (`inference/server.py:18-19`). The net effect is a closed loop: the trainer updates LoRA weights, the inference server hot-loads them, and the next batch of rollouts samples from the updated policy — all while the base 30B-A3B model stays frozen and shared.

The key architectural point for this thesis is that the reward is the *only* component that the treatment arm changes. The environment, the REPL isolation, the proxy, the telemetry plumbing, GRPO, DPPO, and the LoRA hot-loading are identical across arms. The objective lives entirely in the rubric, to which we now turn.

## 4.2 The stock rubric and its usage gates

`RLMTrainRubric` (`training/src/rlm_train/rubric.py:13-126`) is the upstream rubric the released scaffold was trained with. It computes a single scalar correctness reward via an optional user-supplied `correctness` function and surfaces a battery of monitoring metrics. Understanding its gating is a prerequisite for the shaped rubric, which inherits every gate unchanged.

**Correctness wiring.** When a `correctness` callable is supplied, the constructor wraps it with `_make_main_correctness` and registers it as the single reward function with the given `weight` (`rubric.py:31-32`). It also registers a `gated_reward` *metric* (a non-reward diagnostic) and the telemetry metrics enumerated in Section 4.4.

**The usage gates.** Two thresholds guard against degenerate rollouts that never engaged the scaffold. `_passes_gates(state)` (`rubric.py:44-51`) returns `False` unless

$$\texttt{rlm\_iterations} \ge \texttt{min\_iterations} \quad\text{and}\quad \texttt{rlm\_sub\_llm\_calls} \ge \texttt{min\_subcall}.$$

In the thesis configuration `min_iterations = 2` and `min_subcall = 0`, so the binding gate requires at least two REPL turns: a rollout that emits a "final answer" on its very first turn — without ever inspecting the offloaded context — fails the gate. The `min_subcall = 0` setting makes the sub-call count non-binding by default; it exists so that a stricter "must have genuinely called a sub-LLM" gate can be turned on for the shaped arm (Section 4.5).

**The gated value.** The gating behaviour is controlled by `gate_reward`. When `gate_reward = False` (upstream default), `main` simply returns the raw correctness `value` (`rubric.py:68-69`). When `gate_reward = True`, the reward is forced to `0.0` if the rollout fails `_passes_gates`, or if the gated correctness `value < min_reward` (`rubric.py:70-74`). We refer to the value produced by this gated computation — `0.0` if gates fail or `value < min_reward`, else `value` — as the **base value** $\mathrm{base}(c)$. This base value is exactly the upstream reward $R = c$ (modulo the gates), and the shaped rubric is built to reproduce it precisely.

The state keys the gates and metrics read — `rlm_iterations`, `rlm_sub_llm_calls`, `rlm_sub_llm_tokens`, `rlm_sub_llm_usage_missing` — are written by the environment during the rollout (Section 4.1, Section 4.4). The rubric never mutates them; it only reads.

## 4.3 `EfficiencyGatedRubric`: the core contribution

`EfficiencyGatedRubric` (`training/src/rlm_train/shaping.py:106-201`) is the implementation of the Chapter 3 objective. Its module docstring states the design contract directly: with `shaping_coef == 0.0` (the default) the reward is *byte-identical* to the upstream correctness-only reward — the control arm — and with `shaping_coef > 0.0` it adds a bounded efficiency bonus *only* to rollouts that are already correct and pass the same usage gates. It is a strict subclass of `RLMTrainRubric`, so it inherits the gates of Section 4.2 verbatim.

### 4.3.1 Efficiency axes

An efficiency *axis* is a single fewer-is-better budget dimension. The frozen dataclass `EfficiencyAxis` (`shaping.py:43-63`) carries a `state_key`, a `budget`, and a `weight`. Its per-axis efficiency is

$$\texttt{axis\_efficiency}(u, B) = \max\!\big(0,\; 1 - u/B\big) \in [0, 1],$$

computed in `EfficiencyAxis.efficiency` by reading $u = \texttt{used}$ from `state.get(self.state_key)` (defaulting to `0.0`) and returning `max(0.0, 1.0 - used / self.budget)`. Efficiency is $1.0$ at zero usage and decays linearly to $0$ at the budget, then clamps. An axis is **enabled** iff `budget > 0` *and* `weight > 0` (`shaping.py:57-59`); a non-positive budget or weight disables it. This makes opt-in *per axis* the default: a caller must positively supply a budget for an axis to count.

The factory `default_axes` (`shaping.py:66-86`) builds the three standard axes from the config knobs:

| Axis | State key | Budget source | Config value |
|------|-----------|---------------|--------------|
| Turns | `rlm_iterations` | `max_iterations` | $B_{\mathrm{turns}} = 20$ |
| Sub-calls | `rlm_sub_llm_calls` | `subcall_budget` | $B_{\mathrm{calls}} = 32$ |
| Sub-LLM tokens | `rlm_sub_llm_tokens` | `token_budget` | $B_{\mathrm{tok}} = 200000$ |

The turns axis always has a budget because `max_iterations` always exists in the harness; the sub-call and token budgets default to `0.0` in the factory signature, so they are inert unless the config supplies positive values. The thesis config supplies all three with unit weights (`iteration_weight = subcall_weight = token_weight = 1.0`).

### 4.3.2 The efficiency score

`efficiency_score(state, axes)` (`shaping.py:89-103`) reduces the enabled axes to a single $e \in [0,1]$:

$$e = \frac{\sum_{a \,\in\, \mathrm{enabled}} w_a \cdot \texttt{axis\_efficiency}_a}{\sum_{a \,\in\, \mathrm{enabled}} w_a},$$

clamped to $[0,1]$. Critically, **if no axis is enabled the function returns `0.0`** (`shaping.py:96-98`), and likewise if the total weight is non-positive. This makes the bonus a *safe no-op* rather than an accidental free reward: a misconfiguration that disables every axis yields $e = 0$, hence $R = c$, not a spurious reward.

### 4.3.3 Base, shaped, and the gate

The rubric separates the upstream value from the shaped value into two private methods.

`_base_value(state, value)` (`shaping.py:156-165`) replicates the stock gated value exactly: it returns `value` unchanged when `gate_reward` is off; otherwise it returns `0.0` if `_passes_gates` fails or `value < min_reward`, else `value`. This is, by construction, the same $\mathrm{base}(c)$ as Section 4.2.

`_shaped_value(state, base)` (`shaping.py:167-177`) applies the bonus under a triple gate:

1. if shaping is disabled (`shaping_coef == 0`), return `base` unchanged;
2. if `base < correct_threshold`, return `base` (no bonus for not-fully-correct rollouts; with `correct_threshold = 1.0` this means *only* a perfect $c$ qualifies);
3. if `_passes_gates` fails, return `base`;
4. otherwise return $\texttt{base} \cdot (1 + \texttt{shaping\_coef} \cdot e)$ with $e$ from `efficiency_score`.

Composed in `_make_main_correctness` (`shaping.py:179-187`), the reward function is

$$R = \texttt{\_shaped\_value}\big(\texttt{state},\, \texttt{\_base\_value}(\texttt{state}, c)\big),$$

which is exactly the Chapter 3 objective $R = c\,(1 + \lambda e)$ restricted to the gated, fully-correct set, and $R = \mathrm{base}(c)$ everywhere else.

**Parity at $\lambda = 0$.** When `shaping_coef = 0.0`, step 1 of `_shaped_value` returns `base` immediately, so $R = \mathrm{base}(c)$ identically — the byte-identical control. This is not merely asserted; it is pinned by a regression test (Section 4.6). The shaped path therefore *adds* behaviour without *removing* any, satisfying the strict-superset contract.

### 4.3.4 The two registered metrics

When a `correctness` callable is present, the constructor registers two additional metrics (`shaping.py:148-150`), neither of which is a reward function (both have zero weight in the reward sum):

- **`rlm_efficiency_score`** (`shaping.py:200-201`) exposes the raw $e$ for the rollout, so a run can be audited for *where* efficiency is being spent (turns vs. calls vs. tokens) independently of whether the bonus actually fired.
- **`efficiency_bonus`** (`shaping.py:189-198`) reports $\text{shaped} - \text{base}$, i.e. the additive contribution $\lambda \cdot \text{base} \cdot e$ that the bonus made to this rollout's reward. On the control arm this is identically zero; on the treatment arm it is positive only on correct, gate-passing rollouts and zero otherwise.

These two metrics, plus the inherited telemetry of Section 4.4, are what `tools/analyze_pilot.py` reads to decide whether the treatment is genuinely cheaper or merely hacking the scorer.

### 4.3.5 Illustrative reward arithmetic

To make the shaping tangible, consider two *fully correct* ($c = 1$) rollouts under the canonical parameters $\lambda = 0.2$, budgets $B_{\mathrm{turns}} = 20$, $B_{\mathrm{calls}} = 32$, $B_{\mathrm{tok}} = 200000$, equal weights. *(Illustrative arithmetic; not an empirical result.)*

A **lean** rollout using 3 turns, 3 sub-calls, and 35 000 sub-LLM tokens has

$$e_{\mathrm{turns}} = 1 - \tfrac{3}{20} = 0.85,\quad e_{\mathrm{calls}} = 1 - \tfrac{3}{32} = 0.90625,\quad e_{\mathrm{tok}} = 1 - \tfrac{35000}{200000} = 0.825,$$

so $e = 0.86042$ and $R = 1 + 0.2 \cdot 0.86042 = 1.1721$.

A **wasteful** rollout using 18 turns, 30 sub-calls, and 160 000 tokens has $e_{\mathrm{turns}} = 0.10$, $e_{\mathrm{calls}} = 0.0625$, $e_{\mathrm{tok}} = 0.20$, so $e = 0.12083$ and $R = 1.0242$. Both score *exactly* $1.0$ under the correctness-only control, so the control gradient cannot distinguish them; the shaped reward separates them by $\approx 0.148$.

## 4.4 Telemetry: making cost observable

The efficiency reward is only as good as the usage counters it reads, and those counters must be populated identically on both arms so that the control run still *reports* cost even though it does not *reward* it. The accounting lives in the environment, not the rubric, so it is on for every run.

The function `_record_sub_call(state, meta)` (`env.py:254-277`) is the single point of sub-LLM accounting. On each sub-LLM call routed through the proxy it increments `rlm_sub_llm_calls`. If the proxy supplies a `usage` dict in the call metadata, it accumulates `rlm_sub_llm_tokens`, preferring `usage["total_tokens"]` and falling back to `prompt_tokens + completion_tokens` when the total is absent (`env.py:268-274`). When usage is missing or malformed — `meta["usage"]` is not a dict, or the arithmetic raises `TypeError`/`ValueError` — it increments `rlm_sub_llm_usage_missing` instead (`env.py:265, 275-276`). The docstring is explicit that this is *telemetry only*: it does not change the upstream correctness-only reward; the efficiency rubric merely reads these counters and is opt-in.

The stock rubric surfaces these as **zero-weight metrics** — `rlm_iterations`, `rlm_sub_llm_calls`, `rlm_sub_llm_tokens`, `rlm_sub_llm_usage_missing` (`rubric.py:94-104`). Because they carry no reward weight, they do not perturb the GRPO advantage on the control arm, yet they are logged to Weights & Biases every step. The consequence is the property that makes the A/B clean: an *unshaped* control run produces the full cost profile (mean turns, sub-calls, tokens, and the missing-usage rate) for free, so the treatment can be compared against a control on cost even though only the treatment optimizes it.

The `rlm_sub_llm_usage_missing` counter deserves emphasis as a health signal. If a non-trivial fraction of sub-calls report no usage, the token axis is being fed partial data and $e_{\mathrm{tok}}$ is optimistically biased (it under-counts cost). A rising missing-usage rate concurrent with a falling token count is a red flag that the apparent efficiency gain is a measurement artifact, not a real reduction — which is why `analyze_pilot.py` reports it alongside the cost metrics.

## 4.5 Evaluation environments, scorers, and verifier hardening

The training mixture and the evaluation suite use four task families, each with its own scorer and context-aware user prologue, source-traced from PrimeIntellect's research-environments and the LMxLM task descriptions. All four expose the task context as REPL variables and finalize via `answer["content"] = ...` followed by `answer["ready"] = True`, rather than file writes. All four prologues carry a uniform "Plan before you act" orchestrator hint and a context-budget warning ("your model window is only ~16k tokens ... NEVER print, paste, or echo raw context; pass chunks as ARGUMENTS to `llm_query_batched`"), which exists to prevent the REPL from flooding the orchestrator's window with raw context.

| Environment | Scorer (file) | Mechanism |
|-------------|---------------|-----------|
| OOLONG-synth | `_oolong_synth_score` (`oolong/env.py:118-151`) | exact match → numeric $0.75^{\lvert\Delta\rvert}$ decay → date parse → substring fallback |
| OOLONG-Pairs | `_score_pairs` (`oolong_pairs/env.py:114-129`) | F1 over unordered `(id1, id2)` pairs, `<think>` blocks stripped |
| BrowseComp-Plus | `_make_browsecomp_plus_judge_score` / `_score_browsecomp_plus` (`browsecomp_plus/env.py:392-409`) | LLM judge (default) or deterministic containment fallback |
| LongBench-v2 CodeQA | `_extract_choice_letter` (`longbench_codeqa/env.py:78-91`) | regex extract letter A–D, exact match to gold |

**OOLONG-synth** checks an exact string match (`env.py:131-132`), then a numeric answer with $0.75^{\lvert \text{gold} - \text{trimmed}\rvert}$ partial-credit decay (`env.py:135-139`; a unit test pins $0.75^2 = 0.5625$ for off-by-two), then a date parse (`env.py:140-144`), and finally a case-insensitive substring fallback: it returns $1.0$ if `gold_s.lower()` is a substring of the output, excluding the three fixed comparison phrases (`more common than`, `less common than`, `same frequency as`) so a bare comparator cannot satisfy it (`env.py:146-150`). **OOLONG-Pairs** computes precision/recall/F1 over regex-extracted, order-normalized pairs after stripping `<think>` blocks. **BrowseComp-Plus** defaults to `reward_mode = "judge"` with `judge_model = "openai/gpt-4.1"`, prompting the judge for a JSON `is_correct` boolean (`env.py:81-98`); its deterministic fallback `_score_browsecomp_plus` normalizes text and returns $1.0$ on exact match *or bidirectional containment* (`env.py:197-210`). **LongBench-v2 CodeQA** extracts a single A–D letter and returns $1.0$ only if `gold ∈ {A,B,C,D}` and the extracted letter matches (`env.py:87-91`).

### 4.5.1 Why hardening is necessary

Adding an efficiency term amplifies every weakness in $c$: a policy rewarded for the cheapest trajectory the scorer still calls correct will discover and exploit scorer loopholes, which is precisely the reward-hacking failure mode that motivates this thesis [Skalse et al., 2022; Amodei et al., 2016] (Goodhart's law). Two loopholes are known and dangerous *specifically because they admit cheap wrong answers that the scorer rates correct*:

- **OOLONG substring fallback.** `_oolong_synth_score` returns $1.0$ whenever `gold.lower()` appears anywhere in the raw output (`oolong/env.py:146-150`). A wasteful-but-honest rollout and a lean rollout that merely *mentions* the gold string both score $1.0$; under shaping the latter is rewarded *more*, even if it never inspected the context.
- **BrowseComp-Plus containment fallback.** `_score_browsecomp_plus` returns $1.0$ on mutual containment whenever `reward_mode != "judge"` or the judge call fails (`browsecomp_plus/env.py:197-210`). A short, vague answer contained in the gold can pass.

### 4.5.2 Mitigations

The hardening posture for the shaped arm is:

1. **Gate on the strictest available $c$.** For BrowseComp-Plus, require the LLM judge (`reward_mode = "judge"`, `judge_model = "openai/gpt-4.1"`) and treat the containment path as a fallback to be tightened or disabled for the shaped arm, so the efficiency bonus rides on judge agreement, not lexical overlap [BrowseComp-Plus].
2. **Tighten lenient fallbacks** on the shaped arm; optionally gate on a second independent check.
3. **Require minimum genuine work** — not merely `min_iterations >= 2`, but evidence the context was actually inspected (this is the purpose of the otherwise-inert `min_subcall` gate from Section 4.2, which can be raised for the shaped arm).
4. **Treat any cost reduction that coincides with a rising fallback-hit rate as reward hacking, not progress.** This is operationalized as a diagnostic in `analyze_pilot.py` (Section 4.6) and as the SHORTCUT/HEALTH block it prints.

The detailed anti-reward-hacking protocol — keep evaluation unshaped, sweep $\lambda$ and keep only Pareto points, audit trajectories, stress with adversarial canaries — is stated in Chapter 5.

### 4.5.3 The evaluation suite is unshaped

A non-negotiable design choice is that **no eval environment carries `shaping_coef`**: the evaluation suite reports stock correctness and cost only. The pilot and full-scale eval suites both use four unshaped environments — OOLONG TREC-coarse @131k, OOLONG-Pairs @32k, BrowseComp-Plus ($k = 50$, gpt-4.1 judge), and LongBench-v2 CodeQA — and none of their `args` blocks contain `shaping_coef`. Evaluation thus measures the policy's behaviour on the *unmodified* objective, so an improvement on the eval suite cannot be an artifact of the training reward. The full-scale eval uses $n = 50/20/150/50$ per environment; the A40 pilot uses $n = 20$ per environment to fit the smaller allocation while keeping the same datasets, contexts, and $k$ so the numbers stay comparable.

## 4.6 Config wiring and the faithful-twin invariant

The A/B is realized as two TOML configs that are *identical except for the reward knob*. The treatment is `training/configs/rlm-qwen3-30b-thesis.toml` (`shaping_coef = 0.2`); the control is `rlm-qwen3-30b-thesis-control.toml` (`shaping_coef = 0.0`). Both fix the base model `Qwen/Qwen3-30B-A3B-Instruct-2507` [Qwen Team, 2025] with non-thinking sampling (`enable_thinking = false`), the same LoRA adapter (`rank = 32`, `alpha = 64`, `dropout = 0.0`, targets `q_proj, k_proj, v_proj, o_proj`), the same optimizer (`lr = 5e-5`), the same trainer settings (`max_steps = 200`, `seq_len = 8192`, `batch_size = 32`, `group_size = 8`, checkpoint/eval interval $= 20$), the same inference caps (`max_model_len = 16384`, `max_completion_tokens = 4096`, `gpu_memory_utilization = 0.80`, `tp = 2`, `dp = 2`), the same pre-batch filters (repetition with window $3000$ and `prob_threshold = 0.99`; zero-advantage), the same $0.5/0.5$ training mixture of `oolong` (`dataset_name = "spam"`, context sampled in $[32768, 65536]$) and `browsecomp_plus` (`num_examples = 150`, `k = 50`, `reward_mode = "judge"`), and the same unshaped four-environment eval suite. The shared reward budgets are `correct_threshold = 1.0`, `subcall_budget = 32.0`, `token_budget = 200000.0`, unit axis weights, `max_iterations = 20`, `min_iterations = 2`, `min_subcall = 0`.

### 4.6.1 The allowed-diffs set

The faithful-twin invariant is enforced mechanically by `tools/validate_thesis_shaping.py`. It flattens both TOMLs to dotted leaf paths and computes the set of leaves whose values differ (`_flatten` and `_config_diff_keys`, `validate_thesis_shaping.py:64-82`). It then asserts that this diff set is a *subset* of `_CONTROL_ALLOWED_DIFFS` (`validate_thesis_shaping.py:53-61`), which contains exactly seven keys:

```
output_dir
wandb.name
orchestrator.model.lora.name
orchestrator.train.env[0].name
orchestrator.train.env[1].name
orchestrator.train.env[0].args.shaping_coef
orchestrator.train.env[1].args.shaping_coef
```

That is: the output directory, the run name, the two LoRA adapter names, the two training-environment names, and the two `shaping_coef` values. Any other difference — a changed LoRA rank, a different eval environment, a perturbed context length — fails validation and flags the A/B as confounded. The validator additionally checks that `shaping_coef` flips from $> 0$ (treatment) to $0$ (control). This rules out the tempting but wrong choice of using `rlm-qwen3-30b-efficient-eval-suite.toml` as the control: that config is single-environment and would change the *training distribution*, confounding the comparison.

### 4.6.2 What the validator and tests prove

Beyond the diff check, `validate_thesis_shaping.py` runs roughly thirty named conditions and the test suite `training/tests/test_efficiency_shaping.py` pins the Chapter 3 invariants as executable checks:

| Property | Statement | Pinned by |
|----------|-----------|-----------|
| Parity | `RLMTrainRubric(coef=0)` $\equiv$ `EfficiencyGatedRubric(coef=0)` $= 1.0$ | `test_parity_with_stock_rubric_when_coef_zero` (`:58-70`) |
| Gating | $c = 0$ rollout scores $0.0$ regardless of how cheap | `test_no_bonus_for_incorrect_rollout` (`:90-99`) |
| Dominance | expensive-correct $\ge$ cheap-wrong, across usage extremes | `test_correct_always_beats_incorrect` (`:155-173`) |
| Monotonicity | cheap $\ge$ mid $\ge$ pricey $\ge 1.0$ (more usage never pays) | `test_more_usage_never_pays` (`:179-191`) |
| Boundedness | max-efficiency $\Rightarrow R = c\,(1+\lambda)$ | `test_reward_is_bounded` (`:197-210`) |
| Metrics exposed | `efficiency_bonus`, `rlm_efficiency_score`, token + missing counters present | `test_efficiency_metrics_exposed` (`:269-286`) |
| Telemetry | `_record_sub_call` tallies calls/tokens and missing-usage | `test_record_sub_call_accumulates_…` (`:298-313`) |

The structural caps are deliberately *not* a substitute for the reward, and two further tests pin that distinction: `test_structural_caps_leave_correctness_only_rollouts_tied` and `test_efficiency_reward_separates_cap_satisfying_correct_rollouts`. The caps (`max_iterations = 20`, `seq_len = 8192`, `max_model_len = 16384`, `max_completion_tokens = 4096`, `sub_max_tokens = 4096`, the REPL wall-clock timeout) are *ceilings*, not preferences: two fully-correct, cap-satisfying rollouts can have very different cost and tie at $R = c$ under the control. The efficiency reward credit-assigns cost *within* the correct, cap-satisfying set, which is something no ceiling can do.

### 4.6.3 Pilot wiring

The in-flight feasibility study uses an A40-tuned pair, `pilot-l0.toml` (control, $\lambda = 0$) and `pilot-l02.toml` (treatment, $\lambda = 0.2$), with `max_steps = 10`, `tp = 4`, `dp = 1`, `gpu_memory_utilization = 0.90`, `enforce_eager = true`, `group_size = 2`, and $n = 20$ per eval environment, while keeping the same datasets, contexts, $k = 50$, and shaping budgets so the pilot numbers remain comparable to the full-scale run. Both arms are launched sequentially by `scripts/valar_pilot.sh`, which exports `RLM_TRAIN_EXEC_TIMEOUT_S = 180` and `RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S = 180` and invokes `launch_efficient.sh` (`uv run --no-sync rl @ $CFG`) for each config. The arms differ only in `shaping_coef` (`0.0` vs `0.2` at lines 97 and 119 of each config). Results are read back from the offline W&B runs by `tools/analyze_pilot.py`, which prints three blocks: **EVAL ACCURACY** (step-0 base $\to$ final delta per environment), **TRAJECTORY COST** (final iterations / sub-LLM calls / tokens with a percent delta between arms), and **SHORTCUT / HEALTH DIAGNOSTICS** (efficiency, shaped-bonus, missing-usage, gibberish, repetition, truncation, zero-advantage). The diagnostic block operationalizes Section 4.5's hardening rule: a cost reduction is only credited as progress if it is *not* accompanied by rising fallback-hit or missing-usage rates.

> **[RESULTS PENDING]** The pilot is in flight and has produced no completed result artifacts. The only concrete numbers in this chapter are configuration values and the labelled illustrative arithmetic of Section 4.3.5; no accuracies, costs, or deltas are reported. Chapter 6 holds the result skeletons, and Chapter 5 details the experimental protocol that fills them.
