# 6. Results

> **[RESULTS PENDING — A/B pilot in flight; no numbers are reported here]**

This chapter is intentionally a *reporting plan* rather than a results report. The paired A/B experiment specified in Chapter 5 — control ($\lambda=0$, `rlm-qwen3-30b-thesis-control.toml`) versus treatment ($\lambda=0.2$, `rlm-qwen3-30b-thesis.toml`), and its A40-tuned feasibility variant (`pilot-l0.toml` versus `pilot-l02.toml`, launched via `scripts/valar_pilot.sh`) — has not yet produced completed result artifacts. Per the results policy of this thesis, every cell that would contain a measured quantity is rendered as an em dash (—) or `TBD`. The only concrete numbers anywhere below are configuration constants drawn from the canonical configs and the labelled illustrative reward arithmetic of Chapter 3. No empirical outcome is stated as fact.

What this chapter *does* fix in advance is the analysis: the tables, figures, and decision rules are committed here, before the numbers exist, so that the experiment cannot be reinterpreted post hoc to favor a desired conclusion. Each artifact is tied to a specific falsifiable prediction (P1–P4 of Chapter 5) and to a specific column emitted by `tools/analyze_pilot.py`, whose three report blocks — `## EVAL ACCURACY`, `## TRAJECTORY COST`, and `## SHORTCUT / HEALTH DIAGNOSTICS` — are the source of truth for the schemas below.

## 6.1 Reporting plan

The evaluation suite is **unshaped** in both arms: the four eval environments carry no `shaping_coef`, so reported accuracy is stock task correctness $c$ and reported cost is raw trajectory telemetry. This separation of the *training* reward from the *publication* metric is point 7 of the anti-reward-hacking protocol (Chapter 5): the headline of this work is an accuracy-versus-cost frontier on unshaped evaluation, never the shaped training reward.

Accuracy is measured per environment with `analyze_pilot.py`'s `EVAL ACCURACY` block, which reports the step-0 base-model value, the final trained value, and their difference $\Delta = \text{trained} - \text{base}$, separately for control and treatment. Because `skip_first_step=false`, step 0 is a genuine base-model baseline. Cost is read from the `TRAJECTORY COST` block (matched by `COST_PAT`, i.e. keys containing `iter`, `sub_llm`, `subcall`, `sub_call`, or `token`), reported as the final value per arm with a percent change $\mathrm{D\%} = (\text{treat} - \text{ctrl})/\text{ctrl}\times 100$. The three cost axes correspond exactly to the three efficiency budgets of the objective: turns ($B_{\text{turns}}=20$), sub-LLM calls ($B_{\text{calls}}=32$), and sub-LLM tokens ($B_{\text{tok}}=200000$). Reward-hacking guards are read from the `SHORTCUT / HEALTH DIAGNOSTICS` block (matched by `DIAG_PAT`: `usage_missing`, `efficiency`, `shaped`, `gibberish`, `repetition`, `truncat`, `zero_advantage`).

The full A/B uses HF eval cardinalities ($n=50/20/150/50$ across the four environments); the in-flight pilot uses $n=20$ per environment at `max_steps=10` with otherwise identical datasets, context lengths, and $k=50$ so the pilot and full-scale numbers remain directly comparable. Tables below use the HF cardinalities; pilot tables are structurally identical with the smaller $n$.

## 6.2 Accuracy: no-regression test (P1)

Table 6.1 tests **P1** (no correctness regression): treatment accuracy $\geq$ control $- \epsilon$ on the held-out suite. A positive outcome is treatment $\Delta$ within $\epsilon$ of control $\Delta$ on every environment; a negative outcome — treatment accuracy falling outside $\epsilon$ below control on any environment — would mean the efficiency bonus is buying cheapness at the cost of correctness, which the gated objective ($R = c\,(1+\lambda e)$, bonus applied only when $c \geq$ `correct_threshold` $=1.0$) is specifically constructed to prevent. P1 is the precondition for all other claims: P2 and P3 are only meaningful *at equal accuracy*.

**Table 6.1 — Unshaped eval accuracy by environment, control ($\lambda=0$) vs treatment ($\lambda=0.2$).** Values are base (step 0) → trained (final), with $\Delta$.

| Environment | $n$ | Control: base → trained ($\Delta$) | Treatment: base → trained ($\Delta$) | Treat $-$ Control |
|---|---|---|---|---|
| OOLONG `trec_coarse` @131k | 50 | — → — (—) | — → — (—) | TBD |
| OOLONG-Pairs @32k (F1) | 20 | — → — (—) | — → — (—) | TBD |
| BrowseComp-Plus ($k=50$, judge) | 150 | — → — (—) | — → — (—) | TBD |
| LongBench-v2 CodeQA | 50 | — → — (—) | — → — (—) | TBD |
| **Suite mean** | — | — → — (—) | — → — (—) | TBD |

## 6.3 Cost: mean-trajectory test (P2)

Table 6.2 tests **P2** (cheaper trajectories at equal accuracy). It reports the mean of each of the three cost axes per environment and arm, with $\mathrm{D\%}$. A positive outcome is a Pareto improvement: strictly fewer mean turns and/or sub-LLM calls and/or sub-LLM tokens under treatment, *conditioned on* P1 holding in Table 6.1. A negative outcome is treatment cost statistically indistinguishable from (or above) control, which would mean the gradient signal failed to credit-assign cost — consistent with the thesis's rejection of "efficiency is just prompting," since prompting alone provides no such signal.

**Table 6.2 — Mean trajectory cost by environment (lower under treatment is the target).** Axes and budgets: turns ($B=20$), sub-LLM calls ($B=32$), sub-LLM tokens ($B=200000$).

| Environment | Axis | Control mean | Treatment mean | D% |
|---|---|---|---|---|
| OOLONG `trec_coarse` | turns | — | — | TBD |
| | sub-LLM calls | — | — | TBD |
| | sub-LLM tokens | — | — | TBD |
| OOLONG-Pairs | turns / calls / tokens | — / — / — | — / — / — | TBD |
| BrowseComp-Plus | turns / calls / tokens | — / — / — | — / — / — | TBD |
| LongBench-v2 CodeQA | turns / calls / tokens | — / — / — | — / — / — | TBD |

## 6.4 Tail cost: p95 test (P3)

Table 6.3 tests **P3** (lower tail cost). The paper documents long-tailed, high-variance RLM cost in which a minority of trajectories re-verify or re-generate many times [Zhang et al., 2026, §F.2]. P3 predicts that the efficiency reward shrinks the p95 of turns and tokens specifically — not merely the mean. A positive outcome is treatment p95 below control p95 on turns and/or tokens at preserved accuracy, indicating the worst-case trajectories are pulled in. A negative outcome — p95 unchanged while means fall — would suggest the bonus only trims already-lean rollouts and leaves the expensive tail untouched, the most costly regime in practice.

**Table 6.3 — p95 of turns and sub-LLM tokens by environment.** (Percentiles are computed from the per-rollout turn and token telemetry logged during evaluation; the exact percentile keys are to be confirmed against the trainer's metric schema.)

| Environment | Metric | Control p95 | Treatment p95 | D% |
|---|---|---|---|---|
| OOLONG `trec_coarse` | turns | — | — | TBD |
| | sub-LLM tokens | — | — | TBD |
| OOLONG-Pairs | turns / tokens | — / — | — / — | TBD |
| BrowseComp-Plus | turns / tokens | — / — | — / — | TBD |
| LongBench-v2 CodeQA | turns / tokens | — / — | — / — | TBD |

## 6.5 The headline figure: accuracy-vs-cost frontier

**Figure 6.1 — Accuracy-vs-cost Pareto frontier (placeholder).** *Planned content:* a scatter plot with mean per-trajectory cost on the $x$-axis (one panel each for turns, sub-LLM calls, and sub-LLM tokens) and unshaped suite accuracy on the $y$-axis. Each point is one arm at one $\lambda$: the control ($\lambda=0$) anchors the correctness-only baseline, and the treatment points ($\lambda \in \{0.05, 0.1, 0.2\}$, Section 6.7) trace the frontier. Only Pareto-non-dominated points are retained as headline results (protocol point 4). *Reading the figure:* the thesis is supported iff treatment points sit up-and-to-the-left of control — equal-or-higher accuracy at strictly lower cost; a point down-and-to-the-left (lower cost, lower accuracy) would be a P1 violation and is excluded from the frontier. This figure, not the shaped training reward, is the publication metric.

## 6.6 Reward-hacking guard

Table 6.4 is the safety check that distinguishes genuine efficiency from scorer exploitation. Adding an efficiency term amplifies every weakness in $c$, rewarding the cheapest trajectory the scorer still calls correct (Chapter 5). The two known loopholes are the OOLONG substring fallback (`_oolong_synth_score` returns $1.0$ when the gold string is a case-insensitive substring of the output, `training/environments/oolong/oolong/env.py:146-150`) and the BrowseComp-Plus containment fallback (`_score_browsecomp_plus` returns $1.0$ on mutual containment when the judge is bypassed, `training/environments/browsecomp_plus/browsecomp_plus/env.py:197-210`). A *cost reduction that coincides with a rising fallback-hit rate is reward hacking, not progress.* A positive outcome is treatment fallback-hit and shortcut rates flat or lower than control while cost falls; a negative outcome — cost down but fallback-hits up — invalidates the corresponding cost gain.

**Table 6.4 — Fallback-hit / shortcut diagnostics by environment** (from the `SHORTCUT / HEALTH DIAGNOSTICS` block; `rlm_sub_llm_usage_missing` must be $\approx 0$ for the token axis to be live).

| Environment | Diagnostic | Control (last) | Treatment (last) |
|---|---|---|---|
| OOLONG | substring-fallback hit rate | — | — |
| BrowseComp-Plus | containment-fallback hit rate | — | — |
| All | `rlm_sub_llm_usage_missing` | — | — |
| All | `rlm_efficiency_score` | — | — |
| All | `efficiency_bonus` | — | — |
| All | repetition / truncation / `zero_advantage` | — / — / — | — / — / — |

## 6.7 Lambda sweep

Table 6.5 records the $\lambda$ sweep (protocol point 4), which sets the operating point and tests the falsification criterion: the thesis is **falsified if, across reasonable $\lambda$, the treatment cannot beat control on P2/P3 without violating P1.** Each row is one treatment arm; we retain only Pareto-frontier points for Figure 6.1. A positive outcome is a monotone-ish trade — rising $\lambda$ buys lower cost until accuracy begins to erode, exposing a usable knee. A negative outcome is no $\lambda$ achieving a cost win within the P1 accuracy band, which would falsify the headline claim. The canonical treatment uses $\lambda=0.2$; $\lambda=0$ is, by parity, byte-identical to the control.

**Table 6.5 — $\lambda$ sweep: Pareto points.** Accuracy is unshaped suite mean; cost is mean across the three axes (or per-axis in the long form).

| $\lambda$ | Suite accuracy | Mean turns | Mean sub-LLM calls | Mean sub-LLM tokens | On frontier? |
|---|---|---|---|---|---|
| 0.00 (control) | — | — | — | — | TBD |
| 0.05 | — | — | — | — | TBD |
| 0.10 | — | — | — | — | TBD |
| 0.20 (canonical) | — | — | — | — | TBD |

## 6.8 Trajectory audit

Scalar rewards are not audited alone (protocol point 5). For each arm we will hand-audit a stratified sample of rollouts to confirm that observed cost reductions reflect leaner *genuine* problem-solving — evidence the context was actually inspected — rather than premature finalization or a scorer shortcut. The template below pairs each audited trajectory with its telemetry and a qualitative verdict; the illustrative arithmetic of Chapter 3 (at canonical budgets, a lean correct rollout of 3 turns / 3 sub-calls / 35{,}000 tokens scores $R \approx 1.1721$ versus a wasteful correct rollout at $R \approx 1.0242$, both tying at $R=1.0$ under the control) gives the expected shape of a healthy treatment trajectory.

**Trajectory-audit template (per sampled rollout).**

| Field | Value |
|---|---|
| Arm / $\lambda$ | — |
| Environment / prompt id | — |
| Turns / sub-LLM calls / sub-LLM tokens | — / — / — |
| Correctness $c$ | — |
| `rlm_efficiency_score` $e$ | — |
| `efficiency_bonus` ($= R_{\text{shaped}} - R_{\text{base}}$) | — |
| Fallback path triggered? (substring / containment / judge) | — |
| Context genuinely inspected? (Y/N + evidence) | — |
| Adversarial-canary outcome (short answer wrong unless context read) | — |
| Verdict (genuine efficiency / reward-hacking / inconclusive) | — |

Once the pilot completes, `tools/analyze_pilot.py` populates Tables 6.1–6.4 and the sweep harness populates Table 6.5; the frontier of Figure 6.1 and the audit of Section 6.8 then determine whether P1–P4 are supported or the headline claim is falsified. Interpretation of these results is deferred to Chapter 7.
