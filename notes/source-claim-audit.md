# Source claim audit for efficient RLM training

Date: 2026-06-22. Sources refreshed with Firecrawl into `.firecrawl/*latest.md`; upstream GitHub verified with `git ls-remote origin main`.

## Source inventory

| Source | Local evidence | Status |
|---|---|---|
| X thread `a1zhang/status/2059633834094678173` | `.firecrawl/x-a1zhang-2059633834094678173-latest.md` | Read; includes main post, two thread posts, top comments. |
| GitHub `alexzhang13/rlm` | `.firecrawl/github-alexzhang13-rlm-latest.md`; local clone `HEAD=156fd725411b9cae822f5920a6cbf102a5473baa`; `origin/main` equals same commit | Read; local clone is current upstream. |
| Hugging Face model card | `.firecrawl/hf-rlm-qwen3-30b-a3b-v0.1-latest.md`; adapter file `.firecrawl/hf-files/adapter_config.json` | Read; adapter_config fetched and parsed. |
| RLM paper | `papers/2512.24601-rlm.pdf` + extracted `papers/2512.24601-rlm.txt` | Used for failure-case constraints and training caveats. |

## Claim-by-claim verification

### X thread claims

| Claim | Evidence | Verification |
|---|---|---|
| A minimal training harness is built on `prime-rl` and `verifiers`. | X main post says this; GitHub README says the repo includes a `verifiers` training environment based on Prime Intellect `prime-rl`; `training/pyproject.toml` depends on `verifiers`; upstream `training/README.md` says it plugs into `prime-rl`. | **Verified.** |
| Users can train RLMs without sandboxes. | X main post says "without sandboxes"; GitHub training README says the harness does not require or use sandboxes and uses subprocess-isolated local REPL. | **Verified.** Safety caveat: GitHub README still says ideal setup would use sandboxes for safety. |
| All training code is available in the repo `training/` folder. | Upstream has `training/src/rlm_train`, `training/environments/oolong`, and `training/configs/rlm-qwen3-30b-example.toml`. | **Partially verified.** The public repo includes a runnable training harness and one OOLONG example config, but it does **not** include every environment implied by the released model's mixed-suite training (notably BC+/BrowseComp-Plus). |
| RLM-Qwen3-30B-A3B-v0.1 was trained using RL on a separate split of environments (OOLONG-Spam, BC+ split). | X main post says this. HF card says "mixed long-context environment suite" and RL. | **Source-asserted but not fully reproducible from public GitHub.** OOLONG-Spam is present; BC+/BrowseComp-Plus training env/config is not present in upstream `training/`. |
| The run was for a day on 8xA100 using `prime-rl`. | X main post says this; HF card lists Hardware = 8 x A100 and Method = RL with `prime-rl`. | **Source-corroborated by X + HF, not independently reproduced.** No public training log/checkpoint metadata was found in the repo proving duration. |
| Code and model are open source on GitHub/Hugging Face. | GitHub repo and HF model card exist; model card exposes adapter use; adapter_config is fetchable. | **Verified for code + adapter metadata.** The full base model is hosted separately by Qwen; the HF repo is a LoRA adapter, not standalone weights. |
| Training RLMs should improve long-horizon tasks and smaller OSS models. | X thread post says this; paper reports training gains for 8B; HF card reports 30B adapter gains. | **Supported by paper/HF evaluations, not independently rerun here.** |

### GitHub repo claims relevant to training

| Claim | Evidence | Verification |
|---|---|---|
| Repo provides inference engine plus training environment. | GitHub README training section; local tree has `rlm/` and `training/`. | **Verified.** |
| Training harness is for local REPL RLM, depth=1. | `training/README.md` says depth=1; upstream `RLMTrainEnv` starts backend with `depth=1`. | **Verified.** |
| Sub-LM calls are routed through a proxy back to trainer inference server. | `training/README.md`; upstream `training/src/rlm_train/proxy.py`; `RLMTrainEnv.setup_state` registers `SubLLMProxy`. | **Verified.** |
| Code execution happens in a subprocess on training host. | `training/README.md`; upstream `SubprocessReplBackend`; `worker.py`. | **Verified.** |
| Launch command is `uv run rl @ training/configs/rlm-qwen3-30b-example.toml`. | `training/README.md`. | **Verified.** |
| Worked example is OOLONG. | `training/environments/oolong`; `training/configs/rlm-qwen3-30b-example.toml`. | **Verified.** |
| The example config trains Qwen3-30B-A3B-Instruct-2507 on OOLONG spam 32k-65k and evals trec_coarse 131k. | Upstream example config. | **Verified.** |
| Upstream reward includes explicit efficiency/cost shaping. | Upstream `RLMTrainRubric` returns correctness only unless gate mode is enabled; no token-cost penalty. | **False for upstream.** Local untracked files add cost-shaping, but they are not upstream. |

### Hugging Face model-card claims

| Claim | Evidence | Verification |
|---|---|---|
| Model repo is a LoRA adapter, not standalone. | HF card; fetched `adapter_config.json` has `peft_type=LORA`. | **Verified.** |
| Base model is `Qwen/Qwen3-30B-A3B-Instruct-2507`. | HF card; adapter_config `base_model_name_or_path`. | **Verified.** |
| Adapter config is LoRA r=32, alpha=64, dropout=0, targets q/k/v/o. | HF card; adapter_config fetched and parsed. | **Verified.** This differs from upstream example config, which also adapts MLP/expert projections. The efficient replication config follows HF adapter parity. |
| It was trained as an RLM policy via RL on a mixed long-context suite. | HF card. | **Source-asserted.** Public repo verifies RL harness, but public config does not reconstruct the mixed suite. |
| Training env is `RLMTrainEnv`, logically 1:1 with `rlm.RLM.completion`. | HF card; upstream training README; `RLMTrainEnv` mirrors RLM history/REPL/subcalls. | **Verified conceptually from code.** Exact parity requires running trajectories, not done here. |
| Training depth is 1; inference can use depth >1 though not trained. | HF card; upstream training starts backend with `depth=1`; core RLM supports `max_depth`. | **Verified for training code.** Depth>1 inference support exists in canonical harness, but adapter quality at >1 is source-asserted. |
| Training used max_model_len=16384. | HF card; upstream example config uses `max_model_len=16384`. | **Verified against public example and HF.** |
| Training hardware was 8 x A100. | HF card and X post. | **Source-corroborated, not independently reproduced.** |
| Results table improvements. | HF card lists exact numbers. | **Source-asserted only.** No public eval rollouts/logs were found here to independently recompute the table. |
| Some inference flags still TBD. | HF card limitations. | **Verified as a stated uncertainty.** Important for exact replication: orchestrator/prologue details are not fully specified. |

## Paper failure cases that constrain a proper efficient config

| Failure case | Evidence in paper | Config/control implication |
|---|---|---|
| Exploding sub-call costs are a side-effect risk. | §7. | Do not add unbounded subcall fanout; keep `max_iterations=20`; use orchestration prompt; use timeout. |
| Blocking/sequential sub-LM calls make RLMs slow; p95 runtime tail dominated by sequential calls. | Appendix B and §F.2. | Use upstream orchestrator addendum's batching discipline; set `RLM_TRAIN_EXEC_TIMEOUT_S=180`. |
| Qwen3-Coder may use thousands of subcalls if not prompted. | Appendix C/E.3. | Use upstream `ORCHESTRATOR_ADDENDUM`; prefer fat-prompt small batches; no tiny one-line prompts. |
| Thinking models can exhaust output tokens. | Appendix B. | `extra_body={enable_thinking=false}` and `max_completion_tokens=4096`. |
| Final-answer vs thought is brittle. | Appendix B. | Public upstream can monitor this via `gated_reward`/`rlm_below_min_iterations`; true reward gating is not exposed by public OOLONG config. |
| Small models without coding ability struggle. | Appendix B. | Use Qwen3-30B-A3B base for this replication, not 8B. |

## Verdict on the candidate config

`training/configs/rlm-qwen3-30b-efficient.toml` is a **proper public-repo-compatible efficient RLM training config**, not an exact reconstruction of the private/published mixed-suite run.

What it verifies and preserves:

- Uses upstream schema only.
- Uses upstream OOLONG training environment only.
- Uses HF adapter parity: LoRA r=32, alpha=64, dropout=0, q/k/v/o only.
- Uses base model, token caps, max_model_len, depth, local-REPL training style, and 8-GPU deployment consistent with sources.
- Avoids local misleading cost-shaping code.
- Sets launch-time timeout guard for the paper's sequential-call failure mode.

What remains uncertain / not publicly reconstructable from the three sources alone:

- Exact mixed environment list and sampling weights for the released adapter (X says OOLONG-Spam + BC+ split; GitHub only publishes OOLONG env/config).
- Exact training duration details beyond "a day" / "8 x A100".
- Exact orchestrator hint/prologue conditioning used for every evaluation; HF says some flags are TBD.
- Independent reproduction of the HF results table; no rollouts/logs are included in the sources read here.

Therefore, a claim that the config exactly reproduces `mit-oasys/rlm-qwen3-30b-a3b-v0.1` would be too strong. A correct claim is: it is the best upstream-only, source-verified efficient RLM training config possible from the public repo and model card, with missing BC+/mixed-suite details explicitly flagged.
