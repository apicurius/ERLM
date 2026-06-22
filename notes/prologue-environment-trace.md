# RLM eval environments + user_prologue trace

Date: 2026-06-22. Purpose: create local `RLMTrainEnv` environments with the necessary user prologues by tracing LMxLM and PrimeIntellect research-environments, plus the HF README/eval image.

## HF README line-by-line findings

Raw file: `.firecrawl/hf-files/README.md` (line-numbered during inspection).

Load-bearing lines:

- Lines 17-23: model is `RLM Qwen3-30B-A3B v0.1`, LoRA adapter, base `Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA r=32, alpha=64, q/k/v/o.
- Lines 27-31: adapter is intended as the **root LM** inside the RLM harness; it expects RLM system prompt + REPL scaffolding; training depth was 1; exact inference flags include orchestrator hints / per-env user prologues but are marked TBD.
- Lines 35-44: evals used base with and without a `"Plan before you act"` orchestrator hint, and RLM-trained + hint. Eval environments in the HF table/picture: OOLONG trec_coarse @132k (n=50), OOLONG-Pairs @32k (n=20), BrowseComp-Plus test (n=150, k=50 docs), LongBenchv2 Code repo QA (n=50).
- Lines 60-72: training used `rlm/training`, `RLMTrainEnv`, `prime-rl`, verifiable rewards, depth=1, 8xA100.
- Lines 100-116: inference config uses `max_iterations=20`, `max_depth=1`, `max_completion_tokens=4096`, `enable_thinking=False`, `sub_sampling_args={"max_tokens":4096}`, and `orchestrator=True` default matching training.
- Lines 124-128: limitations: depth>1 not trained; no persistent/compaction/budget/timeout/errors/custom-tools used during training; exact per-env flags/prologues are future/TBD.

The HF eval image (`media/hf/rlm_eval_results.png`) visually matches the table: the four envs above, grey base / teal base+plan / orange trained+plan, and footer `evals: full split where feasible; BCP n=150, LongBenchv2 CodeQA n=50`.

## Alex/az commits traced in PrimeIntellect research-environments

Repo cloned at `_pkgs/research-environments`, HEAD `a2b76f6a`. We deepened history and searched author `az <alex.lx.zhang@gmail.com>` / Alex-like authors. Relevant commits:

- `4802f7c5` (2026-03-02) — `Change OOLONG RLM eval to be based on the actual repository (#187)`.
- `ddd5efb0` (2026-03-04) — `OOLONG_RLM: Update reward function, add ability to select subsets + context Len (#195)`.
- `36de2d0f` (2026-04-14) — `update OOLONG with filtering numerical tasks (which are broken)) (#270)`.
- `1c039a33` (2026-04-24) — `Update RLM envs with new RLM harness (#315)`.
- `2fb131b9` (2026-05-31) — `Add rlm_longbenchpro environment (#421)`.
- `570654b4` (2026-06-01) — `Add rlm_oolong_pairs (#422)`.
- `b07ace37` (2026-06-02) — `rlm_oolong defaults to trec_coarse at 256k / 512k (#433)`.

Other Alex RLM env commits seen: `ead5bebc` (MRCRv2 RLM), `fa302cb8` (LongBenchPro RLM), `1f958795` (LongCoT-RLM), `457005b9` (Graphwalks + RLM), `f409d52c` (tau3-bench for RLMs).

## Prime research env prologues/instructions recovered

### `rlm_oolong`

Source: `_pkgs/research-environments/environments/rlm_oolong/rlm_oolong.py`.

Key text recovered:

- `_ENV_TIPS`: context in `/workspace/context.txt`; split into chunks; prompt each chunk; call `llm_batch()` once; aggregate relevant findings.
- `_APPEND_SYSTEM_PROMPT_SYNTH`: final answer only to `/task/answer.txt`, short single token/word/date/label.
- Default synth dataset after Alex commit: `trec_coarse`, context lengths `(262144, 524288)`; but HF eval table uses `trec_coarse @ 132k (n=50)`, so local eval config uses `131072` to match HF.

Local adaptation in `training/environments/rlm_eval_suite/rlm_eval_suite/envs.py`:

- `OOLONG_USER_PROLOGUE` ports the strategy to upstream `RLMTrainEnv`: use REPL variable `context` instead of `/workspace/context.txt`, use `llm_query_batched` instead of CLI `llm_batch`, terminate via `answer["content"]` / `answer["ready"]` instead of `/task/answer.txt`.
- `load_oolong_environment()` builds OOLONG synth rows and uses official-style scoring.

### `rlm_oolong_pairs`

Source: `_pkgs/research-environments/environments/rlm_oolong_pairs/rlm_oolong_pairs.py` and README.

Key text recovered:

- `_ENV_TIPS`: parse each line as `Date: <date> || User: <id> || Instance: <question>`; classify labels; aggregate in Python; output pairs.
- `_APPEND_SYSTEM_PROMPT`: final answer only to `/task/answer.txt`; list every matching pair `(user_id_1, user_id_2)` lower ID first, one per line, or `[]`.
- Dataset: `mit-oasys/oolong-pairs` question/gold files plus `oolongbench/oolong-synth` trec_coarse context. Default context_len `32768`, exactly matching HF eval table OOLONG-Pairs @32k (n=20).

Local adaptation:

- `OOLONG_PAIRS_USER_PROLOGUE` ports the Prime env task tips and answer format to local `RLMTrainEnv`.
- `load_oolong_pairs_environment()` loads `mit-oasys/oolong-pairs` JSONs and streams the needed trec_coarse context from OOLONG synth to avoid materializing the full dataset.
- Scoring is precision/recall/F1 over unordered user-ID pairs; reward is F1.

## LMxLM sources used

Local repo: `/Users/oerdogan/LMxLM`.

- `lmxlm/environments/lm_to_lm/oolong/description.py` and `env.py`: OOLONG question framing, TREC-coarse blurb, official scoring style, verbosity extraction.
- `lmxlm/environments/lm_to_program/oolong_pairs/description.py` and `programs.py`: pairwise aggregation framing, A/B user-set strategy, pair output format.
- `lmxlm/environments/lm_to_program/browsecomp_plus/*`: evidence-doc BrowseComp-Plus protocol, now ported as `rlm-browsecomp-plus-local` with the HF-card small-model setting `k=50`.

## BrowseComp-Plus + LongBench-v2 CodeQA (now implemented)

Both remaining HF-picture eval environments are now ported as local `RLMTrainEnv`
environments (smoke-loaded against the real datasets).

### `rlm-browsecomp-plus-local`

- Source: `Tevatron/browsecomp-plus` (the evidence-doc BrowseComp-Plus that the
  HF table refers to: `test (n=150, k=50 documents)`), matching the LMxLM
  `lm_to_program/browsecomp_plus` setting. The dataset rows are XOR/SHA-256
  decrypted with the public canary (same formula as LMxLM `_data.py`).
- Prompt/answer format traced from Prime `rlm_browsecomp.py`
  (`QUERY_TEMPLATE` + `APPEND_SYSTEM_PROMPT`): simple-evals
  `Explanation / Exact Answer / Confidence`. The verbatim HLE
  `GRADER_TEMPLATE` is preserved as `BROWSECOMP_GRADER_TEMPLATE`.
- Adaptation: evidence docs are exposed as the REPL `context` (list[str]); the
  model finalizes via `answer["content"]`/`answer["ready"]`.
- Scoring caveat: the released run used an HLE/LLM judge; this local port uses a
  deterministic Exact-Answer normalized match/containment proxy (so it runs
  offline-of-judge). Swap in a `JudgeRubric` with `BROWSECOMP_GRADER_TEMPLATE`
  to reproduce the exact released metric.
- Smoke test: `num_examples=2, k=5` loaded; gold decrypted correctly (e.g.
  `Queen Marie of Romania`).

### `rlm-longbench-codeqa-local`

- Source: `THUDM/LongBench-v2`, filtered to `domain == "Code Repository
  Understanding"` (`sub_domain == "Code repo QA"`) — exactly 50 examples,
  matching the HF table `LongBenchv2 Code repo QA (n=50)`.
- It is a native 4-choice MCQ (`choice_A..D`, gold letter in `answer`), so the
  scorer is a deterministic exact-letter match — faithful, no judge needed.
- Adaptation: the long repo is exposed as the REPL `context`; the model
  finalizes a single letter via `answer["content"]`/`answer["ready"]`.
- Smoke test: `num_examples=2` loaded; gold letter present, `Code repo QA`
  sub_domain, ~1.9M-char repo context.

## Created / updated artifacts

- `training/environments/rlm_eval_suite/`: local env package (4 env IDs).
- `.../rlm_eval_suite/envs.py`: `rlm-oolong-local`, `rlm-oolong-pairs-local`,
  `rlm-browsecomp-plus-local`, `rlm-longbench-codeqa-local` loaders, each with a
  source-traced `user_prologue`.
- `.../pyproject.toml`: four `verifiers.environments` entry points.
- `training/configs/rlm-qwen3-30b-efficient-eval-suite.toml`: trains OOLONG-Spam
  and evaluates all FOUR HF-picture envs (OOLONG @132k n=50, OOLONG-Pairs @32k
  n=20, BrowseComp-Plus n=150 k=50, LongBench-v2 CodeQA n=50).
- `tools/validate_rlm_eval_suite.py`: validates all four loaders expose
  `user_prologue`, prologues carry the plan hint + task-format instructions, and
  the config references all four env IDs.

## Small-model k=50 consistency

For the Qwen3-30B / small-model RL setting, BrowseComp-Plus should use `k=50` consistently for both training and evaluation whenever BCP is included. The `k=1000` setting belongs to the RLM paper's GPT-5 long-context stress benchmark, not the HF Qwen3-30B adapter eval. The env default and config are therefore `k=50`.

## BrowseComp-Plus paper clarification

See `notes/browsecomp-plus-paper-rlm-details.md`. Critical distinction: the RLM paper uses BrowseComp+ with **1,000 documents** per task, but the HF Qwen3-30B model-card eval uses **k=50 documents**. The local eval-suite config follows the HF card (`k=50`) and now uses judge mode (`openai/gpt-4.1`) to match the BrowseComp+ paper metric.

## Residual uncertainties

- **Exact per-env prologues**: HF README lines 31/128 explicitly say the exact
  eval-time flags/prologues are TBD. The prologues here are evidence-derived
  ports (Prime + LMxLM + the "Plan before you act" hint), not byte-identical to
  Alex's private eval conditioning.
- **BrowseComp-Plus judge**: judge mode is now wired by default for the config; deterministic Exact-Answer matching remains only as a fallback/proxy if judge calls are unavailable.
- **Doc-sampling k-pool**: when a query's gold+evidence+negative pool is smaller
  than `k=50`, the local port returns the available pool (no full-corpus
  distractor padding), matching LMxLM's default `include_distractors=False`.
