# rlm-eval-suite

Local `RLMTrainEnv` ports of **all four** RLM eval environments shown in the HF
model-card eval picture, traced from:

- `PrimeIntellect-ai/research-environments` commits by Alex Zhang (`az <alex.lx.zhang@gmail.com>`):
  - `4802f7c5` OOLONG RLM eval based on actual repo
  - `ddd5efb0` OOLONG_RLM reward/context subset updates
  - `36de2d0f` OOLONG numerical filtering
  - `1c039a33` update RLM envs with new harness
  - `570654b4` add `rlm_oolong_pairs`
  - `b07ace37` rlm_oolong defaults to `trec_coarse` at 256k/512k
  - `40553027` rlm-browsecomp environment (simple-evals answer format + HLE grader)
  - `2fb131b9` add `rlm_longbenchpro`
- `LMxLM` task descriptions/program prompts for OOLONG, OOLONG-Pairs, and
  BrowseComp-Plus (`lm_to_program/browsecomp_plus`).
- `THUDM/LongBench-v2` `Code Repository Understanding` domain for the code-repo QA env.
- HF model card lines 31/35/39-44/124-128 + the eval image (`media/hf/rlm_eval_results.png`):
  the "Plan before you act" hint, the four eval envs, and per-env prologue uncertainty.

These environments adapt the Prime sandbox style (`/workspace/context.txt`, `/task/answer.txt`) to the upstream `alexzhang13/rlm` training harness, which exposes `context` directly in the REPL and terminates by setting `answer["content"]` and `answer["ready"] = True`.

## Environment IDs (one per HF-picture eval environment)

- `rlm-oolong-local`: OOLONG synth tasks; official OOLONG scoring port. HF eval: `trec_coarse @132k (n=50)`.
- `rlm-oolong-pairs-local`: OOLONG-Pairs; precision/recall/F1 over user-ID pairs. HF eval: `@32k (n=20)`.
- `rlm-browsecomp-plus-local`: BrowseComp-Plus evidence-doc deep-research QA from `Tevatron/browsecomp-plus`
  (decrypted via the public canary), `k` docs/query, simple-evals `Explanation / Exact Answer / Confidence`
  answer format. HF eval: `test (n=150, k=50 documents)`.
- `rlm-longbench-codeqa-local`: `THUDM/LongBench-v2` filtered to `domain == "Code Repository Understanding"`
  (exactly 50 examples), 4-choice MCQ scored by exact letter match. HF eval: `LongBenchv2 Code repo QA (n=50)`.

### Small-model k=50 consistency

For this Qwen3-30B / small-model train+eval setup, `rlm-browsecomp-plus-local` defaults to `k=50`, and the config uses `k=50`. Do not use the RLM paper's `k=1000` BrowseComp+ stress setting for this run unless creating a separate large-context replication config.

### Scoring fidelity caveats

- **BrowseComp-Plus**: use `reward_mode="judge"` for the released/HF-card metric. The local port preserves the verbatim HLE grader prompt (`BROWSECOMP_GRADER_TEMPLATE`) and simple-evals answer format. A deterministic `Exact Answer` normalized-match proxy remains available only as a fallback / offline smoke-test mode (`reward_mode != "judge"`).
- **LongBench-v2 CodeQA**: 4-choice MCQ, so exact-letter scoring is faithful (no judge needed).
- **Per-env prologues**: HF README lines 31/128 state the exact eval-time flags/prologues are TBD; the
  prologues here are evidence-derived ports (Prime + LMxLM + the "Plan before you act" hint), not claimed
  byte-identical to Alex's private eval conditioning.
