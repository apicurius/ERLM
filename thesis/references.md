# References

This bibliography lists every canonical citation key used throughout the thesis, together with additional sources identified from the project repository and its notes. Entries use an author–year style. Inline citations elsewhere in the thesis use the verbatim keys defined here, e.g. `[Zhang et al., 2026]`. Where a bibliographic detail could not be verified against a primary source, it is explicitly flagged "(details to verify)"; arXiv identifiers and Hugging Face slugs are reproduced exactly as recorded in the project repository.

## Primary works

**[Zhang et al., 2026]** Zhang, A. L., Kraska, T., and Khattab, O. (2026). *Recursive Language Models.* arXiv preprint arXiv:2512.24601. (The foundational RLM paper. Defines the task-agnostic paradigm that replaces `llm.completion(prompt, model)` with `rlm.completion(prompt, model)`, offloads context as a REPL variable, and treats sub-LLM calls as first-class functions. Source of the long-tailed-cost observation, §F.2, and the length-generalization result, Observation 6. Released scaffold: `mit-oasys/rlm-qwen3-30b-a3b-v0.1`.) (publication venue and exact author affiliations: details to verify)

**[Wang et al., 2024]** Wang, X., et al. (2024). *CodeAct: Executable Code Actions Elicit Better LLM Agents.* arXiv preprint arXiv:2402.01030. (Design lineage cited for the RLM code-environment harness with function-like sub-LLM calls, in preference to JSON tool-calling.) (full author list: details to verify)

## Reinforcement learning and alignment

**[Shao et al., 2024]** Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv preprint. (Introduces Group Relative Policy Optimization, GRPO, the group-relative advantage estimator $s_i - \mathrm{mean}(s)$ used by the `prime-rl` trainer.) (arXiv id and full author list: details to verify)

**[Ouyang et al., 2022]** Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback (InstructGPT).* Advances in Neural Information Processing Systems (NeurIPS). (RLHF foundation for reward-driven post-training.) (exact volume, pages, and arXiv id: details to verify)

**[Lambert et al., 2024]** Lambert, N., et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* arXiv preprint. (Reinforcement Learning with Verifiable Rewards, RLVR, the verifiable-correctness reward family from which the correctness signal $c$ in $R = c\,(1 + \lambda e)$ descends.) (arXiv id and full author list: details to verify)

**[Ng et al., 1999]** Ng, A. Y., Harada, D., and Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* In *Proceedings of the Sixteenth International Conference on Machine Learning (ICML)*. (Potential-based reward shaping; theoretical reference for adding the efficiency term $\lambda e$ as a correctness-gated, policy-preserving shaping signal.) (exact pages: details to verify)

**[Skalse et al., 2022]** Skalse, J., et al. (2022). *Defining and Characterizing Reward Hacking.* Advances in Neural Information Processing Systems (NeurIPS). (Formal grounding for the anti-reward-hacking protocol in Chapter 5 and the verifier-hardening discussion.) (full author list, exact pages, and arXiv id: details to verify)

**[Amodei et al., 2016]** Amodei, D., et al. (2016). *Concrete Problems in AI Safety.* arXiv preprint arXiv:1606.06565. (Reward gaming, scalable oversight, and Goodhart's law; motivates keeping efficiency strictly subordinate to correctness.) (full author list: details to verify)

## Models, adapters, and parameter-efficient fine-tuning

**[Hu et al., 2021]** Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv preprint arXiv:2106.09685. (Low-rank adapters; the thesis trains `rank=32`, `alpha=64` adapters on the `q_proj`/`k_proj`/`v_proj`/`o_proj` projections.) (full author list: details to verify)

**[Qwen Team, 2025]** Qwen Team (2025). *Qwen3 Technical Report.* arXiv preprint / Alibaba Group technical report. (Base-model family; this work uses `Qwen/Qwen3-30B-A3B-Instruct-2507`, a 30B-A3B mixture-of-experts instruct model, in non-thinking sampling mode.) (arXiv id and exact report title: details to verify)

## Software, frameworks, and infrastructure

**[verifiers]** *Verifiers: Environments for LLM Reinforcement Learning.* PrimeIntellect / willccbb open-source library. (Environment abstraction underlying `RLMTrainEnv`, which mirrors a depth-1 `RLM.completion` call with REPL and sub-LLM-call logging.) (authorship, version, and URL: details to verify)

**[prime-rl]** Prime Intellect. *prime-rl.* Open-source RL trainer. (GRPO/DPPO trainer with vLLM inference, runtime LoRA hot-loading via `/load_lora_adapter`, and multi-environment sampling by relative ratio weight.) (version and URL: details to verify)

## Benchmarks and datasets

**[OOLONG]** *OOLONG: Long-Context Question Answering Benchmark.* (Long-context QA; the thesis uses the `trec_coarse` subset at 131k–262k tokens, plus the `spam` training split. Source-traced from PrimeIntellect-ai `research-environments` (`rlm_oolong`) and the LMxLM OOLONG task description.) (canonical citation, authors, and venue: details to verify)

**[OOLONG-Pairs]** `mit-oasys/oolong-pairs`. Hugging Face dataset. (Pairwise user-interaction classification at ~32k context, n=20, F1-scored over unordered user-ID pairs.) (canonical citation, authors, and venue: details to verify)

**[BrowseComp]** *BrowseComp: A Benchmark for Browsing Agents.* (Parent benchmark extended by BrowseComp-Plus.) (canonical citation, authors, and venue: details to verify)

**[BrowseComp-Plus]** *BrowseComp-Plus.* arXiv preprint arXiv:2508.06600; corpus `Tevatron/browsecomp-plus`. (Fixed curated corpus of 830 queries over 100,195 documents with canary-encrypted gold answers and an LLM-as-judge accuracy metric; the thesis uses $k=50$ documents per query with a `gpt-4.1` judge in the small-model setting.) (full author list and venue: details to verify)

**[LongBench-v2]** *LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks.* `THUDM/LongBench-v2`. (Long-context benchmark; the thesis uses the Code Repository Understanding subset, n=50, 4-choice MCQ with deterministic exact-letter scoring.) (arXiv id, authors, and venue: details to verify)

## Released artifacts

**[RLM-scaffold]** *mit-oasys/rlm-qwen3-30b-a3b-v0.1.* Hugging Face model card. (Released LoRA adapter, `rank=32`, `alpha=64`, `q_proj`/`k_proj`/`v_proj`/`o_proj` projections, over `Qwen/Qwen3-30B-A3B-Instruct-2507`; trained with a correctness-only reward $R = c$ on a mixed OOLONG-Spam + BrowseComp-Plus long-context suite via `RLMTrainEnv` and `prime-rl`.) (training duration and hardware as source-asserted; details to verify)

---

> **[NOTE]** Bibliographic identifiers (arXiv ids, Hugging Face slugs) are reproduced verbatim from the project repository and configuration files. Author lists, venues, and page ranges marked "(details to verify)" must be confirmed against primary sources before final submission.
