# BrowseComp-Plus details from the BrowseComp+ paper and RLM paper

Sources read locally:

- BrowseComp-Plus paper: `papers/2508.06600-browsecomp-plus.pdf` / `.txt`
- RLM paper: `papers/2512.24601-rlm.pdf` / `.txt`
- HF model card README: `.firecrawl/hf-files/README.md`

## What BrowseComp-Plus is (paper `2508.06600`)

BrowseComp-Plus is **not** just the original live-web BrowseComp. It is a controlled,
fixed-corpus benchmark intended to remove the fairness/reproducibility problems of
black-box web search. Key details:

- It extends BrowseComp with a **fixed, curated corpus**.
- Each query has **human-verified supporting/evidence documents** and **hard negative** documents.
- Positive evidence is gathered by decomposing the BrowseComp question into clues,
  scraping cited evidence pages, then human-verifying whether documents justify clues
  and jointly answer the question.
- Annotators also mark **gold documents**: documents that directly contain the final
  answer, possibly semantically/implicitly rather than as an exact string.
- Final corpus statistics: **830 queries**, **100,195 documents**, average per query:
  **6.1 evidence docs**, **76.28 negative docs**, **2.9 gold docs**; average document
  length ~5,179 words / 32,296 chars.
- The agentic-search benchmark gives a retriever tool whose top **k=5** search results
  are returned, each truncated to the first **512 tokens**, for cost reasons.
- End-to-end Accuracy follows BrowseComp: an **LLM-as-judge (`gpt-4.1`)** compares the
  final answer against the gold answer. Agents also output confidence for calibration.
- Appendix prompt format includes:
  - `Explanation: ...`
  - `Exact Answer: ...`
  - `Confidence: ...`
  and the explanation asks for inline evidence-document citations by docid.

## How the RLM paper uses BrowseComp-Plus (`2512.24601`)

The RLM paper's main long-context benchmark is **BrowseComp-Plus (1K documents)**:

- 150 randomly sampled instances.
- Each instance input contains **1000 randomly chosen documents**.
- The **gold and evidence documents are guaranteed to exist** in those 1000 docs.
- Reported metric: percentage correct answers.
- Task length in Table 1: **6M-11M tokens**.
- It is framed as a constant-document-number task, but harder than S-NIAH because the
  answer requires piecing together information from several documents.
- RLM appendix D.2 additionally varies the number of documents on 20 random BCP tasks;
  only iterative methods such as RLM/ReAct remain strong at 100+ docs, and RLM is the
  only one reported to maintain perfect performance at the 1000-doc scale in that ablation.
- RLM trajectory E.1 shows a BrowseComp-Plus query with **1000 unique documents (~8.3M
  total tokens)** containing evidence documents and negatives. GPT-5 root first probes
  the 1000-doc list with regex/keywords, then uses subcalls/verification.

## Important distinction: RLM paper vs HF Qwen3-30B model card

These are **not the same BrowseComp+ input size**:

- RLM paper Table 1: `BrowseComp+ (1K)` = **1000 documents**, 6M-11M tokens.
- HF `mit-oasys/rlm-qwen3-30b-a3b-v0.1` README/eval image: `BrowseComp-Plus test`
  with **n=150, k=50 documents**.

So the local `rlm-browsecomp-plus-local` environment in this repo defaults to the
HF eval / small-model setting (`num_examples=150`, `k=50`) for both train and eval when BCP is used,
not the RLM paper's GPT-5 `1K documents` setting. A separate `k=1000` config would be
needed to reproduce the RLM paper's main BCP experiment, and it must include distractor
padding from the full 100k-doc corpus.

## Implications for the ERLM env/config

Changes made after reading both papers:

1. `rlm-browsecomp-plus-local` now preserves document IDs/URLs in each context string:
   `DOCID: ...\nURL: ...\nTEXT: ...`, so the model can cite evidence docs as the
   BrowseComp+ paper prompt expects.
2. BrowseComp+ eval config now explicitly uses:
   - `reward_mode = "judge"`
   - `judge_model = "openai/gpt-4.1"`
   matching the paper's Accuracy definition.
3. Deterministic `Exact Answer` matching remains available as a fallback/proxy if
   `reward_mode != "judge"` or if judge calls fail, but it is **not** the paper's metric.
4. The current HF-card config uses `k=50`. For RLM-paper `1K` experiments, a distinct
   config should set `k=1000` and enable full-corpus distractor padding; the current env
   does not yet materialize/pad from the full 100,195-document corpus.

## Bottom line

For the **HF Qwen3-30B / small-model train+eval setting**, keep `k=50` and judge scoring.
For the **RLM paper Table 1**, the correct BrowseComp+ setup is **150 tasks × 1000 docs**
with guaranteed gold/evidence inclusion, judged by an LLM; this is a heavier separate config.
