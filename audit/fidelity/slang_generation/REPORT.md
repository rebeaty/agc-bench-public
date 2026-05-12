# slang_generation fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** discuss (manifest-promised judges absent; metric is custom proxy not paper's suite; consider dropping or fully re-implementing as Wu & Sun (2025)'s detection/interpretation/coinage/reuse modes)

## Paper / repo audited
- Paper: https://aclanthology.org/2025.findings-emnlp.348/ (Wu & Sun, EMNLP-Findings 2025) — abstract via ACL Anthology. Paper compares human slang from Online Slang Dictionary vs GPT-4o/Llama-3 outputs across detection, interpretation, lexical coinage, and reuse modes. Does **not** centrally feature a "freeform definition→slang generation" task as its primary evaluation. (note: "skim")
- Repo: https://github.com/siyangwu1/LLM-Slang-Dictionary — `data/conv_slang.txt` (list of (term, definition) tuples) and `code/generation.py::build_prompt_general` referenced by scenario header. (note: "skim")

## Implementation audited
- scenarios/slang_generation_scenario.py — fetches `data/conv_slang.txt`, parses as Python list of (term, definition) tuples. For each definition, asks model to generate one slang word + definition + usage_context as JSON. Reference is the gold term (used loosely; not really a target since task is generative). Single freeform mode; paper's reuse vs coinage variants are NOT exposed.
- metrics/slang_generation_metric.py — `SlangGenerationMetric` is a custom semantic-novelty proxy: parses JSON from model output, looks up generated word's WordNet glosses, computes mean L2 distance between definition embedding (MiniLM-L6-v2 via embedder factory) and gloss embeddings as `slang_semantic_novelty`. Plus parse_rate and novelty_coverage diagnostics. Sophisticated but **not the paper's metric suite**.
- registry_metrics.yaml (lines 2483-2498): two llm_judge metrics — `llm_judge_creativity` and `llm_judge_relevance` — both `judge_model_name: openai/gpt-4`, `judge_prompt: null`, T=0, 256 tokens. **NOT routed to `SlangGenerationMetric`** and prompts are unfilled. The implemented metric (semantic_novelty/parse_rate/novelty_coverage) is **dead code** as registered.
- registry_inference.yaml (lines 620-622): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: Triple problem. (a) Registry advertises two GPT-4 judges with null prompts that don't connect to the implemented metric. (b) The actual implemented metric is a WordNet-distance proxy that the paper does not use — it's our invention. (c) Paper's actual evaluation is detection accuracy / interpretation match / coinage-vs-reuse classification by humans + LLM judges against Online Slang Dictionary attestations; our task framing is "definition → generate slang word" which is closer to a controlled paraphrase task than to anything in the paper.
- [HIGH] A. Dataset source: We use `conv_slang.txt` (conversational slang term-definition pairs) but the paper's primary analysis compares against the broader Online Slang Dictionary attestations. Our slice is a subset that doesn't map onto the paper's evaluation cells.
- [MEDIUM] B. Prompt fidelity: Paraphrased prompt with explicit JSON schema; original `build_prompt_general` was for batched generation in the paper's experiments. Spirit preserved for a generation pretext, but the framing is ours.
- [info] D. Generation config: defaults; for short JSON output, low T would be advisable.

## Notes
This dataset entry is structurally drift from its source paper. The implemented metric is well-engineered (parser-aware, WordNet+embedder pipeline, threading-safe) but it is a custom novelty proxy, not anything the paper reports. The registered judges are unwired. Two paths: (a) **drop** from AGC (cleanest) since the bench essentially measures something the paper doesn't, or (b) **rebuild** to match Wu & Sun's actual evaluation (detection/interpretation/coinage-vs-reuse with proper human-attested gold) — this is a substantial undertaking requiring access to their evaluation harness. Given the broad mismatch, drop is recommended unless this benchmark name is essential to the release set.

## Compared to prior audit (2026-04-25)
- Prior: Tier 3, high, discuss (manifest-promised judges absent; metric is custom proxy not paper's suite)
- Now:   Tier 3, high, discuss (drop or full rebuild)
- Delta: confirmed
