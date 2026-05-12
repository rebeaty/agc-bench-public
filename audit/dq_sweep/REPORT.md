# AGC-Bench Data-Quality Sweep — Primary Release Set

Two independent passes, cross-referenced:

1. **Heuristic pass** (this sweep) — empty / short / refusal / repetitive rates from raw `scenario_state.json` files for every (model, dataset) cell.
2. **On-task audit** (existing) — `data_quality_llm_judge_v4.csv`, three random items per cell rated by `x-ai/grok-4.1-fast` for on-task / garbled.

Release scope: 83 strict-coverage models × 67 text-only datasets = 5,561 expected cells.
Cells in the join: 5,513.

## Cross-classification

| status | count | % |
|--------|-------|---|
| **CLEAN** | 4,990 | 90.5 |
| **AUDIT_ONLY** | 216 | 3.9 |
| **HEURISTIC_ONLY** | 152 | 2.8 |
| **AUDIT_MISSING** | 123 | 2.2 |
| **BOTH** | 32 | 0.6 |

- **CLEAN**: heuristic and audit both green.
- **HEURISTIC_ONLY**: heuristic-flagged but audit on-task ≥ 50 %. Often short-form benchmarks where minimal output is the expected format (e.g. multiple-choice, close-ended exact-match).
- **AUDIT_ONLY**: audit on-task < 50 %, normal length. Subtle off-task content the heuristics cannot see.
- **BOTH**: flagged by both signals — highest-priority cells for action (re-run, exclude, or document).
- **AUDIT_MISSING**: cell ran but did not appear in the audit sample (rare).

## Action priority — 32 cells flagged by both signals

| model | dataset | n | mean chars | audit on-task | heuristic flag |
|-------|---------|---|-----------|---------------|----------------|
| deepcogito/cogito-v2.1-671b | grapheval_ai_researcher | 9 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | analobench | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | chinese_homophonic_puns | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | futuregen | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | historical_analogy | 20 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | humor_transfer | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | lcc_metaphor | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | moh_x | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | science_analogies | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | scimon | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | showerthoughts | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | simile_generation | 50 | 0 | 0% | empty 100% |
| google/gemma-2-27b-it | unfun_corpus | 50 | 0 | 0% | empty 100% |
| meta-llama/llama-3.1-8b-instruct | unfun_corpus | 50 | 78 | 33% | refusal 44% |
| meta-llama/llama-3.2-11b-vision-instruct | unfun_corpus | 50 | 88 | 33% | refusal 44% |
| meta-llama/llama-3.2-3b-instruct | pron_vs_prompt | 50 | 980 | 33% | refusal 68% |
| meta-llama/llama-3.2-3b-instruct | speak_to_structure | 50 | 93 | 0% | refusal 90% |
| meta-llama/llama-3.2-3b-instruct | unfun_corpus | 50 | 110 | 0% | refusal 96% |
| morph/morph-v3-fast | outline_to_story | 50 | 9109 | 0% | empty 40% |
| morph/morph-v3-fast | poetmt | 50 | 99 | 0% | empty 52% |
| morph/morph-v3-fast | riddlesense | 50 | 156 | 0% | empty 38% |
| morph/morph-v3-fast | science_analogies | 50 | 20 | 0% | empty 54% |
| morph/morph-v3-fast | sdat | 50 | 254 | 0% | empty 40% |
| morph/morph-v3-fast | showerthoughts | 50 | 18303 | 0% | repetitive 86% |
| morph/morph-v3-fast | simile_generation | 50 | 35 | 33% | empty 48% |
| morph/morph-v3-fast | ss_gen | 50 | 30296 | 67% | empty 52% |
| morph/morph-v3-fast | story_quality | 50 | 282 | 0% | empty 38% |
| qwen/qwen3-14b | historical_analogy | 20 | 1 | 0% | empty 90% |
| tencent/hunyuan-a13b-instruct | newyorker_humor | 50 | 0 | 0% | empty 100% |
| z-ai/glm-4.5v | futuregen | 50 | 111 | 33% | empty 32% |
| z-ai/glm-4.5v | proparalogy | 50 | 6 | 0% | empty 78% |
| z-ai/glm-4.5v | riddlesense | 50 | 15 | 0% | empty 32% |
