# crowd_vote fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** discuss / rename (rebrand as "marketing_creativity" — original benchmark is pairwise crowd voting; this is single-LLM-judge proxy on a re-curated brand list)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2509.09702 ("Creativity Benchmark: A benchmark for marketing creativity for large language models", Springboards.ai) — "skim" via scenario docstring (paper not re-fetched, but scenario header quotes the prompt templates and dataset construction caveat verbatim)
- Repo: none (proprietary platform at https://creativitybenchmark.ai/; the brand-challenge set is not publicly released)

## Implementation audited
- scenarios/crowd_vote_scenario.py — Hardcoded `_BRANDS_BY_CATEGORY` (lines 60–109) listing 100 brands across the paper's 12 categories. Three task templates `insights`/`ideas`/`wild_ideas` (lines 118–134) match the paper's prompt wording. System prompt (lines 112–116) matches paper. 100 brands × 3 task types = 300 instances. Prepended `system + "\n\n" + task_prompt` rather than using a separate system role.
- metrics/crowd_vote_metric.py — Wraps annotator output `marketing_creativity_judge` into stats: parse_rate + originality, brand_relevance, creative_potential, conciseness, overall. Prefix renamed 2026-04-25 from `crowd_vote_*` to `marketing_creativity_judge_*` to reflect that no actual crowd vote occurs.
- registry_metrics.yaml (line 733): registers `llm_judge_quality` with judge_model_name=`openai/gpt-4`, T=0.0, max_new_tokens=256. `in_helm: false`. Does NOT match the actual annotator/metric prefix used in `crowd_vote_metric.py` ("marketing_creativity_judge").
- registry_inference.yaml (line 181): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: paper's primary metric is **pairwise crowd voting** by 678 advertising professionals across ~11K head-to-head comparisons producing Bradley-Terry-style rankings. Our implementation uses a single LLM judge giving 5 absolute sub-scores (originality, brand_relevance, creative_potential, conciseness, overall). This is a structurally different evaluation, not a judge-model swap.
- [HIGH] A. Dataset/instance source: paper's 100 brand challenges are **proprietary** and not released. The implementation substitutes a hand-curated list of 100 well-known brands (8–9 per category) chosen for "diversity and global recognition", documented in the scenario header. Items therefore do not map 1:1 to the paper's distribution.
- [MEDIUM] C. Metric/scoring fidelity: registry_metrics.yaml registers `llm_judge_quality` (single number, gpt-4 judge) but the actual run_spec (not read here, but cross-referenced from prior audit) wires `marketing_creativity_judge` with 5 sub-scores. Stale yaml relative to implementation.
- [LOW] B. Prompt fidelity: prompts and system prompt are reproduced verbatim from the paper.
- [LOW] D. Generation configuration: defaults; paper does not specify model decoding params for the contestant models.

## Notes
The implementation is competent but it is not a reproduction of the published "Creativity Benchmark". The benchmark identifier `crowd_vote` actively misleads — there is no crowd, no voting. The post-2026-04-25 metric prefix rename to `marketing_creativity_judge_*` is the right direction; the scenario should follow suit (rename to `marketing_creativity_judge` or similar) and either (a) state plainly in the paper that the primary pairwise comparisons are out of scope, or (b) implement a Bradley-Terry rerank from pairwise judge calls. Tier 3 retained.

## Compared to prior audit (2026-04-25)
- Prior: Tier 3, high, keep_as_is_with_renaming (or rework to pairwise voting)
- Now:   Tier 3, high, discuss / rename
- Delta: confirmed
