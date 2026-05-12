# hypogen fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** discuss / keep_as_proxy (already rebranded; document explicitly that this is NOT Si et al. 2024 and is not paper-comparable)

## Paper / repo audited
- Paper: none (registry source_paper deliberately blanked 2026-04-25; the originally-cited arXiv:2409.04109 — Si et al., "Can LLMs Generate Novel Research Ideas?" — describes a full retrieval/expansion/blinded-human-review pipeline that this scenario does NOT reproduce)
- Repo: https://huggingface.co/datasets/UniverseTBD/hypogen-dr1 — "skim" (verified parquet path used by scenario; columns abstract / bit / flip)

## Implementation audited
- scenarios/hypogen_scenario.py — Downloads `data/test-00000-of-00001.parquet` from `UniverseTBD/hypogen-dr1` via direct URL (lines 41–44). Builds prompt at lines 76–83: gives the model the abstract + the "bit" (conventional limitation) and asks it to "Propose a novel research hypothesis or approach that overcomes this limitation (the 'flip')". Reference is the gold flip text (CORRECT_TAG). Skips items missing any of abstract/bit/flip.
- No metric file.
- registry_metrics.yaml (line 1477): registers `llm_judge_novelty` with judge_model_name=`openai/gpt-4`, T=0.0, max_new_tokens=256. `in_helm: false` (annotator presumably wired in run_spec, not verified here).
- registry_master.yaml (line 628): explicitly notes "Rebranded 2026-04-25 (audit Tier-3 fix). The local task is bit-flip completion against the `UniverseTBD/hypogen-dr1` artifact — NOT the Si et al. (2024) AI-Researcher pipeline. Citation has been decoupled."
- registry_inference.yaml (line 333): `_use_defaults: true`.

## Deviations found
- [HIGH] A. Dataset/instance source: although the rebrand correctly disclaims the Si et al. paper, the benchmark identifier `hypogen` itself is borrowed from a different upstream concept (HypoGen as published by Sternlicht et al. and others is a hypothesis-generation framework, not just bit-flip completion). The dataset-derived task is a single-shot completion against released gold flips, not a hypothesis-generation pipeline. The proxy is internally coherent but is not "paper-comparable" against any known published HypoGen leaderboard.
- [HIGH] C. Metric/scoring fidelity: registry registers `llm_judge_novelty` with no judge prompt (`judge_prompt: null`). Without a registered rubric, the judge call cannot be reproduced. Need either (a) a concrete rubric or (b) reference-overlap metrics (BLEU/ROUGE) against the gold flip — currently neither is wired explicitly in the yaml.
- [MEDIUM] B. Prompt fidelity: the scenario header openly says "No exact upstream prompt is published for this dataset derivative, so the local prompt remains a simple standardized instruction." The prompt at lines 76–83 is locally invented but spirit-preserving for the bit-flip task.
- [LOW] D. Generation configuration: defaults; novelty/hypothesis tasks typically use higher T (0.7–1.0) — defaults may be too low for diverse hypothesis generation.

## Notes
The 2026-04-25 rebrand is the right move. Two follow-ups: (1) populate `judge_prompt` in registry_metrics.yaml with the actual rubric used (or replace `llm_judge_novelty` with a reference-overlap metric like ROUGE-L since gold `flip` text is available); (2) consider whether to also report a reference-overlap baseline against the gold flip — that's a cheap, reproducible signal even without an LLM judge. Tier 3 retained because the benchmark identifier + source_paper history will mislead readers without explicit footnoting.

## Compared to prior audit (2026-04-25)
- Prior: Tier 3, high, discuss (rebrand or replace)
- Now:   Tier 3, high, discuss / keep_as_proxy
- Delta: confirmed
