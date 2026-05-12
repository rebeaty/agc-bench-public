# poetmt fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_judge_alignment_and_comet (reconcile registry-vs-annotator judge, add COMET/BLEURT, fix instance-count docstring, populate inline judge_prompt strings)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/poetmt_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.poetmt_metric.PoetMTAutomaticMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **AnnotatorSpec(s):** `llm_judge.poetmt_annotator.PoetMTJudgeAnnotator`, `llm_judge.poetmt_annotator.PoetMTJudgeAnnotator`, `llm_judge.poetmt_annotator.PoetMTJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2408.09945 (Chen et al., EMNLP 2025, "LLMs for Classical Chinese Poetry Translation") - skim
- Repo: https://github.com/andongBlue/PoetMT - skim (uses `all_poems/{tang,song,yuan}.jsonl`)

## Implementation audited
- scenarios/poetmt_scenario.py - clones repo via `git clone`, loads `all_poems/{dynasty}.jsonl`. Fixed instruction (lines 133-137): "Please translate this classical Chinese poem into an English poem. Return only the translated English poem, with no explanation or extra commentary.\nPoem:{src}". Reference = `ref` (human English translation), CORRECT_TAG. dynasty="all" -> Tang+Song+Yuan combined (~790 by docstring; paper Table 1 reports 608).
- metrics/poetmt_metric.py - `PoetMTAutomaticMetric` computes BLEU-1 and BLEU-4 via NLTK `sentence_bleu` with method-1 smoothing on lightly normalized output (preamble stripping for "Translation:" wrappers and code fences).
- llm_judge/poetmt_annotator.py - 3 rubric judges (BS/BF/BM, 1-5 scale) per paper App. B.6/B.7/B.8. Default judge model `google/gemini-3-flash-preview` (env-overridable).
- registry_metrics.yaml: `bleu_1`, `bleu_4` (PoetMTAutomaticMetric); `llm_judge_beauty_of_sound`, `llm_judge_beauty_of_form`, `llm_judge_beauty_of_meaning` -- all wired to `openai/gpt-4`, T=0, max=32. `judge_prompt` is a *citation string* ("paper Appendix B.6 / repo evaluate/poem_evaluate.py") not the actual prompt template.
- registry_inference.yaml: `_use_defaults: true` (T=0.7, max=512, n=1).

## Deviations found
- [HIGH] C. Judge model mismatch: registry pins `openai/gpt-4`; the actual annotator defaults to `google/gemini-3-flash-preview`. The two are not equivalent and registry-driven tooling will misreport which judge ran.
- [HIGH] C. Metric scoring fidelity: paper primary metrics include sacreBLEU + COMET + BLEURT; implementation provides only NLTK sentence-BLEU (1 and 4) plus the BS/BF/BM judges. COMET and BLEURT absent.
- [MEDIUM] A. Instance count: scenario docstring claims Tang 295 / Song 196 / Yuan 299 / total 790; paper Table 1 reports 197 / 189 / 222 / 608. Either docstring or upstream split has drifted.
- [MEDIUM] A. Sentence-level adequacy subset (n=758) and the ACC metric from the paper are not implemented.
- [MEDIUM] C. LLM-Avg aggregate from the paper not computed.
- [MEDIUM] C. `judge_prompt` field in registry contains a citation string instead of the actual template -- runtime tooling that injects the registered prompt will send the literal "paper Appendix B.6..." text to the judge.
- [LOW] B. RAT retrieval/explanation context omitted (zero-shot baseline) -- defensible.
- [LOW] B. Poetry-type metadata not threaded into prompt.
- [LOW] D. T=0.7 default reasonable for creative translation; max=512 may truncate longer poems.

## Notes
Scenario layout, BLEU computation, and judge rubric content are faithful to the paper. Fix priority: (1) reconcile judge model -- pick GPT-4 (paper) or Gemini (current default) and update both annotator and registry to match; (2) replace the citation-string `judge_prompt` values with the actual templates (vendor App. B.6/B.7/B.8 verbatim into the registry); (3) add COMET (HuggingFace `unbabel/wmt22-comet-da`) and BLEURT-20; (4) fix the docstring instance counts or the split selection so they match Table 1; (5) document the sentence-level / ACC / LLM-Avg gaps.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_judge_alignment_and_comet
- Now:   Tier 2, high, patch_with_judge_alignment_and_comet
- Delta: confirmed
