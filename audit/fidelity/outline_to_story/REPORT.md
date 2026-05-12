# outline_to_story fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** keep_as_is_with_caveat (document the RAKE-side outline extraction; flag missing WikiPlots and perplexity)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/outline_to_story_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.outline_to_story_metric.OutlineToStoryMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2101.00822 (Fang et al., "Outline to Story," 2021) — note: skim (abstract; says outlines built by SOTA keyword extraction)
- Repo: https://github.com/fangleai/Outline2Story — note: unread (extraction details not separately fetched)

## Implementation audited
- scenarios/outline_to_story_scenario.py — Loads `euclaise/writingprompts` HF dataset (default `split="test"`, 15,138 examples). Segments gold story into paragraphs with a dialogue/<114-char merge heuristic (lines 56–68). Runs RAKE (`min_length=1, max_length=4`) to extract 2–5 event keywords per paragraph (lines 71–82, count = `min(5, max(2, len/228 + 1.5))`). Builds prompt: "Write a coherent multi-paragraph story that follows the ordered outline events below..." + raw WP prompt + numbered `Paragraph i: ev1; ev2; ...` outline + "Story:" (lines 92–99). Reference = paragraph-merged gold story.
- metrics/outline_to_story_metric.py — `OutlineToStoryMetric`: per-paragraph normalized-substring match of expected RAKE events against generated paragraph. Reports `outline_event_recall` (mean per-paragraph recall) and `outline_paragraph_coverage` (% paragraphs with at least one event match). Faithful operationalization of "did the model follow the outline."
- registry_metrics.yaml (lines 2011–2036): six metrics — `rouge_1`, `rouge_2`, `rouge_l`, `bleu_4` (BasicGenerationMetric), plus the two custom controllability metrics.
- registry_inference.yaml (lines 504–506): `_use_defaults: true`. Run-spec layer reportedly pins T=0.95, max_tokens=1024, zero-shot.

## Deviations found
- [MEDIUM] B./C. Outline extraction substitution: AGC computes outlines on the fly with RAKE (Rapid Automatic Keyword Extraction); paper presumably used a different keyword extractor for their released outline–story pairs. The paragraph-aligned event-control construct is preserved but the actual outline content differs from paper artifacts.
- [MEDIUM] A. Coverage: WikiPlots slice (the paper's other dataset) is not implemented; only WritingPrompts is covered.
- [MEDIUM] C. Metric coverage: Perplexity (one of the paper's reference metrics) is dropped because it is incompatible with API-only models. Defensible, but a documented gap.
- [LOW] C. Custom controllability metrics: `outline_event_recall` and `outline_paragraph_coverage` are HELM-side additions that extend, not replace, the paper's eval. Reasonable.
- [LOW] A. 200-instance subsample (benchmark-wide policy) vs. paper's full test split.
- [info] B. Prompt template is paraphrased AGC-side but preserves the spirit (paragraph-by-paragraph outline-controlled story generation).
- [info] D. Inference T=0.95 is appropriate for open-ended story generation.

## Notes
Faithful in spirit — paragraph-aligned event control, gold-story-derived outline (so the controllability metric is well-posed), reference-based BLEU/ROUGE for surface fidelity, and a sensible custom controllability metric. The RAKE substitution is the main caveat: it changes the input distribution slightly relative to the paper's released outlines, but does not change the construct (controllable multi-paragraph generation). Recommended cleanup: surface the RAKE choice in scenario tags, document the missing WikiPlots slice, and decide whether to ship with or without perplexity.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium-high, keep_as_is (with caveat about RAKE-based outline extraction)
- Now:   Tier 2, medium-high, keep_as_is_with_caveat
- Delta: confirmed
