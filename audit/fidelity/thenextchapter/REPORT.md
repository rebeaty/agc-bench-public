# thenextchapter fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (document metric-set differences vs paper; populate `judge_prompt` placeholders; consider roberta-large for BERTScore)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/thenextchapter_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.meteor_metric.MeteorMetric`, `metrics.bert_score_metric.BertScoreMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **ScenarioSpec args:** `subset=subset`
> - **AnnotatorSpec(s):** `llm_judge.thenextchapter_annotator.TheNextChapterJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2301.09790 (Xie et al., INLG 2023, "The Next Chapter") - skim
- Repo: https://github.com/ZhuohanX/TheNextChapter - skim (sample_human.txt files used)

## Implementation audited
- scenarios/thenextchapter_scenario.py - three subsets (roc/wp/cnn) loaded from `GeneratedStories/{subset}/sample_human.txt`. Zero-shot prompt = subset-specific cue + raw condition (lines 62-75: "Continue the following story...", "Write a short story...", "Continue the following news article:"). Reference = `story` field as CORRECT_TAG. Per docstring: 800 ROC / 1000 WP / 600 CNN -- matches paper Sec 3.2.
- metrics/thenextchapter_metric.py - DOES NOT EXIST (`has_metric_file: false`).
- llm_judge/thenextchapter_annotator.py - 5 judge dimensions (fluency, coherence, relatedness, logicality, interestingness, 1-5 scale, gpt-4, T=0.0, max=256) matching paper's human-evaluation axes.
- registry_metrics.yaml: `bleu_4`, `rouge_{1,2,l}`, `meteor`, `bert_score` (`bert-base-uncased`), plus 5 `thenextchapter_{fluency,coherence,relatedness,logicality,interestingness}` judge metrics (all `judge_model_name: openai/gpt-4`, `judge_prompt: null`, T=0.0, max=256).
- registry_inference.yaml: `_use_defaults: true` (T=0.7, max=512, n=1).

## Deviations found
- [MEDIUM] D. Decoding mismatch: T=0.7 default vs paper's nucleus sampling p=0.95. HELM adapter doesn't expose top_p in the `_use_defaults` block.
- [MEDIUM] B. Paradigm shift: HELM zero-shot+cue vs paper's few-shot/fine-tuned model evaluation.
- [MEDIUM] C. Metric set differences: implementation adds ROUGE-1/2/L + METEOR (not in paper primary metrics); omits BLEURT, BARTScore (BAS), MS-Jaccard (MSJ), BBL, and SBL/D-3/LR-n diversity metrics.
- [MEDIUM] C. BERTScore backbone: registry uses `bert-base-uncased` vs paper's `roberta-large` -- non-comparable absolute values.
- [MEDIUM] C. Judge swap: paper uses AMT + in-house human raters (3 per item, 20 items/subset); implementation substitutes GPT-4 over the full set on the same 5 axes. Defensible scaling choice but not the same measurement.
- [MEDIUM] C. `judge_prompt` is `null` for all 5 judge metrics in registry, even though the annotator file presumably contains the actual prompts -- registry-driven tooling will see no prompt template.
- [info] A. 800/1000/600 instance counts match paper exactly.
- [info] B. Subset-specific cues are minimal but defensible for chat-tuned models that would otherwise ask clarifying questions.

## Notes
Data ingestion and the 5-axis judge dimensions are faithful to the paper's human-evaluation rubric. Real fidelity gaps are: decoding (T=0.7 vs nucleus p=0.95), missing diversity metrics (MSJ/BBL/D-3/LR-n), BERTScore backbone (bert-base-uncased vs roberta-large), and the human->LLM judge swap. Patches: (a) add top_p=0.95 to inference registry, (b) switch BERTScore to roberta-large to match paper, (c) populate `judge_prompt` strings inline in registry, (d) note the diversity-metrics gap as known scope reduction.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_caveats
- Now:   Tier 2, high, keep_as_is_with_caveats
- Delta: confirmed
