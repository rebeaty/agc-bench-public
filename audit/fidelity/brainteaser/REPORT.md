# brainteaser fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_registry_wiring (point `registry_metrics.yaml` at `metrics.brainteaser_metric.BrainteaserMetric` and enumerate grouped metrics)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/brainteaser_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.brainteaser_metric.BrainteaserMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2310.05057 (Jiang et al., EMNLP 2023, BRAINTEASER) - skim (abstract only)
- Repo: https://github.com/1171-jpg/BrainTeaser - skim (data path validated)

## Implementation audited
- scenarios/brainteaser_scenario.py - downloads AES-encrypted `BTDATA.zip` (password `brainteaser`), iterates `sentence_puzzle.npy` + `word_puzzle.npy`, emits one Instance per item (4 choices: answer + 2 distractors + "none-of-the-above"); `extra_data` carries subset/variant/base_id for grouped scoring (~2,307 instances total including SR/CR variants).
- metrics/brainteaser_metric.py - implements `BrainteaserMetric` with grouped Ori/Sem/Con single accuracies plus SR (Ori&Sem) and CR (Ori&Sem&Con) group accuracies, separately for sentence/wordplay/all -- mirrors paper's primary reporting.
- registry_metrics.yaml: lists ONLY `exact_match -> helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics`; the custom `BrainteaserMetric` class is NOT registered.
- registry_inference.yaml: `_use_defaults: true` (T=0.7, max=512, n=1).

## Deviations found
- [HIGH] C. Metric/scoring fidelity: registry omits `BrainteaserMetric`. The paper's primary numbers (Ori/Sem/Con single + SR/CR group accuracies) will not be reported -- only generic exact_match will run. Custom metric file exists but is dead code.
- [LOW] D. Generation config: T=0.7 default for an MCQ task is higher than ideal; 0.0-0.3 would better match deterministic MCQ scoring conventions.
- [info] A. Dataset: full ~2,307-item zero-shot release loaded; matches paper.
- [info] B. Prompts: question text passed verbatim, four choices attached as references; relies on HELM default MCQ adapter which fits paper's zero-shot setup.

## Notes
Scenario and metric implementation are faithful -- the failure is purely registry wiring. Fix: edit `registry_metrics.yaml` `brainteaser` block to register `metrics.brainteaser_metric.BrainteaserMetric` with metric names `brainteaser_{sentence,wordplay,all}_{overall,single_original,single_semantic,single_context,sr,cr}_accuracy`, and verify the run_spec wires it. Drop temperature to ~0.0 in the inference registry since this is MCQ.

## Compared to prior audit (2026-04-25)
- Prior: Tier ?, unknown, no audit conducted
- Now:   Tier 2, high, patch_with_registry_wiring
- Delta: newly classified
