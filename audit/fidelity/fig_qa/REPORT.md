# fig_qa fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_use_validation_as_test_default

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/fig_qa_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
> - **ScenarioSpec args:** `use_validation_as_test=True` (line 15 of run_spec) — directly addresses the HIGH deviation flagged below ("default routes hidden-label test split…"). The released runs evaluate against the 1,094 labeled validation examples per the paper's reported setup.
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Liu et al., NAACL 2022, "Testing the Ability of Language Models to Interpret Figurative Language" (arXiv:2204.12632)
- Repo: github.com/nightingal3/Fig-QA

## Implementation audited
- scenarios/fig_qa_scenario.py — builds `"{startphrase}\n\nWhich interpretation is correct?"` plus two references; HELM `ADAPT_MULTIPLE_CHOICE_JOINT` letters them A/B.
- No custom metric file (uses HELM built-in `compute_reference_metrics`).
- 11,914 paired metaphors in dataset (train 9,674 / val 1,094 / test 1,146 hidden).

## Deviations found
- [HIGH] **Default routes hidden-label test split to TEST_SPLIT without CORRECT_TAG** — accuracy is uncomputable in default config. Fix: set `use_validation_as_test=True` (1,094 labeled val examples) to match paper's reported eval setup.
- [LOW] Minor wording deviation from repo's zero/few-shot template ("Which is more plausible?"). Low impact.
- [INFO] Inference: `_use_defaults`. MC task — low temperature appropriate; HELM defaults fine.

## Notes
Paper reports multiple-choice accuracy. Registry maps `exact_match` via `compute_reference_metrics`, which under MC-joint adaptation equals MC accuracy. Faithful approach. Single fix needed: change default to `use_validation_as_test=True` so accuracy is computable on the labeled validation set; otherwise scenario is faithful to paper's MC accuracy protocol.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_use_validation_as_test_default
- Now:   Tier 2, high, patch_with_use_validation_as_test_default
- Delta: confirmed (verified scenarios/fig_qa_scenario.py lines 63-99: `use_validation_as_test` default False routes hidden-label test to TEST_SPLIT; registry metric is `exact_match` under `compute_reference_metrics`)
