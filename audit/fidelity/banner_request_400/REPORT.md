# banner_request_400 fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium
**Recommendation:** keep_as_is_with_caveat (surface judge prompts in registry; document the local-renderer substitution)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/banner_request_400_run_specs.py`:
>
> - **MetricSpec(s):** `llm_judge.banner_request_400_metric.BannerRequest400Metric`
> - **AnnotatorSpec(s):** `llm_judge.banner_request_400_annotator.BannerRequest400Annotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2503.11060 (BannerAgency, Sony) — note: skim (abstract only)
- Repo: https://github.com/sony/BannerAgency — note: skim (README only; eval.py not fetched directly)

## Implementation audited
- scenarios/banner_request_400_scenario.py — Downloads sony/BannerAgency repo, loads `concrete_5k.json` and iterates all advertisers × 4 pairs (`pair_1`..`pair_4`). Builds a multimedia `Input` with the advertiser PNG logo + a paraphrased text prompt (lines 21–63) instructing the model to act as a 300x250 banner foreground designer and emit a strict JSON blueprint schema. Reference-free (`references=[]`).
- metrics/banner_request_400_metric.py — does not live in this repo's `metrics/` folder; registry references `llm_judge.banner_request_400_metric.BannerRequest400Metric` (an external HELM annotator class). `has_metric_file: false` is correct for this folder.
- registry_metrics.yaml (lines 287–349): 7 paper-rubric judge metrics — `_score` (mean) plus `_taa`, `_lps`, `_aqs`, `_ctae`, `_cpyq`, `_bis` — all `openai/gpt-4o`, T=0.3, max_new_tokens=512, **all `judge_prompt: null`**. Plus 3 HELM-only health metrics (`_blueprint_validity`, `_render_success`, `_valid_judge_rate`) wired to the formula-based metric class.
- registry_inference.yaml (lines 82–84): `_use_defaults: true`.

## Deviations found
- [MEDIUM] B. Prompt fidelity: The generation prompt (lines 21–63) is paraphrased AGC-side, not lifted verbatim from the BannerAgency paper. It preserves intent (canvas size, typography ranges, layout style menu, JSON schema) but adds explicit pixel/percentage guidance that may not match the upstream prompt exactly.
- [MEDIUM] C. Metric/scoring fidelity: The 7 LLM-judge dimensions cover the paper's TAA/LPS/AQS/CTAE/CPYQ/BIS rubric, but `judge_prompt: null` on all 7. Without surfacing the prompts in the registry, paper-rubric reproduction is not auditable from the manifest alone (the actual prompt may live in the external `BannerRequest400Metric` class).
- [MEDIUM] A. Pipeline substitution: Paper renders banners through a Figma plugin; AGC uses a local deterministic blueprint→PNG renderer (referenced via `_render_success` health metric). Different rendering path may shift judge scores.
- [LOW] C. Judge model: pinned to `openai/gpt-4o` rather than canonical Gemini judge.
- [LOW] D. Generation config: paper-unspecified, so `_use_defaults` is acceptable; T=0.3 for judges is reasonable (deterministic-ish).
- [info] A. Instance count: 100 advertisers × 4 pair_keys = 400 instances, matches BannerRequest400 spec.

## Notes
The 6 paper-rubric dimensions plus blueprint/render health metrics are well-shaped. The two real risks are (1) the paraphrased generation prompt and (2) the local renderer substitution, both of which should be flagged in the paper writeup. The 3 health metrics are HELM-only but defensible as plumbing checks. Surfacing the actual judge prompts into `judge_prompt` (or pointing to where they live in the metric class) would close the auditability gap.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, Medium, "Keep with caveat. Faithful on rubric dimensions and instance source; deviates on rendering pipeline and uses an adapted, not verbatim, generation prompt. Surface judge prompt for full auditability."
- Now:   Tier 2, medium, keep_as_is_with_caveat
- Delta: confirmed
