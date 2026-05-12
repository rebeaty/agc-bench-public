# speak_to_structure fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (add MolCustom_BasicProp subtask for full 10/10 coverage)

## Paper / repo audited
- Paper: arXiv:2412.14642 — "Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation"
- Source data: HF `phenixace/S2-TOMG-Bench` (full) / `phenixace/S2-TOMG-Bench-mini`

## Implementation audited
- scenarios/speak_to_structure_scenario.py — prepends chemist system head ("You are working as an assistant of a chemist user…") and appends dataset's `Instruction` field verbatim, with added directive to emit single `Molecule: [SMILES]` final line.
- metrics/speak_to_structure_metric.py — `SpeakToStructureMetric` with three RDKit-based metrics:
  - `validity` — fraction parseable by `Chem.MolFromSmiles` (matches paper).
  - `success_rate` — subtask-aware structural match: atom/bond/group counts for MolCustom; pre/post deltas for MolEdit (Add/Del/Sub); directional property change for MolOpt (LogP/MR/QED).
  - `similarity` — Morgan-2/2048 Tanimoto vs source molecule (MolEdit/MolOpt only).
- 9 of 10 paper subtasks loaded; per-subtask cap 500 (paper full split is ~5,000 ≈ ~556/subtask).

## Deviations found
- [MEDIUM] **MolCustom_BasicProp subtask omitted**: 9 of paper's 10 subtasks loaded. Named in docstring but missing from `_SUBTASKS`.
- [LOW] **Per-subtask cap 500** vs paper's ~556.
- [LOW] **Format-enforcement directive** added (not in paper); ensures parseability without semantic distortion.
- [INFO] Inference defaults (T=0.7, max_tokens=512) used — paper unspecified.

## Notes
SMARTS patterns and atom-list fields mirror upstream evaluator. Faithful core. No LLM judge — fully automatic RDKit cheminformatics evaluation, matching paper. `Instance.references=[]`; ground truth in `extra_data['row']` for evaluator routing. Tier 1.
