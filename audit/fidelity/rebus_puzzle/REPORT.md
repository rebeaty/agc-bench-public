# rebus_puzzle fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (minor: judge model swap is documented)

## Paper / repo audited
- Repo: https://github.com/Kyunnilee/visual_puzzles ("Visual Puzzles: A Probe for Understanding Vision-Language Models")
- Dataset: https://huggingface.co/datasets/Kyunnilee/visual-puzzles
- N=432 hand-annotated English rebus puzzles, 11 cognitive skill categories.

## Implementation audited
- scenarios/rebus_puzzle_scenario.py — verbatim PROMPT from `solvers/utils.py`, including JSON-with-`answer`/`reasoning` instruction.
- metrics/rebus_puzzle_metric.py — `rebus_answer_accuracy` (normalized exact match on parsed JSON `answer`), `rebus_answer_parse_rate`, `rebus_semantic_equivalence` (binary YES/NO judge).
- llm_judge/rebus_puzzle_annotator.py — judge `google/gemini-2.5-flash-lite` (T=0, 16 tokens) with backup. Paper used GPT-4o.
- 432 puzzles loaded from HF + GitHub `answers.json`. Matches paper.

## Deviations found
- [LOW] **Judge model swap**: Gemini 2.5 Flash-Lite vs paper's GPT-4o. Cost/access motivation; binary semantic-equivalence task is identical.
- [LOW] `max_tokens=256` modest given JSON+reasoning format; could truncate long reasoning but `answer` field typically appears early.
- [INFO] Image saved as JPEG (lossy) from PNG source — negligible visual fidelity loss.

## Notes
Zero-shot, T=0 multimodal generation; matches paper's primary condition. Skill metadata preserved in `extra_data` for sub-group analysis. PROMPT verbatim, INSTANCE COUNT exact match. Tier 1.
