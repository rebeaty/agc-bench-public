# rebus_puzzle — _USER_PROMPT

**Category**: scoring
**Source**: `llm_judge/rebus_puzzle_annotator.py`
**Notes**: Module-level string constant `_USER_PROMPT`. Verbatim from source paper rubric.

```text
Determine whether the predicted rebus answer should be accepted as semantically
equivalent to the gold answer.

Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Return exactly one word:
YES
or
NO
```
