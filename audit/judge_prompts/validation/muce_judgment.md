# MuCE judgment prompt (predict human creativity rating)

**Category**: validation
**Source**: `scripts/muce_judgment_pilot.py`
**Judge model**: `each release model under test`
**Notes**: Each release model predicts a human creativity rating on the 1,862-item MuCE subset (paper §3.6 / §4.5).

```text
Task description: {full_prompt}

Response to rate: {response}

A creative or original response should be:
  - novel (unusual, original, unique)
  - high-quality (well-crafted and appropriate for the task)
  - surprising (unexpected, non-obvious)

Rate the {rating_label} of the response on a scale from 10 (not at all) to 50 (extremely {rating_label}). Return only a single integer between 10 and 50, with no other text.
```
