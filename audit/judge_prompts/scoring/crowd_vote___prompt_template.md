# crowd_vote — _PROMPT_TEMPLATE

**Category**: scoring
**Source**: `llm_judge/crowd_vote_annotator.py`
**Notes**: Module-level string constant `_PROMPT_TEMPLATE`. Verbatim from source paper rubric.

```text
You are evaluating a marketing creativity benchmark response.

This local benchmark is a proxy adaptation of a pairwise human-preference benchmark.
Score the single response on four dimensions from 1 to 5.

Brand: {brand}
Category: {category}
Task type: {task_type}
Prompt: {prompt_text}
Response: {response}

Scoring guidance:
- Originality: Is the response surprising, non-cliche, and fresh?
- Brand Relevance: Is it meaningfully tied to this brand rather than generic?
- Creative Potential: Could it become a plausible campaign platform or activation?
- Conciseness: Does it respect the intended short-form constraint for the task?

Return exactly this format:
ORIGINALITY: [1-5]
BRAND_RELEVANCE: [1-5]
CREATIVE_POTENTIAL: [1-5]
CONCISENESS: [1-5]
OVERALL: [1-5]
```
