# tinyfabulist — grammar_score

**Category**: scoring
**Source**: `data/registry/registry_metrics.yaml (tinyfabulist.grammar_score)`
**Judge model**: `openai/o3-mini-2025-01-31`

```text
Evaluate the following fable according to these specific criteria:

1. **Grammar & Style (1-10)**:
   1-3: Significant errors that impede understanding
   4-6: Some errors but generally readable
   7-10: Clean, polished writing with appropriate language and style for a fable

2. **Creativity & Originality (1-10)**:
   1-3: Derivative, predictable, or clichéd
   4-6: Contains some original elements but follows familiar patterns
   7-10: Fresh perspective, innovative approach while maintaining classic fable structure

3. **Moral Clarity (1-10)**:
   1-3: Moral absent, confused, or contradictory
   4-6: Moral present but underdeveloped or lacking impact
   7-10: Clear, meaningful moral that provides genuine insight

4. **Adherence to Prompt (1-10)**:
   1-3: Missing multiple required elements from the prompt
   4-6: Incorporates main elements but overlooks some instructions
   7-10: Thoroughly addresses all prompt requirements while maintaining narrative cohesion

Fable: {generated_response}

Return JSON: {"grammar": <1-10>, "creativity": <1-10>, "moral_clarity": <1-10>, "adherence_to_prompt": <1-10>}
```
