# tinyfabulist — _USER_PROMPT

**Category**: scoring
**Source**: `llm_judge/tinyfabulist_annotator.py`
**Notes**: Module-level string constant `_USER_PROMPT`. Verbatim from source paper rubric.

```text
Evaluate the following fable according to these specific criteria:

1. **Grammar & Style (1-10)**:
   • 1-3: Significant errors that impede understanding
   • 4-6: Some errors but generally readable
   • 7-10: Clean, polished writing with appropriate language and style for a fable

2. **Creativity & Originality (1-10)**:
   • 1-3: Derivative, predictable, or clichéd
   • 4-6: Contains some original elements but follows familiar patterns
   • 7-10: Fresh perspective, innovative approach while maintaining classic fable structure

3. **Moral Clarity (1-10)**:
   • 1-3: Moral absent, confused, or contradictory
   • 4-6: Moral present but underdeveloped or lacking impact
   • 7-10: Clear, meaningful moral that provides genuine insight

4. **Adherence to Prompt (1-10)**:
   • 1-3: Missing multiple required elements from the prompt
   • 4-6: Incorporates main elements but overlooks some instructions
   • 7-10: Thoroughly addresses all prompt requirements while maintaining narrative cohesion

5. **Age Group Fit**:
   Determine which age group this fable is most appropriate for based on:
   • Vocabulary complexity and sentence structure
   • Conceptual difficulty of the moral lesson
   • Story length and complexity
   • Content appropriateness

Age groups are defined as:
  - A: 3 years or under
  - B: 4-7 years
  - C: 8-11 years
  - D: 12-15 years
  - E: 16 years or above

Format your response as valid JSON with this structure:
{{
    "type": "Fable Evaluation",
    "grammar": <integer 1-10>,
    "creativity": <integer 1-10>,
    "moral_clarity": <integer 1-10>,
    "adherence_to_prompt": <integer 1-10>,
    "best_age_group": "<letter: A, B, C, D, or E>",
    "explanation": [
        "<One sentence explaining grammar & style score>",
        "<One sentence explaining creativity & originality score>",
        "<One sentence explaining moral clarity score>",
        "<One sentence explaining adherence to prompt score>",
        "<One sentence explaining why this fable best fits the chosen age group>"
    ]
}}

Be critical but fair. Ensure your entire evaluation is concise yet informative.

Original Prompt:
{prompt}

Fable:
{fable}
```
