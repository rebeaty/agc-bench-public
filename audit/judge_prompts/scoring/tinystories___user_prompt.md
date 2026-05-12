# tinystories — _USER_PROMPT

**Category**: scoring
**Source**: `llm_judge/tinystories_annotator.py`
**Notes**: Module-level string constant `_USER_PROMPT`. Verbatim from source paper rubric.

```text
The student is given a beginning of a story. The student needs to complete it into a full story.
Evaluate the student's completed story as if you were grading a classroom exercise.

Score each dimension from 1 to 10:
- grammar: correctness and clarity of language
- creativity: originality and imagination of the story
- consistency: how well the completion fits the given beginning and remains coherent

Also estimate the age group of the hypothetical student writer using:
- A: 3 or under
- B: 4-5
- C: 6-7
- D: 8-9
- E: 10-12

Return only valid JSON in this exact format:
{{
  "grammar": 1-10 integer,
  "creativity": 1-10 integer,
  "consistency": 1-10 integer,
  "age_group": "A" | "B" | "C" | "D" | "E"
}}

Story beginning:
***
{story_beginning}
***

Student completion:
***
{completion}
***
```
