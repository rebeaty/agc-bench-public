# cpers — _PROMPT_TEMPLATE

**Category**: scoring
**Source**: `llm_judge/cpers_annotator.py`
**Notes**: Module-level string constant `_PROMPT_TEMPLATE`. Verbatim from source paper rubric.

```text
You are evaluating the creativity of a Persian literary sentence using the culturally adapted TTCT-style framework from the CPers paper.

Topic: {topic}
Generated Persian text: {response}

Score each question from 1 to 5.

Originality:
1. Does the sentence demonstrate creativity and originality in expression?
2. Does the sentence avoid cliches and overused expressions?
3. Does the sentence contain at least one literary device such as simile, metaphor, hyperbole, or antithesis?

Fluency:
1. Is the sentence grammatically correct?
2. Does the sentence sound natural to Persian readers?
3. Is the sentence appropriate as a literary sentence?

Flexibility:
1. Does the sentence use multiple ideas or layers to express the topic?
2. Does the sentence look at the topic from a fresh perspective?
3. Does the sentence show stylistic or conceptual variety?

Elaboration:
1. Does the sentence use rich and diverse vocabulary?
2. Does the sentence evoke imagery?
3. Does the sentence effectively convey emotion?

Also detect whether each rhetorical device is present:
- simile
- metaphor
- hyperbole
- antithesis

Return exactly one JSON object in this schema:
{{
  "originality": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "fluency": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "flexibility": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "elaboration": {{"q1": 1, "q2": 1, "q3": 1, "average": 1.0}},
  "overall_creativity": 1.0,
  "devices": {{
    "simile": 0,
    "metaphor": 0,
    "hyperbole": 0,
    "antithesis": 0
  }}
}}
```
