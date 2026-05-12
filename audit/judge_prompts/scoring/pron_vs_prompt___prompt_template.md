# pron_vs_prompt — _PROMPT_TEMPLATE

**Category**: scoring
**Source**: `llm_judge/pron_vs_prompt_annotator.py`
**Notes**: Module-level string constant `_PROMPT_TEMPLATE`. Verbatim from source paper rubric.

```text
You are an expert literature critic. Evaluate the following creative synopsis for an imaginary movie title using the released Pron vs Prompt literary rubric.

Title: {title}
Synopsis: {synopsis}

Return JSON only with integer scores using these exact keys:
- title_attractiveness: 0-3
- style_attractiveness: 0-3
- theme_attractiveness: 0-3
- title_originality: 0-3
- style_originality: 0-3
- plot_originality: 0-3
- relevance: 0-4
- title_creativity: 0-3
- synopsis_creativity: 0-3
- anthology: 0-3
- readers_opinion: 0-3
- critics_opinion: 0-3
- own_voice: 0-3

Guidance:
- Attractiveness: how engaging the title, style, and story/characters are as literary objects.
- Originality: how surprising and non-cliched the title, style, and plot are.
- Relevance: how well the synopsis uses the title as a creative starting point.
- Creativity: overall creativity of the title and synopsis.
- Criticism block: how likely the text is anthology-worthy, aligned with readers and critics, and indicative of a recognizable voice.

Use only integers in the allowed ranges.
```
