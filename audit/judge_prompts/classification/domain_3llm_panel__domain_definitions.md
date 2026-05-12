# Domain classification — DOMAIN_DEFINITIONS (definitions block)

**Category**: classification
**Source**: `scripts/classify_domains.py`
**Judge model**: `gemini-3-flash, x-ai/grok-4.1-fast, openai/gpt-4.1-mini (majority vote)`
**Notes**: Per-benchmark domain assignment. Cohen's κ ≈ 0.85 across pairs (paper §3.7).

```text
Each benchmark in the AGC-Bench release set was screened to be creativity-related
during a PRISMA systematic review. Your job is to classify each one into
exactly one of six creativity domains and to flag whether the task asks
the model to *generate* creative outputs or *evaluate* them.

DOMAIN (pick one):

1. **Brainstorming**
   Open-ended divergent ideation on everyday or general topics that do
   not require specialized domain knowledge. The creative ability is
   producing varied, novel candidates from common conceptual space.

2. **Problem Solving**
   The creativity dimension is convergent insight or lateral thinking:
   solving a well-defined puzzle, riddle, or lateral-thinking problem
   that has a constrained answer space but requires non-obvious
   reasoning. The creative ability is finding the unexpected solution.

3. **STEM**
   Scientific, mathematical, or technical creativity grounded in domain
   knowledge: hypothesis generation, research-idea proposals, novel
   mathematical solutions or proofs, scientific modeling, technical
   writing, or data-driven discovery.

4. **Story / Narrative**
   The creativity dimension is narrative writing: producing or judging
   stories, narratives, or extended creative prose given a prompt, cue,
   outline, or genre. The creative ability is coherent narrative
   construction at paragraph or longer length.

5. **Figurative Language**
   The creativity dimension is figurative language: producing or judging
   metaphors, similes, hyperbole, personification, idioms, slang, or
   similar constructs at the phrase or sentence level rather than full
   narratives.

6. **Humor**
   The creativity dimension is humor: producing or judging jokes, puns,
   caption-contest entries, satire, witty observations, or other humorous
   short-form constructions.

TASK TYPE (separate, pick one):

- **Generation**: the task asks the model to produce a creative output
  (a story, an idea, a metaphor, a joke).
- **Evaluation**: the task asks the model to judge, rate, rank, or
  pairwise-compare creative outputs (e.g., binary creative-or-not
  verdict, Likert creative-quality rating, A-vs-B preference).

Both fields are required. Decide both based on the actual task structure
shown in the sample prompts and model responses, not the source paper's
framing.
```
