# Reasoning on/off intervention

**Category**: intervention
**Source**: `experiments/cap_interventions/reasoning_on_off/run_generation.py`
**Judge model**: `10 reasoning-capable release models`
**Notes**: Toggle reasoning per provider API. d_z = +0.34 length-residualized composite shift. Paper §3.6 / §4.6. Prompts are assembled in `build_prompt(task, prompt_id, item, condition)`; the function source is reproduced verbatim above.

```text
def build_prompt(task: str, prompt_id: int, item: str) -> str:
    """The CAP-creative prompt only — same wording as the 'creative' arm
    of the creative_vs_effective experiment, so the only manipulation is
    reasoning on vs off."""
    if task == "AUT":
        return (f"Come up with unusual and original uses for the following object: {item}\n\n"
                "Give exactly 3 unusual uses, separated by semicolons (;). Each should be a short phrase "
                "(a few words). Give only the uses, no numbering or explanation.\n\nAnswer:")
    if task == "Design":
        return ("Think of unusual and original solutions to this design problem:\n"
                f"{item}\n\nGive exactly 3 unusual solutions, separated by semicolons (;). Each should be a short phrase "
                "(a few words). Give only the solutions, no numbering or explanation.\n\nAnswer:")
    if task == "SCTT":
        return ("Think of unusual and original scientific ideas for the following scenario:\n"
                f"{item}\n\nGive exactly 3 unusual scientific ideas or questions, separated by semicolons (;). "
                "Each should be a short phrase or question. Give only the ideas, no numbering or explanation.\n\nAnswer:")
    if task == "Metaphor":
        return ("Finish the sentence with an unusual and original metaphor:\n"
                f"{item}\n\nGive exactly 1 unusual metaphor completion. It should be a short phrase of 1-5 words. "
                "Give only the completion, no explanation.")
    if task == "Story":
        return (f"Write an unusual and original short story using these three words: {item}\n\n"
                "Give exactly 1 short story of 3-8 sentences. The story should be surprising, original, "
                "and engaging. Give only the story, no title or explanation.")
    raise ValueError(task)
```
