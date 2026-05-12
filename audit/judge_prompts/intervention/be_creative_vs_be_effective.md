# Be-creative vs. be-effective intervention

**Category**: intervention
**Source**: `experiments/cap_interventions/creative_vs_effective/run_generation.py`
**Judge model**: `15 frontier release models`
**Notes**: Prompt-suffix manipulation on the 5 CAP tasks. d_z = +1.40 length-residualized. Paper §3.6 / §4.6. Prompts are assembled in `build_prompt(task, prompt_id, item, condition)`; the function source is reproduced verbatim above.

```text
def build_prompt(task: str, prompt_id: int, item: str, condition: str) -> str:
    if task == "AUT":
        if condition == "creative":
            return (
                f"Come up with unusual and original uses for the following object: {item}\n\n"
                "Give exactly 3 unusual uses, separated by semicolons (;). Each should be a short phrase "
                "(a few words). Give only the uses, no numbering or explanation.\n\n"
                "Answer:"
            )
        return (
            f"Come up with practical and useful purposes for the following object: {item}\n\n"
            "Give exactly 3 practical uses, separated by semicolons (;). Each should be a short phrase "
            "(a few words). Give only the uses, no numbering or explanation.\n\n"
            "Answer:"
        )
    if task == "Design":
        if condition == "creative":
            return (
                "Think of unusual and original solutions to this design problem:\n"
                f"{item}\n\n"
                "Give exactly 3 unusual solutions, separated by semicolons (;). Each should be a short phrase "
                "(a few words). Give only the solutions, no numbering or explanation.\n\n"
                "Answer:"
            )
        return (
            "Think of practical and feasible solutions to this design problem:\n"
            f"{item}\n\n"
            "Give exactly 3 practical solutions, separated by semicolons (;). Each should be a short phrase "
            "(a few words). Give only the solutions, no numbering or explanation.\n\n"
            "Answer:"
        )
    if task == "SCTT":
        if condition == "creative":
            return (
                "Think of unusual and original scientific ideas for the following scenario:\n"
                f"{item}\n\n"
                "Give exactly 3 unusual scientific ideas or questions, separated by semicolons (;). "
                "Each should be a short phrase or question. Give only the ideas, no numbering or explanation.\n\n"
                "Answer:"
            )
        return (
            "Think of rigorous and testable scientific ideas for the following scenario:\n"
            f"{item}\n\n"
            "Give exactly 3 rigorous scientific ideas or questions, separated by semicolons (;). "
            "Each should be a short phrase or question. Give only the ideas, no numbering or explanation.\n\n"
            "Answer:"
        )
    if task == "Metaphor":
        if condition == "creative":
            return (
                "Finish the sentence with an unusual and original metaphor:\n"
                f"{item}\n\n"
                "Give exactly 1 unusual metaphor completion. It should be a short phrase of 1-5 words. "
                "Give only the completion, no explanation."
            )
        return (
            "Finish the sentence with an apt and fitting metaphor:\n"
            f"{item}\n\n"
            "Give exactly 1 fitting metaphor completion. It should be a short phrase of 1-5 words. "
            "Give only the completion, no explanation."
        )
    if task == "Story":
        if condition == "creative":
            return (
                f"Write an unusual and original short story using these three words: {item}\n\n"
                "Give exactly 1 short story of 3-8 sentences. The story should be surprising, original, "
                "and engaging. Give only the story, no title or explanation."
            )
        return (
            f"Write a clear and well-crafted short story using these three words: {item}\n\n"
            "Give exactly 1 short story of 3-8 sentences. The story should be coherent, fitting, "
            "and engaging. Give only the story, no title or explanation."
        )
    raise ValueError(task)
```
