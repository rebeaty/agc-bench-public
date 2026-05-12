"""
HELM Scenario: MoPS Premise Evaluation

Paper: "MoPS: Modular Story Premise Synthesis for Evaluating Creative Writing" (arXiv:2406.05690)
Dataset: ManTle/mops on HuggingFace (https://huggingface.co/datasets/ManTle/mops)

Task: Given structured story components (theme, background, persona, event, ending, twist),
synthesize a coherent story premise sentence.

Prompt format: adapted from the upstream `SYNYHESIZE_PROMPT` template in
`GAIR-NLP/MoPS`.

Fields used: theme, background, persona, event, ending, twist (inputs); premise (reference)
Fields skipped: novel, script (long model-generated story outputs), id
Split: curated (100 highest-quality premises across 14 themes)

Eval: LLM-as-judge — fascination, completeness, originality (0-100 each, GPT-4 style prompt);
      plus set-level breadth/density diversity over sentence embeddings.
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class MoPSPremiseScenario(Scenario):
    name = "mops_premise"
    description = "ManTle/mops"
    tags = ["creativity", "story_generation", "narrative"]

    @staticmethod
    def _strip_prefix(text, prefix):
        """Remove an embedded field label prefix if present (e.g. 'Twist: ...')."""
        stripped = text.strip()
        if stripped.lower().startswith(prefix.lower() + ":"):
            stripped = stripped[len(prefix) + 1:].strip()
        return stripped

    def get_instances(self, output_path):
        dataset = load_dataset("ManTle/mops", split="curated")

        instances = []
        for item in dataset:
            prompt = (
                "The following is the theme, background, persona, main event, final ending "
                "and twist of a novel or script:\n\n"
                "### Theme\n"
                f"{item['theme']}\n\n"
                "### Background\n"
                f"{self._strip_prefix(item['background'], 'background')}\n\n"
                "### Persona\n"
                f"{self._strip_prefix(item['persona'], 'persona')}\n\n"
                "## Event\n"
                f"{self._strip_prefix(item['event'], 'event')}\n\n"
                "## Ending\n"
                f"{self._strip_prefix(item['ending'], 'ending')}\n\n"
                "## Twist\n"
                f"{self._strip_prefix(item['twist'], 'twist')}\n\n"
                "Please combine the aforementioned elements of a novel or script into "
                "one compact, concise, and coherent sentence as a story premise.\n"
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=item["premise"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                extra_data={"source_id": item["id"], "theme": item["theme"]},
            ))

        return instances
