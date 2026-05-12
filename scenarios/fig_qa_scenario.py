"""
HELM Scenario: Fig-QA (Figurative Language Question Answering)

Paper: https://arxiv.org/abs/2204.12632 (NAACL 2022)
       "Testing the Ability of Language Models to Interpret Figurative Language"
Code: https://github.com/nightingal3/Fig-QA
Dataset: https://huggingface.co/datasets/nightingal3/fig-qa

Task: Interpret creative metaphors and figurative language through Winograd schema-style
multiple-choice questions. Models must use commonsense reasoning to understand the
implicit meaning of metaphorical expressions.

Example:
  Metaphor: "Her word had the strength of titanium."
  Options:
    A. Her promises can be believed.
    B. Her promises cannot be trusted.
  Correct: A (titanium is strong, so her word is strong/reliable)

Paired example (showing contrast):
  Metaphor: "Her word had the strength of a wine glass."
  Options:
    A. Her promises can be believed.
    B. Her promises cannot be trusted.
  Correct: B (wine glass is fragile, so her word is weak/unreliable)

Dataset: 11,914 human-written creative metaphors paired as Winograd schemas
  - Train: 9,674 examples
  - Validation: 1,094 examples
  - Test: 1,146 examples (labels hidden)

Evaluation: Multiple-choice accuracy
  - Models must select the correct interpretation (ending1 or ending2)
  - Tests nonliteral reasoning and creative language understanding

Metaphor Types: Similes, implicit metaphors, and other figurative expressions
  - Often structured as comparisons ("as X as Y")
  - Requires understanding both literal properties and metaphorical implications
  - Tests cultural knowledge and commonsense reasoning

Key Challenge: Models must reason about the implicit meanings conveyed through
creative metaphors, going beyond literal interpretation to understand the intended
message or comparison.

Note: The paper shows that while models can be fine-tuned to perform reasonably well,
their few-shot performance falls significantly short of human performance, indicating
this is a challenging creative language understanding task.
"""

from datasets import load_dataset
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, VALID_SPLIT, TEST_SPLIT, TRAIN_SPLIT
)


class FigQAScenario(Scenario):
    name = "fig_qa"
    description = "Figurative Language Question Answering (Metaphor Interpretation)"
    tags = ["creativity", "metaphor", "figurative_language", "commonsense", "winograd"]

    def __init__(self, use_validation_as_test: bool = False):
        """
        Initialize Fig-QA scenario.

        Args:
            use_validation_as_test: If True, use validation split as test (labels available).
                                   If False, use test split (labels hidden).
                                   (default: False)
        """
        super().__init__()
        self.use_validation_as_test = use_validation_as_test

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for Fig-QA benchmark."""
        # Load dataset from HuggingFace
        dataset = load_dataset('nightingal3/fig-qa')

        instances = []

        # Process training split
        for idx, item in enumerate(dataset['train']):
            if item.get('valid', 1) == 1:  # Only include valid examples
                instances.append(self._create_instance(item, idx, TRAIN_SPLIT))

        # Process validation split (always included)
        for idx, item in enumerate(dataset['validation']):
            if item.get('valid', 1) == 1:
                split = TEST_SPLIT if self.use_validation_as_test else VALID_SPLIT
                instances.append(self._create_instance(item, idx, split))

        # Process test split (only if not using validation as test)
        if not self.use_validation_as_test:
            for idx, item in enumerate(dataset['test']):
                if item.get('valid', 1) == 1:
                    # Test split has labels=-1 (hidden), so we'll create instances
                    # without correct answer tags
                    instances.append(self._create_instance(item, idx, TEST_SPLIT, has_labels=False))

        return instances

    def _create_instance(self, item: dict, idx: int, split: str, has_labels: bool = True) -> Instance:
        """Create a HELM Instance from a Fig-QA example."""
        startphrase = item['startphrase']
        ending1 = item['ending1']
        ending2 = item['ending2']
        label = item.get('labels', -1)

        # Input is just the metaphor + question; ADAPT_MULTIPLE_CHOICE_JOINT formats the options
        prompt = f"{startphrase}\n\nWhich interpretation is correct?"

        # References contain the actual option texts; the adapter will letter them A/B
        if has_labels and label != -1:
            ref_a = Reference(Output(text=ending1), tags=[CORRECT_TAG] if label == 0 else [])
            ref_b = Reference(Output(text=ending2), tags=[CORRECT_TAG] if label == 1 else [])
            references = [ref_a, ref_b]
        else:
            references = [
                Reference(Output(text=ending1), tags=[]),
                Reference(Output(text=ending2), tags=[])
            ]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": idx,
                "startphrase": startphrase,
                "ending1": ending1,
                "ending2": ending2,
                "label": label if has_labels else -1
            }
        )
