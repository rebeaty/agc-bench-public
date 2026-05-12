"""
HELM Scenario: NYT Connections

Paper: "NYT-Connections: A Deceptively Simple Text Classification Task that
       Stumps System-1 Thinkers" (COLING 2025, Best Dataset Paper)
       https://arxiv.org/abs/2412.01621
Data: https://huggingface.co/datasets/tm21cy/NYT-Connections

Task: Given 16 words, identify 4 groups of 4 words that share a common theme.
      Tests abstract reasoning, semantic knowledge, and wordplay understanding.

Prompt format (from lechmazur/nyt-connections benchmark):
  Find groups of four items that share something in common.
  Output them in the following format: four total lines.
  On each line, there should be four comma-separated items.

  Words: {words}

Fields used: words, answers
Fields skipped: date, contest (metadata)

Evaluation: Custom (check if all 4 groups match exactly)
  - Each correctly identified group = 0.25 points
  - Perfect solve = 1.0 points

Note: Only train split available; using full dataset as test set.
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class NYTConnectionsScenario(Scenario):
    name = "nyt_connections"
    description = "tm21cy/NYT-Connections"
    tags = ["creativity", "abstract_reasoning", "word_puzzles", "semantic_knowledge"]

    # Prompt from lechmazur/nyt-connections benchmark (p1.txt)
    PROMPT_TEMPLATE = """Find groups of four items that share something in common. Output them in the following format: four total lines. On each line, there should be four comma-separated items. No additional text (like group titles or descriptions) should be in the output. Also, there should not be anything in your output before or after the solution.

Words: {words}

Answer:"""

    def __init__(self, subset: str = "all"):
        super().__init__()
        self.subset = subset

    def get_instances(self, output_path: str):
        dataset = load_dataset("tm21cy/NYT-Connections", split="train")

        instances = []
        for item in dataset:
            words = item['words']
            answers = item['answers']

            # Format words as comma-separated list
            words_str = ", ".join(words)

            # Build prompt
            prompt = self.PROMPT_TEMPLATE.format(words=words_str)

            # Build reference answer (canonical format: 4 lines, each with 4 words)
            # Sort words within each group for consistent comparison
            reference_lines = []
            for answer in answers:
                group_words = sorted(answer['words'])
                reference_lines.append(", ".join(group_words))

            # Sort groups by first word for canonical ordering
            reference_lines.sort()
            reference_text = "\n".join(reference_lines)

            # Create reference with the canonical answer
            references = [
                Reference(Output(text=reference_text), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
