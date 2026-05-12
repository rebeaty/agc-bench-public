"""
HELM Scenario: TinyStories

Paper: https://arxiv.org/abs/2305.07759 (ICLR 2024)
Dataset: roneneldan/TinyStories
Evaluation prompts: https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/Evaluation%20prompts.yaml

Task: Complete short stories written in simple, child-friendly language. Given the
beginning of a story, generate a coherent and creative ending.

Prompt format: Standard story completion
  - Input: First half of a story (ending mid-sentence or at a cliffhanger)
  - Output: Story completion

Fields used: Evaluation_prompts.yaml (44 story beginnings)
Fields skipped: None (training data not used for evaluation)

Evaluation: llm_judge
  - Judge model: GPT-4
  - Dimensions: Grammar, Creativity, Consistency (with story beginning)
  - Scale: 1-10 for each dimension
  - Additional: Age group estimation
  - Format: "Grammar: 8/10, Creativity: 7/10, Consistency: 7/10"
  - See scenarios/tinystories/annotator_notes.md for judge configuration

Note: The TinyStories dataset contains 2.1M training stories and 22K validation
      stories for training small language models. The 44 evaluation prompts from
      Evaluation_prompts.yaml constitute the official test set for evaluating
      story generation capabilities.
"""

import os
import yaml
from urllib.request import urlretrieve
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input,
    TEST_SPLIT
)


class TinyStoriesScenario(Scenario):
    name = "tinystories"
    description = "roneneldan/TinyStories"
    tags = ["creativity", "story_generation", "language_generation"]

    EVAL_PROMPTS_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/Evaluation%20prompts.yaml"

    def _download_evaluation_prompts(self, output_path: str) -> str:
        """Download evaluation prompts YAML file if not already present."""
        prompts_file = os.path.join(output_path, "evaluation_prompts.yaml")

        if not os.path.exists(prompts_file):
            os.makedirs(output_path, exist_ok=True)
            urlretrieve(self.EVAL_PROMPTS_URL, prompts_file)

        return prompts_file

    def get_instances(self, output_path):
        # Download evaluation prompts
        prompts_file = self._download_evaluation_prompts(output_path)

        # Load prompts from YAML
        with open(prompts_file, 'r') as f:
            story_prompts = yaml.safe_load(f)

        instances = []
        for story_beginning in story_prompts:
            prompt = f"""The student is given the beginning of a story and needs to complete it into a full story.
Write the rest of the story in simple language that a young child could understand.

***
{story_beginning}
***"""

            # No reference completions - evaluation is via LLM-as-judge
            # Empty references list for open-ended generation task
            references = []

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                extra_data={"story_beginning": story_beginning},
            ))

        return instances
