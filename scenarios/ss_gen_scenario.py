"""
HELM Scenario: SS-GEN (Social Story Generation)

Paper: SS-GEN: A Social Story Generation Framework with Large Language Models
       https://arxiv.org/abs/2406.15695
Code/Data: https://github.com/MIMIFY/SS-GEN

Task: Generate Social Stories for children with Autism Spectrum Disorder (ASD).
      Social Stories are therapeutic narratives that help children understand social
      situations and develop coping strategies.

Prompt format (from paper Section 4):
  Develop a concise, clear, straightforward, positive and supportive Social Story
  titled "{title}" for children and teens with autism, 200-300 words, that promotes their social understanding and boosts their participation in daily activities, fostering independence and confidence.

Constraints (8 criteria from paper Appendix Figure 13):
  - Structural: Clear title, introduction, main body, conclusion
  - Voice/Tone: Positive, patient, literally accurate
  - Descriptiveness: 2:1 ratio of descriptive to coaching sentences
  - Perspective: First/third person only (never second person)
  - Content: Addresses WH-questions, celebrates achievements

Fields used: title, story_content (reference)
Fields available but not used in prompt: chapter, explanation (metadata), id
  - Note: chapter and explanation could be added to prompt for additional context,
    but paper evaluation used title-only prompts for consistency

Evaluation (from paper Section 4):
  - Traditional metrics: BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L
  - Human evaluation: Structural Clarity (1-5), Descriptive Orientation (Y/N),
    Situational Safety (Y/N)
  - GPT-4 evaluation: Coherence, Descriptiveness, Empathy, Grammaticality,
    Relevance (1-5 each)

Dataset: 5,085 stories across 57 thematic chapters
  - Train: 4,068 examples
  - Dev: 509 examples
  - Test: 508 examples
"""

import json
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class SSGenScenario(Scenario):
    """
    SS-GEN: Social Story Generation benchmark for evaluating LLM creativity
    in constrained therapeutic writing for children with autism.
    """

    name = "ss_gen"
    description = "MIMIFY/SS-GEN"
    tags = ["creativity", "social_stories", "text_generation", "healthcare"]

    # Data URLs from GitHub repository
    GITHUB_BASE_URL = (
        "https://raw.githubusercontent.com/MIMIFY/SS-GEN/main/SS-GEN%20Dataset/"
    )
    TRAIN_FILE = "refined_gpt4story_all_5085_from_gpt4titles_train.jsonl"
    DEV_FILE = "refined_gpt4story_all_5085_from_gpt4titles_dev.jsonl"
    TEST_FILE = "refined_gpt4story_all_5085_from_gpt4titles_test.jsonl"

    def __init__(self):
        super().__init__()

    def _build_prompt(self, title: str) -> str:
        """
        Build the prompt for social story generation.
        Uses the exact instruction from paper Section 4.
        """
        return (
            f'Develop a concise, clear, straightforward, positive and supportive '
            f'Social Story titled "{title}" for children and teens with autism, '
            f'200-300 words, that promotes their social understanding and boosts their participation in daily activities, fostering independence and confidence.'
        )

    def _load_jsonl(self, file_path: str) -> List[dict]:
        """Load JSONL file and return list of examples."""
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        return examples

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load SS-GEN dataset and create HELM instances."""
        # Ensure output directory exists
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        # Download dataset files
        split_files = {
            TRAIN_SPLIT: self.TRAIN_FILE,
            VALID_SPLIT: self.DEV_FILE,
            TEST_SPLIT: self.TEST_FILE,
        }

        instances: List[Instance] = []

        for split, filename in split_files.items():
            file_url = self.GITHUB_BASE_URL + filename
            local_path = os.path.join(data_path, filename)

            # Download file if not already present
            ensure_file_downloaded(
                source_url=file_url,
                target_path=local_path,
            )

            # Load examples from JSONL
            examples = self._load_jsonl(local_path)

            # Create instances
            for example in examples:
                example_id = example["id"]
                title = example["title"]
                story_content = example["story_content"]

                # Build prompt using paper's instruction
                prompt = self._build_prompt(title)

                # Create instance with reference story
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=story_content), tags=[CORRECT_TAG])],
                    split=split,
                    id=f"{split}_{example_id}",
                )
                instances.append(instance)

        return instances
