"""
HELM Scenario: Unfun Corpus

Paper: Getting Serious about Humor: Crafting Humor Datasets with Unfunny Large Language Models
        Zachary Horvitz, Jingru Chen, Rahul Aditya, Harshvardhan Srivastava,
        Robert West, Zhou Yu, Kathleen McKeown
        ACL 2024
        https://arxiv.org/abs/2403.00794

Code: https://github.com/zacharyhorvitz/Getting-Serious-With-LLMs

Dataset: Paired satirical and "unfunned" headlines from The Onion
  - Test: 375 examples
  - Validation: 186 examples
  - Total: 561 evaluation instances

Task: "Unfunning" - Edit satirical headlines to make them realistic/serious.
      This benchmark evaluates humor understanding and manipulation (removing humor)
      rather than humor generation.

Prompt format (from data_generation/prompts/unfun_dataset/few-shot/ and hit_llm_generation_v2.py):
  Chat-style (primary):
    System: "You are a helpful assistant that edits humorous headlines to make them realistic."
    User: {satirical_headline}
    (No explicit "Humorous headline:" or "Realistic headline:" labels)

  Completion-style (alternative):
    "The following humorous headlines can be edited to be realistic:
    {satirical_headline} ->"
    (Uses " ->" separator, not "Realistic version:")

  Note: Paper uses 8-shot prompts with randomly sampled examples from
        `train_for_prompting.tsv`. This scenario follows that few-shot setup by default.

Evaluation:
  - Automatic metrics in the upstream repo include:
    - Edit distance from the humorous source
    - Lexical diversity over generated headlines
    - Humor-classifier accuracy views
  - The paper also reports human ratings (realness, funniness, grammaticality,
    coherence)

Fields used: funny_headline (input), unfunned_headline (reference), url (metadata)

Note: This benchmark tests the ability to understand and remove humor from text,
      which is an asymmetrical task to humor generation. The paper notes that LLMs
      excel at "unfunning" but underperform at generating novel jokes.

Data source: Original Unfun game data from https://github.com/epfl-dlab/unfun
            Processed version in this repo includes no-leakage splits.
"""

import csv
import os
import random
from typing import List, Optional, Tuple
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class UnfunCorpusScenario(Scenario):
    """
    Unfun Corpus for evaluating humor understanding and manipulation.

    Models are tasked with editing satirical headlines from The Onion
    to create realistic, serious versions ("unfunning").
    """

    name = "unfun_corpus"
    description = "zacharyhorvitz/Getting-Serious-With-LLMs"  # GitHub repo
    tags = ["creativity", "humor", "text-editing", "language-understanding"]

    def __init__(self, prompt_style: str = "chat", context_size: int = 8, seed: int = 1234, include_validation: bool = False):
        """
        Args:
            prompt_style: Style of prompt to use. Options: ["chat", "completion"]
            context_size: Number of few-shot pairs to include.
            seed: Base seed for deterministic context sampling.
            include_validation: Whether to append validation examples.
        """
        super().__init__()
        if prompt_style not in ["chat", "completion"]:
            raise ValueError(f"Invalid prompt_style: {prompt_style}. Must be 'chat' or 'completion'")
        self.prompt_style = prompt_style
        self.context_size = context_size
        self.seed = seed
        self.include_validation = include_validation

    def download_dataset(self, output_path: str) -> tuple[str, str, str]:
        """Download the prompt, test, and validation datasets."""
        base_url = "https://raw.githubusercontent.com/zacharyhorvitz/Getting-Serious-With-LLMs/main/datasets/unfun/unfun_processed/paired"

        prompt_url = f"{base_url}/train_for_prompting.tsv"
        test_url = f"{base_url}/test_unique_pairs_no_leakage.tsv"
        val_url = f"{base_url}/val_unique_pairs_no_leakage.tsv"

        prompt_path = os.path.join(output_path, "train_for_prompting.tsv")
        test_path = os.path.join(output_path, "test_unique_pairs_no_leakage.tsv")
        val_path = os.path.join(output_path, "val_unique_pairs_no_leakage.tsv")

        ensure_file_downloaded(source_url=prompt_url, target_path=prompt_path)
        ensure_file_downloaded(source_url=test_url, target_path=test_path)
        ensure_file_downloaded(source_url=val_url, target_path=val_path)

        return prompt_path, test_path, val_path

    def load_dataset(self, file_path: str) -> List[dict]:
        """Load and parse the TSV dataset."""
        examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:
                    examples.append({
                        'unfun_id': row[0],
                        'unfunned': row[1],
                        'funny_id': row[2],
                        'funny': row[3],
                        'url': row[4] if len(row) > 4 else None
                    })

        return examples

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split()).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("…", "...")

    def _sample_context(self, satirical_headline: str, prompt_examples: List[dict]) -> List[Tuple[str, str]]:
        samples = [
            (example["funny"], example["unfunned"])
            for example in prompt_examples
            if self._normalize(example["funny"]) != self._normalize(satirical_headline)
            and self._normalize(example["unfunned"]) != self._normalize(satirical_headline)
        ]
        if len(samples) < self.context_size:
            raise ValueError(f"Not enough prompting samples to draw {self.context_size} examples")
        rng = random.Random(f"{self.seed}:{self._normalize(satirical_headline)}:{self.prompt_style}")
        return rng.sample(samples, self.context_size)

    def create_prompt(self, satirical_headline: str, prompt_examples: List[dict]) -> str:
        """
        Create the prompt based on the selected style.

        This follows the upstream few-shot prompting regime using
        `train_for_prompting.tsv`.
        """
        sample_context = self._sample_context(satirical_headline, prompt_examples)

        if self.prompt_style == "chat":
            lines = ["System: You are a helpful assistant that edits humorous headlines to make them realistic.", ""]
            for funny, unfunned in sample_context:
                lines.append(f"User: {funny}")
                lines.append(f"Assistant: {unfunned}")
                lines.append("")
            lines.append(f"User: {satirical_headline}")
            lines.append("Assistant:")
            return "\n".join(lines)

        lines = ["The following humorous headlines can be edited to be realistic:"]
        for funny, unfunned in sample_context:
            lines.append(f"{funny} -> {unfunned}")
        lines.append(f"{satirical_headline} ->")
        return "\n".join(lines)

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for the Unfun Corpus.

        Creates instances from the held-out test split by default.
        """
        prompt_path, test_path, val_path = self.download_dataset(output_path)

        prompt_examples = self.load_dataset(prompt_path)
        test_examples = self.load_dataset(test_path)
        val_examples = self.load_dataset(val_path) if self.include_validation else []

        instances = []

        # Process test split
        for example in test_examples:
            prompt = self.create_prompt(example['funny'], prompt_examples)

            # Reference is the human-created unfunned headline
            references = [
                Reference(Output(text=example['unfunned']), tags=[CORRECT_TAG])
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"unfun_test_{example['funny_id']}",
                    extra_data={"satirical_headline": example["funny"], "reference_unfunned": example["unfunned"]},
                )
            )

        # Process validation split
        for example in val_examples:
            prompt = self.create_prompt(example['funny'], prompt_examples)

            references = [
                Reference(Output(text=example['unfunned']), tags=[CORRECT_TAG])
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=VALID_SPLIT,
                    id=f"unfun_val_{example['funny_id']}",
                    extra_data={"satirical_headline": example["funny"], "reference_unfunned": example["unfunned"]},
                )
            )

        return instances
