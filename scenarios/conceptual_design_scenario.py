"""
HELM Scenario: Conceptual Design Generation Problems

Paper: "Conceptual Design Generation Using Large Language Models"
       https://arxiv.org/abs/2306.01779
Code: https://github.com/kevinma1515/gpt_idetc

Open-ended creative design ideation: given an engineering design problem,
generate 100 conceptual solutions. 13 design problems from the IDETC 2023
paper, 12 with 100 crowdsourced human reference solutions each (Amazon
Mechanical Turk). Problem 11 (peanut shell removal) excluded — no human refs.

Prompt source: prompt_engineering/zero_shot.py in the GitHub repo. Four
  prompt variants: base, novel, diverse, unique (paper's zero-shot conditions).
  The local runnable path keeps the original "Generate 100 design solutions"
  wording so set-level evaluation can be computed from a single completion.
Fields used: design problem text, human reference solutions (CSV files)
Fields preserved in extra_data: prompt variant, problem index, and reference file
metadata for auditability

Data: Cloned from GitHub repo at runtime (no HuggingFace dataset).
"""

import csv
import os
import subprocess
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class ConceptualDesignScenario(Scenario):
    name = "conceptual_design"
    description = "kevinma1515/gpt_idetc"
    tags = ["creativity", "design", "open_ended", "divergent_thinking"]

    # 13 design problems from zero_shot.py, indexed 0-12
    PROBLEMS = [
        "a lightweight exercise device that can be used while traveling",
        "a lightweight exercise device that can collect energy from human motion",
        "a new way to measure the passage of time",
        "a device that disperses a light coating of a powdered substance over a surface",
        "a device that allows people to get a book that is out of reach",
        "an innovative product to froth milk",
        "a way to minimize accidents from people walking and texting on a cell phone",
        "a device to fold washcloths, hand towels, and small bath towels",
        "a way to make drinking fountains accessible for all people",
        "a measuring cup for the blind",
        "a device to immobilize a human joint",
        "a device to remove the shell from a peanut in areas with no electricity",
        "a device that can help a home conserve energy",
    ]

    # Map problem index -> human reference filename
    # Problem 11 (peanut) has no human references and is excluded
    REFERENCE_FILES = {
        0: "amazonTurkDesPrompt1.csv",
        1: "amazonTurkDesPrompt2.csv",
        2: "amazonTurkDesPrompt3.csv",
        3: "amazonTurkDesPrompt4.csv",
        4: "amazonTurkDesPrompt5.csv",
        5: "amazonTurkDesPrompt6.csv",
        6: "amazonTurkDesPrompt7.csv",
        7: "amazonTurkDesPrompt8.csv",
        8: "amazonTurkDesPrompt9.csv",
        9: "amazonTurkDesPrompt10.csv",
        10: "amazonTurkDesPrompt11.csv",
        12: "amazonTurkDesPromptNA.csv",
    }

    PROMPT_VARIANTS = {
        "base": "Generate 100 design solutions for {problem}",
        "novel": "Generate 100 novel design solutions for {problem}",
        "diverse": "Generate 100 diverse design solutions for {problem}",
        "unique": "Generate 100 unique design solutions for {problem}",
    }

    def __init__(self, prompt_variant: str = "base"):
        """
        Args:
            prompt_variant: One of "base", "novel", "diverse", "unique"
        """
        super().__init__()
        if prompt_variant not in self.PROMPT_VARIANTS:
            raise ValueError(
                f"prompt_variant must be one of "
                f"{list(self.PROMPT_VARIANTS.keys())}, got '{prompt_variant}'"
            )
        self.prompt_variant = prompt_variant

    @staticmethod
    def _clone_repo(output_path):
        """Clone the data repo if not already present."""
        repo_dir = os.path.join(output_path, "gpt_idetc")
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/kevinma1515/gpt_idetc.git", repo_dir],
                check=True, capture_output=True,
            )
        return repo_dir

    @staticmethod
    def _load_references(filepath):
        """Load human reference solutions from a single-column CSV."""
        refs = []
        with open(filepath, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    text = row[0].strip().strip('"')
                    if text:
                        refs.append(text)
        return refs

    def get_instances(self, output_path):
        repo_dir = self._clone_repo(output_path)
        refs_dir = os.path.join(repo_dir, "sentence_embeddings", "data")

        template = self.PROMPT_VARIANTS[self.prompt_variant]
        instances = []

        for idx, problem in enumerate(self.PROBLEMS):
            if idx not in self.REFERENCE_FILES:
                continue

            prompt = template.format(problem=problem)

            ref_file = os.path.join(refs_dir, self.REFERENCE_FILES[idx])
            human_solutions = self._load_references(ref_file)

            references = [
                Reference(Output(text=sol), tags=[CORRECT_TAG])
                for sol in human_solutions
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"conceptual_design_{idx}",
                extra_data={
                    "problem_index": idx,
                    "prompt_variant": self.prompt_variant,
                    "reference_file": self.REFERENCE_FILES[idx],
                },
            ))

        return instances
