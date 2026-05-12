"""
HELM Scenario: AraStories (Arabic Story Generation Evaluation Framework)

Paper: "Arabic Automatic Story Generation with Large Language Models"
       https://arxiv.org/abs/2407.07551 (ArabicNLP 2024 @ ACL)
Code: https://github.com/UBC-NLP/arastories

Prompt format:
  Each prompt is a structured Arabic instruction specifying story constraints
  (age group, setting, ending type, character count, country, etc.).
  Prompts are used directly from the dataset as-is.

Prompt source: GPT-4 prompt generator in notebooks/PromptGeneratorMSA.ipynb;
  prompts already materialized in CSV files.
Fields used: Prompt (story generation instruction)
Fields skipped: Story (GPT-4-generated reference; stored in extra_data for audit),
                unnamed index column

Three Arabic dialect subsets:
  - msa: Modern Standard Arabic (996 instances)
  - egyptian: Egyptian Arabic (1,000 instances)
  - moroccan: Moroccan Arabic (1,000 instances)

Evaluation: LLM-as-judge on 5 dimensions (Fluency, Coherence, Following
Instructions, Consistency, Variety) on a 1-5 scale. See annotator_notes.md.
"""

import csv
import os
import urllib.request
import zipfile

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    TEST_SPLIT,
)

DIALECT_FILES = {
    "msa": "Formatted-MSA-prompts-stories-for-fine-tuning.csv",
    "egyptian": "Formatted-Egyptian-prompts-stories-for-fine-tuning.csv",
    "moroccan": "Formatted-Moroccan-prompts-stories-for-fine-tuning.csv",
}

REPO_ZIP_URL = (
    "https://github.com/UBC-NLP/arastories/archive/refs/heads/main.zip"
)


class AraStoriesScenario(Scenario):
    name = "arastories"
    description = "UBC-NLP/arastories"
    tags = ["creativity", "story_generation", "arabic"]

    def __init__(self, dialect: str = "all"):
        """
        Args:
            dialect: Which Arabic variety to use. One of "msa", "egyptian",
                     "moroccan", or "all" (default: all three).
        """
        super().__init__()
        if dialect not in ("msa", "egyptian", "moroccan", "all"):
            raise ValueError(
                f"Invalid dialect '{dialect}'. "
                "Must be 'msa', 'egyptian', 'moroccan', or 'all'."
            )
        self.dialect = dialect

    def _download_data(self, output_path: str) -> str:
        """Download and extract the AraStories repository."""
        data_dir = os.path.join(output_path, "arastories-main", "data")
        if os.path.exists(data_dir):
            return data_dir

        zip_path = os.path.join(output_path, "arastories.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        return data_dir

    def _load_csv(self, filepath: str) -> list[dict]:
        """Load prompt-story pairs from a CSV file."""
        rows = []
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("Prompt", "").strip()
                story = row.get("Story", "").strip()
                if prompt:
                    rows.append({"prompt": prompt, "story": story})
        return rows

    def get_instances(self, output_path: str) -> list[Instance]:
        data_dir = self._download_data(output_path)

        if self.dialect == "all":
            dialects = ["msa", "egyptian", "moroccan"]
        else:
            dialects = [self.dialect]

        instances = []
        for dialect in dialects:
            filepath = os.path.join(data_dir, DIALECT_FILES[dialect])
            rows = self._load_csv(filepath)

            for idx, row in enumerate(rows):
                instances.append(
                    Instance(
                        input=Input(text=row["prompt"]),
                        # Paper/repo automatic evaluation is reference-free:
                        # judge sees only the prompt plus the generated story.
                        references=[],
                        split=TEST_SPLIT,
                        id=f"{dialect}_{idx}",
                        extra_data={
                            "dialect": dialect,
                            "released_reference_story": row["story"],
                            "source_row_index": idx,
                        },
                    )
                )

        return instances
