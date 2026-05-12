"""
HELM Scenario: PerMPST (Personalized Movie Plot Synopsis and Tags)

Paper: https://arxiv.org/abs/2310.03304
Code: https://github.com/facebookresearch/perse
Dataset: https://dl.fbaipublicfiles.com/perse/PerMPST.tar.gz
Published: October 2023

Task: Personalized story evaluation - predict how a specific reviewer would rate
a movie plot based on their review history. Tests ability to understand individual
preferences and evaluate story quality from personalized perspectives.

Prompt format (already formatted in dataset):
  [User Question] You will be presented with several plot summaries, each
  accompanied by a review from the same critic. Your task is to analyze both
  the plot summaries and the corresponding reviews to discern the reviewer's
  preferences. Afterward, consider a new plot and create a review that you
  believe this reviewer would write based on the established preferences.

  [The Start of Plot 0]
  {historical_plot_0}
  [The End of Plot 0]
  [Review]
  ```json
  {
    "Review": "{historical_review_0}",
    "Score": {historical_score_0}
  }
  ```

  ... [additional historical examples if k > 1]

  [The Start of Plot]
  {new_plot_to_evaluate}
  [The End of Plot]

Expected output:
  ```json
  {
    "Review": "<review text>",
    "Score": <1-10>
  }
  ```

Evaluation: Regression metrics (Pearson, Spearman, Kendall-Tau correlation)
comparing predicted scores to ground truth reviewer scores. See metric_notes.md
for detailed evaluation setup.

Dataset structure:
- examples: List of historical reviews from the same reviewer
  - summ_plot: Summarized movie plot
  - clean_review: Reviewer's review text
  - score: Reviewer's score (1-10)
- prompt: Formatted prompt with historical context + new plot
- completion: Ground truth (review + score in JSON format)

Dataset: 13,251 train, 915 validation
  - k=0: No historical context
  - k=1: 1 historical review (default, 915 val examples)
  - k=2-5: 2-5 historical reviews for richer context
  - 1,412 unique reviewers in training, 92 in validation
  - Scores range from 1-10 scale
"""

import json
import os
import re
import urllib.request
import tarfile
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)


class PerMPSTScenario(Scenario):
    """
    PerMPST: Personalized Movie Plot Synopsis and Tags Evaluation

    Personalized story evaluation task where models predict reviewer-specific
    scores and reviews for movie plots based on reviewer history.
    """

    name = "permpst"
    description = "facebookresearch/perse"
    tags = ["creativity", "personalization", "story-evaluation", "regression"]

    # Number of historical reviews (context)
    VALID_K_VALUES = [0, 1, 2, 3, 4, 5]

    def __init__(self, k: int = 1):
        """
        Args:
            k: Number of historical reviews to include as context.
               k=0: No history, k=1: 1 review (default), up to k=5
        """
        super().__init__()
        if k not in self.VALID_K_VALUES:
            raise ValueError(
                f"Invalid k value '{k}'. Must be one of: {self.VALID_K_VALUES}"
            )
        self.k = k

    def _download_data(self, output_path: str) -> str:
        """
        Download PerMPST dataset from Facebook Research if not already present.

        Returns:
            Path to the extracted data directory
        """
        data_dir = os.path.join(output_path, "permpst_data")
        expected_file = os.path.join(data_dir, f"review.valid.c{self.k}.jsonl")

        # Check if data already exists
        if os.path.exists(expected_file):
            print(f"Data already exists at {data_dir}")
            return data_dir

        print(f"Downloading PerMPST dataset...")
        os.makedirs(data_dir, exist_ok=True)

        # Download tar.gz file
        tar_path = os.path.join(data_dir, "PerMPST.tar.gz")
        dataset_url = "https://dl.fbaipublicfiles.com/perse/PerMPST.tar.gz"

        urllib.request.urlretrieve(dataset_url, tar_path)
        print(f"Downloaded to {tar_path}")

        # Extract
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

        print(f"Extraction complete")
        return data_dir

    def _extract_score_from_completion(self, completion: str) -> float:
        """
        Extract numeric score from completion JSON.

        Args:
            completion: String like '```json{"Review": "...", "Score": 8}```'

        Returns:
            Extracted score (1-10), or 5.0 as default if extraction fails
        """
        try:
            # Remove code block markers if present
            completion = completion.strip()
            if completion.startswith("```"):
                # Remove opening ```json or ```
                completion = re.sub(r"^```(?:json)?\s*", "", completion)
                # Remove closing ```
                completion = re.sub(r"```\s*$", "", completion)

            # Parse JSON
            data = json.loads(completion)
            score = float(data.get("Score", 5.0))

            # Clamp to valid range
            return max(1.0, min(10.0, score))

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Default to middle score if parsing fails
            print(f"Warning: Failed to extract score from completion: {e}")
            return 5.0

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate PerMPST instances from validation split.

        Each instance contains:
        - Input: Formatted prompt with reviewer history + new plot to evaluate
        - Reference: Ground truth score (1-10) extracted from completion
        """
        # Download data if needed
        data_dir = self._download_data(output_path)

        # Load validation data
        valid_file = os.path.join(data_dir, f"review.valid.c{self.k}.jsonl")
        print(f"Loading validation data from {valid_file}")

        instances = []
        with open(valid_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())

                    # Extract prompt (already formatted)
                    prompt_parts = item.get('prompt', [])
                    if not prompt_parts:
                        print(f"Warning: No prompt in line {line_num}, skipping")
                        continue

                    # Prompt is a list with one string element
                    prompt_text = prompt_parts[0] if isinstance(prompt_parts, list) else str(prompt_parts)

                    # Extract ground truth score from completion
                    completion = item.get('completion', '')
                    ground_truth_score = self._extract_score_from_completion(completion)

                    # Get reviewer info for instance ID
                    examples = item.get('examples', [])
                    reviewer_name = examples[0]['reviewer_name'] if examples else "unknown"
                    idx = item.get('idx', line_num)

                    # Create reference with score as text
                    # For regression tasks, reference contains the numeric value
                    references = [
                        Reference(
                            Output(text=f"{ground_truth_score:.1f}"),
                            tags=[CORRECT_TAG]
                        )
                    ]

                    instances.append(
                        Instance(
                            input=Input(text=prompt_text),
                            references=references,
                            split=TEST_SPLIT,
                            id=f"permpst_k{self.k}_{reviewer_name}_{idx}"
                        )
                    )

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

        print(f"Created {len(instances)} PerMPST instances (k={self.k})")
        return instances
