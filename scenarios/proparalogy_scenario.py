"""ProPara-Logy: Natural Language Analogy Generation Benchmark

Citation:
    Sultan, O., Bitton, Y., Yosef, R., & Shahaf, D. (2024).
    ParallelPARC: A Scalable Pipeline for Generating Natural-Language Analogies.
    In Proceedings of NAACL 2024.

Paper: https://arxiv.org/pdf/2403.01139.pdf
Repository: https://github.com/orensul/ParallelPARC
"""

import hashlib
import json
import os
import random
import pandas as pd
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


class ProparaLogyScenario(Scenario):
    """ProPara-Logy: Cross-domain analogical reasoning benchmark.

    The benchmark evaluates models' ability to identify analogous processes across
    different domains. Given a source paragraph describing a process in one domain,
    models must select which of three target paragraphs represents an analogous
    process in another domain.

    Task format: 3-choice multiple choice
    - 1 correct analogous paragraph
    - 1 challenging distractor (similar but not analogous)
    - 1 random distractor (unrelated process)

    Dataset: 310 test examples
    - Source domains: Natural Sciences, Social Sciences, Engineering, Biomedical
    - Target domains: Natural Sciences, Social Sciences, Engineering, Biomedical
    - Analogy types: close analogy (60%), far analogy (40%)
    """

    name = "proparalogy"
    description = "orensul/ParallelPARC"
    tags = ["creativity", "analogical_reasoning", "cross_domain", "process_understanding"]

    DATASET_DOWNLOAD_URL = "https://github.com/orensul/ParallelPARC.git"

    def __init__(self, analogy_type: str = "all"):
        """Initialize ProPara-Logy scenario.

        Args:
            analogy_type: Which analogy type to include:
                - "all": All analogies (default)
                - "close": Close analogies only (same/similar domains)
                - "far": Far analogies only (distant domains)
        """
        super().__init__()
        valid_types = ["all", "close", "far"]
        if analogy_type not in valid_types:
            raise ValueError(f"analogy_type must be one of {valid_types}, got: {analogy_type}")
        self.analogy_type = analogy_type

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load ProPara-Logy dataset and create HELM instances."""
        # Set up data directory
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)

        # Download dataset via git clone if not already present
        repo_dir = os.path.join(data_dir, "ParallelPARC")
        if not os.path.exists(repo_dir):
            os.system(f"git clone {self.DATASET_DOWNLOAD_URL} {repo_dir}")

        # Load test data with all distractors
        dataset_path = os.path.join(
            repo_dir,
            "datasets",
            "gold_test_set",
            "gold_set_analogies_w_challenging_distractors_w_randoms.csv",
        )
        df = pd.read_csv(dataset_path)

        # Filter by analogy type if specified
        if self.analogy_type != "all":
            df = df[df["analogy_type"] == f"{self.analogy_type} analogy"]

        instances = []
        for _, row in df.iterrows():
            source_paragraph = row["source_paragraph"].strip()
            source_domain = row["source_domain"]
            target_domain = row["target_domain"]
            analogy_type = row["analogy_type"]

            # Get the three choices
            correct_answer = row["target_paragraph"].strip()
            challenging_distractor = row["distractor_target_paragraph"].strip()
            random_distractor = row["random_target_paragraph"].strip()

            # Per-instance positional shuffle (paper-fidelity fix, 2026-04-25).
            # The original implementation hardcoded A=correct for every instance,
            # inflating accuracy for any model with a positional prior. The paper
            # randomizes option order. We do the same with a deterministic
            # per-instance seed derived from sample_id so the run is reproducible.
            instance_id = f"{row['sample_id']}_{analogy_type.replace(' ', '_')}"
            seed = int.from_bytes(
                hashlib.sha256(f"proparalogy::{instance_id}".encode("utf-8")).digest()[:8],
                "big",
            )
            options = [
                ("correct", correct_answer),
                ("challenging", challenging_distractor),
                ("random", random_distractor),
            ]
            random.Random(seed).shuffle(options)

            labels = ["A", "B", "C"]
            prompt = (
                f'Given the following process description:\n\n'
                f'"{source_paragraph}"\n\n'
                f'Which of the following describes an analogous process?\n\n'
                + "\n\n".join(f"{label}) {text}" for label, (_kind, text) in zip(labels, options))
                + "\n\nAnswer with A, B, or C."
            )

            # Tag the label that corresponds to the correct answer post-shuffle.
            references = [
                Reference(
                    Output(text=label),
                    tags=[CORRECT_TAG] if kind == "correct" else [],
                )
                for label, (kind, _text) in zip(labels, options)
            ]

            instance = Instance(
                input=Input(text=prompt),
                references=references,
                id=instance_id,
                split="test",
            )
            instances.append(instance)

        return instances


# Example usage and test
if __name__ == "__main__":
    scenario = ProparaLogyScenario(analogy_type="all")
    instances = scenario.get_instances("/tmp/proparalogy_test")
    print(f"Loaded {len(instances)} instances")
    if instances:
        print(f"\nExample instance ID: {instances[0].id}")
        print(f"Prompt preview: {instances[0].input.text[:300]}...")
        print(f"Number of references: {len(instances[0].references)}")
        print(f"Correct answer: {instances[0].references[0].output.text}")
