"""
HELM Scenario: SCAR (Scientific Analogical Reasoning with Structure Abduction)

Paper: Beneath Surface Similarity: Large Language Models Make Reasonable
       Scientific Analogies after Structure Abduction
       (EMNLP 2023 Findings)
       https://aclanthology.org/2023.findings-emnlp.160

Code & Data: https://github.com/siyuyuan/scar

Dataset: SCAR - 400 scientific analogies across 13 domains
- Biology, Physics, Chemistry, Computer Science, Mathematics, Engineering,
- Geography, History, Literature, Philosophy, Economics, Art, Sports

Task: Analogical structure abduction
Given two systems with background information, identify the term mappings
that form analogies between elements in the two systems.

Example:
  System A: Solar System (Newton, Sun, Earth)
  System B: Atom Structure (Nucleus, Faraday, Electron)
  Answer: [['Newton','Faraday'], ['Sun','Nucleus'], ['Earth','Electron']]

Evaluation: Open-ended generation with structured output matching

Prompt format:
  From the paper's template.txt file, using Instruction 1 format.
  Model must generate mappings in the format: [['item_a1', 'item_b1'], ['item_a2', 'item_b2'], ...]

Fields used: system_a, system_b, mappings, system_a_domain, system_b_domain,
            system_a_background, system_b_background, Explanation (optional)
Fields skipped: lang (all English), id (used for tracking only)

Note: Dataset available in the GitHub repository under release/system_analogy_en.json
      The data is in JSONL format (one JSON object per line).

Alternative name: "Relational Structure Identification (RSI) Test" in some references.
"""

from typing import List, Optional
import json
import subprocess
import os

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)


class SCARScenario(Scenario):
    """
    SCAR: Scientific Analogical Reasoning with Structure Abduction

    Evaluates analogical reasoning through identifying structural mappings
    between scientific concepts from different domains.
    """

    name = "scar"
    description = "siyuyuan/scar"
    tags = ["creativity", "reasoning", "analogy", "scientific"]

    def __init__(
        self,
        dataset_path: Optional[str] = None,
    ):
        """
        Args:
            dataset_path: Path to cloned SCAR repository. If None, will clone to temp directory.
        """
        super().__init__()
        self.dataset_path = dataset_path

    def _ensure_dataset(self, output_path: str) -> str:
        """Clone or verify SCAR repository."""
        if self.dataset_path and os.path.exists(self.dataset_path):
            return self.dataset_path

        # Clone to output directory
        repo_path = os.path.join(output_path, "scar")
        if not os.path.exists(repo_path):
            print(f"Cloning SCAR repository to {repo_path}...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/siyuyuan/scar.git", repo_path],
                check=True
            )
        return repo_path

    def _format_mappings_output(self, mappings: List[List[str]]) -> str:
        """Format mappings list as expected output string."""
        # Convert to string representation matching the expected format
        return str(mappings)

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load SCAR dataset and create instances for analogical structure abduction."""

        # Ensure dataset is available
        repo_path = self._ensure_dataset(output_path)
        data_file = os.path.join(repo_path, "release", "system_analogy_en.json")

        # Load JSONL data
        print(f"Loading SCAR dataset from {data_file}...")
        instances = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Extract systems and their information
                system_a = item['system_a']
                system_b = item['system_b']
                system_a_domain = item['system_a_domain']
                system_b_domain = item['system_b_domain']
                system_a_background = item['system_a_background']
                system_b_background = item['system_b_background']
                mappings = item['mappings']

                # Extract items from each system (all unique terms mentioned in mappings)
                items_a = sorted(set(m[0] for m in mappings))
                items_b = sorted(set(m[1] for m in mappings))

                # Build prompt using Instruction 1 format from template.txt
                prompt_text = (
                    "When provided with two distinct scenarios, identify and list the corresponding "
                    "elements within each scenario to create a clear analogy between them. You must "
                    "establish a one-to-one connection between the items in both scenarios. Present your "
                    "findings in the format: [['Scenario 1 Item 1', 'Scenario 2 Item 1'], "
                    "['Scenario 1 Item 2', 'Scenario 2 Item 2'], ...]\n\n"
                    f"Scenario 1: {system_a}\n"
                    f"Domain: {system_a_domain}\n"
                    f"Background: {system_a_background}\n"
                    f"Items in Scenario 1: {', '.join(items_a)}\n\n"
                    f"Scenario 2: {system_b}\n"
                    f"Domain: {system_b_domain}\n"
                    f"Background: {system_b_background}\n"
                    f"Items in Scenario 2: {', '.join(items_b)}\n\n"
                    "Establish a one-to-one mapping between the items in the two scenarios "
                    "and present your findings in the format: [['Scenario 1 Item 1', 'Scenario 2 Item 1'], "
                    "['Scenario 1 Item 2', 'Scenario 2 Item 2'], ...]\n\n"
                    "Answer:"
                )

                # Create reference with correct mapping format
                correct_output = self._format_mappings_output(mappings)

                # Create instance
                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=[
                            Reference(Output(text=correct_output), tags=[CORRECT_TAG])
                        ],
                        split=TEST_SPLIT,  # All 400 instances used as test set
                        id=f"scar_{item['id']}",
                        extra_data={
                            "system_a": system_a,
                            "system_b": system_b,
                            "system_a_domain": system_a_domain,
                            "system_b_domain": system_b_domain,
                            "items_a": items_a,
                            "items_b": items_b,
                            "gold_mappings": mappings,
                            "num_mappings": len(mappings),
                        }
                    )
                )

        print(f"Loaded {len(instances)} SCAR instances")
        print(f"  All instances in TEST split (no train/val splits provided)")

        # Count domain pairs
        domain_pairs = {}
        for inst in instances:
            pair = (inst.extra_data["system_a_domain"], inst.extra_data["system_b_domain"])
            domain_pairs[pair] = domain_pairs.get(pair, 0) + 1

        print(f"  Unique domain pairs: {len(domain_pairs)}")
        print(f"  Mappings per instance: {min(i.extra_data['num_mappings'] for i in instances)}-"
              f"{max(i.extra_data['num_mappings'] for i in instances)} "
              f"(avg: {sum(i.extra_data['num_mappings'] for i in instances)/len(instances):.1f})")

        return instances
