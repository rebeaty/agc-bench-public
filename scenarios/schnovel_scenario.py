"""
HELM Scenario: SchNovel - Scholarly Novelty Assessment

Paper: https://arxiv.org/abs/2409.16605 (AISD@ACL 2025)
Dataset: https://huggingface.co/datasets/ethannlin/SchNovel

Task: Assess which of two scholarly papers is more novel.
Binary classification task where models choose between two papers.

Dataset: 15,000 paper pairs across 6 fields from arXiv
- Computer Science (cs): 2,500 pairs
- Mathematics (math): 2,500 pairs
- Physics (physics): 2,500 pairs
- Quantitative Biology (qbio): 2,500 pairs
- Quantitative Finance (qfin): 2,500 pairs
- Statistics (stat): 2,500 pairs

Publication date spans: 2-10 years apart
Ground truth: More recently published paper (paper1) is assumed to be more novel

Prompt format: Binary choice (1 or 2)
Evaluation: exact_match
"""

import json
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output, Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)


class SchNovelScenario(Scenario):
    """SchNovel: Scholarly Novelty Assessment Scenario

    Evaluates model's ability to assess novelty in scholarly publications
    by choosing which of two papers is more novel.
    """

    name = "schnovel"
    description = "ethannlin/SchNovel"
    tags = ["creativity", "novelty_assessment", "scholarly_evaluation"]

    VALID_FIELDS = ["cs", "math", "physics", "qbio", "qfin", "stat", "all"]

    def __init__(self, field: str = "all"):
        super().__init__()
        if field not in self.VALID_FIELDS:
            raise ValueError(f"Invalid field: {field}. Must be one of {self.VALID_FIELDS}")
        self.field = field

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []

        if self.field == "all":
            fields_to_load = [f for f in self.VALID_FIELDS if f != "all"]
        else:
            fields_to_load = [self.field]

        for field in fields_to_load:
            instances.extend(self._load_field_data(field))

        return instances

    def _load_field_data(self, field: str) -> List[Instance]:
        """Load paper pairs for a specific field"""
        dataset_url = f"https://huggingface.co/datasets/ethannlin/SchNovel/raw/main/{field}_dataset.json"

        with urllib.request.urlopen(dataset_url) as response:
            data = json.loads(response.read().decode())

        instances = []

        # Data is organized by year, then by year gap
        for year in data["data"]:
            year_data = data["data"][year]
            if isinstance(year_data, dict):
                # Data organized by year gap
                for gap in year_data:
                    for pair in year_data[gap]:
                        instances.append(self._create_instance(pair, field))
            else:
                # Data is a list
                for pair in year_data:
                    instances.append(self._create_instance(pair, field))

        return instances

    def _create_instance(self, pair: dict, field: str) -> Instance:
        """Create an instance from a paper pair"""
        paper1 = pair["paper1"]
        paper2 = pair["paper2"]

        # Create prompt using exact Zero-Shot prompt from paper (Appendix A.2)
        prompt = f"""You will be provided with the title and abstract of two research papers. Please determine which of the two articles is more novel. Follow these steps for evaluation.

Step 1: Identify the problem and solution that the research paper attempts to solve.

Step 2: Determine how unique the solution is given the current research landscape in 2024. Does the paper introduce a new idea, theory, or concept that has not been previously discussed in the literature?

Step 3: Determine how creative the solution is given the current research landscape in 2024. Does it apply a known idea in a completely new context or in a way that has not been done before?

Step 4: Using the findings from Steps 1-3, determine which paper is more novel.

In your response, please only state which paper is more novel (e.g., 1 if Paper 1 is more novel; 2 if Paper 2 is more novel).

User Prompt:
• Paper 1 Title: {paper1["title"]}
• Paper 1 Abstract: {paper1["abstract"]}
• Paper 2 Title: {paper2["title"]}
• Paper 2 Abstract: {paper2["abstract"]}"""

        # Paper 1 is always the more recent paper and assumed to be more novel
        references = [
            Reference(output=Output(text="1"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="2"), tags=[]),
        ]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
        )
