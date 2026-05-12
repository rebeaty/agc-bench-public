"""
HELM Scenario: ARN (Analogical Reasoning on Narratives)

Paper: https://arxiv.org/abs/2310.00996 (TACL 2024)
Data: https://bit.ly/3t7qZ3S (Google Drive)

Note: No public code repository exists for this benchmark. The paper does not
      provide a GitHub link; only the dataset is available via Google Drive.

Task: Given a query narrative and two candidate narratives, select the one that
forms a better analogical mapping with the query based on proverbial message
(system similarity) rather than surface features.

Dataset:
  - 1,095 triples (query, analog, distractor)
  - Four partitions based on analogy type (near/far) and distractor similarity (high/low)
  - Human accuracy: 96%

Prompt format (from Paper Appendix E.2 - GPT/Llama format):
  narratives can be mapped to each other in terms of the proverbial message they
  strive to convey. These proverbial messages can be related to traditions, common
  knowledge, or moral principles. We call this mapping analogical mapping. Which
  one of the two narratives (1, 2) can create a better analogical mapping with the
  query narrative? Answer in the template: {{narrative_x, because narrative_x and
  query_narrative are ...}}
  ———
  query_narrative: {query}
  ———
  narrative_1: {choice1}
  ———
  narrative_2: {choice2}
  ##########
  narrative

Subsets:
  "all" - Full benchmark (1,095 examples)
  "near_high" - Near analogies with high-similarity distractors (253 examples)
  "near_low" - Near analogies with low-similarity distractors (294 examples)
  "far_high" - Far analogies with high-similarity distractors (254 examples)
  "far_low" - Far analogies with low-similarity distractors (294 examples)

Fields used: query_narrative, first_choice, second_choice, correct_answer,
             analogy_level, distractor_similarity
Evaluation: exact_match (binary accuracy)
"""

import os

import gdown
import pandas as pd
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class ARNScenario(Scenario):
    name = "arn"
    description = "Sourati et al. ARN (Analogical Reasoning on Narratives)"
    tags = ["creativity", "analogical_reasoning", "narrative_understanding"]

    DRIVE_FOLDER_ID = "1itOPXtorFEgweQCd71m2bIRwWAUHcXuf"
    DATA_FILENAME = "Analogical Reasoning on Narratives (ARN) dataset.xlsx"

    SUBSETS = ["all", "near_high", "near_low", "far_high", "far_low"]

    # Exact prompt from Paper Appendix E.2 (GPT/Llama evaluation format)
    PROMPT_TEMPLATE = """narratives can be mapped to each other in terms of the proverbial message they strive to convey. These proverbial messages can be related to traditions, common knowledge, or moral principles. We call this mapping analogical mapping. Which one of the two narratives (1, 2) can create a better analogical mapping with the query narrative? Answer in the template: {{{{narrative_x, because narrative_x and query_narrative are ...}}}}
———
query_narrative: {query}
———
narrative_1: {choice1}
———
narrative_2: {choice2}
##########
narrative"""

    def __init__(self, subset: str = "all"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown subset: {subset}. Must be one of {self.SUBSETS}")
        self.subset = subset

    def _download_data(self, output_path: str) -> str:
        """Download dataset from Google Drive using gdown."""
        data_dir = os.path.join(output_path, "arn_data")
        xlsx_path = os.path.join(data_dir, self.DATA_FILENAME)

        if not os.path.exists(xlsx_path):
            os.makedirs(data_dir, exist_ok=True)
            gdown.download_folder(
                url=f"https://drive.google.com/drive/folders/{self.DRIVE_FOLDER_ID}",
                output=data_dir,
                quiet=True,
            )
            # File is nested in a subdirectory
            nested_path = os.path.join(data_dir, "Analogical Reasoning over Narratives", self.DATA_FILENAME)
            if os.path.exists(nested_path):
                os.rename(nested_path, xlsx_path)

        return xlsx_path

    def _filter_by_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe based on subset selection."""
        if self.subset == "all":
            return df

        # Parse subset name: {analogy_level}_{distractor_similarity}
        parts = self.subset.split("_")
        analogy_level = parts[0]  # "near" or "far"
        distractor_sim = parts[1]  # "high" or "low"

        return df[
            (df['analogy_level'] == analogy_level) &
            (df['distractor_similarity'] == distractor_sim)
        ]

    def get_instances(self, output_path: str):
        xlsx_path = self._download_data(output_path)
        df = pd.read_excel(xlsx_path)
        df = self._filter_by_subset(df)

        instances = []
        for _, row in df.iterrows():
            query = row['query_narrative']
            choice1 = row['first_choice']
            choice2 = row['second_choice']
            correct = int(row['correct_answer'])  # 1 or 2

            # Format prompt
            prompt = self.PROMPT_TEMPLATE.format(
                query=query,
                choice1=choice1,
                choice2=choice2
            )

            # References: both choices, correct one tagged
            references = [
                Reference(
                    Output(text="1"),
                    tags=[CORRECT_TAG] if correct == 1 else []
                ),
                Reference(
                    Output(text="2"),
                    tags=[CORRECT_TAG] if correct == 2 else []
                ),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"arn_{self.subset}_{len(instances)}",
                extra_data={
                    "subset": self.subset,
                    "correct_answer": str(correct),
                },
            ))

        return instances
