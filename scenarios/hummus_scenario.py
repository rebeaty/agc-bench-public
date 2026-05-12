"""
HELM Scenario: HUMMUS (Humorous Multimodal Metaphor Use)

Paper: https://arxiv.org/abs/2504.02983
Code: https://github.com/xiaoyuisrain/humorous-multimodal-metaphor-use

This benchmark has 4 text-compatible tasks (Tasks 3-4 require visual grounding):

  Subset "classification" (Task 1 - CLAS):
    - Binary classification: does humor involve metaphor use?
    - 940 instances, eval: exact_match
    - Prompt: "Does the humor...involve metaphor use? Answer with Yes or No."

  Subset "naming" (Task 2 - NAME):
    - Identify the conceptual metaphor in "TARGET IS SOURCE" format
    - 589 instances (metaphorical only), eval: open_ended (sentence similarity)
    - Prompt: "Which conceptual metaphor is used? Answer in TARGET DOMAIN IS SOURCE DOMAIN format."

  Subset "caption_highlight" (Task 5 - CAPT):
    - Highlight metaphor-related caption text with <i></i> tags
    - 568 instances (metaphorical only), eval: exact_match (Jaccard index in paper)
    - Prompt: "Which part of the caption is related? Surround it with <i></i> tag."

  Subset "explanation" (Task 6 - EXPL):
    - Explain how metaphor contributes to humor (<=30 words)
    - 628 instances (metaphorical only), eval: open_ended (ROUGE)
    - Prompt: "How does metaphor use contribute to the humor? Explain in no more than 30 words."

Prompt source: model_evaluation/prompts.py (CLAS02, NAME04, CAPT02, EXPL06)
Data source: GitHub repo (test_set.json for classification, hummus_dataset.json for others)

Note: Original benchmark is multimodal. This uses text-only ablation with image
      descriptions from the CapCon corpus.
"""

import json
import os
import pandas as pd
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class HummusScenario(Scenario):
    name = "hummus"
    description = "xiaoyuisrain/humorous-multimodal-metaphor-use"
    tags = ["creativity", "metaphor", "humor"]

    SUBSETS = ["classification", "naming", "caption_highlight", "explanation"]
    BASE_URL = "https://raw.githubusercontent.com/xiaoyuisrain/humorous-multimodal-metaphor-use/main"

    def __init__(self, subset: str = "classification"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"subset must be one of {self.SUBSETS}, got '{subset}'")
        self.subset = subset

    def _download_file(self, url: str, output_path: str, filename: str) -> str:
        """Download a file if not already cached."""
        filepath = os.path.join(output_path, filename)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)
        return filepath

    def _load_image_descriptions(self, output_path: str) -> pd.DataFrame:
        """Load image descriptions indexed by contest_number."""
        desc_path = self._download_file(
            f"{self.BASE_URL}/images/image_descriptions.csv",
            output_path,
            "hummus_image_descriptions.csv"
        )
        df = pd.read_csv(desc_path)
        return df.set_index("contest_number")

    def get_instances(self, output_path: str):
        desc_df = self._load_image_descriptions(output_path)

        if self.subset == "classification":
            return self._get_classification_instances(output_path, desc_df)
        elif self.subset == "naming":
            return self._get_naming_instances(output_path, desc_df)
        elif self.subset == "caption_highlight":
            return self._get_caption_highlight_instances(output_path, desc_df)
        elif self.subset == "explanation":
            return self._get_explanation_instances(output_path, desc_df)

    def _get_classification_instances(self, output_path: str, desc_df: pd.DataFrame):
        """Task 1: Binary classification - metaphor use Yes/No."""
        test_path = self._download_file(
            f"{self.BASE_URL}/model_evaluation/test_set.json",
            output_path,
            "hummus_test_set.json"
        )
        with open(test_path) as f:
            test_data = json.load(f)

        instances = []
        for item in test_data:
            image_desc = desc_df.at[item["contest_number"], "image_description"]

            prompt = (
                f"Image: {image_desc}\n"
                f"Caption: {item['caption']}\n\n"
                f"Does the humor of the given image-and-caption combination "
                f"involve metaphor use? Answer the question with Yes or No."
            )

            # Yes/WIDLII -> Yes, No -> No
            correct = "No" if item["is_met"] == "No" else "Yes"

            references = [
                Reference(Output(text="Yes"), tags=[CORRECT_TAG] if correct == "Yes" else []),
                Reference(Output(text="No"), tags=[CORRECT_TAG] if correct == "No" else []),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances

    def _get_naming_instances(self, output_path: str, desc_df: pd.DataFrame):
        """Task 2: Identify conceptual metaphor in 'TARGET IS SOURCE' format."""
        full_path = self._download_file(
            f"{self.BASE_URL}/hummus_dataset.json",
            output_path,
            "hummus_dataset.json"
        )
        with open(full_path) as f:
            full_data = json.load(f)

        instances = []
        for item_id, item in full_data.items():
            # Only metaphorical items with conceptual_metaphor annotation
            if item.get("met_class") not in ["Yes", "WIDLII"]:
                continue
            if "conceptual_metaphor" not in item:
                continue

            image_desc = desc_df.at[item["contest_number"], "image_description"]

            prompt = (
                f"Image: {image_desc}\n"
                f"Caption: {item['caption']}\n\n"
                f"The humor of the given image-and-caption combination involves metaphor use. "
                f"Which conceptual metaphor is used? "
                f"Answer the question in \"TARGET DOMAIN IS SOURCE DOMAIN\" format "
                f"(e.g., \"LOVE IS A JOURNEY\")."
            )

            # Reference is the gold conceptual metaphor
            references = [
                Reference(Output(text=item["conceptual_metaphor"]), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances

    def _get_caption_highlight_instances(self, output_path: str, desc_df: pd.DataFrame):
        """Task 5: Highlight metaphor-related caption text with <i></i> tags."""
        full_path = self._download_file(
            f"{self.BASE_URL}/hummus_dataset.json",
            output_path,
            "hummus_dataset.json"
        )
        with open(full_path) as f:
            full_data = json.load(f)

        instances = []
        for item_id, item in full_data.items():
            if item.get("met_class") not in ["Yes", "WIDLII"]:
                continue
            if "caption_hl" not in item:
                continue

            image_desc = desc_df.at[item["contest_number"], "image_description"]

            prompt = (
                f"Image: {image_desc}\n"
                f"Caption: {item['caption']}\n\n"
                f"The humor of the given image-and-caption combination involves metaphor use. "
                f"Which part of the caption is related to the metaphor? "
                f"Surround it with a pair of <i></i> tag."
            )

            # Reference is the caption with <i></i> tags
            references = [
                Reference(Output(text=item["caption_hl"]), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances

    def _get_explanation_instances(self, output_path: str, desc_df: pd.DataFrame):
        """Task 6: Explain how metaphor contributes to humor."""
        full_path = self._download_file(
            f"{self.BASE_URL}/hummus_dataset.json",
            output_path,
            "hummus_dataset.json"
        )
        with open(full_path) as f:
            full_data = json.load(f)

        instances = []
        for item_id, item in full_data.items():
            if item.get("met_class") not in ["Yes", "WIDLII"]:
                continue
            if "explanation" not in item:
                continue

            image_desc = desc_df.at[item["contest_number"], "image_description"]

            prompt = (
                f"Image: {image_desc}\n"
                f"Caption: {item['caption']}\n\n"
                f"How does metaphor use contribute to the humor of the given "
                f"image-and-caption combination? Explain in no more than 30 words."
            )

            # Reference is the gold explanation
            references = [
                Reference(Output(text=item["explanation"]), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
