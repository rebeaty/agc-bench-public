"""
HELM Scenario: Humor Transfer Learning — Binary Humor Detection

Paper: "One Joke to Rule them All? On the (Im)possibility of Generalizing Humor"
       https://arxiv.org/abs/2508.19402
Code: https://github.com/morturr/HumorTransferLearning

Two subsets from the paper's humor transfer learning study, onboarded as
binary humor classification tasks:

  - sarcasm_headlines: Sarcastic (The Onion) vs real (HuffPost) news headlines.
    Source: Misra & Arora, 2023. HuggingFace: raquiba/Sarcasm_News_Headline
  - amazon_questions: Humorous vs non-humorous product questions from Amazon.
    Source: Ziser et al., 2020. AWS S3: humor-detection-pds

Prompt source: Paper Appendix C instruction fine-tuning prompt (full template).
  Uses the complete Alpaca-style format with preamble, ### Instruction,
  ### Input, and ### Response sections. Labels: Yes/No per the paper.
Fields used:
  sarcasm_headlines — headline, is_sarcastic
  amazon_questions — question, product_description, label
Fields skipped:
  sarcasm_headlines — article_link
  amazon_questions — image_url
"""

import csv
import io
import urllib.request
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class HumorTransferScenario(Scenario):
    name = "humor_transfer"
    description = "Binary humor detection from humor transfer learning study"
    tags = ["creativity", "humor", "classification"]

    SUBSETS = ["sarcasm_headlines", "amazon_questions"]

    # Paper Appendix C: full instruction fine-tuning prompt (Alpaca-style)
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a sentiment analysis task.\n\n"
        "### Instruction:\n"
        "Given the following text, please determine if it should be "
        "classified as funny or not funny. Base your classification on "
        "humor elements such as wit, irony, absurdity, or comedic timing.\n\n"
        "### Input:\n"
        "{text}\n\n"
        "### Response:"
    )

    def __init__(self, subset: str = "sarcasm_headlines"):
        """
        Args:
            subset: One of "sarcasm_headlines", "amazon_questions"
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"subset must be one of {self.SUBSETS}, got '{subset}'"
            )
        self.subset = subset

    def _load_sarcasm_headlines(self):
        """Load Sarcasm Headlines from HuggingFace (test split, 26,709 examples)."""
        dataset = load_dataset(
            "raquiba/Sarcasm_News_Headline", split="test"
        )
        items = []
        for row in dataset:
            items.append({
                "text": row["headline"],
                "label": row["is_sarcastic"],
            })
        return items

    def _load_amazon_questions(self):
        """Load Amazon Questions from AWS S3 (19,142 examples)."""
        items = []
        for url in [
            "https://humor-detection-pds.s3.amazonaws.com/Humorous.csv",
            "https://humor-detection-pds.s3.amazonaws.com/Non-humorous-unbiased.csv",
        ]:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp:
                data = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(data))
            for row in reader:
                text = (
                    f"Product: {row['product_description']}\n"
                    f"Question: {row['question']}"
                )
                items.append({
                    "text": text,
                    "label": int(row["label"]),
                })
        return items

    def get_instances(self, output_path):
        if self.subset == "sarcasm_headlines":
            items = self._load_sarcasm_headlines()
        else:
            items = self._load_amazon_questions()

        instances = []
        for item in items:
            prompt = self.PROMPT_TEMPLATE.format(text=item["text"])

            label = item["label"]
            references = [
                Reference(
                    Output(text="Yes"),
                    tags=[CORRECT_TAG] if label == 1 else [],
                ),
                Reference(
                    Output(text="No"),
                    tags=[CORRECT_TAG] if label == 0 else [],
                ),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"humor_transfer_{self.subset}_{len(instances):06d}",
                extra_data={"subset": self.subset, "gold_label": "Yes" if label == 1 else "No"},
            ))

        return instances
