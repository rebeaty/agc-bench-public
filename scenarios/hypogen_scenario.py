"""
HELM Scenario: HypoGen bit-flip proxy

Paper: Can LLMs Generate Novel Research Ideas?
       https://arxiv.org/abs/2409.04109
Upstream repo: https://github.com/NoviScl/AI-Researcher
Dataset: UniverseTBD/hypogen-dr1 (test split)

This local benchmark is intentionally a dataset-derived proxy over the released
`hypogen-dr1` bit-flip artifacts. It is not the full AI-Researcher ideation
pipeline from the paper/repo, which operates over research topics, retrieval,
idea generation, proposal expansion, ranking, and filtering.

Proxy task:
  Given a paper abstract and a description of the conventional limitation
  (the "bit"), generate a novel hypothesis or approach that overcomes the
  limitation (the "flip").

Prompt source:
  No exact upstream prompt is published for this dataset derivative, so the
  local prompt remains a simple standardized instruction over the released
  `abstract`, `bit`, and `flip` fields.
"""

import os
import urllib.request
from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_TEST_PARQUET_URL = (
    "https://huggingface.co/datasets/UniverseTBD/hypogen-dr1/resolve/main/"
    "data/test-00000-of-00001.parquet"
)


class HypoGenScenario(Scenario):
    """
    Dataset-derived bit-flip hypothesis generation proxy.
    """

    name = "hypogen"
    description = "UniverseTBD/hypogen-dr1 bit-flip proxy"
    tags = ["creativity", "scientific_creativity", "hypothesis_generation", "open_ended", "proxy"]

    def _load_test_split(self, output_path: str):
        scenario_dir = os.path.join(output_path, self.name)
        os.makedirs(scenario_dir, exist_ok=True)
        parquet_path = os.path.join(scenario_dir, "hypogen_dr1_test.parquet")
        if not os.path.exists(parquet_path):
            urllib.request.urlretrieve(_TEST_PARQUET_URL, parquet_path)
        return load_dataset("parquet", data_files=parquet_path, split="train")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = self._load_test_split(output_path)

        instances = []
        for index, item in enumerate(dataset):
            abstract = (item["abstract"] or "").strip()
            bit = (item["bit"] or "").strip()
            flip = (item["flip"] or "").strip()

            if not abstract or not bit or not flip:
                continue

            prompt = (
                "The following is an abstract from a research paper and a description "
                "of a conventional approach or limitation (the \"bit\").\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Conventional approach / limitation (bit):\n{bit}\n\n"
                "Propose a novel research hypothesis or approach that overcomes "
                "this limitation (the \"flip\"):"
            )

            instances.append(Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=flip), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                id=f"hypogen_{index}",
                extra_data={
                    "task_form": "bit_flip_proxy",
                    "paper_id": (item.get("paper_id") or "").strip(),
                    "title": (item.get("title") or "").strip(),
                    "venue": (item.get("venue") or "").strip(),
                    "year": (item.get("year") or "").strip(),
                    "url": (item.get("url") or "").strip(),
                },
            ))

        return instances
