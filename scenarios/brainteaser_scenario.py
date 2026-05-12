"""HELM Scenario: BrainTeaser lateral-thinking MCQ.

Zero-shot lateral-thinking multiple-choice benchmark from
`1171-jpg/BrainTeaser` (Jiang et al., EMNLP 2023). Items pose puzzles that
require breaking standard semantic associations to find the correct answer.
Scoring is exact-match accuracy on the MCQ label.
"""
import io
import os

import numpy as np
import pyzipper
from helm.common.general import ensure_file_downloaded
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)

class BrainteaserScenario(Scenario):
    name = "brainteaser"
    description = "1171-jpg/BrainTeaser EMNLP zero-shot data"
    tags = ["creativity", "lateral_thinking", "multiple_choice"]
    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/1171-jpg/BrainTeaser/main/data/BTDATA.zip"
    DATASET_PASSWORD = b"brainteaser"

    def get_instances(self, output_path):
        zip_path = os.path.join(output_path, "BTDATA.zip")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=zip_path,
            unpack=False,
        )

        instances = []
        with pyzipper.AESZipFile(zip_path) as archive:
            archive.pwd = self.DATASET_PASSWORD
            sources = [
                ("sentence_puzzle.npy", "sentence"),
                ("word_puzzle.npy", "wordplay"),
            ]
            for archive_name, subset in sources:
                data = np.load(io.BytesIO(archive.read(archive_name)), allow_pickle=True)
                for item in data:
                    item_id = item["id"]
                    variant = "original"
                    if item_id.endswith("_SR"):
                        variant = "semantic"
                    elif item_id.endswith("_CR"):
                        variant = "context"

                    choices = item.get("choice_list")
                    if choices is None:
                        choices = [
                            item["answer"],
                            item["distrator1"],
                            item["distrator2"],
                            item["distrator(unsure)"],
                        ]

                    references = []
                    for idx, choice in enumerate(choices):
                        tags = [CORRECT_TAG] if idx == int(item["label"]) else []
                        references.append(Reference(Output(text=choice), tags=tags))

                    base_id = item_id.split("-")[1].split("_")[0]
                    instances.append(
                        Instance(
                            input=Input(text=item["question"]),
                            references=references,
                            split=TEST_SPLIT,
                            id=item_id,
                            extra_data={
                                "subset": subset,
                                "variant": variant,
                                "base_id": base_id,
                            },
                        )
                    )

        return instances
