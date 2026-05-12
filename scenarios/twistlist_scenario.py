"""
HELM Scenario: TwistList - Tongue Twister Generation

Paper: https://arxiv.org/abs/2306.03457 (ACL 2023)
Dataset: https://github.com/tangg555/TwistList
Dropbox: https://www.dropbox.com/scl/fi/dcur1b74jtugkrtq4eqfc/datasets.zip

Task: Generate tongue twisters from keywords/key phrases.
Tongue twisters are phonetically conditioned text that maximizes sound overlap
whilst maintaining semantic consistency with input topics and grammatical correctness.

Dataset: TwistList 1.0 - 2,125 human-authored tongue twisters
- Train: 1,912 examples
- Val: 106 examples
- Test: 107 examples

Keywords extracted using RAKE (Rapid Automatic Keyword Extraction)

Prompt format: Open-ended generation
Evaluation: open_ended (BLEU, ROUGE, BERTScore)
  Additional phonology metrics in paper: PO (Phoneme Overlap), iPED/oPED (Phonemic Edit Distance)
"""

import os
import urllib.request
import zipfile
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output, Reference,
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
    TRAIN_SPLIT,
)


class TwistListScenario(Scenario):
    """TwistList: Tongue Twister Generation Scenario

    Generates tongue twisters from keywords, evaluating models' ability
    to create phonetically challenging yet semantically consistent text.
    """

    name = "twistlist"
    description = "tangg555/TwistList"
    tags = ["creativity", "phonetic_generation", "wordplay"]

    # Dropbox direct download link (dl=1 triggers download)
    DATASET_URL = "https://www.dropbox.com/scl/fi/dcur1b74jtugkrtq4eqfc/datasets.zip?rlkey=wgj72wsfrlff3xxifgqt6cngv&dl=1"

    def __init__(self, use_prompts: bool = False):
        """
        Args:
            use_prompts: If True, use tt-prompt-data (with "Generate tongue twisters about key words:" prefix)
                        If False, use tt-data (keywords only)
        """
        super().__init__()
        self.use_prompts = use_prompts

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download and extract dataset
        dataset_dir = self._download_and_extract_dataset(output_path)

        # Choose data directory
        data_subdir = "tt-prompt-data" if self.use_prompts else "tt-data"
        data_path = os.path.join(dataset_dir, "datasets", "tongue_twister", data_subdir)

        instances = []

        # Load train, val, and test splits
        for split_name, split_constant in [
            ("train", TRAIN_SPLIT),
            ("val", VALID_SPLIT),
            ("test", TEST_SPLIT),
        ]:
            source_file = os.path.join(data_path, f"{split_name}.source.txt")
            target_file = os.path.join(data_path, f"{split_name}.target.txt")

            with open(source_file, "r", encoding="utf-8") as f_src, open(
                target_file, "r", encoding="utf-8"
            ) as f_tgt:
                sources = [line.strip() for line in f_src]
                targets = [line.strip() for line in f_tgt]

            assert len(sources) == len(targets), f"Mismatch in {split_name} split sizes"

            for keywords, tongue_twister in zip(sources, targets):
                instances.append(
                    self._create_instance(keywords, tongue_twister, split_constant)
                )

        return instances

    def _create_instance(
        self, keywords: str, tongue_twister: str, split: str
    ) -> Instance:
        """Create an instance from keywords and tongue twister"""

        # Create prompt - if not using prompt version, add our own instruction
        if self.use_prompts:
            # Keywords already have "Generate tongue twisters about key words:" prefix
            prompt = keywords
        else:
            # Add explicit instruction for keyword-only version
            prompt = f"Generate a tongue twister using these key words: {keywords}"

        # Create reference with the ground truth tongue twister
        references = [Reference(output=Output(text=tongue_twister), tags=[CORRECT_TAG])]

        return Instance(input=Input(text=prompt), references=references, split=split)

    def _download_and_extract_dataset(self, output_path: str) -> str:
        """Download dataset from Dropbox and extract it"""
        dataset_dir = os.path.join(output_path, "twistlist")
        zip_path = os.path.join(dataset_dir, "datasets.zip")
        extracted_marker = os.path.join(dataset_dir, ".extracted")

        # Check if already extracted
        if os.path.exists(extracted_marker):
            return dataset_dir

        # Create directory
        os.makedirs(dataset_dir, exist_ok=True)

        # Download zip file if not present
        if not os.path.exists(zip_path):
            print(f"Downloading TwistList dataset from Dropbox...")
            urllib.request.urlretrieve(self.DATASET_URL, zip_path)

        # Extract zip file
        print(f"Extracting TwistList dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Create marker file to indicate successful extraction
        with open(extracted_marker, "w") as f:
            f.write("extracted")

        return dataset_dir
