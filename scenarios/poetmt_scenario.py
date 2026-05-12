"""
HELM Scenario: PoetMT (Classical Chinese Poetry Translation)

Paper: https://arxiv.org/abs/2408.09945
Code: https://github.com/andongBlue/PoetMT
Published: EMNLP 2025

The local HELM port follows the released repo's baseline-style `all_poems`
files. It does not reconstruct the paper's separate discourse- and
sentence-level subsets, but it keeps the core task of translating a classical
Chinese poem into an English poem without RAT retrieval context.
"""

import json
import os
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


class PoetMTScenario(Scenario):
    """
    PoetMT: Classical Chinese Poetry Translation Benchmark

    Evaluates translation quality across three dimensions:
    adequacy, fluency, and elegance (creative expression).
    """

    name = "poetmt"
    description = "andongBlue/PoetMT"
    tags = ["creativity", "translation", "poetry"]

    VALID_DYNASTIES = ["tang", "song", "yuan", "all"]

    def __init__(self, dynasty: str = "all"):
        """
        Args:
            dynasty: Which dynasty to evaluate. Options:
                - "tang": Tang Dynasty poems (295 examples)
                - "song": Song Dynasty poems (196 examples)
                - "yuan": Yuan Dynasty poems (299 examples)
                - "all": All dynasties combined (790 examples)
        """
        super().__init__()
        if dynasty not in self.VALID_DYNASTIES:
            raise ValueError(
                f"Invalid dynasty '{dynasty}'. Must be one of: {self.VALID_DYNASTIES}"
            )
        self.dynasty = dynasty

    def _download_data(self, output_path: str) -> str:
        """
        Download PoetMT dataset from GitHub if not already present.

        Returns:
            Path to the data directory
        """
        data_dir = os.path.join(output_path, "poetmt_data")

        # Check if data already exists
        if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
            print(f"Data already exists at {data_dir}")
            return data_dir

        # Clone the repository
        import subprocess
        print("Downloading PoetMT dataset from GitHub...")
        os.makedirs(output_path, exist_ok=True)

        repo_url = "https://github.com/andongBlue/PoetMT.git"
        subprocess.run(
            ["git", "clone", repo_url, data_dir],
            check=True,
            capture_output=True
        )
        print("Download complete")

        return data_dir

    def _load_poems(self, data_dir: str, dynasty: str) -> List[dict]:
        """
        Load poems from JSONL files for specified dynasty.

        Args:
            data_dir: Path to the data directory
            dynasty: Dynasty name ("tang", "song", "yuan")

        Returns:
            List of poem dictionaries
        """
        file_path = os.path.join(data_dir, "all_poems", f"{dynasty}.jsonl")

        poems = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                poems.append(json.loads(line.strip()))

        return poems

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate PoetMT instances for the specified dynasty.

        Each instance contains:
        - Input: Classical Chinese poem with translation prompt
        - Reference: Expert English translation
        """
        data_dir = self._download_data(output_path)

        all_poems = []
        if self.dynasty == "all":
            for dynasty in ["tang", "song", "yuan"]:
                poems = self._load_poems(data_dir, dynasty)
                all_poems.extend([(poem, dynasty) for poem in poems])
        else:
            poems = self._load_poems(data_dir, self.dynasty)
            all_poems = [(poem, self.dynasty) for poem in poems]

        instances = []
        for idx, (poem, dynasty_name) in enumerate(all_poems):
            chinese_poem = poem["src"]
            reference_translation = poem["ref"]
            if not chinese_poem or not reference_translation:
                continue

            prompt = (
                "Please translate this classical Chinese poem into an English poem.\n"
                "Return only the translated English poem, with no explanation or extra commentary.\n"
                f"Poem:{chinese_poem}"
            )

            references = [
                Reference(
                    Output(text=reference_translation),
                    tags=[CORRECT_TAG]
                )
            ]

            instance_id = f"poetmt_{dynasty_name}_{idx}"
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=instance_id,
                    extra_data={
                        "source_poem": chinese_poem,
                        "reference_translation": reference_translation,
                        "title": poem.get("title", ""),
                        "author": poem.get("author", ""),
                        "dynasty": dynasty_name,
                    },
                )
            )

        return instances
