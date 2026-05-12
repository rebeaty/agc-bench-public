"""
HELM Scenario: Metaphor Generation from Literal Sentences

Paper: MERMAID: Metaphor Generation with Symbolism and Discriminative Decoding
       https://aclanthology.org/2021.naacl-main.336/
Code:  https://github.com/tuhinjubcse/MetaphorGenNAACL2021

The released runnable benchmark surface is the paper's 156-example human test
set: literal inputs from `human1test.txt` paired with metaphorical rewrites from
`human2test.txt`. The upstream files mark substituted verbs with `<V>` tags; the
scenario strips those tags before storing the reference text because the scorer
operates on the cleaned sentence surface rather than the annotation markup.
"""

import os
import re
from typing import List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)

_VERB_TAG_RE = re.compile(r"</?V>")


class MetaphorGenerationScenario(Scenario):
    """Metaphor Generation Scenario."""

    name = "metaphor_generation"
    description = "tuhinjubcse/MetaphorGenNAACL2021"
    tags = ["creativity", "metaphor", "figurative_language"]

    # Raw GitHub URLs for test data
    LITERAL_URL = "https://raw.githubusercontent.com/tuhinjubcse/MetaphorGenNAACL2021/main/fairseq/human1test.txt"
    METAPHORICAL_URL = "https://raw.githubusercontent.com/tuhinjubcse/MetaphorGenNAACL2021/main/fairseq/human2test.txt"

    @staticmethod
    def _download(output_path: str, url: str, filename: str) -> str:
        data_dir = os.path.join(output_path, "data", "metaphor_generation")
        ensure_directory_exists(data_dir)
        local_path = os.path.join(data_dir, filename)
        ensure_file_downloaded(source_url=url, target_path=local_path)
        return local_path

    @staticmethod
    def _read_lines(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle.read().splitlines()]

    def get_instances(self, output_path: str) -> List[Instance]:
        literal_path = self._download(output_path, self.LITERAL_URL, "human1test.txt")
        metaphor_path = self._download(output_path, self.METAPHORICAL_URL, "human2test.txt")
        literal_sentences = self._read_lines(literal_path)
        metaphorical_sentences = self._read_lines(metaphor_path)

        assert len(literal_sentences) == len(
            metaphorical_sentences
        ), f"Mismatch: {len(literal_sentences)} literal vs {len(metaphorical_sentences)} metaphorical"

        instances = []
        for index, (literal, metaphorical) in enumerate(zip(literal_sentences, metaphorical_sentences)):
            if literal and metaphorical:  # Skip empty lines
                instances.append(self._create_instance(index, literal, metaphorical))

        return instances

    def _create_instance(self, index: int, literal_sentence: str, metaphorical_sentence: str) -> Instance:
        """Create an instance from a literal-metaphorical sentence pair."""

        metaphorical_clean = _VERB_TAG_RE.sub("", metaphorical_sentence).strip()
        references = [Reference(output=Output(text=metaphorical_clean), tags=[CORRECT_TAG])]

        return Instance(
            input=Input(text=literal_sentence),
            references=references,
            split=TEST_SPLIT,
            id=str(index),
            extra_data={
                "literal_sentence": literal_sentence,
                "metaphorical_reference": metaphorical_clean,
                "raw_metaphorical_reference": metaphorical_sentence,
            },
        )
