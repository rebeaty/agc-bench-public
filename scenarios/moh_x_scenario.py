"""
HELM Scenario: MOH-X Metaphor Detection

Original dataset paper:
  Mohammad, S., Shutova, E., & Turney, P. (2016). "Metaphor as a Medium for
  Emotion: An Empirical Study." *SEM 2016.
  https://aclanthology.org/S16-2003/

MOH-X preprocessing:
  Gao, G., Choi, E., Choi, Y., & Zettlemoyer, L. (2018). "Neural Metaphor
  Detection in Context." EMNLP 2018.
  https://aclanthology.org/D18-1060/
  https://github.com/gao-g/metaphor-in-context

Referenced in LAG paper (arXiv:2504.11190) as a standard metaphor detection
benchmark.

Task: Binary classification of whether a target verb in a sentence is used
metaphorically or literally. MOH-X is a verb-focused subset of the original
MOH corpus, constructed by extracting verb–subject and verb–direct object
argument triples. Each example consists of a sentence and a specific target
verb whose usage must be classified.

Dataset:
  - 647 sentence–verb pairs
  - Labels: 0 (literal), 1 (metaphorical) — balanced ~50/50
  - ~214 unique verbs; average sentence length ~7.4 tokens
  - No predefined train/test split; standard evaluation uses 10-fold CV
  - All 647 examples used as TEST_SPLIT here (no labels withheld)
  - Source: data.zip from Google Drive (file ID: 1-v_sUlupDrKq8ERlnh5RwdI7Pyk-6cxn)
    containing MOH-X_formatted_svo_cleaned.csv

Prompt format (standard binary classification; no LLM prompt specified in
source papers — LAG paper uses complex ontology pipelines not suitable for
HELM; standard format used following LCC Metaphor precedent):

  Is the word "{verb}" used metaphorically in the following sentence?

  Sentence: {sentence}

  Answer (Yes or No):

Fields used:   verb (target word), sentence (context)
Fields skipped: arg1, arg2 (syntactic roles, not needed for prompt);
                verb_idx (position index, not needed — verb named explicitly)

Evaluation: exact_match (Yes/No binary accuracy)
"""

import csv
import glob
import os
import subprocess
import zipfile
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_DRIVE_FILE_ID = "1-v_sUlupDrKq8ERlnh5RwdI7Pyk-6cxn"
_CSV_FILENAME = "MOH-X_formatted_svo_cleaned.csv"

_PROMPT_TEMPLATE = (
    'Is the word "{verb}" used metaphorically in the following sentence?\n\n'
    "Sentence: {sentence}\n\n"
    "Answer (Yes or No):"
)


class MOHXScenario(Scenario):
    """
    MOH-X verb metaphor detection: classify a target verb as metaphorical or
    literal in context. 647 sentence–verb pairs, balanced classes.
    """

    name = "moh_x"
    description = "gao-g/metaphor-in-context (MOH-X)"
    tags = ["creativity", "metaphor", "figurative_language", "classification"]

    def _download_data(self, output_path: str) -> str:
        """Download and extract MOH-X CSV from Google Drive zip."""
        zip_path = os.path.join(output_path, "mohx_data.zip")

        if not os.path.exists(zip_path):
            subprocess.run(
                ["pip", "install", "-q", "gdown"],
                check=True, capture_output=True
            )
            subprocess.run(
                ["gdown", f"https://drive.google.com/uc?id={_DRIVE_FILE_ID}",
                 "-O", zip_path],
                check=True, capture_output=True
            )

        # Extract the MOH-X CSV (may be at root or in a subdirectory)
        csv_path = os.path.join(output_path, _CSV_FILENAME)
        if not os.path.exists(csv_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                match = next(
                    (n for n in zf.namelist() if _CSV_FILENAME in n), None
                )
                if match is None:
                    raise FileNotFoundError(
                        f"{_CSV_FILENAME} not found in zip. "
                        f"Contents: {zf.namelist()}"
                    )
                # Extract to output_path and move to flat path if nested
                zf.extract(match, output_path)
                extracted = os.path.join(output_path, match)
                if extracted != csv_path:
                    os.rename(extracted, csv_path)

        return csv_path

    def get_instances(self, output_path: str) -> List[Instance]:
        os.makedirs(output_path, exist_ok=True)
        csv_path = self._download_data(output_path)

        instances = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                verb = row["verb"].strip()
                sentence = row["sentence"].strip()
                label = int(row["label"])  # 0 = literal, 1 = metaphorical

                prompt = _PROMPT_TEMPLATE.format(verb=verb, sentence=sentence)

                is_metaphor = label == 1
                references = [
                    Reference(
                        Output(text="Yes"),
                        tags=[CORRECT_TAG] if is_metaphor else [],
                    ),
                    Reference(
                        Output(text="No"),
                        tags=[] if is_metaphor else [CORRECT_TAG],
                    ),
                ]

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                    )
                )

        return instances
