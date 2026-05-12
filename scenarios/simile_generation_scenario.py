"""
HELM scenario for SCOPE simile generation.

Paper: https://aclanthology.org/2020.emnlp-main.524/
Repo: https://github.com/tuhinjubcse/SimileGeneration-EMNLP2020

The released benchmark surface is the 150-example `SimileEMNLP.csv` evaluation
set. The paper's automatic metrics score only the generated VEHICLE after
discarding the shared sentence prefix, so this scenario stores the released
vehicle labels from `human_labels.csv` as HELM references while preserving the
full-sentence human similes in `extra_data` for audit.
"""

import csv
import os
import re
import urllib.request
from typing import List, Tuple

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)

_BRACKETED_PROPERTY_RE = re.compile(r"^(.*)\[(.+?)\](.*)$")


class SimileGenerationScenario(Scenario):
    name = "simile_generation"
    description = "tuhinjubcse/SimileGeneration-EMNLP2020"
    tags = ["creativity", "figurative_language", "simile", "style_transfer"]

    DATA_URL = "https://raw.githubusercontent.com/tuhinjubcse/SimileGeneration-EMNLP2020/master/SimileEMNLP.csv"
    VEHICLE_LABELS_URL = (
        "https://raw.githubusercontent.com/tuhinjubcse/SimileGeneration-EMNLP2020/master/human_labels.csv"
    )

    @staticmethod
    def _download(url: str, output_path: str, filename: str) -> str:
        os.makedirs(output_path, exist_ok=True)
        local_path = os.path.join(output_path, filename)
        if not os.path.exists(local_path):
            urllib.request.urlretrieve(url, local_path)
        return local_path

    @staticmethod
    def _parse_literal(literal: str) -> Tuple[str, str, str]:
        match = _BRACKETED_PROPERTY_RE.match(literal)
        if not match:
            return literal, "", ""
        prefix, property_text, suffix = match.groups()
        return prefix, property_text, suffix

    @staticmethod
    def _valid_vehicle(text: str) -> bool:
        cleaned = (text or "").strip()
        return bool(cleaned and cleaned != "------")

    def _load_rows(self, output_path: str) -> Tuple[List[dict], List[List[str]]]:
        simile_path = self._download(self.DATA_URL, output_path, "SimileEMNLP.csv")
        labels_path = self._download(self.VEHICLE_LABELS_URL, output_path, "human_labels.csv")

        with open(simile_path, "r", encoding="utf-8") as handle:
            simile_rows = list(csv.DictReader(handle))

        with open(labels_path, "r", encoding="utf-8") as handle:
            label_rows = list(csv.reader(handle))

        if label_rows and label_rows[0] and label_rows[0][0] == "Human1":
            label_rows = label_rows[1:]

        if len(simile_rows) != len(label_rows):
            raise ValueError(
                f"Expected aligned SCOPE release files, got {len(simile_rows)} simile rows and {len(label_rows)} label rows."
            )

        return simile_rows, label_rows

    def get_instances(self, output_path: str) -> List[Instance]:
        simile_rows, label_rows = self._load_rows(output_path)
        instances: List[Instance] = []

        for index, (row, vehicle_row) in enumerate(zip(simile_rows, label_rows)):
            literal = (row.get("Literal Sense") or "").strip()
            human1 = (row.get("Human1") or "").strip()
            human2 = (row.get("Human2") or "").strip()
            scope = (row.get("SCOPE") or "").strip()

            vehicle_refs = []
            for vehicle_text in vehicle_row[:2]:
                if self._valid_vehicle(vehicle_text):
                    vehicle_refs.append(Reference(Output(text=vehicle_text.strip()), tags=[CORRECT_TAG]))

            if not literal or not vehicle_refs:
                continue

            prefix, property_text, suffix = self._parse_literal(literal)
            prompt = (
                "Rewrite the literal sentence below as exactly one simile sentence.\n"
                "Replace the bracketed literal property with a simile vehicle while keeping the rest of the sentence natural.\n"
                "Return only the rewritten sentence.\n\n"
                f"Literal sentence: {literal}\n"
                "Simile sentence:"
            )

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=vehicle_refs,
                    split=TEST_SPLIT,
                    id=str(index),
                    extra_data={
                        "literal_sentence": literal,
                        "literal_prefix": prefix,
                        "literal_property": property_text,
                        "literal_suffix": suffix,
                        "full_sentence_human_references": [text for text in [human1, human2] if self._valid_vehicle(text)],
                        "scope_reference": scope,
                    },
                )
            )

        return instances
