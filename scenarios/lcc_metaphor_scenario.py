"""
HELM Scenario: LCC Metaphor Detection

Paper: https://aclanthology.org/L16-1668/ (LREC 2016, original LCC)
       https://aclanthology.org/2022.acl-long.144/ (ACL 2022, probing study)
Code: https://github.com/EhsanAghazadeh/Metaphors_in_PLMs

Task: Binary classification of whether a target word in a sentence is used
metaphorically or literally.

Dataset:
  - LCC (Language Computer Corporation) Metaphor Corpus
  - 8,028 test examples (balanced: 4,014 metaphor, 4,014 non-metaphor)
  - Fields: text, span1 (word indices), label

Prompt format (standard binary classification - no paper-specified prompt):
  Note: Original paper is a probing study, not LLM prompting. Using standard format.

  Is the word "{target_word}" used metaphorically in the following sentence?
  Sentence: {sentence}
  Answer (Yes or No):

Subsets:
  "en" - English (8,028 test examples)
  "es" - Spanish
  "ru" - Russian
  "fa" - Farsi

Evaluation: exact_match (binary accuracy)
"""

import ast
import csv
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class LCCMetaphorScenario(Scenario):
    name = "lcc_metaphor"
    description = "EhsanAghazadeh/Metaphors_in_PLMs (LCC corpus)"
    tags = ["creativity", "metaphor", "figurative_language", "classification"]

    BASE_URL = "https://raw.githubusercontent.com/EhsanAghazadeh/Metaphors_in_PLMs/main/data/multi_ling"

    SUBSETS = ["en", "es", "ru", "fa"]

    # Standard binary classification prompt (paper is probing study, no LLM prompt specified)
    PROMPT_TEMPLATE = """Is the word "{target_word}" used metaphorically in the following sentence?

Sentence: {sentence}

Answer (Yes or No):"""

    def __init__(self, subset: str = "en"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown subset: {subset}. Must be one of {self.SUBSETS}")
        self.subset = subset

    def _extract_target_word(self, text: str, span_str: str) -> str:
        """Extract the target word(s) from text using span indices."""
        span = ast.literal_eval(span_str)  # Convert "[13, 14]" to [13, 14]
        words = text.split()
        return ' '.join(words[span[0]:span[1]])

    def get_instances(self, output_path: str):
        # Download test data
        data_url = f"{self.BASE_URL}/lcc_{self.subset}/test.csv"
        data_path = os.path.join(output_path, f"lcc_{self.subset}_test.csv")

        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(data_url, data_path)

        instances = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row['text']
                span = row['span1']
                label = row['label']

                # Extract target word
                target_word = self._extract_target_word(text, span)

                # Format prompt
                prompt = self.PROMPT_TEMPLATE.format(
                    target_word=target_word,
                    sentence=text
                )

                # Correct answer: "Yes" for Metaphor, "No" for Non-metaphor
                correct = "Yes" if label == "Metaphor" else "No"

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
