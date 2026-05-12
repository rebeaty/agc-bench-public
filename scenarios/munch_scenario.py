"""
HELM Scenario: MUNCH (Metaphor Understanding Challenge)

Paper: https://arxiv.org/abs/2403.11810 (ACL 2024)
       "Metaphor Understanding Challenge Dataset for LLMs"
Code: https://github.com/xiaoyuisrain/metaphor-understanding-challenge
License: CC-BY-4.0

Task: Evaluate metaphor understanding through word/sentence substitution judgement.
Given a sentence with a highlighted word, determine which substitution options
preserve the sentence meaning.

Dataset:
  - 1,492 examples for judgement tasks
  - Each example has two options labeled 'apt' (correct) or 'inapt' (incorrect)

Prompt format (from prompts.md - WOTG20/WOTG33 for word, SSTA94/GASW94 for sentence):
  Word judgement (implicit):
    Which of the given options can replace the highlighted word in the given
    sentence without altering the sentence's meaning?
    Sentence: {sentence}
    Option A: {option_a}
    Option B: {option_b}
    Option C: Both Option A and Option B
    Option D: Neither Option A nor Option B
    Correct answer: Option

  Word judgement (M-word - metaphorical word explicitly marked):
    Which of the given options can replace the highlighted metaphorically used
    word in the given sentence without altering the sentence's meaning?
    ...

Subsets:
  "word_implicit" - Word substitution, metaphor not explicitly indicated (1,492)
  "word_mword" - Word substitution, metaphorical word indicated (1,492)
  "sent_implicit" - Sentence paraphrase, metaphor not indicated (1,492)
  "sent_mword" - Sentence paraphrase, metaphorical word indicated (1,492)

Answer mapping:
  - A only apt → "A"
  - B only apt → "B"
  - Both apt → "C"
  - Neither apt → "D"

Evaluation: exact_match (4-way accuracy)
"""

import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class MUNCHScenario(Scenario):
    name = "munch"
    description = "xiaoyuisrain/metaphor-understanding-challenge"
    tags = ["creativity", "metaphor", "word_substitution", "figurative_language"]

    BASE_URL = "https://raw.githubusercontent.com/xiaoyuisrain/metaphor-understanding-challenge/main/tasks"

    SUBSETS = ["word_implicit", "word_mword", "sent_implicit", "sent_mword"]

    # Prompt templates from prompts.md (WOTG variants - clearest format)
    WORD_PROMPT_IMPLICIT = """Which of the given options can replace the highlighted word in the given sentence without altering the sentence's meaning?
Sentence: {sentence}
Option A: {option_a}
Option B: {option_b}
Option C: Both Option A and Option B
Option D: Neither Option A nor Option B
Correct answer: Option"""

    WORD_PROMPT_MWORD = """Which of the given options can replace the highlighted metaphorically used word in the given sentence without altering the sentence's meaning?
Sentence: {sentence}
Option A: {option_a}
Option B: {option_b}
Option C: Both Option A and Option B
Option D: Neither Option A nor Option B
Correct answer: Option"""

    SENT_PROMPT_IMPLICIT = """Select sentences that are semantically equivalent to the following sentence.
Sentence: {sentence}
Option A: {option_a}
Option B: {option_b}
Option C: Both Option A and Option B
Option D: Neither Option A nor Option B
Correct answer: Option"""

    SENT_PROMPT_MWORD = """Given a sentence where the highlighted word is metaphorically used, select sentences that are semantically equivalent to this sentence.
Sentence: {sentence}
Option A: {option_a}
Option B: {option_b}
Option C: Both Option A and Option B
Option D: Neither Option A nor Option B
Correct answer: Option"""

    def __init__(self, subset: str = "word_implicit"):
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown subset: {subset}. Must be one of {self.SUBSETS}")
        self.subset = subset

    def _get_data_file_and_prompt(self):
        """Return the appropriate data file URL and prompt template."""
        if self.subset == "word_implicit":
            return f"{self.BASE_URL}/word_judge.json", self.WORD_PROMPT_IMPLICIT
        elif self.subset == "word_mword":
            return f"{self.BASE_URL}/word_judge.json", self.WORD_PROMPT_MWORD
        elif self.subset == "sent_implicit":
            return f"{self.BASE_URL}/sent_judge_implicit.json", self.SENT_PROMPT_IMPLICIT
        elif self.subset == "sent_mword":
            return f"{self.BASE_URL}/sent_judge_mword.json", self.SENT_PROMPT_MWORD

    def _determine_correct_answer(self, options):
        """Determine the correct answer (A, B, C, or D) based on option labels."""
        a_apt = options[0]['label'] == 'apt'
        b_apt = options[1]['label'] == 'apt'

        if a_apt and b_apt:
            return "C"  # Both
        elif a_apt and not b_apt:
            return "A"  # Only A
        elif not a_apt and b_apt:
            return "B"  # Only B
        else:
            return "D"  # Neither

    def get_instances(self, output_path: str):
        data_url, prompt_template = self._get_data_file_and_prompt()

        # Download data
        filename = data_url.split('/')[-1]
        data_path = os.path.join(output_path, filename)
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(data_url, data_path)

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []
        for item in data:
            sentence = item['s0']
            options = item['options']

            # Format prompt
            prompt = prompt_template.format(
                sentence=sentence,
                option_a=options[0]['text'],
                option_b=options[1]['text']
            )

            # Determine correct answer
            correct = self._determine_correct_answer(options)

            # Create references for all 4 options
            references = [
                Reference(Output(text="A"), tags=[CORRECT_TAG] if correct == "A" else []),
                Reference(Output(text="B"), tags=[CORRECT_TAG] if correct == "B" else []),
                Reference(Output(text="C"), tags=[CORRECT_TAG] if correct == "C" else []),
                Reference(Output(text="D"), tags=[CORRECT_TAG] if correct == "D" else []),
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))

        return instances
