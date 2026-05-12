"""
HELM Scenario: Slang Generation Evaluative Framework

Paper: An Evaluative Framework for Creativity in LLM-Generated Slang
       https://aclanthology.org/2025.findings-emnlp.348/
Code:  https://github.com/siyangwu1/LLM-Slang-Dictionary

This local benchmark keeps the paper's freeform slang-generation slice:
given a definition, generate one slang usage with a word, a definition, and
a usage example. It does not yet expose the paper's separate reuse and
coinage modes as independent run variants.

Prompt source: `code/generation.py::build_prompt_general()`, simplified to
one entry and tightened to an explicit JSON response contract so HELM can
parse the structured fields before scoring semantic novelty.
"""

import ast
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_DATA_URL = (
    "https://raw.githubusercontent.com/siyangwu1/LLM-Slang-Dictionary"
    "/main/data/conv_slang.txt"
)


class SlangGenerationScenario(Scenario):
    """Freeform slang generation over the released conversational slang list."""

    name = "slang_generation"
    description = "siyangwu1/LLM-Slang-Dictionary"
    tags = ["creativity", "language_generation", "slang", "lexical_creativity"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "conv_slang.txt")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(_DATA_URL, data_path)

        instances = []
        with open(data_path, encoding="utf-8") as f:
            # File is a single Python list of (term, definition) tuples
            entries = ast.literal_eval(f.read())

        for term, definition in entries:
            definition = definition.strip()
            term = term.strip()
            if not definition or not term:
                continue

            prompt = (
                "You are a creative slang dictionary generator.\n"
                f"Generate 1 novel slang usage in English that expresses the definition: {definition}\n\n"
                "Return valid JSON with this schema:\n"
                "{\n"
                '  "word": "one slang term",\n'
                '  "definition": "brief slang definition",\n'
                '  "usage_context": "one usage example sentence"\n'
                "}\n"
                "Return only the JSON object."
            )

            references = [
                Reference(Output(text=term), tags=[CORRECT_TAG])
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"slang_generation_{len(instances)}",
                extra_data={
                    "gold_term": term,
                    "target_definition": definition,
                    "generation_mode": "freeform",
                },
            ))

        return instances
