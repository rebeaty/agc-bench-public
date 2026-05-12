"""
HELM Scenario: SciMON

Paper: SciMON: Scientific Inspiration Machines Optimized for Novelty
       Wang et al., ACL 2024. arXiv:2305.14259
Code:  https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty

Task: Scientific inspiration sentence generation. Given background context from a
scientific paper and a source entity involved in a "used-for" relationship, generate
a sentence describing the novel scientific method, dataset, or approach involved.
Tests scientific creativity — the ability to propose scientifically grounded ideas
given a problem context.

Dataset: gold_subset/idea_sentence.json from GitHub (194 human-verified instances
  from 2022 ACL Anthology papers). All instances passed 5 human quality criteria:
    - Output is not trivially overlapping with the context
    - IE extraction is of sufficient quality (not generic, correct)
    - Context contains relevant information for the target relation
    - Relation is part of the main idea proposed by the paper
    - Sentence generation label = 1 (suitable for generation evaluation)

Each instance provides a background context (sentences from the paper's abstract),
a source entity (e.g., "paraphrase generation"), a relation type (always "used for"),
and a gold reference sentence (rel_sent) from the paper expressing the relation.

Prompt format (reference-based HELM adaptation of the released GPT zero-shot style):

  Scientific context: {context}

  Relationship: {input}

  Based on the above context, write exactly one sentence describing the novel
  scientific method, dataset, or approach involved in this relationship.

Fields used:
  - context: background sentences from the paper abstract (input)
  - input:   entity + relation + type (e.g., "grammar is used for OtherScientificTerm"),
             used to convey relation direction and entity type in the prompt
  - rel_sent: gold reference sentence from the paper (CORRECT_TAG)

Fields skipped:
  - output:      short entity phrase (2-3 words); rel_sent is a richer generation target
  - neg_sample:  not present in gold_subset (only in full test set)
  - forward:     relation direction encoded in `input` field directly
  - cos_sim:     quality-filtering metadata (all examples passed all criteria)
  - annotation flags: all 1 (quality gate already applied)

Evaluation: open_ended — BLEU-4, ROUGE-L, SciBERT-aligned BERTScore against rel_sent gold reference
  Note: automatic metrics measure semantic similarity, not novelty per se.
  The paper's primary evaluation is human assessment of novelty, relevance, and
  technical depth. For novelty-sensitive scoring, see annotator_notes.md.
"""

import json
import os
import urllib.request
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

_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty"
    "/main/data/gold_subset.zip"
)


class SciMONScenario(Scenario):
    """
    Scientific inspiration sentence generation from SciMON gold subset.

    Given a background context about a scientific problem and a source entity
    involved in a "used-for" relation, generate a sentence describing the novel
    scientific contribution. Gold reference is the actual sentence from the paper.

    194 human-verified instances from 2022 ACL Anthology papers (held-out from
    training data which covers papers through 2021).
    """

    name = "scimon"
    description = "https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty"
    tags = ["creativity", "scientific_ideation", "open_ended"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_dir = os.path.join(output_path, "scimon")
        os.makedirs(data_dir, exist_ok=True)

        gold_json = os.path.join(data_dir, "gold_subset", "idea_sentence.json")
        if not os.path.exists(gold_json):
            zip_path = os.path.join(data_dir, "gold_subset.zip")
            urllib.request.urlretrieve(_DATA_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)

        with open(gold_json) as f:
            data = json.load(f)

        instances = []
        for item in data:
            context = item["context"].strip()
            input_rel = item["input"].strip()   # e.g. "grammar is used for OtherScientificTerm"
            rel_sent = item["rel_sent"].strip()

            prompt = (
                f"Scientific context: {context}\n\n"
                f"Relationship: {input_rel}\n\n"
                "Based on the above context, write exactly one sentence describing "
                "the novel scientific method, dataset, or approach involved in this "
                "relationship."
            )

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=rel_sent), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
