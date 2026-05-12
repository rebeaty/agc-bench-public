"""
HELM Scenario: PunEval — Pun Understanding Evaluation

Paper: "A good pun is its own reword": Can Large Language Models Understand Puns?
       arXiv:2404.13599, EMNLP 2024, pages 11766–11782
Code:  https://github.com/Zhijun-Xu/PunEval

Tasks implemented (task parameter):
  generation  — Given a pun keyword and its two meanings, generate a pun sentence.
                Prompt: Method 1 (direct generation, no contextual words) from Notebook 5.
  explanation — Given a pun sentence, explain why it is a pun.
                Prompt: CoT recognition prompt from Notebook 2 (side="pun").
                Note: Paper derives explanations from the Reason field of the CoT
                recognition output; no standalone explanation prompt exists in the paper.

Task skipped:
  recognition — Binary pun/non-pun classification; not a creativity task.

Pun types (pun_type parameter):
  hom — Homographic: same spelling, different meanings (1,443 examples)
        e.g., "kid" = joke / young goat
  het — Heterographic: different spellings, similar sounds (1,146 examples)
        e.g., "clove" ≈ "clothes"
  all — Both combined (2,589 examples, default)

Data source: GitHub raw URL (no HuggingFace dataset)
  dataset/hom_dataset.json + dataset/het_dataset.json
  Merged from SemEval-2017 Task 7 + ExPun (Amazon Science)

Dataset note: 2,589 total entries, but 1,132 are SemEval-only entries lacking ExPun
  augmentation — they contain only human_text with no word-sense or explanation fields.
  Both tasks filter to the 1,457 fully annotated entries (pun_word present).

Fields used:
  generation:  pun_word, pun_sense, alter_word, alter_sense (prompt inputs)
               human_text (gold pun sentence; soft reference for BLEU/ROUGE)
  explanation: human_text (pun sentence input)
               human_explanation (gold explanation; reference for BLEU/ROUGE)

Fields skipped:
  pun_sense_key, alter_sense_key — internal WordNet sense IDs, not needed for prompts
  pun_word_ind    — SemEval entry ID, metadata only
  human_keywords  — used in Method 2 (contextually-constrained generation); not used here
  human_rating    — human quality rating of the pun; for calibration only
  (unannotated entries with only human_text) — 1,132 SemEval entries without ExPun fields

Evaluation:
  generation:  llm_judge (pun detection — is the output a valid pun?) + custom metrics
               Ambiguity, Distinctiveness, Surprise, Unusualness (see metric_notes.md)
  explanation: open_ended (BLEU/ROUGE vs human_explanation as reference)
"""

import json
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Reference,
    Scenario,
    CORRECT_TAG,
    Input,
    Instance,
    Output,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded

_HOM_URL = (
    "https://raw.githubusercontent.com/Zhijun-Xu/PunEval/main/dataset/hom_dataset.json"
)
_HET_URL = (
    "https://raw.githubusercontent.com/Zhijun-Xu/PunEval/main/dataset/het_dataset.json"
)

# Verbatim from Notebook 5, method 1 (direct pun generation, no contextual words)
_GENERATION_DEFINITION = (
    "<*Definition*>\n"
    "Puns are a form of wordplay exploiting different meanings of a word or similar-sounding"
    " words, while non-puns are jokes or statements that don't rely on such linguistic ambiguities."
)

_GENERATION_INSTRUCTION = (
    "<*Instruction*>\n"
    "Below is a keyword and two of its meanings. Please generate a pun sentence with punchline"
    " on the keyword that conveys both given meanings simultaneously. Except for the keyword, the"
    " pun sentence must not utilize any words from either of the two meanings. Besides, once a"
    " keyword is used, it's strictly prohibited to use it again in the latter half of the"
    ' sentence. You must output the current status in a parsable JSON format. An example output'
    ' looks like:\n{"Sentence": "XXX"}'
)

# Verbatim from Notebook 2 CoT recognition prompt with side="pun"
_EXPLANATION_DEFINITION = (
    "<*Definition*>\n"
    "Puns are a form of wordplay exploiting different meanings of a word or similar-sounding"
    " words, while non-puns are jokes or statements that don't rely on such linguistic ambiguities."
)

_EXPLANATION_INSTRUCTION = (
    "<*Instruction*>\n"
    'Determine whether the given Text is a pun. Give your reasons first, then make your final'
    ' decision clearly. You should either say "The given text is a pun" or say'
    ' "The given text is a non-pun". You must output the current status in a parsable JSON'
    ' format. An example output looks like:\n'
    '{"Reason": "XXX", "Choice": "The given text is a XXX"}'
)

_VALID_TASKS = ("generation", "explanation")
_VALID_PUN_TYPES = ("hom", "het", "all")


def _load_entries(pun_type: str, output_path: str) -> List[dict]:
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    downloads = []
    if pun_type in ("hom", "all"):
        downloads.append((_HOM_URL, os.path.join(data_dir, "pun_eval_hom_dataset.json")))
    if pun_type in ("het", "all"):
        downloads.append((_HET_URL, os.path.join(data_dir, "pun_eval_het_dataset.json")))

    entries = []
    for url, local_path in downloads:
        ensure_file_downloaded(source_url=url, target_path=local_path)
        with open(local_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key, item in data.items():
            item["_id"] = key
            entries.append(item)

    # Filter to ExPun-annotated entries only. 1,132 SemEval-only entries have just
    # human_text with no word-sense fields or human_explanation (not usable for either task).
    entries = [e for e in entries if e.get("pun_word") is not None]
    return entries  # 1,457 fully annotated entries


class PunEvalScenario(Scenario):
    """
    PunEval: LLM pun creativity evaluation across generation and explanation tasks.

    generation (default): given a pun keyword + two meanings, generate a pun sentence.
    explanation: given a pun sentence, explain why it is a pun.

    Pun type: hom (homographic), het (heterographic), or all (both, default).
    Dataset: 1,443 homographic + 1,146 heterographic = 2,589 total.
    """

    name = "pun_eval"
    description = "github.com/Zhijun-Xu/PunEval (arXiv:2404.13599)"
    tags = ["creativity", "humor", "wordplay", "language_creativity"]

    def __init__(self, task: str = "generation", pun_type: str = "all"):
        super().__init__()
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS!r}, got {task!r}")
        if pun_type not in _VALID_PUN_TYPES:
            raise ValueError(f"pun_type must be one of {_VALID_PUN_TYPES!r}, got {pun_type!r}")
        self.task = task
        self.pun_type = pun_type

    def get_instances(self, output_path: str) -> List[Instance]:
        entries = _load_entries(self.pun_type, output_path)

        instances = []
        for item in entries:
            if self.task == "generation":
                prompt = (
                    f"{_GENERATION_DEFINITION}\n\n"
                    f"{_GENERATION_INSTRUCTION}\n\n"
                    f"<*Your Response*>\n"
                    f"Keyword: {item['pun_word']}\n"
                    f"Meaning 1: {item['pun_word']} <{item['pun_sense']}>\n"
                    f"Meaning 2: {item['alter_word']} <{item['alter_sense']}>\n"
                    f"Output:"
                )
                # human_text is the gold pun sentence; used as soft reference for BLEU/ROUGE.
                # Note: many valid puns exist per keyword — LLM judge is the primary eval.
                references = [Reference(Output(text=item["human_text"]), tags=[CORRECT_TAG])]

            else:  # explanation
                prompt = (
                    f"{_EXPLANATION_DEFINITION}\n\n"
                    f"{_EXPLANATION_INSTRUCTION}\n\n"
                    f"<*Your Response*>\n"
                    f"Text: {item['human_text']}\n"
                    f"Output:"
                )
                # human_explanation is the gold explanation reference
                references = [
                    Reference(Output(text=item["human_explanation"]), tags=[CORRECT_TAG])
                ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=item["_id"],
                    extra_data={
                        "task": self.task,
                        "pun_type": "hom" if item["_id"].startswith("hom_") else "het",
                        "pun_word": item.get("pun_word"),
                        "pun_sense": item.get("pun_sense"),
                        "alter_word": item.get("alter_word"),
                        "alter_sense": item.get("alter_sense"),
                        "human_text": item.get("human_text"),
                    },
                )
            )

        return instances  # ~1,457 annotated total (all); subset of hom+het
