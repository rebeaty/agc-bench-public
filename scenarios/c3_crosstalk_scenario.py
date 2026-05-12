"""
HELM Scenario: Chinese Comical Crosstalk (C3) Generation Benchmark

Paper: C3: Towards Realistic Benchmark for Chinese Comical Crosstalk Generation
       ACL 2023 Findings
       https://arxiv.org/abs/2207.00735
Code:  https://github.com/FreedomIntelligence/crosstalk-generation

Task: Given a 10-utterance context from a Chinese crosstalk (相声/Xiangsheng)
dialogue, generate the next 10 utterances continuing the comedic exchange.
Crosstalk (相声) is a traditional Chinese verbal comedy art form involving two
performers trading witty, punning, and culturally rich dialogue. The task tests
a model's ability to generate contextually coherent, humorous Chinese dialogue.

Note on language: This is a Chinese-language benchmark. HELM supports multilingual
evaluation — many LLMs (GPT-4, Claude, Qwen, GLM) handle Chinese fluently.

Dataset:
  - Test: 50 dialogues × 20 utterances = 1,000 utterances total
    (test_filter_50x20.txt)
  - Each instance: first 10 utterances as context, last 10 as gold continuation
  - 50 test instances total
  - Local automatic evaluation uses the benchmark-specific machine metric port
    in metrics/c3_crosstalk_metric.py: BLEU-1/2/4, GLEU, ROUGE-1/2/L, and
    distinct-1/2 with the upstream character-level scoring convention
  - Published human ratings cover general quality, humor, coherence, and an
    ethically-risky flag; see metric_notes for the local documentation

Prompt format (no LLM prompt specified in paper — paper used seq2seq models;
  standard Chinese instruction used here):

  以下是一段相声对话的开头，请续写接下来的对话内容（约10句）。

  {10-utterance context from meta_prompt.json}

  请续写：

Prompt source: Standard instruction format (paper used seq2seq models, not LLMs;
  context text taken verbatim from eval_data/human_eval/data/meta_prompt.json)
Fields used: meta_prompt.json[prompt] (10-utterance context),
  test_filter_50x20.txt lines 10-19 per dialogue (gold continuation)
Fields skipped: generate_completions.json (model-generated outputs, not ground truth),
  meta/train/dev splits (not evaluation data)

Evaluation: open_ended
  Automatic: benchmark-specific BLEU-1/2/4, GLEU, ROUGE-1/2/L, and
  distinct-1/2 against the gold continuation using the upstream evaluator's
  character-level convention
  Human: general quality, humor, coherence, and ethically-risky documented in
  metric_notes
"""

import json
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_META_PROMPT_URL = (
    "https://raw.githubusercontent.com/FreedomIntelligence/crosstalk-generation"
    "/main/eval_data/human_eval/data/meta_prompt.json"
)
_TEST_FILE_URL = (
    "https://raw.githubusercontent.com/FreedomIntelligence/crosstalk-generation"
    "/main/src/common_data/test_filter_50x20.txt"
)

# Each dialogue has 20 utterances; the first 10 form the context, the last 10
# are the gold continuation to predict.
_CONTEXT_TURNS = 10


class C3CrosstalkScenario(Scenario):
    """
    Chinese Comical Crosstalk (C3) generation: continue a crosstalk dialogue.

    50 test instances. Each instance provides a 10-utterance Chinese crosstalk
    context; the model must generate the next ~10 utterances.
    """

    name = "c3_crosstalk"
    description = "FreedomIntelligence/crosstalk-generation"
    tags = ["creativity", "dialogue_generation", "humor", "chinese"]

    def get_instances(self, output_path: str) -> List[Instance]:
        os.makedirs(output_path, exist_ok=True)

        meta_path = os.path.join(output_path, "meta_prompt.json")
        test_path = os.path.join(output_path, "test_filter_50x20.txt")

        if not os.path.exists(meta_path):
            urllib.request.urlretrieve(_META_PROMPT_URL, meta_path)
        if not os.path.exists(test_path):
            urllib.request.urlretrieve(_TEST_FILE_URL, test_path)

        # Load context prompts (first 10 utterances, pre-formatted)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        records = meta["RECORDS"]  # list of {id, prompt, desc}

        # Parse test file into 50 dialogues of 20 utterances each
        with open(test_path, encoding="utf-8") as f:
            raw = f.read()

        dialogues: List[List[str]] = []
        current: List[str] = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped == "":
                if current:
                    dialogues.append(current)
                    current = []
            else:
                current.append(stripped)
        if current:
            dialogues.append(current)

        instances = []
        for record, dialogue in zip(records, dialogues):
            context_text = record["prompt"].strip()

            prompt = (
                "以下是一段相声对话的开头，请续写接下来的对话内容（约10句）。\n\n"
                f"{context_text}\n\n"
                "请续写："
            )

            # Gold continuation: utterances 10–19
            gold_utterances = dialogue[_CONTEXT_TURNS:_CONTEXT_TURNS * 2]
            gold_text = "\n".join(gold_utterances)

            references = [Reference(Output(text=gold_text), tags=[CORRECT_TAG])]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
            ))

        return instances
