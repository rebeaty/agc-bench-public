"""
HELM Scenario: Chinese Homophonic Puns — Punchline Entity Recognition (PER-Task)

Paper: "DuanzAI: Slang-Enhanced Large Language Model with Prompt for Humor
        Understanding" (arXiv:2405.15818)
Code:  https://github.com/YesianRohn/DuanzAI

Task: Given a Chinese homophonic pun joke, identify the single punchline word/phrase
that creates the humor. The punchline replaces one or more characters in a common
Chinese idiom or phrase with phonetically similar (homophonic) characters, creating
wordplay.

Example:
  Joke:      老王一看就知道柿子是坏的，因为老王有一双透柿眼。
  Punchline: 透柿眼  (phonetic substitute for 透视眼, "X-ray eyes")
  Primitive: 透视眼

Note on language: This is a Chinese-language benchmark. HELM supports multilingual
evaluation — many LLMs (GPT-4, Claude, Qwen, GLM) handle Chinese fluently.

Dataset: upstream `task_1.json` fetched from GitHub raw. The current upstream file
contains 2546 Chinese homophonic pun jokes, matching the paper's "nearly 2500"
description more closely than the older local `1,000`-example comment.

Prompt format (adapted from prompt.py zero-shot task_1 template; original designed
for batched processing with line-numbered output — simplified here for single-instance
HELM evaluation):

  你现在的任务是从下面的笑话中找出其中幽默来源的那一个词语。
  当你无法找到时请输出未知，否则只输出该词语，不要有其他内容。

  笑话：{text}
  答案：

Prompt source: Adapted from prompt.py zero-shot template (args.task='1', args.type='1').

Fields used:   text (joke input), punchline (ground truth)
Fields skipped: glm_0shot_out, glm_0shot_punchline, gpt_0shot_punchline,
                gpt_5shot_punchline (pre-computed model outputs, not ground truth);
                baidu (alternative annotation, not needed for evaluation)

Evaluation: Custom fuzzy similarity metric (see metric_notes.md).
  Exact match + fuzzy similarity (difflib.SequenceMatcher + fuzzywuzzy.fuzz.ratio)
  as defined in evaluatePunchline.py.
"""

import json
import os
import urllib.request
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
    "https://raw.githubusercontent.com/YesianRohn/DuanzAI/main/data/task_1.json"
)


class ChineseHomophonicPunsScenario(Scenario):
    """
    Chinese Homophonic Puns — Punchline Entity Recognition (PER-Task).

    Upstream `task_1.json` currently contains 2546 Chinese homophonic pun jokes.
    Each instance provides a joke in Chinese;
    the model must identify the single punchline word/phrase that creates humor
    through phonetic substitution of a common idiom or expression.
    """

    name = "chinese_homophonic_puns"
    description = "YesianRohn/DuanzAI"
    tags = ["creativity", "humor", "wordplay", "chinese", "pun"]

    def get_instances(self, output_path: str) -> List[Instance]:
        os.makedirs(output_path, exist_ok=True)

        data_path = os.path.join(output_path, "task_1.json")
        if not os.path.exists(data_path):
            urllib.request.urlretrieve(_DATA_URL, data_path)

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        instances = []
        for item in data:
            text = item["text"].strip()
            punchline = item["punchline"].strip()

            prompt = (
                "你现在的任务是从下面的笑话中找出其中幽默来源的那一个词语。"
                "当你无法找到时请输出未知，否则只输出该词语，不要有其他内容。\n\n"
                f"笑话：{text}\n"
                "答案："
            )

            references = [Reference(Output(text=punchline), tags=[CORRECT_TAG])]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    extra_data={"dataset_size": len(data)},
                )
            )

        return instances
