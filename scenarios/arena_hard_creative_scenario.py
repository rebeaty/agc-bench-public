"""HELM Scenario: Arena-Hard v2.0 creative-writing subset.

Pairwise-preference creative-writing comparison from `lmarena/arena-hard-auto`,
restricted to the `creative_writing` category and using `gemini-2.0-flash-001`
as the upstream baseline opponent. Scoring is win-rate against the baseline
under an LLM-judge pairwise rubric.
"""
import json
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_file_downloaded

_QUESTION_URL = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/question.jsonl"
)
_BASELINE_URL = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl"
)

_CREATIVE_WRITING_CATEGORY = "creative_writing"
_BASELINE_MODEL = "gemini-2.0-flash-001"


class ArenaHardCreativeScenario(Scenario):
    """Arena-Hard v2.0 creative-writing subset with upstream baseline answers."""

    name = "arena_hard_creative"
    description = "github.com/lmarena/arena-hard-auto (Arena-Hard v2.0 creative_writing subset)"
    tags = ["creativity", "open_ended_generation", "creative_writing", "multilingual"]

    def get_instances(self, output_path: str) -> List[Instance]:
        question_path = os.path.join(output_path, "arena_hard_v2_question.jsonl")
        baseline_path = os.path.join(output_path, "arena_hard_v2_creative_baseline_gemini_2_0_flash_001.jsonl")

        ensure_file_downloaded(source_url=_QUESTION_URL, target_path=question_path, unpack=False)
        ensure_file_downloaded(source_url=_BASELINE_URL, target_path=baseline_path, unpack=False)

        with open(question_path, "r", encoding="utf-8") as f:
            data = f.read()

        baseline_answers = {}
        with open(baseline_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                baseline_answers[record["uid"]] = record["messages"][-1]["content"]["answer"]

        instances: List[Instance] = []
        buf = ""
        for line in data.splitlines():
            if line.startswith("{") and buf:
                buf = ""
            buf = line if not buf else buf + "\n" + line
            try:
                record = json.loads(buf)
            except json.JSONDecodeError:
                continue  # incomplete JSON — keep accumulating
            buf = ""
            if record.get("category") != _CREATIVE_WRITING_CATEGORY:
                continue
            uid = record["uid"]
            baseline_output = baseline_answers.get(uid)
            if baseline_output is None:
                continue

            instances.append(
                Instance(
                    input=Input(text=record["prompt"]),
                    references=[Reference(Output(text=baseline_output), tags=[])],
                    split=TEST_SPLIT,
                    id=f"arena_hard_creative_{uid}",
                    extra_data={
                        "uid": uid,
                        "category": record["category"],
                        "subcategory": record.get("subcategory", ""),
                        "baseline_model": _BASELINE_MODEL,
                    },
                )
            )

        return instances
