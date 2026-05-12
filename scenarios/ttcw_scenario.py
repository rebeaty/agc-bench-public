"""HELM Scenario: TTCW evaluator benchmark on released open-text stories."""

from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from helm.benchmark.scenarios.scenario import Input, Instance, Scenario, TEST_SPLIT


_BASE_URL = "https://raw.githubusercontent.com/salesforce/creativity_eval/main/Art_or_Artifice"
_PROMPT_URL = f"{_BASE_URL}/prompts/with_background.txt"
_STORIES_URL = f"{_BASE_URL}/stories/ttcw_short_stories.json"
_TESTS_URL = f"{_BASE_URL}/tests/ttcw_all_tests.json"
_ANNOTATIONS_URL = f"{_BASE_URL}/annotations/ttcw_annotations.json"


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _background_from_full_prompt(full_prompt: str) -> str:
    marker = "\n\nQ)"
    if marker in full_prompt:
        return full_prompt.split(marker, 1)[0].strip()
    return full_prompt.strip()


class TTCWScenario(Scenario):
    """
    TTCW evaluator benchmark on the released open-text subset.

    The official release contains 48 stories, but only 36 have full text in the
    open repository. This scenario uses those released story texts directly and
    evaluates whether a model can reproduce expert TTCW Yes/No judgments for the
    14 creativity tests.
    """

    name = "ttcw"
    description = "Salesforce/creativity_eval TTCW open-text evaluator benchmark"
    tags = ["creativity", "creative_writing", "evaluation", "llm_as_judge"]

    def _load_assets(self, output_path: str) -> Tuple[str, List[dict], List[dict], List[dict]]:
        cache_dir = os.path.join("benchmark_output", "scenarios", "ttcw")
        prompt_template = open(_download(_PROMPT_URL, os.path.join(cache_dir, "with_background.txt"))).read()
        stories = json.load(open(_download(_STORIES_URL, os.path.join(cache_dir, "ttcw_short_stories.json"))))
        tests = json.load(open(_download(_TESTS_URL, os.path.join(cache_dir, "ttcw_all_tests.json"))))
        annotations = json.load(open(_download(_ANNOTATIONS_URL, os.path.join(cache_dir, "ttcw_annotations.json"))))
        return prompt_template, stories, tests, annotations

    def get_instances(self, output_path: str) -> List[Instance]:
        prompt_template, stories, tests, annotations = self._load_assets(output_path)

        open_text_stories = {
            story["story_id"]: story
            for story in stories
            if not str(story["content"]).startswith("http")
        }
        tests_by_id = {int(test["ttcw_idx"]): test for test in tests}

        grouped_annotations: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        for annotation in annotations:
            story_id = annotation["story_id"]
            ttcw_idx = int(annotation["ttcw_idx"])
            if story_id in open_text_stories:
                grouped_annotations[(story_id, ttcw_idx)].append(annotation["binary_verdict"])

        instances: List[Instance] = []
        for (story_id, ttcw_idx), expert_labels in sorted(grouped_annotations.items()):
            story = open_text_stories[story_id]
            test = tests_by_id[ttcw_idx]
            majority_label = Counter(expert_labels).most_common(1)[0][0]
            background = _background_from_full_prompt(test["full_prompt"])
            prompt = (
                prompt_template.replace("[STORY]", story["content"])
                .replace("[BACKGROUND]", background)
                .replace("[QUESTION]", test["question"])
            )

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"{story_id}_ttcw_{ttcw_idx:02d}",
                    extra_data={
                        "story_id": story_id,
                        "story_name": story["story_name"],
                        "ttcw_idx": ttcw_idx,
                        "ttcw_category": test["category"],
                        "torrance_dimension": test["torrance_dimension"],
                        "question": test["question"],
                        "majority_label": majority_label,
                        "expert_labels": expert_labels,
                    },
                )
            )

        return instances
