"""
HELM Scenario: Historical Analogy Acquisition

Paper: Past Meets Present: Creating Historical Analogy with Large Language Models (ACL 2025)
       https://aclanthology.org/2025.acl-long.200/
       https://arxiv.org/abs/2409.14820
Code: https://github.com/Nianqi-Li/Historical-Analogy-of-LLMs

Task: Given a contemporary or recent historical event, find an analogous historical event
      from the past that shares similar abstract patterns, themes, or dynamics.

Dataset: 20 manually curated famous historical analogies from web and articles
- Input: Event name + detailed event introduction (from Wikipedia or similar sources)
- Output: Name of the analogous historical event
- Examples:
  - Arab Spring → Revolutions of 1989
  - COVID-19 pandemic → Spanish flu
  - Russian Revolution → French Revolution

Prompt format: Based on direct_generation.py from the repository
  You are a historical analogy bot. For input events, your goal is to find
  the event that best fits the analogy.

  Input Event:
  {event_name}
  {event_intro}
  Historical Analogies Events:

Evaluation: llm_judge
  - Pass@1: Fuzzy match via Wikipedia search (exact match with variants)
  - Multi-dimensional similarity: GPT-4 scoring on abstract similarity (1-4 scale)
    Dimensions: topic similarity, general situation similarity, detail/focus overlap
  See annotator_notes.md for complete evaluation setup

Fields used: event_name, event_intro, target_event (from popular_analogy.jsonl)
Fields skipped: event_type (metadata, not needed for task)
Dataset composition: 20 test examples (popular analogies with ground truth)
"""

import json
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class HistoricalAnalogyScenario(Scenario):
    """Historical Analogy Acquisition Scenario

    Evaluates models' ability to identify analogous historical events by recognizing
    abstract patterns, similar themes, and comparable dynamics across time periods.
    """

    name = "historical_analogy"
    description = "Nianqi-Li/Historical-Analogy-of-LLMs"
    tags = ["creativity", "analogy", "historical_reasoning"]

    POPULAR_ANALOGY_URL = "https://raw.githubusercontent.com/Nianqi-Li/Historical-Analogy-of-LLMs/main/dataset/popular_analogy.jsonl"

    def _download_dataset(self, output_path: str) -> str:
        scenario_dir = os.path.join(output_path, self.name)
        os.makedirs(scenario_dir, exist_ok=True)
        dataset_path = os.path.join(scenario_dir, "popular_analogy.jsonl")
        ensure_file_downloaded(
            source_url=self.POPULAR_ANALOGY_URL,
            target_path=dataset_path,
            unpack=False,
        )
        return dataset_path

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset_path = self._download_dataset(output_path)
        with open(dataset_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        instances = []
        for idx, line in enumerate(lines):
            if line.strip():
                data = json.loads(line)
                instances.append(self._create_instance(data, idx))

        return instances

    def _create_instance(self, data: dict, index: int) -> Instance:
        """Create an instance from a historical analogy data point"""

        event_name = data["event_name"]
        event_intro = data["event_intro"]
        target_event = data["target_event"]

        # Build prompt following the paper's format (from direct_generation.py)
        prompt = f"""You are a historical analogy bot. For input events, your goal is to find the event that best fits the analogy. Here is a case:

==== case 
Input Event:
2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history. COVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
Historical Analogies Events:
Spanish flu

==== Answer the following questions using the format given above
Input Event:
{event_name}
{event_intro}
Historical Analogies Events:
"""

        references = [Reference(output=Output(text=target_event), tags=[CORRECT_TAG])]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=TEST_SPLIT,
            id=f"historical_analogy_{index}",
            extra_data={
                "event_name": event_name,
                "target_event": target_event,
            },
        )
