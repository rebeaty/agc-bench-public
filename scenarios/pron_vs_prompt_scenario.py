"""HELM Scenario: Pron vs Prompt."""

import csv
import os
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Scenario,
)

_DATA_URL = "https://raw.githubusercontent.com/grmarco/pron-vs-prompt/main/data/synopses_texts.csv"

_SYSTEM_CONTEXT = (
    "We are going to do an experiment in which we are going to compare your creative writing"
    " skills with those of a prestigious novelist, Patricio Pron. The task is to generate"
    " synopses for movie titles that do not exist. The synopses must be creative and appealing"
    " to both critics and the general audience, and must have literary value in and of"
    " themselves.\n\n"
    "Here are some details of the novelist you will be competing with:\n"
    "Patricio Pron (Rosario, December 9, 1975) is a writer and literary critic. Granta magazine"
    " selected him in 2010 as one of the 22 best young writers in Spanish. He won the"
    " twenty-second Alfaguara Novel Prize in 2019 for his work Mañana tendremos otros nombres."
)

_USER_TEMPLATE = (
    'The proposed title is: "{title}". Please write a synopsis of about 600 words for that'
    " title that meets the above specifications."
)

_VALID_LANGUAGES = ("en", "es")
_VALID_ORIGINS = ("all", "patricio", "machine")


class PronVsPromptScenario(Scenario):
    """Creative ~600-word story synopsis from an imaginary title."""

    name = "pron_vs_prompt"
    description = "github.com/grmarco/pron-vs-prompt (EMNLP 2024)"
    tags = ["creativity", "creative_writing", "fiction", "literary_quality", "open_ended_generation"]

    def __init__(self, language: str = "en", title_origin: str = "all"):
        super().__init__()
        if language not in _VALID_LANGUAGES:
            raise ValueError(f"language must be one of {_VALID_LANGUAGES!r}, got {language!r}")
        if title_origin not in _VALID_ORIGINS:
            raise ValueError(f"title_origin must be one of {_VALID_ORIGINS!r}, got {title_origin!r}")
        self.language = language
        self.title_origin = title_origin

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "pron_vs_prompt_synopses_texts.csv")
        if not os.path.exists(data_path):
            urllib.request.urlretrieve(_DATA_URL, data_path)

        with open(data_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            instances: List[Instance] = []
            index = 0
            for row in reader:
                origin = row.get("title_origin", "").strip()
                if self.title_origin != "all" and origin != self.title_origin:
                    continue

                title = row["english_title"].strip() if self.language == "en" else row["title"].strip()
                if not title:
                    continue

                prompt = f"{_SYSTEM_CONTEXT}\n\n{_USER_TEMPLATE.format(title=title)}"
                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=[],
                        split=TEST_SPLIT,
                        id=f"pron_vs_prompt_{self.language}_{origin}_{index}",
                        extra_data={
                            "title": title,
                            "title_origin": origin,
                            "language": self.language,
                        },
                    )
                )
                index += 1

        return instances
