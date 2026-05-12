"""HELM scenario for Fann or Flop music-fan prediction tasks."""

import json
from typing import List

from huggingface_hub import hf_hub_download

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    CORRECT_TAG,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

_INSTRUCTION = (
    "أنت خبير في الأدب العربي والشعر. سيُقدَّم إليك قصيدة عربية كاملة، "
    "والمطلوب منك تقديم شرح مفصّل لكل بيت من أبياتها. "
    "ينبغي أن يتضمّن شرحك: المعنى الحرفي، والعمق الموضوعي، والسياق الثقافي، "
    "والصور الأدبية، والأسلوب التعبيري."
)

_PROMPT_TEMPLATE = """{instruction}

عنوان القصيدة: {title}
الشاعر: {author}
الحقبة الأدبية: {era}
البحر: {meter}
النوع: {genre}

القصيدة:
{poem_verses}

اشرح كل بيت من أبيات القصيدة بيتاً بيتاً."""


class FannOrFlopScenario(Scenario):
    """Fann or Flop: Arabic poetry verse-explanation generation.

    The public release includes both a paragraph-form gold explanation and a
    verse-aligned explanation list. This scenario keeps `raw_explanation` as the
    HELM reference text for overlap metrics while preserving the structured
    `explanation` list in `extra_data` for the benchmark-specific judge.
    """

    name = "fann_or_flop"
    description = "omkarthawakar/FannOrFlop (arXiv:2505.18152)"
    tags = ["creativity", "arabic", "poetry", "multilingual", "open_ended_generation"]

    def __init__(self, era: str = "all", genre: str = "all"):
        super().__init__()
        self.era = era
        self.genre = genre

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset_path = hf_hub_download(
            repo_id="omkarthawakar/FannOrFlop",
            filename="dataset.json",
            repo_type="dataset",
        )
        with open(dataset_path, "r", encoding="utf-8") as handle:
            dataset = json.load(handle)

        instances = []
        for item in dataset:
            if self.era != "all" and item.get("era") != self.era:
                continue
            if self.genre != "all" and item.get("genre") != self.genre:
                continue

            prompt = _PROMPT_TEMPLATE.format(
                instruction=_INSTRUCTION,
                title=item["title"] or "",
                author=item["author"] or "",
                era=item["era"] or "",
                meter=item["meter"] or "",
                genre=item["genre"] or "",
                poem_verses=item["poem_verses"] or "",
            )

            references = [
                Reference(Output(text=item["raw_explanation"] or ""), tags=[CORRECT_TAG])
            ]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=str(item["id"]),
                    extra_data={
                        "title": item.get("title") or "",
                        "author": item.get("author") or "",
                        "era": item.get("era") or "",
                        "meter": item.get("meter") or "",
                        "genre": item.get("genre") or "",
                        "verse_count": int(item.get("verse_count") or 0),
                        "verse_explanations": item.get("explanation") or [],
                    },
                )
            )

        return instances
