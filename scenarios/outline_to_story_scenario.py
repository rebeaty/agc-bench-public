"""HELM Scenario: Outline-to-Story on WritingPrompts with paragraph events."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from rake_nltk import Rake


def _ensure_nltk_resources() -> None:
    import nltk

    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    )
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


def _clean_prompt(prompt: str) -> str:
    return re.sub(r"\[\s*.*?\s*\]", "", prompt).strip()


def _wp_preprocess(text: str) -> str:
    text = text.replace("<newline>", "\n")
    text = text.replace("``", '"')
    text = text.replace("''", '"')
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ('|\.|,|:|\?|\!|;)", r"\1", text)
    text = re.sub(r'" ([^"]*) "', r'"\1"', text)
    text = text.replace(" n't", "n't")
    return text.strip()


def _detect_dialog(text: str) -> bool:
    return text.startswith(('"', "'", "``", "`", "''", "“", "’", "‘", "”"))


def _split_reference_paragraphs(story: str) -> List[str]:
    raw_paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", story.strip()) if part.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [story.strip()]

    merged: List[str] = [raw_paragraphs[0]]
    for paragraph in raw_paragraphs[1:]:
        if _detect_dialog(paragraph) or len(paragraph) < 114:
            merged[-1] = f"{merged[-1]}\n{paragraph}".strip()
        else:
            merged.append(paragraph)

    return [_wp_preprocess(paragraph) for paragraph in merged]


def _extract_keywords(paragraph: str, rake: Rake) -> List[str]:
    rake.extract_keywords_from_text(paragraph)
    num_keywords = min(5, max(2, int(len(paragraph) / 228.0 + 1.5)))
    cleaned_keywords: List[str] = []
    for phrase in rake.get_ranked_phrases():
        normalized = re.sub(r" ('|\.|,|:|\?|\!|;)", r"\1", phrase.strip("'.,:?!;\" "))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized:
            cleaned_keywords.append(normalized)
        if len(cleaned_keywords) >= num_keywords:
            break
    return cleaned_keywords


def _build_outline_prompt(prompt: str, outline_events: List[List[str]]) -> str:
    outline_lines = []
    for index, paragraph_events in enumerate(outline_events, start=1):
        event_text = "; ".join(paragraph_events) if paragraph_events else "(no extracted events)"
        outline_lines.append(f"Paragraph {index}: {event_text}")

    outline_block = "\n".join(outline_lines)
    return (
        "Write a coherent multi-paragraph story that follows the ordered outline "
        "events below. Use one paragraph per outline item, keep the same order, "
        "and stay grounded in the prompt.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Outline Events:\n{outline_block}\n\n"
        "Story:\n"
    )


class OutlineToStoryScenario(Scenario):
    """
    Outline-to-Story (O2S) on WritingPrompts with paragraph-level event controls.

    This restores the core paper semantics for the WritingPrompts slice by
    segmenting the reference story into paragraphs, extracting RAKE-style event
    phrases for each paragraph, and prompting the model with the ordered event
    outline instead of only the raw Reddit prompt.
    """

    name = "outline_to_story"
    description = "WritingPrompts with paragraph-level outline events"
    tags = ["creativity", "story_generation", "long_form_generation", "outline_control"]

    def __init__(self, split: str = "test", max_instances: int = 0):
        super().__init__()
        self.split = split
        self.max_instances = max_instances
        _ensure_nltk_resources()
        self._rake = Rake(min_length=1, max_length=4)

    def _build_instance(self, example: Dict[str, Any], idx: int, helm_split: str) -> Instance:
        prompt = _clean_prompt(example["prompt"])
        reference_paragraphs = _split_reference_paragraphs(example["story"])
        outline_events = [_extract_keywords(paragraph, self._rake) for paragraph in reference_paragraphs]
        reference_story = "\n\n".join(reference_paragraphs)

        return Instance(
            input=Input(text=_build_outline_prompt(prompt, outline_events)),
            references=[Reference(Output(text=reference_story), tags=[CORRECT_TAG])],
            split=helm_split,
            id=f"o2s_{self.split}_{idx}",
            extra_data={
                "prompt": prompt,
                "outline_events": outline_events,
                "reference_paragraphs": reference_paragraphs,
            },
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        helm_split = VALID_SPLIT if self.split == "validation" else TEST_SPLIT
        dataset = load_dataset("euclaise/writingprompts", split=self.split)
        if self.max_instances > 0:
            dataset = dataset.select(range(min(self.max_instances, len(dataset))))

        return [self._build_instance(example, idx, helm_split) for idx, example in enumerate(dataset)]
