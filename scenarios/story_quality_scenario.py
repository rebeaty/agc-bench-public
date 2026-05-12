"""
HELM Scenario: Story Quality Rating (ScratchPlot)

Paper: "Plot Writing From Pre-Trained Language Models"
       (INLG 2022, arXiv:2206.03021)
Code: https://github.com/YipingNUS/scratchplot-story-generation
Dataset: Crowdsource annotations in repo crowdsource/ directory
         (Toloka platform, 3 annotators per story)

Story quality judgment task. Given a machine-generated story, rate its quality
on a 1-5 Likert scale. 100 stories from two generation systems (ScratchPlot
and plan-and-write, 50 content plans each), rated by 3 human annotators on
three dimensions.

Subsets:
  - interestingness (default): How interesting and engaging is the story?
  - coherence: How coherent and well-structured is the story?
  - naturalness: How natural and fluent is the story?

Ground truth: Rounded average of 3 annotator ratings per story.

Prompt source: Standard rating instruction (paper specifies 1-5 Likert scale
  but does not publish exact annotator instructions).

Fields used: INPUT:story, INPUT:content_plan_id, INPUT:system, OUTPUT:category
Fields skipped: GOLDEN:category (empty), HINT:* (empty), ASSIGNMENT:* (worker metadata)
"""

import csv
import os
import urllib.request
from collections import defaultdict

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_RAW_BASE = (
    "https://raw.githubusercontent.com/YipingNUS/scratchplot-story-generation"
    "/master/crowdsource"
)

_DIMENSION_FILES = {
    "interestingness": "interestingness/full-annotation-result-new.tsv",
    "coherence": "coherence/full-annotation-result-new.tsv",
    "naturalness": "naturalness/full-annotation-result-new.tsv",
}

_DIMENSION_DESCRIPTIONS = {
    "interestingness": "How interesting and engaging is this story?",
    "coherence": "How coherent and well-structured is this story?",
    "naturalness": "How natural and fluent is this story?",
}

_PROMPT_TEMPLATE = (
    "Read the following story and rate its {dimension} on a scale of 1 to 5, "
    "where 1 is the lowest and 5 is the highest.\n\n"
    "{dimension_description}\n\n"
    "Story:\n{story}\n\n"
    'Output your rating as a single number from 1 to 5.\n'
    "Rating:"
)


class StoryQualityScenario(Scenario):
    name = "story_quality"
    description = "YipingNUS/scratchplot-story-generation"
    tags = ["creativity", "story_generation", "judgment"]

    SUBSETS = list(_DIMENSION_FILES.keys())

    def __init__(self, subset: str = "interestingness"):
        """
        Args:
            subset: "interestingness", "coherence", or "naturalness".
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"subset must be one of {self.SUBSETS}, got '{subset}'"
            )
        self.subset = subset

    def _download_annotations(self, output_path: str) -> list[dict]:
        """Download annotation TSV and parse rows."""
        filename = _DIMENSION_FILES[self.subset]
        local_name = filename.replace("/", "_")
        filepath = os.path.join(output_path, local_name)
        if not os.path.exists(filepath):
            url = f"{_RAW_BASE}/{filename}"
            urllib.request.urlretrieve(url, filepath)

        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                story = (row.get("INPUT:story") or "").strip()
                system = (row.get("INPUT:system") or "").strip()
                plan_id = (row.get("INPUT:content_plan_id") or "").strip()
                rating = (row.get("OUTPUT:category") or "").strip()
                if story and rating:
                    rows.append({
                        "story": story,
                        "system": system,
                        "plan_id": plan_id,
                        "rating": int(rating),
                    })
        return rows

    def get_instances(self, output_path: str):
        rows = self._download_annotations(output_path)

        # Aggregate ratings per unique story (plan_id + system)
        story_ratings = defaultdict(list)
        story_text = {}
        for row in rows:
            key = (row["plan_id"], row["system"])
            story_ratings[key].append(row["rating"])
            story_text[key] = row["story"]

        instances = []
        for key, ratings in story_ratings.items():
            plan_id, system = key
            avg_rating = round(sum(ratings) / len(ratings))
            avg_rating = max(1, min(5, avg_rating))

            prompt = _PROMPT_TEMPLATE.format(
                dimension=self.subset,
                dimension_description=_DIMENSION_DESCRIPTIONS[self.subset],
                story=story_text[key],
            )

            references = [
                Reference(
                    Output(text=str(r)),
                    tags=[CORRECT_TAG] if r == avg_rating else [],
                )
                for r in range(1, 6)
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"story_quality_{self.subset}_{plan_id}_{system}",
            ))

        return instances
