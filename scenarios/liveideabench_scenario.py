"""
HELM Scenario: LiveIdeaBench

Paper: https://arxiv.org/abs/2412.17596 (arXiv 2024)
Code: https://github.com/x66ccff/liveideabench

LiveIdeaBench evaluates LLMs' scientific creativity and idea generation using
single-keyword prompts across 1,180 scientific keywords spanning 22 domains.
Models must generate a concise (<100 word) scientific idea for each keyword.

Prompt source: utils/prompts.json in repository (exact text used below)
  - idea_prompt.description: Main prompt template
  - idea_prompt.fallback_description: Extended prompt (for models that refuse)

Fields used: Keyword (column B from keywordsEverywhere20241216.xlsx)
Fields skipped: Search Volume, CPC, Competition, Trend (SEO metadata)

Evaluation: LLM-as-judge across originality, feasibility, clarity (1-10 scale),
plus fluency (A-D pairwise comparison). See annotator_notes.md.
"""

import os
import urllib.request
import openpyxl
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    TEST_SPLIT
)


class LiveIdeaBenchScenario(Scenario):
    name = "liveideabench"
    description = "x66ccff/liveideabench"
    tags = ["creativity", "scientific_ideation", "divergent_thinking"]

    DATA_URL = "https://github.com/x66ccff/liveideabench/raw/main/keywords_data/keywordsEverywhere20241216.xlsx"

    # Exact prompt from utils/prompts.json (idea_prompt.description)
    IDEA_PROMPT = (
        'I\'ll be submitting your next responses to a "Good Scientific Idea" '
        "expert review panel. If they consider your idea to be a good one, "
        "you'll receive a reward. Your assigned keyword is: "
        '"{keyword}". You may provide background information. The idea MUST '
        "be concisely expressed within 100 words total (including any "
        "background information). (Note: good scientific ideas should be "
        "original (novel contribution), feasible (technically implementable), "
        "clearly articulated, and address meaningful problems in the field.)."
    )

    def _download_data(self, output_path: str) -> str:
        filepath = os.path.join(output_path, "liveideabench_keywords.xlsx")
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(self.DATA_URL, filepath)
        return filepath

    def _load_keywords(self, filepath: str) -> list[str]:
        wb = openpyxl.load_workbook(filepath, read_only=True)
        ws = wb.active
        keywords = []
        # Row 1: header note, Row 2: column headers, Row 3+: data
        # Column B (index 2) contains keywords
        for row in ws.iter_rows(min_row=3, max_col=2, values_only=True):
            kw = row[1]
            if kw and isinstance(kw, str) and kw.strip():
                keywords.append(kw.strip())
        wb.close()
        return keywords

    def get_instances(self, output_path: str):
        filepath = self._download_data(output_path)
        keywords = self._load_keywords(filepath)

        instances = []
        for index, keyword in enumerate(keywords):
            prompt = self.IDEA_PROMPT.format(keyword=keyword)

            # Open-ended generation: no gold reference
            instances.append(Instance(
                id=f"liveideabench_{index}",
                input=Input(text=prompt),
                references=[],
                split=TEST_SPLIT,
                extra_data={
                    "keyword": keyword,
                    "prompt_template": "idea_prompt.description",
                },
            ))

        return instances
