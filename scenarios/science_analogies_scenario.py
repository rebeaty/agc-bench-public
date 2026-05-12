"""
HELM Scenario: Science Analogy Generation (SAQA)

Paper: "Analogy Generation by Prompting Large Language Models: A Case Study
       of InstructGPT" (INLG 2022, arXiv:2210.04186)
Code: https://github.com/Bhaavya/InstructGPT-Analogies
Dataset: GitHub repo data/saqa.txt (148 reference analogies, 109 target concepts)

Science analogy generation from target concepts. Given a science concept
(e.g., "dna", "proteins"), generate an analogy explanation. The SAQA dataset
provides reference analogies collected from academic Q&A sites (Chegg, Study.com)
covering high-school science (biology, chemistry, physics, computer science).

Subsets:
  - nosrc (default): Generate an analogy for a target concept without a given
    source. 109 unique target concepts, each with 1-4 reference analogies.
    Prompt from paper's NO_SRC setting (plm_generator.py prompt type 1).
  - wsrc: Generate an analogy using a specified source concept. 148 instances,
    each a (source, target, explanation) triple.
    Prompt from paper's WSRC setting (plm_generator.py prompt type 2).

Prompt source: Exact templates from plm_generator.py (prompts_nosrc[0],
  prompts_wsrc[0]).

Fields used: Source, Target, Explanation (from saqa.txt)
Fields skipped: saqa_full.txt full analogy text (contains source reveal),
  model-generated outputs in saqa_davinci/, human eval in amt_res.xlsx
"""

import csv
import os
import urllib.request
from collections import defaultdict

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)

_RAW_URL = (
    "https://raw.githubusercontent.com/Bhaavya/InstructGPT-Analogies"
    "/main/data/saqa.txt"
)

# Exact prompt templates from plm_generator.py
_NOSRC_PROMPT = "Explain {target} using an analogy."
_WSRC_PROMPT = "Explain {target} using an analogy involving {source}."


class ScienceAnalogiesScenario(Scenario):
    name = "science_analogies"
    description = "Bhaavya/InstructGPT-Analogies"
    tags = ["creativity", "analogy", "science", "open_ended"]

    SUBSETS = ["nosrc", "wsrc"]

    def __init__(self, subset: str = "nosrc"):
        """
        Args:
            subset: "nosrc" for generation without source concept (109 instances),
                    "wsrc" for generation with source concept (148 instances).
        """
        super().__init__()
        if subset not in self.SUBSETS:
            raise ValueError(
                f"subset must be one of {self.SUBSETS}, got '{subset}'"
            )
        self.subset = subset

    def _download_data(self, output_path: str) -> list[dict]:
        """Download saqa.txt and parse tab-separated rows."""
        scenario_dir = os.path.join(output_path, self.name)
        os.makedirs(scenario_dir, exist_ok=True)
        filepath = os.path.join(scenario_dir, "saqa.txt")
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(_RAW_URL, filepath)

        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                source = (row.get("Source") or "").strip()
                target = (row.get("Target") or "").strip()
                explanation = (row.get("Explanation") or "").strip()
                if target and explanation:
                    rows.append({
                        "source": source,
                        "target": target,
                        "explanation": explanation,
                    })
        return rows

    def _build_nosrc_instances(self, rows):
        """One instance per unique target; multiple references if available."""
        by_target = defaultdict(list)
        for row in rows:
            by_target[row["target"]].append(row["explanation"])

        instances = []
        for target, explanations in by_target.items():
            prompt = _NOSRC_PROMPT.format(target=target)
            references = [
                Reference(Output(text=exp), tags=[CORRECT_TAG])
                for exp in explanations
            ]
            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"saqa_nosrc_{target}",
                extra_data={"subset": "nosrc", "target": target, "source": ""},
            ))
        return instances

    def _build_wsrc_instances(self, rows):
        """One instance per (source, target) pair."""
        instances = []
        for idx, row in enumerate(rows):
            prompt = _WSRC_PROMPT.format(
                target=row["target"],
                source=row["source"],
            )
            references = [
                Reference(Output(text=row["explanation"]), tags=[CORRECT_TAG])
            ]
            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"saqa_wsrc_{idx}",
                extra_data={
                    "subset": "wsrc",
                    "target": row["target"],
                    "source": row["source"],
                },
            ))
        return instances

    def get_instances(self, output_path):
        rows = self._download_data(output_path)

        if self.subset == "nosrc":
            return self._build_nosrc_instances(rows)
        else:
            return self._build_wsrc_instances(rows)
