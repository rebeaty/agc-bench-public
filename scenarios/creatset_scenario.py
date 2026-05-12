"""
HELM Scenario: CreataSet — Chinese Creative Writing Generation

Paper: "Evaluating Text Creativity across Diverse Domains: A Dataset and Large
       Language Model Evaluator"
       arXiv:2505.19236, ICLR 2026
Code:  https://github.com/Aman-4-Real/CrEval
Data:  https://huggingface.co/datasets/Aman/CreataSet

Task: Given a Chinese creative writing instruction, generate a creative response.
The benchmark covers 8 domains: Short Texts, Lyrics, Modern Poetry, Ancient Poetry,
Prose, RuoZhiBa, Oorigi-Go, and Infinity-Instruct (50 instances each, 400 total).

Dataset: HuggingFace Aman/CreataSet, file CreataSet-test_with_labeling_400.jsonl
         Loaded via hf_hub_download (load_dataset fails — schema mismatch with the
         paired test file test_paired_3196.jsonl in the same split).

Fields used:
  instruction — Chinese creative writing prompt (input)
  source      — domain label (Short Texts / Lyrics / Modern Poetry / etc.)

Fields preserved in extra_data:
  output          — released reference text from the HF file
  gen_resp_1–4    — released candidate responses from the four source models
  gen_resp_order  — model-name ordering metadata for the released candidates
  avg_score       — Bradley-Terry win-rate calibration scores
  labeling        — 30-annotator pairwise judgment matrix
  title           — internal title metadata

Prompt source: No explicit prompt template in the paper; instruction field is used
  directly as it is already a complete Chinese creative writing request.

Evaluation: pairwise llm_judge against a released baseline response
            (see annotator_notes.md for judge configuration and baseline choice).
            avg_score and labeling remain available as calibration metadata.

Note: Originally flagged "not_suitable" for (1) non-English — invalid, Chinese is
  supported; (2) meta-evaluation — partially valid for the pairwise judging task
  (test_paired_3196.jsonl), but this scenario uses the generation subset (400 items).
  Pairwise judging task skipped as it evaluates evaluator quality, not LLM creativity.

Parameters:
  domain: "all" (default) | "Short Texts" | "Lyrics" | "Modern Poetry" |
          "Ancient Poetry" | "Prose" | "RuoZhiBa" | "Oorigi-Go" | "Infinity-Instruct"
"""

import json
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Scenario,
)

_HF_REPO = "Aman/CreataSet"
_HF_FILENAME = "CreataSet-test_with_labeling_400.jsonl"

_VALID_DOMAINS = (
    "all",
    "Short Texts",
    "Lyrics",
    "Modern Poetry",
    "Ancient Poetry",
    "Prose",
    "RuoZhiBa",
    "Oorigi-Go",
    "Infinity-Instruct",
)


class CreataSetScenario(Scenario):
    """
    CreataSet: Chinese creative writing generation across 8 domains.

    400 instances (50 per domain) from the human-labeled test split. Each
    instruction is a Chinese creative writing prompt. The output field is
    a curated/canonical reference. Human pairwise scores (avg_score, labeling)
    are available for judge calibration — see annotator_notes.md.
    """

    name = "creatset"
    description = "Aman/CreataSet (arXiv:2505.19236)"
    tags = ["creativity", "chinese", "multilingual", "open_ended_generation", "poetry"]

    def __init__(self, domain: str = "all"):
        super().__init__()
        if domain not in _VALID_DOMAINS:
            raise ValueError(f"domain must be one of {_VALID_DOMAINS!r}, got {domain!r}")
        self.domain = domain

    def get_instances(self, output_path: str) -> List[Instance]:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_FILENAME,
            repo_type="dataset",
        )

        instances = []
        with open(local_path, encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                item = json.loads(line)
                if self.domain != "all" and item.get("source") != self.domain:
                    continue

                instruction = item["instruction"].strip()
                if not instruction:
                    continue

                instances.append(
                    Instance(
                        input=Input(text=instruction),
                        references=[],
                        split=TEST_SPLIT,
                        extra_data={
                            "domain": item.get("source", ""),
                            "output": item.get("output", "").strip(),
                            "gen_resp_1": item.get("gen_resp_1", "").strip(),
                            "gen_resp_2": item.get("gen_resp_2", "").strip(),
                            "gen_resp_3": item.get("gen_resp_3", "").strip(),
                            "gen_resp_4": item.get("gen_resp_4", "").strip(),
                            "gen_resp_order": item.get("gen_resp_order", []),
                            "avg_score": item.get("avg_score", []),
                            "labeling": item.get("labeling", []),
                            "title": item.get("title", ""),
                            "source_row": row_index,
                        },
                    )
                )

        return instances  # 400 total (all); 50 per domain
