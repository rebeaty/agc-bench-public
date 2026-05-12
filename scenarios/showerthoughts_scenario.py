"""
HELM Scenario: Showerthoughts Generation and Evaluation

Paper: https://aclanthology.org/2024.starsem-1.23/ (*SEM 2024)
      "Investigating Wit, Creativity, and Detectability of Large Language Models
       in Domain-Specific Writing Style Adaptation of Reddit's Showerthoughts"
Code: https://github.com/aiintelligentsystems/showerthoughts-dataset

Task: Generate one short Showerthought: a witty, clever, creative observation
about everyday life in the style of Reddit's r/Showerthoughts community.

Dataset note:
- The paper collected 411,189 Reddit Showerthoughts via Pushshift.
- The current public repo no longer redistributes that original corpus after
  Reddit access-policy changes.
- The released repo does ship a mixed detector file with 3,000 genuine and
  3,000 generated held-out Showerthoughts. This scenario uses the genuine
  entries from that file only to define evaluation slots and audit metadata.

Evaluation:
- The paper's survey rated standalone Showerthoughts on six 1-6 Likert axes:
  general score, logical validity, creativity, humor, cleverness, and whether
  the text seems written by a real person.
- This scenario therefore does NOT attach references. The local judge should
  see only the generated Showerthought, matching the paper's standalone rating
  setup.

Prompt source:
- Paper Section 4.1 uses a ChatGPT prompt that asks for 100 Showerthoughts.
- HELM adapts that to one-at-a-time generation while preserving the same style
  guidance. An invisible per-instance nonce is appended to prevent request
  caching from collapsing repeated trials into the same output.
"""

import json
import os
import urllib.request
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, TEST_SPLIT
)


_TRIAL_NONCE = "\u200b"


class ShowerthoughtsScenario(Scenario):
    name = "showerthoughts"
    description = "aiintelligentsystems/showerthoughts-dataset"
    tags = ["creativity", "generation", "wit", "humor", "reddit"]

    # Test data contains both genuine and AI-generated examples (50/50 split)
    DATA_URL = "https://raw.githubusercontent.com/aiintelligentsystems/showerthoughts-dataset/main/generated/roberta_test_data_mixed.ndjson"

    def __init__(self, num_instances: int = 300):
        """
        Initialize Showerthoughts scenario.

        Args:
            num_instances: Number of standalone generation slots to include.
        """
        super().__init__()
        self.num_instances = num_instances

    def get_instances(self, output_path: str):
        # Download the data file
        data_path = os.path.join(output_path, "showerthoughts_test.ndjson")
        if not os.path.exists(data_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(self.DATA_URL, data_path)

        # Load genuine Showerthoughts only. The released mixed file is not used
        # as references; it simply provides authentic held-out examples so the
        # benchmark can define a stable number of evaluation slots and record
        # audit metadata about the public release it is derived from.
        genuine_showerthoughts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item['label'] == 'genuine':
                    genuine_showerthoughts.append(item['title'])

        # Limit to requested number of instances
        genuine_showerthoughts = genuine_showerthoughts[:self.num_instances]

        # Adapted from the paper's Section 4.1 ChatGPT prompt, which asked for
        # 100 Showerthoughts per request. The local HELM path keeps the same
        # style guidance while constraining the model to one short response.
        prompt = (
            "Please generate one Showerthought, which is inspired by the Reddit community "
            "r/Showerthoughts. Try to be clever, creative, and funny. The Showerthought should "
            "be relatable and connected to things that people might encounter during mundane tasks. "
            "Return only the Showerthought text as one concise standalone statement.\n\n"
            "Showerthought:"
        )

        instances = []
        for idx, showerthought in enumerate(genuine_showerthoughts, start=1):
            # The zero-width nonce is execution-only: it preserves the repeated
            # single-item generation design without letting HELM replay cached
            # duplicates across evaluation slots.
            instances.append(
                Instance(
                    input=Input(text=f"{prompt}{_TRIAL_NONCE * idx}"),
                    references=[],
                    split=TEST_SPLIT,
                    id=f"showerthoughts_eval_{idx:04d}",
                    extra_data={
                        "trial_index": idx,
                        "public_release_source": "generated/roberta_test_data_mixed.ndjson",
                        "source_label": "genuine",
                        "source_title": showerthought,
                    },
                )
            )

        return instances
