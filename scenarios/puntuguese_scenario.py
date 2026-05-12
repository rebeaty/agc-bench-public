"""HELM Scenario: Puntuguese.

Paper: https://aclanthology.org/2024.lrec-main.1167/ (LREC-COLING 2024)
Repo: https://github.com/Superar/Puntuguese
Dataset: https://huggingface.co/datasets/Superar/Puntuguese

The paper introduces a Portuguese pun corpus with humor-recognition and
pun-location annotations. The published experiments are supervised classifiers,
not prompted chat models, so the local HELM benchmark is an explicit zero-shot
Portuguese adaptation of the humor-recognition task only.

Task kept locally:
- binary humor recognition over the released `text` / `label` pairs
- evaluation on the official Hugging Face `test` split

Fields used:
- `id`
- `text`
- `label` (1 = humorous, 0 = non-humorous)

Fields skipped:
- `tokens`
- `labels`
  These support the separate pun-location task and are intentionally left out of
  the local runnable benchmark.
"""

import os

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)


class PuntugueseScenario(Scenario):
    name = "puntuguese"
    description = "Superar/Puntuguese"
    tags = ["creativity", "humor", "puns", "portuguese", "multilingual"]

    def get_instances(self, output_path):
        cache_dir = os.path.join(output_path, "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)

        dataset = load_dataset("Superar/Puntuguese", split="test", cache_dir=cache_dir)

        instances = []
        for item in dataset:
            text = item["text"].strip()
            label = int(item["label"])

            prompt = (
                f"Texto: {text}\n\n"
                "Este texto é humorístico?\n"
                "Responda apenas com Sim ou Não."
            )

            references = [
                Reference(
                    output=Output(text="Sim"),
                    tags=[CORRECT_TAG] if label == 1 else []
                ),
                Reference(
                    output=Output(text="Não"),
                    tags=[CORRECT_TAG] if label == 0 else []
                )
            ]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=str(item.get("id", "")) or None,
                extra_data={
                    "gold_label": "Sim" if label == 1 else "Não",
                    "task": "humor_recognition",
                    "has_pun_location_annotations": True,
                },
            ))

        return instances
