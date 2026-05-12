"""
HELM Scenario: II-Bench (Image Implication Understanding Benchmark)

Paper: https://arxiv.org/abs/2406.05862 (NeurIPS 2024 Datasets and Benchmarks)
Code: https://github.com/II-Bench/II-Bench
Dataset: https://huggingface.co/datasets/m-a-p/II-Bench

Task: Image implication understanding - evaluating higher-order perception, reasoning,
and comprehension abilities when presented with complex implication images including
abstract artworks, comics, and posters. Questions address metaphors, symbolism, and
detailed understanding requiring visual reasoning beyond surface-level recognition.

Dataset composition:
  - 1,222 images across 6 domains (Life, Art, Society, Psychology, Environment, Others)
  - 1,434 multiple-choice questions (6 options each, single correct answer)
  - Split: 1,399 test questions (answers hidden), 35 dev questions (answers available)
  - Images manually collected and annotated by 50 undergraduate students

Performance gap: MLLMs achieve ~75% accuracy vs. humans at ~90% (peak 98%), with
particular challenges in abstract domains like Art and Psychology.

Prompt format (zero-shot baseline):
  <image>
  {question}
  A. {option1}
  B. {option2}
  C. {option3}
  D. {option4}
  E. {option5}
  F. {option6}
  Answer:

Prompt source: Standard multiple-choice format (paper evaluates multiple prompt modes:
zero-shot, CoT, few-shot 1/2/3, domain/emotion/rhetoric hints)

Fields used: image, question, option1-6, answer, correct_option
Fields skipped: id, image_type, difficulty, domain, emotion, rhetoric, explanation, local_path (metadata)

Note: Test split has hidden answers (for EvalAI leaderboard). This scenario uses the
dev split (35 examples) as the primary evaluation set, similar to other benchmarks
where test labels are unavailable (e.g., RiddleSense).
"""

import os
from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class IIBenchScenario(Scenario):
    name = "ii_bench"
    description = "m-a-p/II-Bench"
    tags = ["creativity", "visual_reasoning", "implication", "metaphor", "multimodal", "vision"]

    def __init__(self, use_cot: bool = False):
        """
        Args:
            use_cot: If True, add "Let's think step by step." for chain-of-thought prompting.
                     Paper shows CoT improves performance on this benchmark.
        """
        super().__init__()
        self.use_cot = use_cot

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load II-Bench instances from HuggingFace.

        Note: Using dev split (35 examples) as test split has hidden answers.
        """
        # Load dev split (answers available)
        dataset = load_dataset("m-a-p/II-Bench", split="dev")

        instances = []
        images_dir = os.path.join(output_path, "images")
        os.makedirs(images_dir, exist_ok=True)
        for idx, item in enumerate(dataset):
            instance = self._create_instance(item, output_path, idx)
            instances.append(instance)

        return instances

    def _create_instance(self, item: dict, output_path: str, idx: int) -> Instance:
        """Create a single Instance from a dataset item."""
        # Extract fields
        pil_image = item['image']  # PIL Image object
        # Save PIL image to disk so HELM can reference it by path
        image_path = os.path.join(output_path, "images", f"{idx}.jpg")
        if not os.path.exists(image_path):
            pil_image.save(image_path)

        question = item['question']
        options = [
            item['option1'],
            item['option2'],
            item['option3'],
            item['option4'],
            item['option5'],
            item['option6']
        ]
        answer_letter = item['answer']  # 'A', 'B', 'C', 'D', 'E', or 'F'
        correct_option_text = item['correct_option']  # The full text of correct answer

        # Build prompt text
        prompt_text = f"{question}\n\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, E, F
            prompt_text += f"{letter}. {option}\n"

        prompt_text += "\nSelect exactly one option and reply with a single letter from A to F.\n"

        if self.use_cot:
            prompt_text += "\nLet's think step by step.\n"

        prompt_text += "\nAnswer:"

        # Create multimedia content: image + question + options
        multimedia_content = MultimediaObject([
            MediaObject(
                content_type="image/jpeg",
                location=image_path
            ),
            MediaObject(
                content_type="text/plain",
                text=prompt_text
            )
        ])

        # Build references: all 6 options, correct one tagged
        # Map answer letter to index
        answer_index = ord(answer_letter) - ord('A')

        references = []
        for i in range(6):
            letter = chr(65 + i)  # A, B, C, D, E, F
            is_correct = (i == answer_index)
            tags = [CORRECT_TAG] if is_correct else []
            references.append(
                Reference(Output(text=letter), tags=tags)
            )

        return Instance(
            input=Input(multimedia_content=multimedia_content),
            references=references,
            split=TEST_SPLIT,  # Using dev split but marking as TEST_SPLIT for evaluation
            extra_data={
                "gold_label": answer_letter,
                "options": options,
                "split_name": "dev",
            },
        )
