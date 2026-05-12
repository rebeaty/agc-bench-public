"""
HELM Scenario: YesBut (Real Satirical Photographs Dataset)

Paper: ***YesBut***: A High-Quality Annotated Multimodal Dataset for evaluating
       Satire Comprehension capability of Vision-Language Models
       https://arxiv.org/abs/2409.13592
       EMNLP 2024

Dataset: https://huggingface.co/datasets/bansalaman18/yesbut
Website: https://yesbut-dataset.github.io/

Task: Satirical Image Understanding (`WHYFUNNY` slice only)
Models are shown composite satirical images and asked to generate explanations
of why the image is funny/satirical.

Prompt format (from paper):
  [Image]
  Why is this image funny/satirical?

The dataset contains 1,084 satirical images with:
- left_image: Description of first half
- right_image: Description of second half
- overall_description: Ground truth explanation of the satire
- difficulty_in_understanding: EASY (954), MEDIUM (107), HARD (14)

Fields used: image, overall_description, difficulty_in_understanding
Fields skipped: left_image, right_image (textual descriptions, not needed for vision task),
                stage (all examples are 'stage2')

Evaluation: Open-ended generation for the Understanding task (BLEU, ROUGE-L,
METEOR, BERTScore in the paper). This local benchmark intentionally scopes
itself to the `WHYFUNNY` explanation slice rather than the paper's full
Detection + Understanding + Completion benchmark family.
Reference answer: overall_description field

Note: Images are PIL objects from HuggingFace dataset, saved to temp files for MediaObject.
Pattern follows humordb.py implementation.
"""

import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject
from datasets import load_dataset


class YesButScenario(Scenario):
    """
    YesBut: Satirical Image Understanding

    Evaluates vision-language models on explaining visual satire and humor
    in composite 'yes...but' photographs showing ironic juxtapositions.
    """

    name = "yesbut"
    description = "bansalaman18/yesbut"
    tags = ["creativity", "multimodal", "vision", "humor", "satire"]

    def __init__(self, difficulty: str = "all"):
        """
        Args:
            difficulty: Filter by difficulty level - "all", "easy", "medium", or "hard"
                       Default "all" includes all 1,084 examples
        """
        super().__init__()
        if difficulty == "all":
            difficulty = os.environ.get("YESBUT_DIFFICULTY", "all")
        valid_difficulties = {"all", "easy", "medium", "hard"}
        if difficulty.lower() not in valid_difficulties:
            raise ValueError(
                f"difficulty must be one of {sorted(valid_difficulties)}, got '{difficulty}'"
            )
        self.difficulty = difficulty.upper() if difficulty != "all" else "all"

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load YesBut dataset and create multimodal instances.

        Each instance contains:
        - Image: Satirical composite photograph
        - Prompt: "Why is this image funny/satirical?"
        - Reference: Ground truth explanation from overall_description
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("bansalaman18/yesbut", split="train")

        # Create temporary directory for images
        temp_dir = os.path.join(output_path, "yesbut_images")
        os.makedirs(temp_dir, exist_ok=True)

        instances = []
        for idx, item in enumerate(dataset):
            # Filter by difficulty if specified
            item_difficulty = (item.get('difficulty_in_understanding') or '').upper()
            if self.difficulty != "all" and item_difficulty != self.difficulty:
                continue

            # Skip examples with no difficulty label (9 examples)
            if not item_difficulty:
                continue

            # Save PIL image to temporary file
            # Images are JPEG format, RGB mode, ~695x695 pixels
            temp_image_path = os.path.join(temp_dir, f"image_{idx:04d}.jpg")
            item['image'].save(temp_image_path, format='JPEG')

            # Create multimedia content: image + prompt
            # Prompt from paper: "Why is this image funny/satirical?"
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="image/jpeg",
                    location=temp_image_path
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\nWhy is this image funny/satirical?"
                )
            ])

            # Reference answer: overall_description explains the satire
            # Example: "The images are funny since they show how the prettiest
            #          footwears like high heels, end up causing a lot of physical
            #          discomfort to the user, all in the name fashion"
            references = [
                Reference(
                    Output(text=item['overall_description']),
                    tags=[CORRECT_TAG]
                )
            ]

            # Create instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"yesbut_{idx:04d}",
                    extra_data={
                        "task_variant": "understanding_whyfunny",
                        "difficulty": item_difficulty,
                    },
                )
            )

        return instances
