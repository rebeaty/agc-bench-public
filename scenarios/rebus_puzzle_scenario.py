"""
HELM Scenario: Rebus Puzzle Probe Dataset

Paper: Visual Puzzles: A Probe for Understanding Vision-Language Models
GitHub: https://github.com/Kyunnilee/visual_puzzles
Dataset: https://huggingface.co/datasets/Kyunnilee/visual-puzzles

Prompt format (from solvers/utils.py):
  Please solve the rebus puzzle represented by the image. Respond with ONLY a valid JSON object containing two keys:
  1. 'answer': the string value of your solution
  2. 'reasoning': a detailed explanation of how you arrived at this answer, including the meaning of each visual element and how they combine

Prompt source: GitHub repository solvers/utils.py (PROMPT variable)

Dataset structure:
  - 432 rebus puzzle images (dataset/image/*.png)
  - Ground truth answers (dataset/answers.json)
  - 11 cognitive skill categories per puzzle

Fields used: image (PNG file), answer (ground truth text), skills (cognitive categories)
Fields skipped: None (gpt4o-caption.json is model output, not used)

Note: This is an open-ended generation task. Models generate free-form text answers
that are evaluated via exact string matching (case/space-insensitive) or LLM-as-judge.
"""

import json
import os
from typing import List
from datasets import load_dataset
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


# Skill category mapping from solvers/utils.py
SKILLS_ID_TO_NAME = {
    0: 'Absence or Negation',
    1: 'Font Style/Size',
    2: 'Image Recognition',
    3: 'Letter and Word Manipulation',
    4: 'Phonetics and Wordplay',
    5: 'Quantitative or Mathematical Reasoning',
    6: 'Spatial and Positional Reasoning',
    7: 'Symbolic Substitution',
    8: 'Text Orientation',
    9: 'Text Recognition (OCR + Typography/Layout)',
    10: 'Visual Metaphors and Cultural References',
}


class RebusPuzzleScenario(Scenario):
    """
    Rebus puzzle solving benchmark for vision-language models.

    Tests abstract reasoning, compositional thinking, and cultural/phonetic inference
    through visual riddles that encode phrases via images, wordplay, and spatial logic.
    """

    name = "rebus_puzzle"
    description = "Kyunnilee/visual-puzzles"
    tags = ["creativity", "multimodal", "vision", "reasoning", "wordplay"]

    def __init__(self, subset: str = "all"):
        """
        Args:
            subset: Filter by cognitive skill category. Options:
                    "all" (default) - all 432 puzzles
                    skill name (e.g., "Phonetics and Wordplay") - filter by skill
                    skill ID (e.g., "4") - filter by skill ID
        """
        super().__init__()
        self.subset = subset
        limit_hint = os.environ.get("REBUS_PUZZLE_INSTANCE_LIMIT_HINT", "").strip()
        self.instance_limit_hint = int(limit_hint) if limit_hint.isdigit() and int(limit_hint) > 0 else None

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Loads rebus puzzles from GitHub repo (images + answers.json).

        Note: Since the HuggingFace dataset only contains images without metadata,
        we need to access the GitHub repository for the complete dataset including
        ground truth answers and skill annotations.
        """

        # Load the HuggingFace dataset for images
        dataset = load_dataset("Kyunnilee/visual-puzzles", split="train")

        # Download answers.json from GitHub if not already present
        answers_path = os.path.join(output_path, "rebus_answers.json")
        if not os.path.exists(answers_path):
            import urllib.request
            answers_url = "https://raw.githubusercontent.com/Kyunnilee/visual_puzzles/main/dataset/answers.json"
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(answers_url, answers_path)

        # Load answers and metadata
        with open(answers_path, 'r') as f:
            answers_data = json.load(f)

        # Create mapping from image filename to metadata
        answers_dict = {item['image']: item for item in answers_data}

        instances = []
        for idx, item in enumerate(dataset):
            # Get corresponding metadata
            image_filename = f"{idx + 1}.png"
            if image_filename not in answers_dict:
                continue

            metadata = answers_dict[image_filename]
            answer = metadata['answer']
            skills = metadata['skills']

            # Filter by subset if specified
            if self.subset != "all":
                # Check if subset matches skill name or ID
                skill_match = False
                for skill_id in skills:
                    skill_name = SKILLS_ID_TO_NAME.get(skill_id, "")
                    if (self.subset == str(skill_id) or
                        self.subset.lower() in skill_name.lower()):
                        skill_match = True
                        break
                if not skill_match:
                    continue

            # Save PIL image to temp file for MediaObject
            temp_image_dir = os.path.join(output_path, "temp_images")
            os.makedirs(temp_image_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_image_dir, f"rebus_{idx + 1}.jpg")

            # Save PIL image
            pil_image = item['image']
            pil_image.convert("RGB").save(temp_image_path, "JPEG")

            # Build prompt with image
            prompt_text = """Please solve the rebus puzzle represented by the image. Respond with ONLY a valid JSON object containing two keys:
1. 'answer': the string value of your solution
2. 'reasoning': a detailed explanation of how you arrived at this answer, including the meaning of each visual element and how they combine"""

            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="image/jpeg",
                    location=temp_image_path
                ),
                MediaObject(
                    content_type="text/plain",
                    text=f"\n\n{prompt_text}"
                )
            ])

            # For open-ended generation, reference contains the ground truth answer
            references = [
                Reference(
                    Output(text=answer),
                    tags=[CORRECT_TAG]
                )
            ]

            # Add skill categories to instance metadata
            skill_names = [SKILLS_ID_TO_NAME[sid] for sid in skills]

            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"rebus_{idx + 1}",
                    extra_data={
                        "skills": skill_names,
                        "skill_ids": skills
                    }
                )
            )

            if self.instance_limit_hint is not None and len(instances) >= self.instance_limit_hint:
                break

        return instances
