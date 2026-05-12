"""
HELM Scenario: Creation-MMBench (Assessing Context-Aware Creative Intelligence in MLLMs)

Paper: https://arxiv.org/abs/2503.14478 (ICCV 2025)
Code: https://github.com/open-compass/Creation-MMBench
Dataset: https://huggingface.co/datasets/opencompass/Creation-MMBench

Task: Multimodal creative generation across 51 fine-grained tasks in 4 categories:
  1. Literary Writing (8 tasks) - Poetry, dialogues, stories based on visual content
  2. Common Functional Writing (18 tasks) - Social media posts, daily inquiries, personal summaries
  3. Professional Functional Writing (19 tasks) - Lesson plans, marketing strategies, design analysis
  4. Creative Multimodal Understanding (6 tasks) - Advertisement interpretation, document analysis

Dataset composition:
  - 765 test cases (15 per task)
  - 1,001 images spanning 25+ categories
  - Variable image counts: 1-9 images per question (most have 1 image: 677/765)
  - 356/765 examples have ground_truth reference answers

Evaluation: GPT-4o as judge using instance-specific criteria for subjective quality
and visual factuality alignment. "Dual evaluation" swaps response positions to reduce bias.
Metrics: Visual Factuality Score (VFS, 1-10 scale) and Reward (-100 to +100).

Prompt format (template structure):
  Assume you are [Role] [Background]
  Please follow the requirements below to [Instruction].
  [Requirement]

Prompt source: Dataset provides complete prompts in 'question' field; template structure
described in paper Section 3.2 and Appendix.

Fields used: question, task_name, role, background, instruction, requirement, criteria,
             image, image_path, reference_answer_by_gpt4o, ground_truth
Fields skipped: source (dataset provenance), index (metadata)

Note: This is an LLM-as-judge benchmark. Reference answers are from GPT-4o (746 examples)
or human-provided ground_truth (356 examples). Evaluation requires GPT-4o with
instance-specific criteria from the 'criteria' field (parsed as dict with keys:
'subjective requirement', 'groundtruth alignment').
"""

import ast
import os
from typing import List
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class CreationMMBenchScenario(Scenario):
    name = "creation_mmbench"
    description = "opencompass/Creation-MMBench"
    tags = ["creativity", "multimodal", "vision", "creative_writing", "functional_writing"]

    def __init__(self, task_filter: str = "all"):
        """
        Args:
            task_filter: Filter by task name or category. Options:
                - "all": All 765 examples (default)
                - Specific task name (e.g., "story_continue", "poetry_creation")
                - Category prefix:
                    - "literary": Literary Writing tasks (8 tasks, ~120 examples)
                    - "common": Common Functional Writing tasks (18 tasks, ~270 examples)
                    - "professional": Professional Functional Writing tasks (19 tasks, ~285 examples)
                    - "creative_understanding": Creative Multimodal Understanding tasks (6 tasks, ~90 examples)
        """
        super().__init__()
        self.task_filter = task_filter

        # Category mappings (based on paper Section 3.2)
        self.literary_tasks = {
            "story_continue", "poetry_creation", "art_inspired_prose",
            "children_book_illustration_dialogue_creation", "daily_conversation_creation",
            "photography_appreciation", "architecture_appreciation", "landscape_appreciation"
        }

        self.creative_understanding_tasks = {
            "advertisement_explanation", "document_understanding",
            "snapshot_analysis", "travel_itinerary_planning",
            "recipe_infer_and_guide", "clothing_match_design"
        }

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load Creation-MMBench instances from HuggingFace.

        Note: Each instance may contain 1-9 images. Most (677/765) have a single image.
        """
        dataset = load_dataset("opencompass/Creation-MMBench", split="test")

        instances = []
        for item in dataset:
            # Apply task filter
            task_name = item['task_name']
            if not self._should_include_task(task_name):
                continue

            instance = self._create_instance(item, output_path)
            instances.append(instance)

        return instances

    def _should_include_task(self, task_name: str) -> bool:
        """Determine if task should be included based on filter."""
        if self.task_filter == "all":
            return True
        elif self.task_filter == "literary":
            return task_name in self.literary_tasks
        elif self.task_filter == "creative_understanding":
            return task_name in self.creative_understanding_tasks
        elif self.task_filter == "common":
            # Common functional writing: not literary, not creative understanding, not professional
            return (task_name not in self.literary_tasks and
                    task_name not in self.creative_understanding_tasks and
                    not self._is_professional_task(task_name))
        elif self.task_filter == "professional":
            return self._is_professional_task(task_name)
        else:
            # Exact task name match
            return task_name == self.task_filter

    def _is_professional_task(self, task_name: str) -> bool:
        """Identify professional functional writing tasks by common patterns."""
        professional_keywords = [
            "lesson_plan", "marketing", "business", "company", "letter_of",
            "public_welfare", "drafting_announcements", "promotional_words",
            "job_posting", "product_", "corporate_"
        ]
        return any(keyword in task_name for keyword in professional_keywords)

    def _create_instance(self, item: dict, output_path: str) -> Instance:
        """Create a single Instance from a dataset item."""
        # Extract fields
        question = item['question']  # Complete prompt with role, background, instruction
        images = item['image']  # List of PIL Image objects
        task_name = item['task_name']

        # Parse criteria (stored as string representation of dict)
        criteria_str = item['criteria']
        try:
            criteria_dict = ast.literal_eval(criteria_str)
        except:
            criteria_dict = {}

        # Get reference answer (prefer ground_truth if available, else GPT-4o reference)
        ground_truth = item.get('ground_truth', '')
        reference_answer = ground_truth if ground_truth else item.get('reference_answer_by_gpt4o', '')

        # Create multimedia content: images + question text
        multimedia_objects = []

        # Save PIL images to disk (HELM expects file paths, not PIL objects)
        import os as _os
        images_dir = _os.path.join(output_path, "images")
        _os.makedirs(images_dir, exist_ok=True)
        task_id = "".join(c if c.isalnum() else "_" for c in task_name)[:60]

        # Add all images first
        for idx, img in enumerate(images):
            image_path = _os.path.join(images_dir, f"{task_id}_{idx}.jpg")
            if not _os.path.exists(image_path):
                img.convert("RGB").save(image_path, "JPEG")
            multimedia_objects.append(
                MediaObject(
                    content_type="image/jpeg",
                    location=image_path
                )
            )

        # Add question text
        multimedia_objects.append(
            MediaObject(
                content_type="text/plain",
                text=question
            )
        )

        multimedia_content = MultimediaObject(multimedia_objects)

        # Build references
        # For open-ended creative generation, we include the reference answer
        # The criteria dict is stored in tags for use by the judge
        references = []
        if reference_answer:
            references.append(
                Reference(
                    Output(text=reference_answer),
                    tags=[CORRECT_TAG, "reference_answer"]
                )
            )

        # Add criteria as metadata (for potential use in evaluation)
        # Store as additional reference with special tag
        if criteria_dict:
            criteria_text = f"Subjective: {criteria_dict.get('subjective requirement', 'N/A')}\n"
            criteria_text += f"Groundtruth Alignment: {criteria_dict.get('groundtruth alignment', 'N/A')}"
            references.append(
                Reference(
                    Output(text=criteria_text),
                    tags=["evaluation_criteria"]
                )
            )

        return Instance(
            input=Input(multimedia_content=multimedia_content),
            references=references,
            split=TEST_SPLIT
        )
