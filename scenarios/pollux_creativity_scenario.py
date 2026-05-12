"""
HELM Scenario: POLLUX - Russian Creativity Evaluation

Paper: Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX
arXiv: https://arxiv.org/abs/2505.24616
Dataset: https://huggingface.co/datasets/ai-forever/POLLUX

POLLUX is a comprehensive benchmark for evaluating Russian-language LLMs across 35 task
types. This scenario focuses on the creativity-related tasks from the benchmark.

Task Types Included:
- Literary Text Generation (Написать художественный текст): 6,764 examples
- Creative Brainstorming (Творческий брейншторминг): 5,558 examples
- Text Interpretation - Subjective (Интерпретация текста): 6,081 examples
- Style Transfer (Стайл-трансфер): 5,977 examples
- AI as a Character (ИИ как персонаж): 6,025 examples
- Applied Brainstorming (Прикладной брейншторминг): 6,452 examples

Total: ~36,000 creativity-related examples

Evaluation: LLM-as-a-Judge with detailed criteria (0-4 scale)
- Creativity criteria: originality, inventiveness
- Literary criteria: dramaturgy, dialogue expressiveness, genre appropriateness
- Technical criteria: linguistic competence, formatting quality

Key finding from paper: "Even top-tier models like Claude 3.5 Sonnet and OpenAI o1
still lag behind human experts in tasks that heavily rely on creativity."

Prompt format:
  {instruction}

Fields used: instruction, task_type, difficulty, domain, criteria_name, criteria_score
Fields skipped: reference_answer (not always provided), answer (model outputs),
                annotations (evaluator comments), model_id (model identifier)

Evaluation: LLM-as-a-Judge (existing annotations) or custom judge evaluation.
            See annotator_notes.md for criteria details.
"""

from typing import List
import pandas as pd
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
)
from datasets import load_dataset


class POLLUXCreativityScenario(Scenario):
    """
    POLLUX Creativity Evaluation - Russian Language

    Evaluates LLMs on creative tasks in Russian including literary text generation,
    creative brainstorming, style transfer, and subjective interpretation.

    The POLLUX benchmark contains 2,115 expert-authored prompts with 161,076
    evaluation samples across 7 models using LLM-as-a-Judge methodology.
    """

    name = "pollux_creativity"
    description = "Russian creativity tasks from POLLUX benchmark (ai-forever/POLLUX)"
    tags = ["creativity", "russian", "literary_generation", "brainstorming", "llm_judge"]

    # Creative task types in Russian
    CREATIVE_TASK_TYPES = [
        "Написать художественный текст",  # Write literary text
        "Творческий брейншторминг",  # Creative brainstorming
        "Интерпретация текста (субъективная оценка)",  # Text interpretation (subjective)
        "Стайл-трансфер",  # Style transfer
        "ИИ как персонаж (экспертная ситуация)",  # AI as character (expert situation)
        "ИИ как персонаж (неформальная ситуация)",  # AI as character (informal)
        "Прикладной брейншторминг",  # Applied brainstorming
    ]

    # Creativity-specific evaluation criteria
    CREATIVITY_CRITERIA = [
        "Креативность",  # Creativity
        "Драматургия",  # Dramaturgy
        "Выразительность диалога",  # Dialogue expressiveness
        "Качество рифмы",  # Rhyme quality
        "Литературные акценты",  # Literary accents
        "Соблюдение образа персонажа",  # Character adherence
        "Размер стиха",  # Verse meter
        "Попадание в жанр",  # Genre appropriateness
    ]

    PARQUET_URL = (
        "https://huggingface.co/datasets/ai-forever/POLLUX/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    )

    def __init__(self, include_all_creative: bool = True, task_type: str = None):
        """
        Args:
            include_all_creative: If True, include all creative task types
            task_type: If specified, only include this task type (Russian name)
        """
        super().__init__()
        self.include_all_creative = include_all_creative
        self.task_type = task_type

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load POLLUX creativity tasks and create instances.

        Each instance is a unique instruction with its evaluation criteria.
        Since the dataset contains multiple model outputs per instruction,
        we deduplicate by instruction and use the criteria/scores as references.
        """

        # Load the published parquet shard directly. This avoids older
        # `datasets` feature-schema incompatibilities with the repository card.
        dataset = pd.read_parquet(self.PARQUET_URL).to_dict(orient="records")

        print(f"Loaded {len(dataset)} total POLLUX examples")

        # Filter for creative tasks
        if self.task_type:
            # Single task type
            creative_examples = [
                ex for ex in dataset
                if ex['task_type'] == self.task_type
            ]
        elif self.include_all_creative:
            # All creative task types
            creative_examples = [
                ex for ex in dataset
                if ex['task_type'] in self.CREATIVE_TASK_TYPES
            ]
        else:
            # Default: just literary text
            creative_examples = [
                ex for ex in dataset
                if ex['task_type'] == "Написать художественный текст"
            ]

        print(f"Filtered to {len(creative_examples)} creative examples")

        # Deduplicate by instruction (multiple models generate outputs per instruction)
        seen_instructions = {}
        for ex in creative_examples:
            instruction = ex['instruction']
            if instruction not in seen_instructions:
                seen_instructions[instruction] = ex

        print(f"Deduplicated to {len(seen_instructions)} unique instructions")

        # Create instances
        instances = []
        for instruction, ex in seen_instructions.items():
            # Build prompt (instructions are self-contained in Russian)
            prompt = instruction

            # Reference: Use reference_answer if available, otherwise empty
            # Note: Many creative tasks don't have single "correct" answers
            reference_text = ex.get('reference_answer', '') or ''

            # Tag references if they exist
            references = []
            if reference_text:
                references.append(
                    Reference(output=Output(text=reference_text), tags=[CORRECT_TAG])
                )

            # Create instance with metadata for evaluation
            instance = Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT,
                id=f"{ex['task_type']}_{len(instances)}",
                extra_data={
                    'task_type': ex['task_type'],
                    'task_subtype': ex.get('task_subtype', ''),
                    'difficulty': ex['difficulty'],
                    'domain': ex['domain'],
                    'criteria_name': ex['criteria_name'],
                    'criteria_description': ex.get('criteria_description', ''),
                    'rubrics': ex.get('rubrics', ''),
                    'expected_score': ex['criteria_score'],  # Expert annotation
                    'is_provocative': ex.get('is_provocative', False),
                },
            )

            instances.append(instance)

        return instances
