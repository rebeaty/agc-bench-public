"""HELM scenario for the New Yorker humor-understanding benchmark family.

Paper: https://aclanthology.org/2023.acl-long.41/
Dataset/repo: https://huggingface.co/datasets/jmhessel/newyorker_caption_contest
            / https://github.com/jmhessel/caption_contest_corpus

The paper defines three tasks: matching, ranking, and explanation. Matching and
ranking are accuracy tasks over answer letters. Explanation is generation-only
in HELM and remains caveated because the paper's main comparison is human
pairwise evaluation rather than a leaderboard-style automatic metric.
"""

import os
from typing import Iterable, List, Optional

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)


class NewYorkerHumorScenario(Scenario):
    name = "newyorker_humor"
    description = "New Yorker Cartoon Caption Contest Humor Understanding Benchmarks"
    tags = ["creativity", "humor", "caption", "commonsense", "explanation"]

    def __init__(
        self,
        task: str = "matching",
        cross_val_fold: Optional[int] = None,
        use_uncanny_description: bool = True,
        prefer_official_from_description: bool = True,
    ):
        """
        Initialize New Yorker Humor Understanding scenario.

        Args:
            task: One of "matching", "ranking", or "explanation"
            cross_val_fold: Cross-validation fold (1-4) or None for default split
            use_uncanny_description: If True, include the "uncanny" description along
                                    with standard description (recommended for better
                                    humor understanding)
        """
        super().__init__()

        if task not in ["matching", "ranking", "explanation"]:
            raise ValueError(f"task must be 'matching', 'ranking', or 'explanation', got: {task}")

        if cross_val_fold is not None and cross_val_fold not in [1, 2, 3, 4]:
            raise ValueError(f"cross_val_fold must be None or 1-4, got: {cross_val_fold}")

        self.task = task
        self.cross_val_fold = cross_val_fold
        self.use_uncanny_description = use_uncanny_description
        self.prefer_official_from_description = prefer_official_from_description

        # Construct dataset configuration name
        if cross_val_fold is None:
            self.config_name = task
        else:
            self.config_name = f"{task}_{cross_val_fold}"

    def _format_cartoon_description(self, item: dict) -> str:
        """Use the official packed description when available, then fall back."""
        if self.prefer_official_from_description and item.get("from_description"):
            return str(item["from_description"]).strip()

        parts = []

        # Location description
        if item.get("image_location"):
            parts.append(f"Location: {item['image_location']}")

        # Standard description
        if item.get("image_description"):
            parts.append(f"Scene: {item['image_description']}")

        # Uncanny/humor-focused description (often captures what makes it funny)
        if self.use_uncanny_description and item.get("image_uncanny_description"):
            parts.append(f"Notable: {item['image_uncanny_description']}")

        # Entities in the scene
        if item.get("entities"):
            entities = item["entities"]
            if entities:
                parts.append(f"Entities: {', '.join(entities)}")

        # Questions about the scene (help guide understanding)
        if item.get("questions"):
            questions = item["questions"]
            if questions:
                parts.append(f"Questions: {' | '.join(questions)}")

        return "\n".join(parts)

    def _load_split_iterable(self, split_name: str, output_path: str) -> Iterable[dict]:
        cache_dir = os.path.join(output_path, "hf_datasets_cache", self.config_name)
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset(
            "jmhessel/newyorker_caption_contest",
            self.config_name,
            split=split_name,
            streaming=True,
            cache_dir=cache_dir,
        )
        return dataset

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for New Yorker Humor benchmarks."""
        instances = []

        # Process each split
        for split_name, helm_split in [
            ("train", TRAIN_SPLIT),
            ("validation", VALID_SPLIT),
            ("test", TEST_SPLIT),
        ]:
            for idx, item in enumerate(self._load_split_iterable(split_name, output_path)):
                if self.task == "matching":
                    instance = self._create_matching_instance(item, idx, helm_split)
                elif self.task == "ranking":
                    instance = self._create_ranking_instance(item, idx, helm_split)
                else:  # explanation
                    instance = self._create_explanation_instance(item, idx, helm_split)

                instances.append(instance)

        return instances

    def _create_matching_instance(self, item: dict, idx: int, split: str) -> Instance:
        """Create instance for matching task (5-way multiple choice)."""
        cartoon_desc = self._format_cartoon_description(item)
        caption_choices = item["caption_choices"]
        label = item["label"]

        # Format as multiple choice (A, B, C, D, E)
        choice_letters = ["A", "B", "C", "D", "E"]
        choices_text = "\n".join([
            f"{letter}. {caption}"
            for letter, caption in zip(choice_letters, caption_choices)
        ])

        prompt = (
            "You will be given a text-only description of a New Yorker cartoon and five caption choices.\n\n"
            f"{cartoon_desc}\n\n"
            "Which of the 5 options (A, B, C, D, or E) is the caption that truly corresponds to the cartoon?\n\n"
            f"{choices_text}\n\n"
            "Answer:"
        )

        # Create references for each choice
        references = []
        for i, letter in enumerate(choice_letters[:len(caption_choices)]):
            tags = [CORRECT_TAG] if label == letter else []
            references.append(Reference(Output(text=letter), tags=tags))

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": item.get("instance_id", f"{split}_{idx}"),
                "contest_number": item.get("contest_number"),
                "task": "matching",
                "caption_choices": caption_choices,
                "gold_label": label,
                "from_description": item.get("from_description"),
            },
        )

    def _create_ranking_instance(self, item: dict, idx: int, split: str) -> Instance:
        """Create instance for ranking task (2-way comparison)."""
        cartoon_desc = self._format_cartoon_description(item)
        caption_choices = item["caption_choices"]
        label = item["label"]

        # For ranking, we have exactly 2 captions to compare
        prompt = (
            "You will be given a text-only description of a New Yorker cartoon and two caption choices.\n\n"
            f"{cartoon_desc}\n\n"
            "Which of the 2 options (A or B) is funnier for the given cartoon?\n\n"
            f"A. {caption_choices[0]}\n"
            f"B. {caption_choices[1]}\n\n"
            "Answer:"
        )

        # Create references
        ref_a = Reference(Output(text="A"), tags=[CORRECT_TAG] if label == "A" else [])
        ref_b = Reference(Output(text="B"), tags=[CORRECT_TAG] if label == "B" else [])
        references = [ref_a, ref_b]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": item.get("instance_id", f"{split}_{idx}"),
                "contest_number": item.get("contest_number"),
                "task": "ranking",
                "caption_choices": caption_choices,
                "gold_label": label,
                "winner_source": item.get("winner_source"),
                "from_description": item.get("from_description"),
            },
        )

    def _create_explanation_instance(self, item: dict, idx: int, split: str) -> Instance:
        """Create instance for explanation task (open-ended generation)."""
        cartoon_desc = self._format_cartoon_description(item)
        caption = item["caption_choices"]
        reference_explanation = item["label"]

        prompt = (
            "You will be given a text-only description of a New Yorker cartoon and its caption.\n\n"
            f"{cartoon_desc}\n\n"
            f"Caption: \"{caption}\"\n\n"
            "In a few sentences, explain why this caption is funny for this cartoon:"
        )

        # For explanation task, we provide the reference explanation
        references = [Reference(Output(text=reference_explanation), tags=[CORRECT_TAG])]

        return Instance(
            input=Input(text=prompt),
            references=references,
            split=split,
            extra_data={
                "instance_id": item.get("instance_id", f"{split}_{idx}"),
                "contest_number": item.get("contest_number"),
                "task": "explanation",
                "caption": caption,
                "reference_explanation": reference_explanation,
                "from_description": item.get("from_description"),
            },
        )
