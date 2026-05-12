"""
HELM Scenario: MARS (Multimodal Analogical Reasoning dataSet)

Paper: Multimodal Analogical Reasoning over Knowledge Graphs
       (ICLR 2023)
       https://arxiv.org/abs/2210.00312
       https://openreview.net/forum?id=NRHajbzg8y0P

Code & Data: https://github.com/zjunlp/MKG_Analogy

Dataset: MARS - Multimodal Analogical Reasoning over Knowledge Graphs
- 10,641 training instances
- 1,020 validation instances
- 1,362 test instances

Task: Analogical reasoning in the form (e_h, e_t) : (e_q : ?)
Given an example entity pair (e_h, e_t) that demonstrates an analogical relationship,
and a question entity (e_q), predict the answer entity (e_a) such that the analogy holds.

Example:
  (clinical_trial, calcium_deficiency) : (exposure, ?)
  Answer: radiation_exposure

Modalities: Text descriptions + Images (from Wikidata)
- MarKG contains 11,292 entities, 192 relations, 76,424 images
- 2,063 analogical entities with images

Prompt format:
  Based on the paper's task formulation. No explicit prompt template provided,
  using standard analogical reasoning format.

Fields used: example (entity pair), question (query entity), answer (target entity),
            relation (Wikidata property), entity2text (descriptions)
Fields skipped: mode (pattern type - not used for basic evaluation)

Images: Available via separate download from Google Drive or Baidu Pan
       https://drive.google.com/file/d/1AqnyrA05vKngfEbhw1mxY5qEoaqiKsC1/view
       Baidu Pan code: 7hoc

Note: This scenario requires cloning the GitHub repository and optionally downloading
      the image dataset. Set `images_path` to the location of downloaded images.
"""

from typing import List, Optional
import json
import os
import subprocess
from pathlib import Path

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class MARSScenario(Scenario):
    """
    MARS: Multimodal Analogical Reasoning over Knowledge Graphs

    Evaluates analogical reasoning capabilities using entity pairs from a knowledge graph.
    """

    name = "mars"
    description = "zjunlp/MKG_Analogy"
    tags = ["creativity", "multimodal", "vision", "reasoning", "analogy"]

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        images_path: Optional[str] = None,
        use_images: bool = False,
    ):
        """
        Args:
            dataset_path: Path to cloned MKG_Analogy repository. If None, will clone to temp directory.
            images_path: Path to downloaded MARS images directory. Required if use_images=True.
            use_images: Whether to include images in the multimodal prompts.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.images_path = images_path
        self.use_images = use_images

        if self.use_images and not self.images_path:
            raise ValueError(
                "images_path must be provided when use_images=True. "
                "Download images from: "
                "https://drive.google.com/file/d/1AqnyrA05vKngfEbhw1mxY5qEoaqiKsC1/view "
                "or Baidu Pan (code: 7hoc): https://pan.baidu.com/s/1WZvpnTe8m0m-976xRrH90g"
            )

    def _ensure_dataset(self, output_path: str) -> str:
        """Clone or verify MKG_Analogy repository."""
        if self.dataset_path and os.path.exists(self.dataset_path):
            return self.dataset_path

        # Clone to output directory
        repo_path = os.path.join(output_path, "MKG_Analogy")
        if not os.path.exists(repo_path):
            print(f"Cloning MKG_Analogy repository to {repo_path}...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/zjunlp/MKG_Analogy.git", repo_path],
                check=True
            )
        return repo_path

    def _load_entity_descriptions(self, data_dir: str) -> dict:
        """Load entity ID to text description mapping."""
        entity2text = {}
        entity2text_path = os.path.join(data_dir, "MarKG", "entity2text.txt")

        with open(entity2text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0]
                    text = parts[1]
                    entity2text[entity_id] = text

        return entity2text

    def _load_relation_descriptions(self, data_dir: str) -> dict:
        """Load relation ID to text description mapping."""
        relation2text = {}
        relation2text_path = os.path.join(data_dir, "MarKG", "relation2text.txt")

        with open(relation2text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    relation_id = parts[0]
                    text = parts[1]
                    relation2text[relation_id] = text

        return relation2text

    def _get_entity_image_path(self, entity_id: str) -> Optional[str]:
        """Get path to entity image if it exists."""
        if not self.use_images or not self.images_path:
            return None

        # Images are organized as images/Q{id}/*.jpg
        entity_dir = os.path.join(self.images_path, entity_id)
        if not os.path.exists(entity_dir):
            return None

        # Get first image in directory
        image_files = [f for f in os.listdir(entity_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            return os.path.join(entity_dir, image_files[0])

        return None

    def _create_multimodal_input(
        self,
        entity_id: str,
        entity_text: str,
        label: str = "Entity"
    ) -> List:
        """Create multimodal content for an entity (text + optional image)."""
        content = []

        # Add text description
        content.append(MediaObject(
            content_type="text/plain",
            text=f"{label}: {entity_text}"
        ))

        # Add image if available
        if self.use_images:
            image_path = self._get_entity_image_path(entity_id)
            if image_path and os.path.exists(image_path):
                content.append(MediaObject(
                    content_type="image/jpeg",
                    location=image_path
                ))

        return content

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load MARS dataset and create instances for analogical reasoning."""

        # Ensure dataset is available
        repo_path = self._ensure_dataset(output_path)
        data_dir = os.path.join(repo_path, "MarT", "dataset")

        # Load entity and relation mappings
        print("Loading entity and relation descriptions...")
        entity2text = self._load_entity_descriptions(data_dir)
        relation2text = self._load_relation_descriptions(data_dir)

        instances = []

        # Load all splits
        splits_info = [
            ("train.json", TRAIN_SPLIT),
            ("dev.json", VALID_SPLIT),
            ("test.json", TEST_SPLIT)
        ]

        for filename, split in splits_info:
            filepath = os.path.join(data_dir, "MARS", filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    item = json.loads(line)

                    # Extract entities and relation
                    example_h_id = item['example'][0]
                    example_t_id = item['example'][1]
                    question_id = item['question']
                    answer_id = item['answer']
                    relation_id = item['relation']

                    # Get text descriptions
                    example_h_text = entity2text.get(example_h_id, example_h_id)
                    example_t_text = entity2text.get(example_t_id, example_t_id)
                    question_text = entity2text.get(question_id, question_id)
                    answer_text = entity2text.get(answer_id, answer_id)
                    relation_text = relation2text.get(relation_id, relation_id)

                    # Build multimodal prompt
                    multimedia_content = []

                    # Task instruction
                    multimedia_content.append(MediaObject(
                        content_type="text/plain",
                        text="Complete the following analogy:\n\n"
                    ))

                    # Example pair
                    multimedia_content.extend(self._create_multimodal_input(
                        example_h_id, example_h_text, "Example A"
                    ))

                    multimedia_content.append(MediaObject(
                        content_type="text/plain",
                        text=f"\nrelates to\n\n"
                    ))

                    multimedia_content.extend(self._create_multimodal_input(
                        example_t_id, example_t_text, "Example B"
                    ))

                    multimedia_content.append(MediaObject(
                        content_type="text/plain",
                        text=f"\nby the relation: {relation_text}\n\n"
                             f"Similarly,\n\n"
                    ))

                    # Question entity
                    multimedia_content.extend(self._create_multimodal_input(
                        question_id, question_text, "Question"
                    ))

                    multimedia_content.append(MediaObject(
                        content_type="text/plain",
                        text="\nrelates to what?\n\nAnswer:"
                    ))

                    # Create multimedia object
                    if self.use_images:
                        input_obj = Input(multimedia_content=MultimediaObject(multimedia_content))
                    else:
                        # For text-only, concatenate all text parts
                        text_parts = [m.text for m in multimedia_content if hasattr(m, 'text') and m.text]
                        input_obj = Input(text="".join(text_parts))

                    # Reference: correct answer (can match by ID or text)
                    references = [
                        Reference(Output(text=answer_id), tags=[CORRECT_TAG]),
                        Reference(Output(text=answer_text), tags=[CORRECT_TAG])
                    ]

                    # Create instance
                    instances.append(
                        Instance(
                            input=input_obj,
                            references=references,
                            split=split,
                            id=f"mars_{split}_{line_num}",
                            extra_data={
                                "example_h": example_h_id,
                                "example_t": example_t_id,
                                "question": question_id,
                                "answer": answer_id,
                                "relation": relation_id,
                                "mode": item.get('mode', 0)
                            }
                        )
                    )

        print(f"Loaded {len(instances)} MARS instances")
        print(f"  Train: {sum(1 for i in instances if i.split == TRAIN_SPLIT)}")
        print(f"  Valid: {sum(1 for i in instances if i.split == VALID_SPLIT)}")
        print(f"  Test: {sum(1 for i in instances if i.split == TEST_SPLIT)}")

        if self.use_images:
            print(f"Images path: {self.images_path}")
        else:
            print("Running in text-only mode (no images)")

        return instances
