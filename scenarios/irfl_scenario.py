"""
HELM Scenario: IRFL (Image Recognition of Figurative Language)

Paper: https://arxiv.org/abs/2303.15445
Code: https://github.com/irfl-dataset/IRFL
Data: https://huggingface.co/datasets/lampent/IRFL
Project: https://irfl-dataset.github.io/

Task: Choose the image that best visualizes the meaning of a figurative expression.
Given a figurative phrase (idiom, metaphor, or simile) with its definition/context,
select which of 4 candidate images best represents the FIGURATIVE meaning (not literal).

Evaluation: Accuracy (4-way multiple choice)
Human performance: 97% (idioms), 99.7% (metaphors), 100% (similes)
Best model (2023): 22% (idioms), 30% (metaphors), 66% (similes)

Prompt format:
  Choose the image that best visualizes the meaning of the figurative expression: "{phrase}"
  Definition: {definition/query}

  [Image A]
  [Image B]
  [Image C]
  [Image D]

  Answer:

Images: 10,062 JPEG files downloaded from HuggingFace (IRFL_images.zip)
Image location: {output_path}/images/{uuid}.jpeg

Fields used: query, phrase, definition, answer, distractors, images_metadata
Fields skipped: None (figurative_type and type used for filtering)

Task types:
  - idiom-detection-task (200 examples)
  - metaphor-detection-task (333 examples)
  - simile-detection-task (277 examples)
"""

import json
import os
import random
import zipfile
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


class IRFLScenario(Scenario):
    """
    IRFL: Image Recognition of Figurative Language

    Multimodal benchmark for figurative language understanding.
    Models must select which image best visualizes a figurative expression
    from 4 candidates (figurative, partial literal, literal, or random).
    """

    name = "irfl"
    description = "lampent/IRFL"
    tags = ["creativity", "multimodal", "vision", "figurative-language"]

    # Configuration names on HuggingFace
    VALID_CONFIGS = [
        "idiom-detection-task",
        "metaphor-detection-task",
        "simile-detection-task",
        "open-simile-detection-task"
    ]

    def __init__(self, config: str = "idiom-detection-task"):
        """
        Args:
            config: Which detection task to load. Options:
                - "idiom-detection-task" (200 examples)
                - "metaphor-detection-task" (333 examples)
                - "simile-detection-task" (277 examples)
                - "open-simile-detection-task" (277 examples)
        """
        super().__init__()
        if config not in self.VALID_CONFIGS:
            raise ValueError(
                f"Invalid config '{config}'. Must be one of: {self.VALID_CONFIGS}"
            )
        self.config = config

    def _download_images(self, output_path: str) -> str:
        """
        Download and extract IRFL images if not already present.

        Returns:
            Path to the images directory
        """
        images_dir = os.path.join(output_path, "images")

        # Check if images already exist
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 10000:
            print(f"Images already exist at {images_dir}")
            return images_dir

        # Download the zip file
        import urllib.request
        zip_path = os.path.join(output_path, "IRFL_images.zip")

        if not os.path.exists(zip_path):
            print("Downloading IRFL images (1.6 GB)...")
            url = "https://huggingface.co/datasets/lampent/IRFL/resolve/main/IRFL_images.zip"
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete")

        # Extract the zip file
        print(f"Extracting images to {images_dir}...")
        os.makedirs(output_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print("Extraction complete")

        return images_dir

    def _format_prompt(self, phrase: str, query: str, definition: str, figurative_type: str) -> str:
        """
        Format the prompt based on task type and available information.

        Args:
            phrase: The figurative expression
            query: Context or definition from the query field
            definition: Explicit definition (may be same as query for idioms)
            figurative_type: 'idiom', 'metaphor', or 'simile'

        Returns:
            Formatted prompt text
        """
        prompt = f'Choose the image that best visualizes the meaning of the figurative expression: "{phrase}"\n\n'

        # For idioms, use explicit definition if available
        if figurative_type == "idiom" and definition:
            # Parse definition if it's a JSON list
            if definition.startswith("["):
                definition = json.loads(definition)[0]
            prompt += f"Definition: {definition}\n\n"
        # For metaphors and similes, use the query as context
        elif query and query != phrase:
            prompt += f"Context: {query}\n\n"

        prompt += "Select exactly one option and answer with a single letter: A, B, C, or D.\n\n"
        return prompt

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate IRFL instances for the specified configuration.

        Each instance contains:
        - Text prompt with the figurative expression and definition
        - 4 candidate images (1 correct, 3 distractors)
        - References A, B, C, D with correct answer tagged
        """
        # Download images if needed
        images_dir = self._download_images(output_path)

        # Load the dataset
        dataset = load_dataset(self.description, self.config, split="test")

        instances = []
        for idx, item in enumerate(dataset):
            # Parse the answer and distractors
            answer_uuid = json.loads(item['answer'])[0]
            distractor_uuids = json.loads(item['distractors'])

            # Deterministically shuffle answer and distractors so the benchmark
            # doesn't leak the correct answer position while remaining reproducible.
            all_uuids = [answer_uuid] + distractor_uuids
            shuffle_rng = random.Random(f"{self.config}:{idx}:{item['phrase']}")
            shuffle_rng.shuffle(all_uuids)
            correct_index = all_uuids.index(answer_uuid)

            # Format the text prompt
            text_prompt = self._format_prompt(
                phrase=item['phrase'],
                query=item['query'],
                definition=item.get('definition', ''),
                figurative_type=item['figurative_type']
            )

            # Create multimedia content: text + 4 images
            multimedia_elements = [
                MediaObject(content_type="text/plain", text=text_prompt)
            ]

            # Add the 4 images with labels
            for i, uuid in enumerate(all_uuids):
                letter = chr(65 + i)  # A, B, C, D
                image_path = os.path.join(images_dir, f"{uuid}.jpeg")

                # Add a text label for the choice
                multimedia_elements.append(
                    MediaObject(content_type="text/plain", text=f"\n{letter})")
                )

                # Add the image
                multimedia_elements.append(
                    MediaObject(
                        content_type="image/jpeg",
                        location=image_path
                    )
                )

            # Add final prompt text
            multimedia_elements.append(
                MediaObject(content_type="text/plain", text="\n\nAnswer:")
            )

            multimedia_content = MultimediaObject(multimedia_elements)

            # Build references: A, B, C, D (correct one is at index 0 = 'A')
            references = []
            for i in range(4):
                letter = chr(65 + i)
                is_correct = (i == correct_index)
                tags = [CORRECT_TAG] if is_correct else []
                references.append(
                    Reference(Output(text=letter), tags=tags)
                )

            # Create instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"{self.config}_{idx}",
                    extra_data={
                        "config": self.config,
                        "figurative_type": item["figurative_type"],
                        "correct_letter": chr(65 + correct_index),
                    },
                )
            )

        return instances
