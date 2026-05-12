"""
HELM Scenario: ESP Dataset (Example Styled Prompts)

Paper: Fusing Pre-Trained Language Models with Multimodal Prompts through Reinforcement Learning
        Youngjae Yu*, Jiwan Chung*, et al. (* equal contribution)
        CVPR 2023
        https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Fusing_Pre-Trained_Language_Models_With_Multimodal_Prompts_Through_Reinforcement_Learning_CVPR_2023_paper.html

Code: https://github.com/JiwanChung/esper

Dataset: 996 images from COCO 2014 validation set, each with captions in multiple styles:
  - caption_sns: Social media style
  - caption_blog: Blog post style
  - caption_news: News article style
  - caption_story: Story/narrative style
  - caption_instruction: Instructional style

Task: Given an image and a target style, generate text in that specified style.

Prompt format (based on code patterns in train/code/finetune*.py):
  [Image]
  {style}:

  Where {style} is one of: sns, blog, news, story, instruction

Evaluation: Open-ended generation evaluated with BLEU-4, METEOR, CIDEr
            (standard caption metrics)

Fields used: caption_sns, caption_blog, caption_news, caption_story, caption_instruction,
             image_id (mapped to COCO URLs)

Note: Not all images have all 5 caption styles. Instances are created only for available styles.
      This benchmark evaluates stylistic adaptation and creative text generation rather than
      traditional ideational creativity.
"""

import json
import os
from typing import List, Optional
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
    CORRECT_TAG,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class ESPDatasetScenario(Scenario):
    """
    ESP (Example Styled Prompts) dataset for style-conditioned image captioning.

    Evaluates models' ability to generate text in different styles (social media,
    blog, news, story, instruction) for the same image.
    """

    name = "esp_dataset"
    description = "JiwanChung/esper"  # GitHub repo as data source
    tags = ["creativity", "multimodal", "vision", "style-transfer", "text-generation"]

    # Style mappings
    STYLE_FIELDS = {
        "sns": "caption_sns",
        "blog": "caption_blog",
        "news": "caption_news",
        "story": "caption_story",
        "instruction": "caption_instruction",
    }
    INSTANCE_LIMIT_ENV = "ESP_DATASET_INSTANCE_LIMIT"

    def __init__(self, styles: Optional[List[str]] = None):
        """
        Args:
            styles: List of styles to include. If None, includes all available styles.
                   Options: ["sns", "blog", "news", "story", "instruction"]
        """
        super().__init__()
        self.styles = styles if styles else list(self.STYLE_FIELDS.keys())

        # Validate styles
        invalid_styles = set(self.styles) - set(self.STYLE_FIELDS.keys())
        if invalid_styles:
            raise ValueError(f"Invalid styles: {invalid_styles}. "
                           f"Valid options: {list(self.STYLE_FIELDS.keys())}")

    def download_dataset(self, output_path: str) -> str:
        """Download the ESP dataset JSON file."""
        dataset_url = "https://raw.githubusercontent.com/JiwanChung/esper/master/data/dataset_v_0_2.json"
        dataset_path = os.path.join(output_path, "dataset_v_0_2.json")

        ensure_file_downloaded(
            source_url=dataset_url,
            target_path=dataset_path,
        )

        return dataset_path

    def load_dataset(self, dataset_path: str) -> tuple:
        """Load and parse the ESP dataset."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Build image_id to URL mapping
        image_map = {}
        for img in data['images']:
            image_map[img['id']] = img['coco_url']

        return data['annotations'], image_map

    def _cache_image(self, output_path: str, image_id: int, image_url: str) -> str:
        image_dir = os.path.join(output_path, "images")
        ensure_directory_exists(image_dir)
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ensure_file_downloaded(source_url=image_url, target_path=image_path)
        return image_path

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Generate instances for the ESP dataset.

        Creates one instance per (image, style) pair where that style caption exists.
        """
        # Download dataset
        dataset_path = self.download_dataset(output_path)

        # Load data
        annotations, image_map = self.load_dataset(dataset_path)

        instances = []
        instance_id = 0
        instance_limit_raw = os.environ.get(self.INSTANCE_LIMIT_ENV)
        instance_limit = int(instance_limit_raw) if instance_limit_raw else None

        for ann in annotations:
            image_id = ann['image_id']

            # Skip if image URL not found
            if image_id not in image_map:
                continue

            image_url = image_map[image_id]
            image_path = self._cache_image(output_path, image_id, image_url)

            # Create instances for each requested style that has a caption
            for style in self.styles:
                field_name = self.STYLE_FIELDS[style]

                # Skip if this style caption doesn't exist for this image
                if field_name not in ann or not ann[field_name]:
                    continue

                caption = ann[field_name].strip()

                # Create multimodal prompt: image + style prefix
                multimedia_content = MultimediaObject([
                    MediaObject(
                        content_type="image/jpeg",
                        location=image_path
                    ),
                    MediaObject(
                        content_type="text/plain",
                        text=(
                            "\nDescribe this image in "
                            f"{style} style. Return only the {style}-style caption."
                        )
                    )
                ])

                # Reference is the styled caption (marked CORRECT so reference metrics can compute)
                references = [
                    Reference(Output(text=caption), tags=[CORRECT_TAG])
                ]

                instances.append(
                    Instance(
                        input=Input(multimedia_content=multimedia_content),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"esp_{instance_id}_{style}_{image_id}"
                    )
                )
                instance_id += 1
                if instance_limit is not None and instance_id >= instance_limit:
                    return instances

        return instances
