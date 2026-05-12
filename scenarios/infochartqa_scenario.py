"""
HELM Scenario: InfoChartQA

Paper: https://arxiv.org/abs/2505.19028
Repo:  https://github.com/CoolDawnAnt/InfoChartQA
Data:  https://huggingface.co/datasets/Jietson/InfoChartQA

InfoChartQA evaluates multimodal question answering on infographic charts with
visual elements such as icons, pictograms, and metaphors. The released dataset
contains three benchmark slices:
  - text: chart-only factual and reasoning QA
  - visual_basic: QA that depends on cropped chart elements
  - visual_metaphor: QA about metaphorical visual design

The upstream README evaluates each item with:
  - input_image: the chart URL
  - extra_input_image: zero or more crops from extra_input_figure_bboxes
  - input_text: question + instructions
  - qtype-aware evaluation keyed by question_type_id

This HELM scenario mirrors that setup by downloading the chart locally,
materializing any released bbox crops as additional media inputs, and feeding
the chart image(s) plus question text through HELM's multimodal adapter.
"""

from __future__ import annotations

import hashlib
import os
import sys
from io import BytesIO
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import load_dataset
from PIL import Image, ImageFile

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject


ImageFile.LOAD_TRUNCATED_IMAGES = True


class InfoChartQAScenario(Scenario):
    """InfoChartQA with local chart downloads and bbox-derived extra figures."""

    name = "infochartqa"
    description = "Jietson/InfoChartQA"
    tags = ["creativity", "chart_qa", "multimodal", "vision"]

    VALID_SUBSETS = ("text", "visual_metaphor", "visual_basic", "all")

    def __init__(self, subset: str = "all"):
        super().__init__()
        requested_subset = (os.environ.get("INFOCHARTQA_SUBSET") or subset).strip() or "all"
        if requested_subset not in self.VALID_SUBSETS:
            raise ValueError(f"subset must be one of {self.VALID_SUBSETS}, got '{requested_subset}'")
        self.subset = requested_subset

        limit_hint = os.environ.get("INFOCHARTQA_INSTANCE_LIMIT_HINT", "").strip()
        self.instance_limit_hint = int(limit_hint) if limit_hint.isdigit() and int(limit_hint) > 0 else None

    @staticmethod
    def _normalize_url(url: str) -> str:
        normalized = (url or "").strip()
        while normalized.startswith("hhttp"):
            normalized = normalized[1:]
        if normalized and not normalized.startswith(("http://", "https://")):
            normalized = "https://" + normalized
        raw_prefix = "https://raw.githubusercontent.com/"
        if normalized.startswith(raw_prefix) and "/refs/heads/" in normalized:
            suffix = normalized[len(raw_prefix) :].split("/")
            if len(suffix) >= 5 and suffix[2] == "refs" and suffix[3] == "heads":
                normalized = raw_prefix + "/".join(suffix[:2] + [suffix[4]] + suffix[5:])
        return normalized

    @staticmethod
    def _sanitize_stem(name: str) -> str:
        stem = os.path.splitext(name)[0]
        stem = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return stem or "infochartqa_image"

    def _download_chart(self, url: str, target_path: str) -> str:
        if os.path.exists(target_path):
            return target_path

        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            payload = response.read()

        image = Image.open(BytesIO(payload))
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        image.save(target_path, format="PNG")
        return target_path

    def _materialize_images(self, item: dict, output_path: str) -> List[MediaObject]:
        images_dir = os.path.join(output_path, "images")
        crops_dir = os.path.join(output_path, "crops")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

        figure_ids = list(item.get("figure_id") or [])
        chart_id = figure_ids[0] if figure_ids else hashlib.sha1(item["url"].encode("utf-8")).hexdigest()
        chart_path = os.path.join(images_dir, f"{self._sanitize_stem(chart_id)}.png")

        chart_path = self._download_chart(self._normalize_url(item["url"]), chart_path)
        media_objects: List[MediaObject] = [MediaObject(content_type="image/png", location=chart_path)]

        bboxes = list(item.get("extra_input_figure_bboxes") or [])
        if not bboxes:
            return media_objects

        with Image.open(chart_path) as chart_image:
            for crop_index, bbox in enumerate(bboxes):
                if len(bbox) != 4:
                    continue

                x, y, width, height = [int(value) for value in bbox]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(chart_image.width, x + width)
                y2 = min(chart_image.height, y + height)
                if x2 <= x1 or y2 <= y1:
                    continue

                extra_ids = list(item.get("extra_input_figure_ids") or [])
                crop_id = (
                    extra_ids[crop_index]
                    if crop_index < len(extra_ids) and extra_ids[crop_index]
                    else f"{self._sanitize_stem(chart_id)}_crop_{crop_index}"
                )
                crop_path = os.path.join(crops_dir, f"{self._sanitize_stem(crop_id)}.png")
                if not os.path.exists(crop_path):
                    chart_image.crop((x1, y1, x2, y2)).save(crop_path, format="PNG")
                media_objects.append(MediaObject(content_type="image/png", location=crop_path))

        return media_objects

    @staticmethod
    def _build_prompt(item: dict, extra_image_count: int) -> str:
        question = str(item["question"]).replace("\\n", "\n").strip()
        instructions = str(item.get("instructions") or "").replace("\\n", "\n").strip()
        prompt_parts = []
        if extra_image_count:
            prompt_parts.append(
                "The first image is the full chart. Any additional images are cropped figures "
                "from that same chart, in the order referenced by the question."
            )
        prompt_parts.append(question)
        if instructions:
            prompt_parts.append(instructions)
        return "\n\n".join(part for part in prompt_parts if part)

    def get_instances(self, output_path: str) -> List[Instance]:
        splits = ["text", "visual_metaphor", "visual_basic"] if self.subset == "all" else [self.subset]
        instances: List[Instance] = []

        for split_name in splits:
            dataset = load_dataset("Jietson/InfoChartQA", split=split_name)
            for item in dataset:
                try:
                    media_objects = self._materialize_images(item, output_path)
                except (HTTPError, URLError, OSError, ValueError) as exc:
                    question_id = item.get("question_id", "unknown")
                    print(
                        f"InfoChartQAScenario: skipping {question_id} due to image preparation failure: {exc}",
                        file=sys.stderr,
                    )
                    continue

                prompt_text = self._build_prompt(item, extra_image_count=max(0, len(media_objects) - 1))
                media_objects.append(MediaObject(content_type="text/plain", text=prompt_text))

                instances.append(
                    Instance(
                        input=Input(multimedia_content=MultimediaObject(media_objects)),
                        references=[Reference(output=Output(text=str(item["answer"])), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                        id=str(item["question_id"]),
                        extra_data={
                            "question_type_id": int(item["question_type_id"]),
                            "question_type_name": str(item["question_type_name"]),
                            "split_name": split_name,
                            "difficulty": str(item["difficulty"]),
                            "chart_type": str(item["chart_type"]),
                            "bbox_count": len(item.get("extra_input_figure_bboxes") or []),
                            "figure_ids": list(item.get("figure_id") or []),
                        },
                    )
                )

                if self.instance_limit_hint is not None and len(instances) >= self.instance_limit_hint:
                    return instances

        return instances
