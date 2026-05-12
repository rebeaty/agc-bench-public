"""HELM scenario for ArtInsight artwork description generation."""

import os
import urllib.parse
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject

_REPO_RAW = (
    "https://raw.githubusercontent.com/makeabilitylab/ArtInsight/main"
    "/Artwork-Description-Scoring/images"
)

# All 30 artwork image filenames from the repository
_IMAGE_FILENAMES = [
    "Abstract_Work_Blue.jpeg",
    "Airplane.jpeg",
    "CountryBalls.jpeg",
    "IMG_2271.jpeg",
    "IMG_2272.jpeg",
    "IMG_2273.jpeg",
    "P6 IMG_0511.jpeg",
    "P6 IMG_0512.jpeg",
    "Vampire.jpeg",
    "Xmas_Artwork.jpeg",
    "abstract_blue_white_red.jpeg",
    "bulldog.jpeg",
    "centaur.jpeg",
    "craft_flower.jpeg",
    "creature_hello.jpeg",
    "imprints.jpeg",
    "love_monster.jpeg",
    "mandala.jpeg",
    "mandala_cat.jpeg",
    "mom-daughter.jpeg",
    "pandas.jpeg",
    "peeps.jpeg",
    "pink_flower.jpeg",
    "poms.jpeg",
    "rainbow.jpeg",
    "skull_face_jones.jpeg",
    "sunflower.jpeg",
    "turkey.jpeg",
    "two_people.jpeg",
    "xmas_tree.jpeg",
]

_PROMPT = (
    "Generate a descriptive description of the artwork in paragraph form (no bullets or numbered points). "
    "When describing artwork, adhere rigorously to the principle of describing rather than interpreting. "
    "Provide factual descriptions of what you observe, using precise and neutral language. Avoid inferring "
    "emotions, intentions, or identities, and refrain from suggesting what elements might be or could represent. "
    "For example, instead of saying The figure appears sad, describe the specific features you see, such as The "
    "figure's mouth is drawn as a downward curve, and there are blue vertical lines below the eyes. Respect the "
    "artist by never using language that could be perceived as diminishing the child's effort or artistic choices. "
    "Avoid terms like simple, rough, messy, or childish. Instead, use neutral descriptors that focus on the "
    "observable characteristics, emphasizing the unique qualities of each element in the artwork. Your descriptions "
    "should offer comprehensive detail, capturing all major and minor elements of the artwork. Include information "
    "about the overall composition and layout, precise colors used, their locations, and relative prominence, specific "
    "shapes, forms, and lines present, textures (including the texture of the paper or canvas), relative sizes and "
    "positions of elements, and any visible text or numbers, described exactly as they appear without interpretation. "
    "Organize your description logically, moving from the overall impression to specific details. Use clear, concise "
    "language that a blind parent can easily visualize. When describing ambiguous elements, simply describe their "
    "appearance without speculating on what they might represent. Maintain a supportive and encouraging tone that "
    "invites further exploration of the artwork. Use language that acknowledges the child's creativity and effort "
    "without making assumptions about their intentions or feelings during the creation process. Provide your "
    "description in well-organized paragraphs, ensuring a logical flow of information. Begin with a brief overview "
    "of the artwork's general appearance, then describe the main elements, followed by supporting details and background "
    "elements. Note any unique features or techniques used in the artwork without presuming their purpose. Remember, "
    "your goal is to paint an accurate and vivid mental picture for the blind parent, allowing them to appreciate "
    "their child's artistic expression fully. Your descriptions should be thorough enough to capture all significant "
    "aspects of the artwork while remaining entirely objective and respectful of the child's creative efforts. Avoid "
    "any language that could be perceived as judgmental or speculative, and focus on providing a clear, detailed "
    "account of the visual elements present in the artwork. Do not ask questions or suggest interpretations in your "
    "descriptions. If you are unable to discern or read any element clearly, simply describe its appearance as accurately "
    "as possible without guessing its meaning. Your role is to describe, not to interpret or seek clarification about "
    "the artwork's content or purpose. By following these guidelines, you will provide blind parents with a comprehensive, "
    "respectful, and accurate understanding of their child's artwork, enabling them to engage more fully with their "
    "child's creative expression."
)


class ArtInsightScenario(Scenario):
    """
    ArtInsight: describe children's artwork for blind/low-vision parents.

    30 instances, one per artwork image. Open-ended generation evaluated by
    LLM-as-judge on a 0-16 rubric (see annotator_notes.md).
    """

    name = "artinsight"
    description = "makeabilitylab/ArtInsight"
    tags = ["creativity", "multimodal", "vision", "description_generation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        images_dir = os.path.join(output_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        instances = []
        for filename in _IMAGE_FILENAMES:
            local_path = os.path.join(images_dir, filename)

            if not os.path.exists(local_path):
                # URL-encode spaces in filenames
                encoded = urllib.parse.quote(filename)
                url = f"{_REPO_RAW}/{encoded}"
                urllib.request.urlretrieve(url, local_path)

            multimedia_content = MultimediaObject([
                MediaObject(content_type="text/plain", text=_PROMPT),
                MediaObject(content_type="image/jpeg", location=local_path),
            ])

            # Open-ended generation — no gold text reference; judge evaluates output
            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=[],
                split=TEST_SPLIT,
                id=f"artinsight_{os.path.splitext(filename)[0]}",
                extra_data={"image_filename": filename},
            ))

        return instances
