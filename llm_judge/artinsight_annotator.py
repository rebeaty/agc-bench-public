"""ArtInsight-specific multimodal judge annotator."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.request import Request

_DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
_BACKUP_JUDGE_MODEL = "openai/gpt-4o-mini"
_JUDGE_MODEL_OVERRIDE = os.environ.get("ARTINSIGHT_JUDGE_MODEL_OVERRIDE", "").strip() or None

_SYSTEM_PROMPT = (
    "There are some descriptions that have been written about pieces of artwork. "
    "This descriptions are designed to help blind parents understand their children's visual artwork. "
    "They are supposed to provide detailed, respectful descriptions of the artwork, focusing on descriptive "
    "aspects such as orientation, scenery, number of artifacts or figures, main colors, and themes and "
    "avoiding reductive or overly simplifying language that minimizes the child's effort and does not assume "
    "interpretations if uncertain. For example, the description should say, 'The person has a frown, and "
    "there are tears falling from their eyes' instead of 'The person appears to be sad.' The description "
    "should have a respectful, supportive, and engaging tone, encouraging open dialogue about the artwork. "
    "The descriptions should avoid making assumptions about names or identities based on any text in the artwork. "
    "The description should be in paragraph form.\n\n"
    "You are a scorer for these descriptions of artwork. Your job is to give each description a score on the "
    "scale of 0-16 based on the following criteria:\n\n"
    "There are 4 criteria and each of these criteria is 4 points, therefore giving the scale of 0-16.\n\n"
    "0 points indicates that the work does not meet the criteria at all. It is significantly lacking in key "
    "areas, with major components missing or entirely incorrect, showing little to no effort or understanding. "
    "1 point means that the work minimally meets the criteria. It addresses some aspects but is incomplete or "
    "contains several errors, demonstrating only a basic understanding of the required skills or knowledge. "
    "2 points indicate that the work partially meets the criteria. It covers most aspects but still has notable "
    "gaps or inaccuracies, showing a moderate understanding and application of the necessary skills or knowledge. "
    "3 points means that the work meets the criteria satisfactorily. Most components are present and correctly "
    "executed, with only minor errors, demonstrating a solid understanding and competent application of the "
    "required skills or knowledge. 4 points indicates that the work exceeds the criteria. It fully addresses "
    "and goes beyond the expectations, with all components well-developed and executed with high accuracy. "
    "This score demonstrates a deep understanding and proficient application of the required skills or knowledge, "
    "showing exceptional effort, creativity, and insight.\n\n"
    "1) Is the description being presumptive, i.e. when it doesn't know something is it making inferences or "
    "assumptions about what they could be? For ex: \"The main figure in the artwork is a large, dark gray shape "
    "in the center. It's hard to say for sure what it is, but it might be a person or animal.\" --> Ideally the "
    "description should just say, \"the main figure in the artwork is a large, dark gray shape in the center.\" "
    "(4 points)\n\n"
    "2) Is it being reductive, i.e. is it ever minimizing the effort or drawing style of the child? For ex, "
    "in the past I've had descriptions say things like, \"this is a drawing of simple stick figures\" where the "
    "parent has disliked the use of the word \"simple\". Or another example: \"this is a rough rectangle\" -- "
    "parents don't like it when descriptions use terms like 'rough' that diminish the work the child has put in. "
    "(4 points)\n\n"
    "3) Is it being too simple, i.e. only saying things like: \"This is a child's drawing of a forest and some "
    "animals\". Ideally the description goes into detail about the artwork. (4 points)\n\n"
    "4) Are all the major elements of the artwork captured? (4 points)\n\n"
    "There is also a miscellaneous section that can subtract points.\n"
    "5. Miscellaneous (Are there any other parts of the response which take away from the overall quality?)"
)

_FINAL_USER_PREFIX = "Score the description for this piece of artwork.\n\nArtwork:"
_FINAL_DESCRIPTION_PREFIX = "\n\nDescription:\n"

_FEWSHOT_EXAMPLES = [
    (
        "Score the description for this piece of artwork.\n\nArtwork:",
        "\n\nDescription:\nThis artwork is quite expressive and engaging! The main feature of this creation is a "
        "handprint in the center, boldly outlined in black and filled with a blend of yellow and grey colors. "
        "The fingers are distinctly marked, and the palm is highlighted with vibrant yellow, giving it a warm, "
        "glowing effect.\n\nAround the handprint, there are various streaks and splashes of color which include "
        "shades of red, pink, and green. These colors seem to be dynamically brushed around, suggesting a sense "
        "of movement or perhaps the joy associated with creation.\n\nIn addition to the handprint and color "
        "splashes, there is text written in a playful, childlike handwriting using a pinkish-purple hue. The "
        "letters \"HBD\" are visible, typically an abbreviation for \"Happy Birthday,\" which might suggest that "
        "this artwork was intended as a birthday greeting or celebration.\n\nThe overall impression of the artwork "
        "is lively and colorful, radiating a sense of personal touch and happiest moments. The use of the handprint "
        "could symbolize identity or personal expression, while the surrounding colors enhance the festive, celebratory "
        "vibe of the piece.",
        "1. Is the description being presumptive? (2/4)\n[The description is partly factual but makes multiple "
        "interpretations such as \"suggesting a sense of movement or perhaps the joy associated with creation\" "
        "and \"could symbolize identity or personal expression\". It also presumes \"HBD\" stands for \"Happy "
        "Birthday\" and that the drawing might be a birthday greeting. These presume the artist's intent.]\n\n"
        "2. Is it being reductive? (4/4)\n[The description is respectful and does not minimize or reduce the "
        "child's effort or style. It uses positive language like \"expressive,\" \"engaging,\" and \"vibrant.\"]\n\n"
        "3. Is it being too simple? (4/4)\n[The description is detailed and goes beyond a simplistic overview. "
        "It covers colors, shapes, and possible interpretations without being too vague or brief]\n\n"
        "4. Are all the major elements captured? (2/4)\n[The description does not mention all the elements such "
        "as the crumpled texture of the paper, the specific red and pink streaks on the left, or the color placement "
        "more accurately. This omission means not all significant elements are captured.]\n\n"
        "5. Miscellaneous (0)\n[No points subtracted.]\n\nTotal Score: (12/16)",
    ),
    (
        "Score the description for this piece of artwork.\n\nArtwork:",
        "\nDescription:\nThis vibrant artwork displays a bold handprint at its center, using deep black paint "
        "against a crisp white background. The fingers of the handprint are distinct, and within the palm, "
        "there's a splash of brilliant yellow, adding a vivid burst of color. Surrounding this central motif "
        "are playful strokes and dabs of pink and red paint that give a sense of dynamic motion to the piece."
        "Interestingly, at the corners of the paper, there are smudges of green paint that provide a slight "
        "contrast to the overall warm tones. The artwork also includes handwritten text in magenta around the "
        "handprint. The letters \"HBD\" are repeated, hinting that this piece might have been created as a joyful "
        "birthday greeting. The artwork appears to be crafted with enthusiasm, each color and stroke contributing "
        "to an expression of celebration and personal touch. This piece exudes a cheerful vibe and suggests a "
        "festive intention, perhaps as a unique and heartfelt gift.",
        "1. Is the description being presumptive? (3/4)\n[While the description is mostly detailed, it makes "
        "assumptions about the intention behind the artwork, such as suggesting it was created for a \"joyful "
        "birthday greeting\" and that it \"exudes a cheerful vibe\" or suggests \"a festive intention\". These "
        "are presumptive statements about the artist's intent.]\n\n"
        "2. Is it being reductive? (4/4)\n[The description is respectful and acknowledges the child's effort and "
        "creativity. Terms like \"playful strokes,\" \"vivid burst,\" and \"crafted with enthusiasm\" highlight "
        "the care and thought put into the artwork without being dismissive.]\n\n"
        "3. Is it being too simple? (4/4)\n[The description is thorough, covering various elements, colors, and "
        "even finer details like the smudges of green paint at the corners, providing a well-rounded understanding "
        "of the artwork.]\n\n"
        "4. Are all the major elements captured? (3/4)\n[The description captures most major elements, including "
        "the handprint, colors, and general texture, but it could mention the crumpled texture of the paper and "
        "possibly the exact positioning of the handprint and text for more completeness.]\n\n"
        "5. Miscellaneous (0)\n[No points subtracted.]\n\nTotal Score: (14/16)",
    ),
    (
        "Score the description for this piece of artwork.\n\nArtwork:\n\nDescription:\nScore the description for "
        "this piece of artwork.\n\nArtwork:",
        "\n\nDescription:\nYour child's artwork features a vibrant and colorful handprint at its center. The "
        "handprint is primarily outlined with black paint, filled in with a yellow color in the palm area, and "
        "some red accents on the fingers. Surrounding the handprint, there are various freeform streaks and "
        "splashes of colors including pink, green, and red. These brushstrokes give the artwork a lively feel. "
        "Additionally, below the handprint, there are some letters painted in purple and red, though they appear "
        "somewhat abstract and might be intended to convey a creative message or initials. The background paper "
        "is slightly crumpled, which adds texture to the piece. Overall, the artwork combines bold lines and "
        "energetic splashes of color, showcasing your child's creativity and enthusiasm.",
        "1. Is the description being presumptive? (4/4)\n[The description avoids making assumptions about the "
        "intent behind the artwork. Instead of assuming meanings, it describes what is visible without inferring "
        "underlying messages or reasons such as \"might be intended to convey a creative message or initials.\"]\n\n"
        "2. Is it being reductive? (4/4)\n[The description is respectful and acknowledges the child's creativity "
        "enthusiastically, using positive terms like \"vibrant,\" \"colorful,\" \"lively feel,\" and \"showcasing "
        "your child's creativity and enthusiasm.\" It does not use any diminishing language.]\n\n"
        "3. Is it being too simple? (3/4)\n[The description doesn't understand the letters and instead takes a "
        "guess at it might be intended to convey, which it should not do.]\n\n"
        "4. Are all the major elements captured? (2/4)\n[The description captures all the major elements: the "
        "handprint, colors, strokes, letters, and the crumpled texture of the paper, providing a thorough overview "
        "of the artwork. However, it is not able to understand the text in the image.]\n\n"
        "5. Miscellaneous (-1)\n[Not able to read text in the image.]\n\nTotal Score: (12/16)",
    ),
    (
        "Score the description for this piece of artwork.\n\nArtwork:",
        "\n\nDescription:\nYour child's artwork features a vibrant and colorful handprint at its center. The "
        "handprint is primarily outlined with black paint, filled in with a yellow color in the palm area, and "
        "some red accents on the fingers. Surrounding the handprint, there are various freeform streaks and "
        "splashes of colors including pink, green, and red. These brushstrokes give the artwork a lively feel. "
        "Additionally, below the handprint, there are some letters painted in purple and red, though they appear "
        "somewhat abstract and might be intended to convey a creative message or initials. The background paper "
        "is slightly crumpled, which adds texture to the piece. Overall, the artwork combines bold lines and "
        "energetic splashes of color, showcasing your child's creativity and enthusiasm.",
        "1. Is the description being presumptive? (4/4)\n[The description avoids making assumptions.]\n\n"
        "2. Is it being reductive? (4/4)\n[The description is respectful and celebrates the child's effort and "
        "creativity with positive language like \"vibrant,\" \"colorful,\" \"lively feel,\" \"bold lines,\" and "
        "\"energetic splashes of color.\" It does not use any diminishing terms.]\n\n"
        "3. Is it being too simple? (4/4)\n[The description is detailed and covers several aspects of the artwork, "
        "such as colors, brushstrokes, letters, and texture.]\n\n"
        "4. Are all the major elements captured? (4/4)\n[The description captures most of the major elements like "
        "the handprint and surrounding colors.]\n\n"
        "5. Miscellaneous (0)\n[No points subtracted.]\n\nTotal Score: (16/16)",
    ),
    (
        "Score the description for this piece of artwork.\n\nArtwork:",
        "\n\nDescription:\nYour child made a colorful and interesting artwork! The main figure in the artwork is "
        "a large, dark gray shape in the center. It's hard to say for sure what it is, but it might be a person "
        "or animal. The figure is surrounded by other colorful shapes and lines in various shades of pink, red, "
        "and yellow. There are also three thin, vertical lines in green, orange, and purple. At the top of the "
        "artwork, there is the word \"HOR\" written in purple, which might be part of a word. There is a lot of "
        "white space in the artwork, and the paper appears to be slightly wrinkled. Do you know what your child "
        "was thinking about when they made this artwork?",
        "1. Is the description being presumptive? (2/4)\n[The description makes assumptions about the main figure, "
        "suggesting it might be a \"person or animal.\" It also assumes the word \"HOR\" might be part of a word, "
        "both of which are unnecessary and speculative.]\n\n"
        "2. Is it being reductive? (3/4)\n[The description is mostly respectful but uses terms like \"it's hard "
        "to say for sure,\" which could come across as dismissive. Offering a clearer description of the shape "
        "would be more respectful of the child's effort.]\n\n"
        "3. Is it being too simple? (3/4)\n[The description covers several elements but lacks depth in describing "
        "the colors and various brushstrokes. It mentions colors and shapes but doesn't provide a comprehensive "
        "view of the artwork's vibrancy or details of each element.]\n\n"
        "4. Are all the major elements captured? (1/4)\n[The description captures most major elements but misses "
        "the detail about the specific letters \"HBD\" in the bottom part of the artwork, which are clearer than "
        "the presumed \"HOR\" at the top. The HOR was read wrong. It is HBD.]\n\n"
        "5. Miscellaneous (-2)\n[The phrase \"Do you know what your child was thinking about when they made this "
        "artwork?\" detracts from the assessment and introduces a subjective element. The description should not "
        "ask questions. It also read the HBD letters wrong as HOR instead.]\n\nTotal Score: (7/16)",
    ),
]

_TOTAL_SCORE_RE = re.compile(r"Total Score:\s*\(([-]?\d+)\s*/\s*16\)", re.IGNORECASE)
_CRITERION_RES = {
    "artinsight_presumptive_score": re.compile(r"1\..*?\(([-]?\d+)\s*/\s*4\)", re.DOTALL),
    "artinsight_reductive_score": re.compile(r"2\..*?\(([-]?\d+)\s*/\s*4\)", re.DOTALL),
    "artinsight_detail_score": re.compile(r"3\..*?\(([-]?\d+)\s*/\s*4\)", re.DOTALL),
    "artinsight_elements_score": re.compile(r"4\..*?\(([-]?\d+)\s*/\s*4\)", re.DOTALL),
    "artinsight_misc_deduction": re.compile(r"5\..*?Miscellaneous\s*\(([-]?\d+)\)", re.DOTALL | re.IGNORECASE),
}


def _parse_judge_output(text: str) -> Dict[str, float] | None:
    """Parse the ArtInsight rubric scoresheet. Each criterion is bounded
    (0-4 for criteria 1-4, total 0-16, misc deduction free-form negative).
    Returns None on any missing match OR out-of-range integer; the caller
    converts None to -100 sentinels so we never silently ascribe a 0 score
    to a parse failure (0 is a legitimate rubric value)."""
    total_match = _TOTAL_SCORE_RE.search(text)
    if total_match is None:
        return None
    total = int(total_match.group(1))
    if not (-4 <= total <= 16):  # Allow some negative slack for misc deductions
        return None

    parsed: Dict[str, float] = {"artinsight_score": float(total)}
    for metric_name, pattern in _CRITERION_RES.items():
        match = pattern.search(text)
        if match is None:
            return None
        value = int(match.group(1))
        if metric_name == "artinsight_misc_deduction":
            # Misc is a deduction (typically <=0), allow modest range
            if not (-8 <= value <= 0):
                return None
        else:
            if not (0 <= value <= 4):
                return None
        parsed[metric_name] = float(value)
    return parsed


class ArtInsightAnnotator(Annotator):
    """Run the upstream-style ArtInsight multimodal scorer through HELM."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str = _DEFAULT_JUDGE_MODEL,
        judge_temperature: float = 0.0,
        judge_max_new_tokens: int = 1024,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self._allow_backup = _JUDGE_MODEL_OVERRIDE is None
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.name = "artinsight_judge"

    def _build_multimodal_prompt(self, target_image_path: str, description: str) -> MultimediaObject:
        anchor_image_path = str(Path(target_image_path).with_name("IMG_2272.jpeg"))
        media_objects: List[MediaObject] = [MediaObject(content_type="text/plain", text=_SYSTEM_PROMPT)]
        for index, (before_image_text, after_image_text, assistant_text) in enumerate(_FEWSHOT_EXAMPLES, start=1):
            media_objects.extend(
                [
                    MediaObject(
                        content_type="text/plain",
                        text=(
                            f"\n\nExample {index}\n"
                            f"{before_image_text}\n"
                            "Reference artwork:\n"
                        ),
                    ),
                    MediaObject(content_type="image/jpeg", location=anchor_image_path),
                    MediaObject(
                        content_type="text/plain",
                        text=(
                            f"{after_image_text}\n\n"
                            "Reference score:\n"
                            f"{assistant_text}"
                        ),
                    ),
                ]
            )

        media_objects.extend(
            [
                MediaObject(
                    content_type="text/plain",
                    text="\n\nNow score the target description using the same rubric.\n\nArtwork:\n",
                ),
                MediaObject(content_type="image/jpeg", location=target_image_path),
                MediaObject(
                    content_type="text/plain",
                    text=(
                        f"{_FINAL_DESCRIPTION_PREFIX}{description}\n\n"
                        "Return the same criterion-by-criterion format and final total score."
                    ),
                ),
            ]
        )
        return MultimediaObject(media_objects=media_objects)

    def _call_judge(self, model_name: str, target_image_path: str, description: str) -> str:
        request = Request(
            model=model_name,
            model_deployment=model_name,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
            multimodal_prompt=self._build_multimodal_prompt(target_image_path, description),
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}: {result.error}")
        return result.completions[0].text.strip()

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text.strip()
        multimedia = request_state.instance.input.multimedia_content
        # 0 is a valid rubric score, so use -100 sentinel for parse failures
        # / missing inputs to distinguish from a legitimate "criteria not met."
        parse_failed = {
            "artinsight_score": -100.0,
            "artinsight_presumptive_score": -100.0,
            "artinsight_reductive_score": -100.0,
            "artinsight_detail_score": -100.0,
            "artinsight_elements_score": -100.0,
            "artinsight_misc_deduction": -100.0,
        }
        if multimedia is None or not multimedia.media_objects:
            return parse_failed

        image_path = None
        for media_object in multimedia.media_objects:
            if media_object.is_type("image"):
                image_path = media_object.location
                break
        if image_path is None:
            return parse_failed

        parsed = None
        try:
            parsed = _parse_judge_output(self._call_judge(self.judge_model_name, image_path, completion))
        except Exception:
            parsed = None

        if parsed is None and self._allow_backup and self.judge_model_name != _BACKUP_JUDGE_MODEL:
            try:
                parsed = _parse_judge_output(self._call_judge(_BACKUP_JUDGE_MODEL, image_path, completion))
            except Exception:
                parsed = None

        if parsed is None:
            return parse_failed

        return parsed
