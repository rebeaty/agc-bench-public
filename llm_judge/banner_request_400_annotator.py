"""BannerRequest400-specific multimodal image judge annotator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.request import Request

from llm_judge.banner_request_400_renderer import parse_blueprint, render_blueprint_to_png

_DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
_JUDGE_MODEL_OVERRIDE = os.environ.get("BANNER_REQUEST_400_JUDGE_MODEL_OVERRIDE", "").strip() or None

_RUBRICS: Dict[str, str] = {
    "banner_request_400_taa": (
        "Definition: Measures how well the generated banner ad aligns with the given request, "
        "including the theme, target audience, and primary purpose.\n"
        "Instructions for Scoring:\n"
        "5 - Perfectly aligns with the request (theme, audience, purpose are all clearly reflected).\n"
        "4 - Mostly aligns, but minor details could be improved.\n"
        "3 - Somewhat aligns, but key elements are missing or unclear.\n"
        "2 - Barely aligns, with major missing or incorrect elements.\n"
        "1 - Does not align with the request at all.\n"
        "Justification Required: Explain how well the banner captures the requested theme and audience."
    ),
    "banner_request_400_lps": (
        "Definition: Evaluates whether the logo is well-integrated into the design in terms of visibility, "
        "size, and positioning.\n"
        "Instructions for Scoring:\n"
        "5 - Logo is well-placed, clearly visible, proportionate, and blends seamlessly.\n"
        "4 - Logo is well-placed but could be slightly improved.\n"
        "3 - Logo is visible but not ideally placed.\n"
        "2 - Logo placement is poor.\n"
        "1 - Logo is either missing or completely misplaced.\n"
        "Justification Required: Explain how the logo is positioned and whether it contributes to brand identity."
    ),
    "banner_request_400_aqs": (
        "Definition: Measures the visual appeal, including color harmony, layout balance, typography, "
        "and overall design quality.\n"
        "Instructions for Scoring:\n"
        "5 - Visually outstanding, professional design, well-balanced, with harmonious colors and readable text.\n"
        "4 - Well-designed, but small refinements could enhance it.\n"
        "3 - Acceptable but has notable design flaws.\n"
        "2 - Visually weak, with noticeable design mistakes.\n"
        "1 - Poor design, lacks professionalism or coherence.\n"
        "Justification Required: Explain what makes the design appealing or unappealing."
    ),
    "banner_request_400_ctae": (
        "Definition: Evaluates whether the Call-to-Action (CTA) is clear, engaging, and visually emphasized.\n"
        "Instructions for Scoring:\n"
        "5 - CTA is clear, compelling, well-placed, and visually prominent.\n"
        "4 - CTA is effective but could be slightly improved.\n"
        "3 - CTA is present but lacks emphasis or clarity.\n"
        "2 - CTA is weak, hard to notice, or poorly worded.\n"
        "1 - No clear CTA is present.\n"
        "Justification Required: Explain how effective the CTA is in prompting user action."
    ),
    "banner_request_400_cpyq": (
        "Definition: Evaluates the effectiveness of the headline, subheadline, and any other text in the banner ad, "
        "focusing on clarity, readability, persuasiveness, and grammatical correctness.\n"
        "Instructions for Scoring:\n"
        "5 - Copy is clear, engaging, grammatically correct, and persuasive, making the message effective.\n"
        "4 - Copy is well-written but could be slightly improved.\n"
        "3 - Copy is somewhat effective but has issues in clarity, grammar, or persuasiveness.\n"
        "2 - Copy is weak, hard to read, contains noticeable grammatical mistakes, or lacks impact.\n"
        "1 - Copy is unclear, irrelevant, or difficult to read due to poor design or bad wording.\n"
        "Justification Required: Is the copy easy to read against the background? Does it match the banner's "
        "purpose and target audience? Is it persuasive and action-driven? Are there grammatical or spelling errors?"
    ),
    "banner_request_400_bis": (
        "Definition: Measures how well the banner ad visually and stylistically aligns with the brand's identity "
        "beyond just logo placement.\n"
        "Instructions for Scoring:\n"
        "5 - Strong brand consistency; the banner design aligns well with the provided logo and conveys a "
        "recognizable brand identity.\n"
        "4 - Mostly aligns, but minor refinements could improve brand consistency.\n"
        "3 - Somewhat aligns, but noticeable inconsistencies exist.\n"
        "2 - Weak brand alignment, only the logo represents the brand while other design choices feel unrelated.\n"
        "1 - No brand identity is reflected; the banner appears generic or disconnected from the brand.\n"
        "Justification Required: Are the colors and typography in line with the brand's usual style? Does the "
        "overall aesthetic feel like it belongs to the brand, or does it look generic?"
    ),
}

_SYSTEM_PROMPT = (
    "You are an expert in advertising design, marketing, and visual communication. "
    "Your task is to evaluate a banner ad image based on the following principle given the advertiser's "
    "logo and banner request. You should rate on a scale of 1 to 5, where 1 is poor and 5 is excellent. "
    "You should also provide a brief justification for your score.\n\n"
    "{score_principle}\n\n"
    "Please start evaluating the banner ad image.\n"
    "Output your answer in the format of "
    '{{"score": 1, "explanation": "explain concisely why you gave this score"}}'
)


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_score(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate the JSON judge payload. Score must be an int in [1,5];
    bare floats/strings are rejected to surface format drift rather than
    coerce a junk value to a clamped score."""
    score = parsed.get("score")
    explanation = parsed.get("explanation")
    # Accept int or float that round-trips to int (e.g., 3.0)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    if isinstance(score, float) and not score.is_integer():
        return None
    score_int = int(score)
    if not (1 <= score_int <= 5):
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    return {"score": float(score_int), "explanation": explanation.strip()}


class BannerRequest400Annotator(Annotator):
    """Evaluate rendered BannerRequest400 outputs with the official image-based rubrics."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str = _DEFAULT_JUDGE_MODEL,
        judge_temperature: float = 0.3,
        judge_max_new_tokens: int = 512,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.name = "banner_request_400_judge"

    def _render_output(self, instance_id: str, completion: str, logo_path: str) -> tuple[Optional[str], bool]:
        blueprint = parse_blueprint(completion)
        if blueprint is None:
            return None, False

        digest = hashlib.sha1(f"{instance_id}\n{completion}".encode("utf-8")).hexdigest()[:12]
        render_dir = Path("benchmark_output/rendered_banners/banner_request_400")
        render_path = render_dir / f"{instance_id}_{digest}.png"
        success = render_blueprint_to_png(blueprint, logo_path, str(render_path))
        return (str(render_path) if success else None), success

    def _build_multimodal_prompt(self, rubric: str, banner_image_path: str, logo_path: str, banner_request: str) -> MultimediaObject:
        return MultimediaObject(
            media_objects=[
                MediaObject(content_type="text/plain", text=_SYSTEM_PROMPT.format(score_principle=rubric)),
                MediaObject(content_type="text/plain", text="\n\nThe banner image to be evaluated is:\n"),
                MediaObject(content_type="image/png", location=banner_image_path),
                MediaObject(content_type="text/plain", text="\n\nFor your reference, the advertiser logo is:\n"),
                MediaObject(content_type="image/png", location=logo_path),
                MediaObject(
                    content_type="text/plain",
                    text=f"\n\nFor your reference, the advertiser banner request is: {banner_request}",
                ),
            ]
        )

    def _call_judge(self, rubric: str, banner_image_path: str, logo_path: str, banner_request: str) -> Optional[Dict[str, Any]]:
        request = Request(
            model=self.judge_model_name,
            model_deployment=self.judge_model_name,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
            multimodal_prompt=self._build_multimodal_prompt(rubric, banner_image_path, logo_path, banner_request),
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            return None
        parsed = _extract_json_dict(result.completions[0].text.strip())
        if parsed is None:
            return None
        return _normalize_score(parsed)

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text.strip()
        extra_data = request_state.instance.extra_data or {}
        logo_path = extra_data.get("logo_path", "")
        banner_request = extra_data.get("banner_request", "")

        render_path, render_success = self._render_output(request_state.instance.id, completion, logo_path)
        # 1 is the minimum legitimate rubric score, so use -100 sentinels for
        # parse / render failures rather than 0.0 (which would be an
        # impossibly-low score that pollutes downstream means).
        base_output: Dict[str, Any] = {
            "banner_request_400_blueprint_validity": 1.0 if parse_blueprint(completion) is not None else 0.0,
            "banner_request_400_render_success": 1.0 if render_success else 0.0,
            "banner_request_400_render_path": render_path or "",
            "banner_request_400_valid_judge_rate": 0.0,
            "banner_request_400_score": -100.0,
        }
        for metric_name in _RUBRICS:
            base_output[metric_name] = -100.0
            base_output[f"{metric_name}_explanation"] = ""

        if not render_success or not render_path:
            return base_output

        valid_scores = []
        for metric_name, rubric in _RUBRICS.items():
            judged = self._call_judge(rubric, render_path, logo_path, banner_request)
            if judged is None:
                # Leave the -100 sentinel in place; do not coerce to 0.
                continue
            base_output[metric_name] = judged["score"]
            base_output[f"{metric_name}_explanation"] = judged["explanation"]
            valid_scores.append(judged["score"])

        if valid_scores:
            base_output["banner_request_400_valid_judge_rate"] = len(valid_scores) / len(_RUBRICS)
            # Mean only over the metrics that actually parsed; -100 sentinels
            # would otherwise drag the aggregate negative.
            base_output["banner_request_400_score"] = sum(valid_scores) / len(valid_scores)

        return base_output
