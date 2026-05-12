"""Creation-MMBench-specific multimodal judge annotator."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import BACKUP_JUDGE_MODEL

_DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
_JUDGE_MODEL_OVERRIDE = os.environ.get("CREATION_MMBENCH_JUDGE_MODEL_OVERRIDE", "").strip() or None

_SYSTEM_PROMPT = """You are an expert judge for Creation-MMBench.

You will be shown the task images, the task question, the instance-specific
criteria, the model response, and a reference response.

Evaluate the MODEL RESPONSE against the question, images, criteria, and
reference response.

Return only a compact JSON object with these keys:
- "model_vfs": integer from 1 to 10
- "reward": integer from -100 to 100
- "reason": short string

Scoring guidance:
- "model_vfs" is the visual factuality / creative-quality score for the model response.
- "reward" should be positive when the model response is better than the reference response.
- Use the instance-specific criteria exactly as written.
- Be conservative and do not invent details not supported by the images or prompt.
"""

_NUMBER_RE = r"(-?\d+(?:\.\d+)?)"


def _extract_text_and_images(multimedia: MultimediaObject | None) -> Tuple[str, List[str]]:
    if multimedia is None or not multimedia.media_objects:
        return "", []

    text_parts: List[str] = []
    image_paths: List[str] = []
    for media_object in multimedia.media_objects:
        if media_object.is_type("image") and media_object.location:
            image_paths.append(media_object.location)
        elif media_object.is_type("text") and media_object.text:
            text_parts.append(media_object.text.strip())
    return "\n".join(part for part in text_parts if part), image_paths


def _extract_reference_and_criteria(request_state: RequestState) -> Tuple[str, str]:
    reference_text = ""
    criteria_text = ""
    for reference in request_state.instance.references:
        if "reference_answer" in reference.tags and not reference_text:
            reference_text = reference.output.text.strip()
        if "evaluation_criteria" in reference.tags and not criteria_text:
            criteria_text = reference.output.text.strip()
    if not reference_text and request_state.instance.references:
        reference_text = request_state.instance.references[0].output.text.strip()
    return reference_text, criteria_text


def _build_prompt(
    question_text: str,
    image_paths: List[str],
    model_response: str,
    reference_response: str,
    criteria_text: str,
    swapped: bool,
) -> MultimediaObject:
    media_objects: List[MediaObject] = [MediaObject(content_type="text/plain", text=_SYSTEM_PROMPT)]

    prompt_text = ["Question:", question_text]
    if criteria_text:
        prompt_text.extend(["", "Instance criteria:", criteria_text])
    prompt_text.extend(["", "Score the MODEL RESPONSE relative to the REFERENCE RESPONSE."])
    prompt_text.extend(
        [
            "The MODEL RESPONSE is the answer to evaluate.",
            "The REFERENCE RESPONSE is the benchmark reference.",
            "Return JSON only.",
        ]
    )
    media_objects.append(MediaObject(content_type="text/plain", text="\n".join(prompt_text)))

    for index, image_path in enumerate(image_paths, start=1):
        media_objects.append(MediaObject(content_type="text/plain", text=f"\nImage {index}:"))
        media_objects.append(MediaObject(content_type="image/jpeg", location=image_path))

    if swapped:
        media_objects.append(
            MediaObject(
                content_type="text/plain",
                text=(
                    "\nReference response:\n"
                    f"{reference_response}\n\n"
                    "Model response:\n"
                    f"{model_response}\n\n"
                    "Remember: score the MODEL RESPONSE, even though it is shown second here.\n"
                    "Return only JSON."
                ),
            )
        )
    else:
        media_objects.append(
            MediaObject(
                content_type="text/plain",
                text=(
                    "\nModel response:\n"
                    f"{model_response}\n\n"
                    "Reference response:\n"
                    f"{reference_response}\n\n"
                    "Return only JSON."
                ),
            )
        )

    return MultimediaObject(media_objects=media_objects)


def _parse_scores(text: str) -> Dict[str, float] | None:
    """Parse the dual-pass judge JSON. Validates ranges (vfs in [1,10],
    reward in [-100,100]) and rejects out-of-range numbers as parse failures
    instead of silently clamping. Returns None on any failure; the caller
    converts None to -100 sentinels."""
    clean_text = text.strip()
    if not clean_text:
        return None

    json_candidate = clean_text
    if "```" in json_candidate:
        json_candidate = re.sub(r"^```(?:json)?", "", json_candidate, flags=re.IGNORECASE).strip()
        json_candidate = re.sub(r"```$", "", json_candidate).strip()

    parsed_json = None
    try:
        parsed_json = json.loads(json_candidate)
    except Exception:
        match = re.search(r"\{.*\}", clean_text, flags=re.DOTALL)
        if match is not None:
            try:
                parsed_json = json.loads(match.group(0))
            except Exception:
                parsed_json = None

    def _in_range(value: Any, lo: float, hi: float) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        f = float(value)
        if not (lo <= f <= hi):
            return None
        return f

    if isinstance(parsed_json, dict):
        model_vfs = parsed_json.get("model_vfs", parsed_json.get("vfs"))
        reward = parsed_json.get("reward")
        vfs_val = _in_range(model_vfs, 1.0, 10.0)
        reward_val = _in_range(reward, -100.0, 100.0)
        if vfs_val is not None and reward_val is not None:
            return {"model_vfs": vfs_val, "reward": reward_val}

    vfs_match = re.search(r"(?:model_vfs|vfs|visual_factuality_score)\s*[:=]\s*{num}".format(num=_NUMBER_RE), clean_text, flags=re.IGNORECASE)
    reward_match = re.search(r"reward\s*[:=]\s*{num}".format(num=_NUMBER_RE), clean_text, flags=re.IGNORECASE)
    if vfs_match and reward_match:
        vfs_val = _in_range(float(vfs_match.group(1)), 1.0, 10.0)
        reward_val = _in_range(float(reward_match.group(1)), -100.0, 100.0)
        if vfs_val is not None and reward_val is not None:
            return {"model_vfs": vfs_val, "reward": reward_val}

    return None


class CreationMMBenchAnnotator(Annotator):
    """Source-grounded dual-pass multimodal judge for Creation-MMBench."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str = _DEFAULT_JUDGE_MODEL,
        judge_temperature: float = 0.0,
        judge_max_new_tokens: int = 2048,
        **_: Any,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self._allow_backup = _JUDGE_MODEL_OVERRIDE is None
        self.name = "creation_mmbench_judge"

    def _call_judge(self, model_name: str, multimodal_prompt: MultimediaObject) -> str:
        request = Request(
            model=model_name,
            model_deployment=model_name,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
            multimodal_prompt=multimodal_prompt,
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}: {result.error}")
        return result.completions[0].text.strip() if result.completions else ""

    def _score_once(
        self,
        model_name: str,
        question_text: str,
        image_paths: List[str],
        model_response: str,
        reference_response: str,
        criteria_text: str,
        swapped: bool,
    ) -> Dict[str, float] | None:
        prompt = _build_prompt(
            question_text=question_text,
            image_paths=image_paths,
            model_response=model_response,
            reference_response=reference_response,
            criteria_text=criteria_text,
            swapped=swapped,
        )
        try:
            raw_text = self._call_judge(model_name, prompt)
        except Exception:
            return None
        parsed = _parse_scores(raw_text)
        if parsed is None:
            return None

        parsed["raw_text_length"] = float(len(raw_text))
        return parsed

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        model_response = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
        question_text, image_paths = _extract_text_and_images(request_state.instance.input.multimedia_content)
        reference_response, criteria_text = _extract_reference_and_criteria(request_state)

        # 1 is the minimum legitimate vfs and 0 is a real reward midpoint, so
        # use -100 sentinels for parse / input failures rather than 0.0.
        parse_failed = {
            "creation_mmbench_vfs": -100.0,
            "creation_mmbench_reward": -100.0,
            "creation_mmbench_dual_eval_gap": -100.0,
            "judge_parse_rate": 0.0,
        }
        if not model_response or not question_text or not image_paths or not reference_response:
            return parse_failed

        pass_scores: List[Dict[str, float]] = []
        for swapped in (False, True):
            parsed = self._score_once(
                self.judge_model_name,
                question_text,
                image_paths,
                model_response,
                reference_response,
                criteria_text,
                swapped=swapped,
            )
            if parsed is None and self._allow_backup and self.judge_model_name != BACKUP_JUDGE_MODEL:
                parsed = self._score_once(
                    BACKUP_JUDGE_MODEL,
                    question_text,
                    image_paths,
                    model_response,
                    reference_response,
                    criteria_text,
                    swapped=swapped,
                )
            if parsed is not None:
                pass_scores.append(parsed)

        if not pass_scores:
            return parse_failed

        # _parse_scores already enforces vfs in [1,10] and reward in [-100,100]
        # and rejects out-of-range values as parse failures, so no clamping
        # here (clamping would silently coerce a junk parse to a valid score).
        vfs_scores = [score["model_vfs"] for score in pass_scores]
        reward_scores = [score["reward"] for score in pass_scores]

        # Dual gap is only meaningful when both passes parsed; otherwise it
        # is undefined and we use the -100 sentinel.
        if len(reward_scores) >= 2:
            dual_gap = abs(reward_scores[0] - reward_scores[1])
        else:
            dual_gap = -100.0

        return {
            "creation_mmbench_vfs": sum(vfs_scores) / len(vfs_scores),
            "creation_mmbench_reward": sum(reward_scores) / len(reward_scores),
            "creation_mmbench_dual_eval_gap": dual_gap,
            "judge_parse_rate": len(pass_scores) / 2.0,
        }
