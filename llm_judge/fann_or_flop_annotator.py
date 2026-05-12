"""Fann or Flop poem-level judge annotator."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)

_JUDGE_MODEL_OVERRIDE = os.environ.get("FANN_OR_FLOP_JUDGE_MODEL_OVERRIDE", "").strip() or None

_SYSTEM_PROMPT = """You are an expert Arabic linguist and literary evaluator.

Your task is to evaluate a full Arabic poem's verse-by-verse explanations. You will compare ground-truth
(human-written) explanations with generated explanations from an AI model.

Evaluate the generated explanation holistically across all verses and return a JSON object with three
scores from 1 to 5:

- faithfulness_score: Does the generated explanation faithfully convey the meaning of each verse?
  5 = Deeply faithful, captures poetic imagery and precise meaning
  3 = Generally aligned but loses some nuance or imagery
  1 = Misinterprets verse meaning or invents content

- fluency_score: Is the generated Arabic well-formed Modern Standard Arabic?
  5 = Fluent, grammatically correct, natural MSA
  3 = Understandable but with minor grammatical issues
  1 = Awkward, incomplete, or ungrammatical

- overall_score: Holistic quality assessment combining faithfulness, fluency, and interpretive depth

Return valid JSON only in this format:
{
  "faithfulness_score": <1-5>,
  "fluency_score": <1-5>,
  "overall_score": <1-5>
}"""

_ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_HEADING_REGEX = re.compile(
    r"(?im)^(?:\s*(?:البيت|بيت|verse)\s*(?:\d+|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)\s*[:：\-]?\s*|\s*\d+\s*[\.\):\-]\s*)"
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_heading(text: str) -> str:
    text = re.sub(
        r"(?im)^\s*(?:البيت|بيت|verse)\s*(?:\d+|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)\s*[:：\-]?\s*",
        "",
        text,
    )
    text = re.sub(r"(?im)^\s*\d+\s*[\.\):\-]\s*", "", text)
    text = re.sub(r"(?im)^\s*(?:شرح|التفسير)\s*[:：\-]?\s*", "", text)
    return text.strip()


def _split_with_headings(text: str) -> List[str]:
    matches = list(_HEADING_REGEX.finditer(text))
    if len(matches) < 2:
        return []

    parts: List[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = _strip_heading(text[start:end])
        if chunk:
            parts.append(chunk)
    return parts


def _split_paragraphs(text: str) -> List[str]:
    parts: List[str] = []
    for block in re.split(r"\n\s*\n+", text):
        cleaned = _strip_heading(block)
        if cleaned and len(cleaned) >= 20:
            parts.append(cleaned)
    return parts


def _parse_generated_verses(text: str, expected_count: int) -> List[Dict[str, str]]:
    normalized = unicodedata.normalize("NFKC", text or "").translate(_ARABIC_DIGIT_MAP).strip()
    if not normalized:
        return []

    segments = _split_with_headings(normalized)
    if not segments:
        segments = _split_paragraphs(normalized)
    if not segments:
        segments = [_strip_heading(normalized)]

    verses: List[Dict[str, str]] = []
    for index, segment in enumerate(segments[:expected_count], start=1):
        cleaned = _norm(segment)
        if cleaned:
            verses.append({"v": index, "text": cleaned})
    return verses


def _extract_json_blob(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _parse_scores(text: str) -> Dict[str, float]:
    payload = _extract_json_blob(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {}

    def _score(name: str) -> float:
        # Rubric is 1-5; 0 is NOT a valid score, so silent-zeroing on
        # parse failure would corrupt downstream means. Use -100 sentinel
        # for missing key, non-numeric value, or out-of-range.
        if name not in parsed:
            return -100
        raw = parsed[name]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return -100
        if not (1.0 <= value <= 5.0):
            return -100
        return value

    return {
        "faithfulness_score": _score("faithfulness_score"),
        "fluency_score": _score("fluency_score"),
        "overall_score": _score("overall_score"),
        "judge_parse_rate": 1.0 if parsed else 0.0,
    }


class FannOrFlopAnnotator(Annotator):
    """Judge generated poem explanations against verse-aligned gold explanations."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        **_: Any,
    ):
        self._auto_client = auto_client
        self.judge_model_name = _JUDGE_MODEL_OVERRIDE or judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.name = "fann_or_flop_poem_judge"

    def _call_judge(self, model_name: str, prompt: str) -> str:
        if _JUDGE_OVERRIDE:
            return _call_openrouter_direct(
                _JUDGE_OVERRIDE,
                prompt,
                self.judge_temperature,
                self.judge_max_new_tokens,
            ).strip()

        request = Request(
            model=model_name,
            model_deployment=model_name,
            prompt=prompt,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}")
        return result.completions[0].text.strip() if result.completions else ""

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
        extra_data = request_state.instance.extra_data or {}
        gold_items = extra_data.get("verse_explanations") or []

        ground_truth = [
            {"v": index, "text": _norm(item.get("explanation", ""))}
            for index, item in enumerate(gold_items, start=1)
            if _norm(item.get("explanation", ""))
        ]
        generated = _parse_generated_verses(completion, len(ground_truth))
        aligned_count = min(len(ground_truth), len(generated))
        verse_parse_rate = (aligned_count / len(ground_truth)) if ground_truth else 0.0

        if aligned_count == 0:
            # No verses aligned — the judge was never called. Use -100
            # sentinel for the score fields (1-5 rubric, 0 is invalid)
            # so this row does not silently zero downstream means.
            return {
                "judge_model": self.judge_model_name,
                "raw_judge_text": "",
                "gold_verse_count": float(len(ground_truth)),
                "generated_verse_count": float(len(generated)),
                "aligned_verse_count": 0.0,
                "verse_parse_rate": 0.0,
                "judge_parse_rate": 0.0,
                "faithfulness_score": -100,
                "fluency_score": -100,
                "overall_score": -100,
            }

        payload = {
            "id": request_state.instance.id,
            "poem_title": extra_data.get("title", ""),
            "ground_truth": ground_truth[:aligned_count],
            "generated": generated[:aligned_count],
        }
        prompt = f"{_SYSTEM_PROMPT}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False)}"

        judge_model = self.judge_model_name
        try:
            raw_judge_text = self._call_judge(judge_model, prompt)
        except Exception:
            judge_model = BACKUP_JUDGE_MODEL
            raw_judge_text = self._call_judge(judge_model, prompt)

        scores = _parse_scores(raw_judge_text)
        return {
            "judge_model": judge_model,
            "raw_judge_text": raw_judge_text,
            "gold_verse_count": float(len(ground_truth)),
            "generated_verse_count": float(len(generated)),
            "aligned_verse_count": float(aligned_count),
            "verse_parse_rate": float(verse_parse_rate),
            "judge_parse_rate": float(scores["judge_parse_rate"]),
            "faithfulness_score": float(scores["faithfulness_score"]),
            "fluency_score": float(scores["fluency_score"]),
            "overall_score": float(scores["overall_score"]),
        }
