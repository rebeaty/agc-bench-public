"""WritingBench-specific checklist-aware LLM judge annotator."""

import json
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request

from llm_judge.generic_llm_judge_annotator import (
    BACKUP_JUDGE_MODEL,
    _JUDGE_OVERRIDE,
    _call_openrouter_direct,
)


EVALUATE_SYSTEM = (
    "You are an expert evaluator with extensive experience in evaluating "
    "response of given query."
)

EVALUATE_PROMPT = """
Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

- Provide reasons for each score by indicating specific strengths or deficiencies within the Response.
- Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.
- Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.
- Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.
- Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.
- Scoring Range: Assign an integer score between 1 to 10

** Output format **
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format **
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```
""".strip()


def _strip_thinking(text: str) -> str:
    marker = "</think>\n\n"
    marker_pos = text.find(marker)
    if marker_pos != -1:
        return text[marker_pos + len(marker) :]
    return text


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_reason(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


class WritingBenchAnnotator(Annotator):
    """Judge a WritingBench completion against all five checklist criteria."""

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_top_p: float,
        judge_max_new_tokens: int,
        max_retries: int = 3,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_top_p = judge_top_p
        self.judge_max_new_tokens = judge_max_new_tokens
        self.max_retries = max_retries
        self.name = "writingbench_judge"

    def _call_judge(self, model_name: str, query: str, response: str, criteria: Dict[str, Any]) -> str:
        criteria_text = json.dumps(criteria, ensure_ascii=False)
        prompt = EVALUATE_PROMPT.format(
            query=query,
            response=_strip_thinking(response),
            criteria=criteria_text,
        )

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
            temperature=self.judge_temperature,
            top_p=self.judge_top_p,
            max_tokens=self.judge_max_new_tokens,
            num_completions=1,
            messages=[
                {"role": "system", "content": EVALUATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"Judge call failed for model {model_name}")
        return result.completions[0].text.strip()

    def _score_with_model(
        self, model_name: str, query: str, response: str, criteria: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        for _ in range(self.max_retries):
            judge_output = self._call_judge(model_name, query, response, criteria)
            parsed = _extract_json_dict(judge_output)
            if parsed is None:
                continue

            score = parsed.get("score")
            reason = _normalize_reason(parsed.get("reason"))
            if isinstance(score, int) and 1 <= score <= 10 and reason:
                return {
                    "criterion_name": criteria.get("name", ""),
                    "score": score,
                    "reason": reason,
                }
        return None

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None

        completion = request_state.result.completions[0].text.strip()
        query = request_state.instance.input.text
        checklist: List[Dict[str, Any]] = request_state.instance.extra_data.get("checklist", [])

        evaluations: List[Dict[str, Any]] = []
        for criterion in checklist:
            result = None
            try:
                result = self._score_with_model(self.judge_model_name, query, completion, criterion)
            except Exception:
                result = None

            if result is None and self.judge_model_name != BACKUP_JUDGE_MODEL:
                try:
                    result = self._score_with_model(BACKUP_JUDGE_MODEL, query, completion, criterion)
                except Exception:
                    result = None

            if result is not None:
                evaluations.append(result)

        valid_count = len(evaluations)
        total_count = len(checklist)
        # Use -100 sentinel when no criterion successfully parsed: the
        # rubric's valid range is 1-10, so 0.0 is out-of-range AND would
        # silently bias downstream means toward the floor. -100 follows
        # the project convention for "rating unavailable".
        if evaluations:
            score = sum(item["score"] for item in evaluations) / len(evaluations)
        else:
            score = -100.0

        valid_rate = valid_count / total_count if total_count else 0.0

        return {
            "writingbench_score": score,
            "writingbench_valid_criteria_rate": valid_rate,
            "writingbench_criteria_count": float(valid_count),
            "writingbench_evaluations": evaluations,
        }
