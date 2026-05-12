"""Generic per-metric LLM-as-Judge annotator for the AGC-Bench."""
import os
import re
from typing import Any, Dict

from google import genai
from google.genai import types as genai_types
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


# When AGC_JUDGE_OVERRIDE is set the user has pinned ALL judge traffic to a
# single deployment (typically AGC-Judge). Re-route the backup at it too, so
# annotators that fall back on a parse failure don't silently hit a third-party
# API the user didn't bill for. (For most parse-failure cases the second call
# will also fail, eventually returning -100 — the integrator filters those.)
BACKUP_JUDGE_MODEL = (
    os.environ.get("AGC_JUDGE_OVERRIDE", "").strip()
    or "google/gemini-3-flash-preview"
)


def _infer_rubric_range(rubric: str) -> tuple[int | None, int | None]:
    """Infer the integer score range from rubric text. Looks for anchor lines
    like "1 = ...", "Score 5: ...", "0-2 scale", etc. Returns (lo, hi) or
    (None, None) if no range is detectable. Used to reject stray integers
    when the judge rambles past the requested score."""
    nums: set[int] = set()
    for m in re.finditer(r'(?:^|\n)\s*(?:Score\s+)?(\d+)\s*[=:]', rubric):
        nums.add(int(m.group(1)))
    if len(nums) >= 2:
        return min(nums), max(nums)
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*(?:scale|likert|rubric|integer|point)',
                  rubric, re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)
    return None, None


def _parse_score_robust(text: str, valid_lo: int | None, valid_hi: int | None) -> int:
    """Extract an integer score from a judge reply.

    Strategy (in order):
      1. If the entire reply is just an integer, use it.
      2. Match labeled patterns ("Score: N", "Rating: 4", "= 5", "**5**").
      3. If a valid range is known, return the LAST integer in the reply
         that falls within [valid_lo, valid_hi].
      4. Else return the LAST integer in the reply.
      5. -100 sentinel if no integer found or no in-range integer found.

    First-integer parsing is unsafe with verbose judge replies: line numbers,
    year references, and word counts in the reasoning text can obscure the
    actual rating. The score is usually at the end of the reply or behind a
    "Score:" / "Rating:" label.
    """
    s = text.strip()
    if not s:
        return -100
    # Case 1: pure integer
    m = re.fullmatch(r'-?\d+', s)
    if m:
        v = int(m.group())
        if valid_lo is None or valid_lo <= v <= valid_hi:
            return v
        return -100
    # Case 2: labeled patterns, scanned right-to-left so the LAST match wins
    label_pat = re.compile(
        r'(?:score|rating|rate|final|answer|verdict|grade)\s*'
        r'(?:is|=|:|\*\*)?\s*\**\s*(-?\d+)',
        re.IGNORECASE,
    )
    candidates = [int(m.group(1)) for m in label_pat.finditer(s)]
    if candidates:
        if valid_lo is not None:
            in_rng = [v for v in candidates if valid_lo <= v <= valid_hi]
            if in_rng:
                return in_rng[-1]
        else:
            return candidates[-1]
    # Case 3+4: fall back to integers in the reply, prefer in-range, prefer last
    all_ints = [int(x) for x in re.findall(r'-?\d+', s)]
    if not all_ints:
        return -100
    if valid_lo is not None:
        in_rng = [v for v in all_ints if valid_lo <= v <= valid_hi]
        if in_rng:
            return in_rng[-1]
        return -100  # Out-of-range stray integers are not a valid rating
    return all_ints[-1]

# If set, ALL judges (regardless of what run_specs say) route to this model
# through a provider-aware direct call. For Google models, that means the
# direct Google API, not OpenRouter.
_JUDGE_OVERRIDE = os.environ.get("CREATIVITY_JUDGE_OVERRIDE", "").strip() or None

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for direct Gemini judge calls")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _call_openrouter_chat_direct(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Direct OpenRouter chat-completion call (bypasses HELM's client routing).
    Per-call client with explicit short timeout to prevent worker threads from
    hanging indefinitely on slow API responses.

    Auto-disables reasoning/thinking traces for known reasoning models so the
    judge returns the score directly instead of paying for ~500-1000 hidden
    chain-of-thought tokens (which made grok-4.1-fast 4-10x slower than its
    advertised throughput on real judge prompts)."""
    import openai
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("CREATIVITY_JUDGE_OVERRIDE set but OPENROUTER_API_KEY missing")
    client = openai.OpenAI(api_key=api_key,
                             base_url="https://openrouter.ai/api/v1",
                             timeout=60.0, max_retries=1)
    extra_body: dict = {}
    effective_max_tokens = max_tokens
    if "grok" in model.lower():
        extra_body["reasoning"] = {"enabled": False}
    if "/gpt-5" in model.lower() or model.lower().startswith("gpt-5"):
        extra_body["reasoning"] = {"effort": "minimal"}
        # Even at minimal effort, gpt-5 family eats ~15-20 tokens on hidden
        # reasoning before emitting the visible answer. Benches with tight
        # budgets (data_narrative judge_max_new_tokens=16) starve and return
        # empty content. Floor at 64 so the visible score actually fits.
        effective_max_tokens = max(max_tokens, 64)
    kwargs = dict(model=model,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=temperature,
                  max_tokens=effective_max_tokens)
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _normalize_google_model_name(model: str) -> str:
    if model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


_GEMINI_MIN_OUTPUT_TOKENS = 8


def _call_google_direct(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    normalized_model = _normalize_google_model_name(model)
    # gemini-3* defaults to thinking-on; without thinking_budget=0 the API
    # consumes max_output_tokens on hidden thoughts and returns empty content
    # (finishReason=MAX_TOKENS), which the judge parser sees as nan preference.
    thinking_config = (
        genai_types.ThinkingConfig(thinking_budget=0)
        if normalized_model.startswith("gemini-3")
        else None
    )
    # HELM judge specs sometimes hardcode max_tokens=1 (works for OpenAI's
    # top_logprobs path but Gemini cannot deliver any visible token under 8
    # for prompts of typical judge size — empirically tested 2026-04-25).
    # The judge parsers only consume the first character anyway, so floor.
    effective_max_tokens = max(max_tokens, _GEMINI_MIN_OUTPUT_TOKENS)
    response = _get_gemini_client().models.generate_content(
        model=normalized_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=effective_max_tokens,
            thinking_config=thinking_config,
        ),
    )
    return (response.text or "").strip()


def _call_direct_model(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    # Explicit "openrouter/" prefix forces OR routing even for Google models —
    # needed when the Google API key is on a tier with insufficient quota.
    if model.startswith("openrouter/"):
        return _call_openrouter_chat_direct(
            model[len("openrouter/"):], prompt, temperature, max_tokens)
    if model.startswith("google/"):
        return _call_google_direct(model, prompt, temperature, max_tokens)
    return _call_openrouter_chat_direct(model, prompt, temperature, max_tokens)


def _call_openrouter_direct(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Backward-compatible provider-aware direct call helper for judge overrides."""
    return _call_direct_model(model, prompt, temperature, max_tokens)


class GenericLLMJudgeAnnotator(Annotator):
    """Calls an LLM judge once per metric with a rubric-specific prompt.

    Each instance handles exactly ONE metric dimension. ``self.name`` is set
    per-instance from ``metric_name`` so multiple annotator specs on the same
    run do not overwrite each other's annotations in ``request_state.annotations``.

    The ``auto_client`` parameter is auto-injected by HELM's AnnotatorFactory.
    """

    def __init__(
        self,
        auto_client: AutoClient,
        judge_model_name: str,
        judge_temperature: float,
        judge_max_new_tokens: int,
        metric_name: str,
        rubric: str,
    ):
        self._auto_client = auto_client
        self.judge_model_name = judge_model_name
        self.judge_temperature = judge_temperature
        self.judge_max_new_tokens = judge_max_new_tokens
        self.metric_name = metric_name
        self.rubric = rubric
        self.name = f"generic_llm_judge_{metric_name}"
        # Inferred from rubric anchor lines like "1 = ..." / "Score 5: ...".
        # Used to validate parsed scores so chatty replies don't grab stray ints.
        self._valid_lo, self._valid_hi = _infer_rubric_range(rubric)

    def _call_judge(self, model_name: str, prompt: str) -> int:
        # Global override: route ALL judge calls to the configured direct path.
        if _JUDGE_OVERRIDE:
            score_text = _call_direct_model(
                _JUDGE_OVERRIDE, prompt, self.judge_temperature, self.judge_max_new_tokens
            ).strip()
        else:
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
            score_text = result.completions[0].text.strip()
        return _parse_score_robust(score_text, self._valid_lo, self._valid_hi)

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()
        input_text = request_state.instance.input.text

        reference_text = ""
        if request_state.instance.references:
            reference_text = request_state.instance.references[0].output.text

        prompt = (
            f"{self.rubric}\n\n"
            f"Instruction:\n{input_text}\n\n"
        )
        if reference_text:
            prompt += f"Reference response:\n{reference_text}\n\n"
        prompt += (
            f"Generated response:\n{completion}\n\n"
            f"Provide only the integer score (e.g., 3):"
        )

        try:
            score = self._call_judge(self.judge_model_name, prompt)
        except Exception:
            # When AGC_JUDGE_OVERRIDE is set, the user has explicitly pinned
            # ALL judge traffic to a single deployment (typically AGC-Judge).
            # Falling back to BACKUP_JUDGE_MODEL would silently route to a
            # different external API, which would make the run harder to
            # reproduce and audit.
            # Mark the cell as parse-failure (-100) and continue.
            if os.environ.get('AGC_JUDGE_OVERRIDE', '').strip():
                score = -100
            else:
                try:
                    score = self._call_judge(BACKUP_JUDGE_MODEL, prompt)
                except Exception:
                    score = -100

        return {self.metric_name: score}
