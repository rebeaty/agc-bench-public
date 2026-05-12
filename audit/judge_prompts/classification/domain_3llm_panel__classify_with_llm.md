# Domain classification — classify_with_llm()

**Category**: classification
**Source**: `scripts/classify_domains.py`
**Judge model**: `gemini-3-flash, x-ai/grok-4.1-fast, openai/gpt-4.1-mini (majority vote)`
**Notes**: Per-benchmark domain assignment. Cohen's κ ≈ 0.85 across pairs (paper §3.7). Function reproduced verbatim; final prompt is sys + DOMAIN_DEFINITIONS + JSON schema.

```text
def classify_with_llm(client_call, bench_card: str) -> dict:
    """client_call(messages) -> string. Returns dict with domain + rationale."""
    sys_prompt = (
        "You are classifying creativity benchmarks. Read the benchmark "
        "card carefully and focus on the actual task structure shown in "
        "the sample prompts and model responses.\n\n"
        + DOMAIN_DEFINITIONS
        + "\n\nRespond in JSON exactly:\n"
        + '{"domain": "<one of: Brainstorming, Problem Solving, STEM, Story / Narrative, Figurative Language, Humor>", '
        + '"task_type": "<Generation or Evaluation>", '
        + '"rationale": "<one sentence citing what in the task structure determined the domain and task type>"}'
    )
    user_prompt = bench_card
    raw = client_call([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    # Parse JSON
    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        raw_clean = re.sub(r"^```\w*\n", "", raw_clean)
        raw_clean = re.sub(r"\n```$", "", raw_clean)
    try:
        parsed = json.loads(raw_clean)
    except Exception:
        md = re.search(r'"domain"\s*:\s*"([^"]+)"', raw_clean)
        mt = re.search(r'"task_type"\s*:\s*"([^"]+)"', raw_clean)
        mr = re.search(r'"rationale"\s*:\s*"([^"]+)"', raw_clean)
        parsed = {
            "domain": md.group(1) if md else None,
            "task_type": mt.group(1) if mt else None,
            "rationale": mr.group(1) if mr else raw_clean[:200],
        }
    return {"raw": raw,
            "domain": parsed.get("domain"),
            "task_type": parsed.get("task_type"),
            "rationale": parsed.get("rationale", "")}
```
