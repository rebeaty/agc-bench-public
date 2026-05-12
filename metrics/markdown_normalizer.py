"""Shared text normalizer for chat-tuned model outputs.

Chat-tuned models often wrap their answers in markdown formatting:
  - **bold**, *italic*, _emphasis_
  - ### headers, --- horizontal rules
  - `inline code`
  - Numbered/bulleted lists with leading whitespace

For metrics that compare the prediction against a literal gold answer
(e.g., MCQ classification, exact-match generation), this formatting
breaks the parser even though the *content* is correct.

`normalize_markdown` strips the formatting wrapper without altering the
semantic content. It's intentionally conservative — it removes only
characters that are unambiguously markdown decoration.
"""
from __future__ import annotations

import re

_MD_BOLD_OR_EMPH = re.compile(r"(\*\*|__|\*|_)")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s*", flags=re.MULTILINE)
_MD_HR = re.compile(r"^\s*-{3,}\s*$", flags=re.MULTILINE)
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_LEADING_BULLET = re.compile(r"^\s*[-*•]\s+", flags=re.MULTILINE)


def normalize_markdown(text: str) -> str:
    """Strip common markdown formatting; preserve underlying text content."""
    if not text:
        return text
    out = text
    out = _MD_HEADER.sub("", out)
    out = _MD_HR.sub("", out)
    out = _MD_INLINE_CODE.sub(r"\1", out)
    out = _MD_BOLD_OR_EMPH.sub("", out)
    out = _LEADING_BULLET.sub("", out)
    return out.strip()
