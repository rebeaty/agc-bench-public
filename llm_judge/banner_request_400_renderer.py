"""Renderer for BannerRequest400 JSON blueprints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont

CANVAS_SIZE = (300, 250)
_DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEFAULT_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _find_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_blueprint(text: str) -> Dict[str, Any] | None:
    candidate = _find_json_block(text)
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("elements"), list):
        return None
    return parsed


def _safe_color(value: Any, default: str) -> Tuple[int, int, int]:
    if isinstance(value, str):
        try:
            return ImageColor.getrgb(value)
        except ValueError:
            return ImageColor.getrgb(default)
    return ImageColor.getrgb(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _DEFAULT_FONT_BOLD if bold else _DEFAULT_FONT
    try:
        return ImageFont.truetype(path, max(8, size))
    except Exception:
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if max_width <= 0 or "\n" in text:
        return text

    words = text.split()
    if not words:
        return text

    lines: List[str] = []
    current_line = words[0]
    for word in words[1:]:
        candidate = f"{current_line} {word}"
        candidate_bbox = draw.multiline_textbbox((0, 0), candidate, font=font, spacing=2)
        candidate_width = candidate_bbox[2] - candidate_bbox[0]
        if candidate_width <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return "\n".join(lines)


def render_blueprint_to_png(blueprint: Dict[str, Any], logo_path: str, output_path: str) -> bool:
    image = Image.new("RGB", CANVAS_SIZE, _safe_color(blueprint.get("background_color"), "#FFFFFF"))
    draw = ImageDraw.Draw(image)

    for element in blueprint.get("elements", []):
        if not isinstance(element, dict):
            continue
        element_type = str(element.get("type", "")).lower()
        position = element.get("position", {}) or {}
        size = element.get("size", {}) or {}
        x = _safe_int(position.get("x"), 0)
        y = _safe_int(position.get("y"), 0)
        width = max(1, _safe_int(size.get("width"), 80))
        height = max(1, _safe_int(size.get("height"), 30))
        bbox = [x, y, x + width, y + height]

        if element_type == "shape":
            draw.rounded_rectangle(
                bbox,
                radius=max(0, _safe_int(element.get("corner_radius"), 0)),
                fill=_safe_color(element.get("fill_color"), "#DDDDDD"),
            )
            continue

        if element_type == "logo":
            try:
                logo = Image.open(logo_path).convert("RGBA")
                logo.thumbnail((width, height))
                paste_x = x + max(0, (width - logo.width) // 2)
                paste_y = y + max(0, (height - logo.height) // 2)
                image.paste(logo, (paste_x, paste_y), logo)
            except Exception:
                continue
            continue

        if element_type == "cta_button":
            draw.rounded_rectangle(
                bbox,
                radius=max(0, _safe_int(element.get("corner_radius"), 8)),
                fill=_safe_color(element.get("fill_color"), "#1F6FEB"),
            )

        text = str(element.get("text", "")).strip()
        if not text:
            continue

        font_size = _safe_int(element.get("font_size"), 18)
        font_weight = str(element.get("font_weight", "regular")).lower()
        font = _font(font_size, bold=font_weight == "bold")
        text_color = _safe_color(element.get("text_color"), "#111111")
        align = str(element.get("align", "left")).lower()
        text = _wrap_text(draw, text, font, width)

        text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if align == "center":
            text_x = x + max(0, (width - text_width) // 2)
        elif align == "right":
            text_x = x + max(0, width - text_width)
        else:
            text_x = x
        text_y = y + max(0, (height - text_height) // 2)
        draw.multiline_text((text_x, text_y), text, fill=text_color, font=font, spacing=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return os.path.exists(output_path)
