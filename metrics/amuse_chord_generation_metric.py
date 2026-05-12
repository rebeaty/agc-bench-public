"""Corpus-level Amuse chord-generation metrics."""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.spatial.distance import jensenshannon

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.common.general import ensure_file_downloaded


_HOOKTHEORY_URL = "https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz"
_NOTES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_NOTE_CONVERTER = {
    "C#": "Db",
    "D#": "Eb",
    "Gb": "F#",
    "G#": "Ab",
    "A#": "Bb",
}
_CHORD_DEGREES_TO_NAME = {
    (4, 3): "",
    (3, 4): "m",
    (3, 4, 3): "m7",
    (4, 3, 3): "7",
    (4, 3, 4): "maj7",
    (4, 3, 7): "add9",
    (3, 3): "dim",
    (5, 2): "sus4",
    (2, 5): "sus2",
    (3, 3, 4): "m7b5",
    (5, 2, 3): "7sus4",
    (3, 4, 3, 4): "m9",
    (3, 4, 7): "madd9",
    (4, 3, 4, 3): "maj9",
    (4, 3, 3, 4, 3): "11",
    (2, 5, 3): "7sus2",
    (3, 4, 3, 4, 3): "m11",
    (3, 3, 3): "dim7",
    (2, 5, 4): "maj7sus2",
    (4, 3, 3, 4): "9",
    (2, 3, 2): "sus2sus4",
    (4, 4): "aug",
    (7,): "5",
    (3, 4, 3, 7): "m7add11",
    (6, 1): "sus#4",
    (4, 3, 3, 3): "7b9",
    (4, 3, 14): "6",
    (4, 3, 10): "add11",
    (3, 4, 4): "minmaj7",
    (4, 3, 3, 4, 3, 4): "13",
    (3, 3, 4, 3, 4): "m11b5b9",
    (4, 3, 3, 5): "7#9",
    (4, 3, 4, 3, 4): "maj9#11",
    (4, 3, 3, 10): "7b13",
    (4, 4, 3): "maj7#5",
}
_SCALE_DEGREES_TO_NAME = {
    (2, 2, 1, 2, 2, 2): "Maj",
    (2, 1, 2, 2, 2, 1): "Dor",
    (1, 2, 2, 2, 1, 2): "Phr",
    (2, 2, 2, 1, 2, 2): "Lyd",
    (2, 2, 1, 2, 2, 1): "Mix",
    (2, 1, 2, 2, 1, 2): "Min",
    (1, 2, 2, 1, 2, 2): "Loc",
    (2, 1, 2, 2, 1, 3): "Hmin",
    (1, 3, 1, 2, 1, 2): "Phdm",
}
_CHORD_NAMES_TO_DEGREES = {value: key for key, value in _CHORD_DEGREES_TO_NAME.items()}
_LEADING_ENUM_RE = re.compile(r"^\s*(?:[-*]|\d+[\).\:-])\s*")
_CODE_FENCE_RE = re.compile(r"^```|```$", re.MULTILINE)


def _example_to_events(example: dict, remove_duplicates: bool = True) -> list[list[tuple]]:
    key_beatstamps = [key["beat"] for key in example["annotations"]["keys"]]
    events: list[list[tuple]] = [
        [("key", key["tonic_pitch_class"], tuple(key["scale_degree_intervals"]))]
        for key in example["annotations"]["keys"]
    ]

    current_key_idx = 0
    last_chord = None
    for chord in example["annotations"]["harmony"]:
        onset = float(chord["onset"])
        while current_key_idx + 1 < len(key_beatstamps) and onset >= key_beatstamps[current_key_idx + 1]:
            current_key_idx += 1
            last_chord = None

        if tuple(chord["root_position_intervals"]) == ():
            break

        processed_chord = (
            "chord",
            chord["root_pitch_class"],
            tuple(chord["root_position_intervals"]),
            chord["inversion"],
        )
        if not remove_duplicates or processed_chord != last_chord:
            events[current_key_idx].append(processed_chord)
            last_chord = processed_chord

    return events


def _transpose_chord_events(chord_progression: list[tuple], target_key: int = 0) -> list[tuple]:
    def transpose_note(note: int, steps: int) -> int:
        return (note + steps) % 12

    converted_progression: list[tuple] = []
    transpose_steps = None
    for event in chord_progression:
        if event[0] == "key":
            key, scale_degrees = event[1], event[2]
            transpose_steps = (target_key - key) % 12
            converted_progression.append(("key", transpose_note(key, transpose_steps), scale_degrees))
        elif event[0] == "chord":
            assert transpose_steps is not None
            root, chord_intervals, inversion = event[1], event[2], event[3]
            converted_progression.append(("chord", transpose_note(root, transpose_steps), chord_intervals, inversion))
    return converted_progression


def _event_to_text(event: tuple) -> str:
    event_type, root_idx, *details = event
    root = _NOTES[root_idx]
    if event_type == "key":
        mode = _SCALE_DEGREES_TO_NAME[details[0]]
        return f"{root} {mode}"
    intervals, inversion = details
    quality = _CHORD_DEGREES_TO_NAME.get(intervals)
    if quality is None:
        return ""
    if inversion > 0:
        chord_notes = [(root_idx + sum(intervals[:i])) % 12 for i in range(len(intervals) + 1)]
        bass_note = _NOTES[chord_notes[inversion]]
        return f"{root}{quality}/{bass_note}"
    return f"{root}{quality}"


def _text_to_event(text: str) -> tuple | None:
    stripped = text.strip()
    if not stripped:
        return None
    if "/" in stripped:
        chord_text, bass_note = stripped.split("/", 1)
        bass_note_idx = _NOTES.index(_NOTE_CONVERTER.get(bass_note, bass_note)) if _NOTE_CONVERTER.get(bass_note, bass_note) in _NOTES else 0
    else:
        chord_text, bass_note_idx = stripped, 0

    for note in _NOTES + list(_NOTE_CONVERTER.keys()):
        if chord_text.startswith(note):
            root = _NOTE_CONVERTER.get(note, note)
            quality = chord_text[len(note) :]
            if quality not in _CHORD_NAMES_TO_DEGREES:
                continue
            root_idx = _NOTES.index(root)
            return ("chord", root_idx, _CHORD_NAMES_TO_DEGREES[quality], bass_note_idx)
    return None


def _normalize_line(line: str) -> str:
    return _LEADING_ENUM_RE.sub("", line.strip())


def _parse_progressions(text: str) -> tuple[list[list[str]], list[list[tuple]]]:
    cleaned = _CODE_FENCE_RE.sub("", text or "").strip()
    parsed_tokens: list[list[str]] = []
    parsed_events: list[list[tuple]] = []
    for raw_line in cleaned.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        tokens = [token.strip() for token in line.split() if token.strip()]
        if len(tokens) != 4:
            continue
        events = [_text_to_event(token.replace("min", "m")) for token in tokens]
        if any(event is None for event in events):
            continue
        parsed_tokens.append(tokens)
        parsed_events.append(list(events))  # type: ignore[arg-type]
    return parsed_tokens, parsed_events


def _compute_self_bleu(progressions: list[list[str]]) -> float:
    if len(progressions) <= 1:
        return 0.0
    smoothie = SmoothingFunction().method1
    scores: list[float] = []
    for index, hypothesis in enumerate(progressions):
        references = progressions[:index] + progressions[index + 1 :]
        if not references:
            continue
        scores.append(sentence_bleu(references, hypothesis, smoothing_function=smoothie))
    return sum(scores) / len(scores) if scores else 0.0


def _ngram_distribution(progressions: list[list[tuple]], n: int) -> pd.Series:
    if n == 1:
        items = [item for progression in progressions for item in progression]
    else:
        items = [tuple(zip(progression[:-1], progression[1:])) for progression in progressions if len(progression) >= 2]
        items = [item for progression in items for item in progression]
    if not items:
        return pd.Series(dtype=float)
    return pd.Series(items).value_counts(normalize=True)


def _compute_jsd(reference_progressions: list[list[tuple]], generated_progressions: list[list[tuple]], n: int) -> float:
    reference_dist = _ngram_distribution(reference_progressions, n)
    generated_dist = _ngram_distribution(generated_progressions, n)
    if reference_dist.empty or generated_dist.empty:
        return 0.0
    all_keys = set(reference_dist.index) | set(generated_dist.index)
    reference_aligned = [reference_dist.get(key, 0.0) for key in all_keys]
    generated_aligned = [generated_dist.get(key, 0.0) for key in all_keys]
    return float(jensenshannon(reference_aligned, generated_aligned))


def _resolve_hooktheory_path(cache_dir: str) -> str:
    override_path = os.environ.get("AMUSE_HOOKTHEORY_PATH", "").strip()
    if override_path and os.path.exists(override_path):
        return override_path

    repo_candidates = [
        "dataset/Hooktheory/Hooktheory.json.gz",
        "data/Hooktheory/Hooktheory.json.gz",
    ]
    for candidate in repo_candidates:
        if os.path.exists(candidate):
            return candidate

    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, "Hooktheory.json.gz")
    ensure_file_downloaded(_HOOKTHEORY_URL, target_path, unpack=False)
    return target_path


@lru_cache(maxsize=1)
def _load_hooktheory_progressions(cache_dir: str) -> tuple[list[list[tuple]], int]:
    os.makedirs(cache_dir, exist_ok=True)
    target_path = _resolve_hooktheory_path(cache_dir)

    with gzip.open(target_path, "rb") as f:
        raw_data = json.load(f)

    hooktheory_progressions: list[list[tuple]] = []
    for example in raw_data.values():
        if "HARMONY" not in example["tags"]:
            continue
        for progression in _example_to_events(example):
            if len(progression[1:]) < 4:
                continue
            if progression[0][1] == 0:
                chord_progression = progression[1:]
            else:
                chord_progression = _transpose_chord_events(progression, 0)[1:]
            if chord_progression:
                hooktheory_progressions.append(chord_progression)
    return hooktheory_progressions, len(raw_data)


class AmuseChordGenerationMetric(EvaluateInstancesMetric):
    """Compute chord-aware Self-BLEU and Hooktheory-backed JSD for Amuse batches."""

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        parsed_progression_counts: list[int] = []
        self_bleu_scores: list[float] = []
        generated_event_progressions: list[list[tuple]] = []
        valid_batches = 0
        total_batches = 0

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            total_batches += 1
            completion = request_state.result.completions[0].text.strip() if request_state.result.completions else ""
            token_progressions, event_progressions = _parse_progressions(completion)
            parsed_progression_counts.append(len(event_progressions))
            if event_progressions:
                valid_batches += 1
                generated_event_progressions.extend(event_progressions)
            if token_progressions:
                self_bleu_scores.append(_compute_self_bleu(token_progressions))

        reference_available = 1.0
        reference_download_blocked = 0.0
        hooktheory_progressions: list[list[tuple]] = []
        raw_song_count = 0
        try:
            hooktheory_progressions, raw_song_count = _load_hooktheory_progressions(
                os.path.join(eval_cache_path, "amuse_chord_generation_reference_cache")
            )
            unigram_jsd = _compute_jsd(hooktheory_progressions, generated_event_progressions, 1)
            bigram_jsd = _compute_jsd(hooktheory_progressions, generated_event_progressions, 2)
        except Exception:
            reference_available = 0.0
            reference_download_blocked = 1.0
            unigram_jsd = 0.0
            bigram_jsd = 0.0
        mean_self_bleu = sum(self_bleu_scores) / len(self_bleu_scores) if self_bleu_scores else 0.0
        mean_parsed_progressions = (
            sum(parsed_progression_counts) / len(parsed_progression_counts) if parsed_progression_counts else 0.0
        )
        valid_batch_rate = valid_batches / total_batches if total_batches else 0.0

        return [
            Stat(MetricName("self_bleu")).add(mean_self_bleu),
            Stat(MetricName("jensen_shannon_divergence_unigram")).add(unigram_jsd),
            Stat(MetricName("jensen_shannon_divergence_bigram")).add(bigram_jsd),
            Stat(MetricName("amuse_valid_progressions_per_instance")).add(mean_parsed_progressions),
            Stat(MetricName("amuse_valid_batch_rate")).add(valid_batch_rate),
            Stat(MetricName("amuse_generated_progression_count")).add(float(len(generated_event_progressions))),
            Stat(MetricName("amuse_hooktheory_progression_count")).add(float(len(hooktheory_progressions))),
            Stat(MetricName("amuse_hooktheory_song_count")).add(float(raw_song_count)),
            Stat(MetricName("amuse_reference_available")).add(reference_available),
            Stat(MetricName("amuse_reference_download_blocked")).add(reference_download_blocked),
        ]
