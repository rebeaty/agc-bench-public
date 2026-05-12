"""
Creativity score metric: Divergent Semantic Integration (DSI) adapted from
  text-machine-lab/diverse-not-short
  (eval_utils.py::fast_dsi, https://github.com/text-machine-lab/diverse-not-short)

Adaptation notes vs. the original:
- The original uses token-level hidden states from a BERT-style model (layer 6).
  This implementation uses sentence-transformer embeddings (one vector per segment),
  which is the natural equivalent when a SentenceTransformer is available.
- DSI = mean pairwise cosine distance across all unique segment pairs in a response.
  Cosine distance = 1 - cosine_similarity.  Higher → more semantically diverse.
- Segmentation logic is ported directly from the original (DAT, AUT, CWT, RAT, PGT).
  The CDAT benchmark uses DAT-style responses (numbered word lists).
- Model is loaded lazily on first call and cached on the instance so that HELM can
  instantiate the metric class without triggering a model download.

CDAT additional metrics (appropriateness, novelty):
- novelty: mean pairwise cosine distance between the 10 generated words (same as DSI)
- appropriateness: mean cosine similarity between each generated word and the cue word
- creativity_score: novelty * appropriateness (CDAT combined score)
- Cue word is extracted from instance input via regex on the prompt template.
- Registry model (FastText crawl-300d-2M-subword) unavailable; sentence-transformers
  used as equivalent embedding source.
"""

import re
import string
import threading
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# ---------------------------------------------------------------------------
# Segmentation helpers  (ported from eval_utils.py)
# ---------------------------------------------------------------------------

def _dat_response_to_words(text: str) -> List[str]:
    """Parse a DAT response (numbered word list) into a list of clean words."""
    words = []
    if "1." in text and "2." in text:
        body = text.split("1.")[-1]
        for part in body.split(". "):
            w = part.strip()
            w = w.translate(str.maketrans("", "", string.punctuation))
            w = "".join(c for c in w if c.isalpha())
            if w:
                words.append(w.lower())
    else:
        for part in text.split(","):
            w = part.split()[-1].strip() if part.split() else ""
            w = w.translate(str.maketrans("", "", string.punctuation))
            w = "".join(c for c in w if c.isalpha())
            if w:
                words.append(w.lower())
    return words


def _aut_response_to_uses(text: str) -> List[str]:
    """Parse an AUT response (numbered use list) into a list of use descriptions."""
    uses = []
    if "1." in text and "2." in text:
        cur = text
        for idx in range(10, -1, -1):
            marker = f"{idx}."
            if marker in cur:
                parts = cur.split(marker)
                cur, use = parts[0], parts[-1].strip()
                uses.append(use)
        uses = uses[::-1]
        uses = [u for u in uses if u]
    if not uses:
        # Fallback: 16-word windows
        tokens = text.split()
        window = 16
        for start in range(0, len(tokens), window):
            chunk = " ".join(tokens[start : start + window])
            uses.append(chunk)
    return uses


def _cwt_response_to_sentences(text: str) -> List[str]:
    """Split creative writing into sentences using a simple rule-based splitter."""
    # Lightweight split on sentence-ending punctuation; avoids nltk dependency.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _pgt_response_to_words(text: str) -> List[str]:
    """Parse a PGT response (colon-separated goal: description pairs)."""
    words = []
    for seg in text.split(","):
        if ":" in seg:
            words.append(seg.split(":")[1].strip())
    if not words:
        clean = text.translate(str.maketrans("", "", string.punctuation))
        words = [clean]
    return words


def _segment_response(text: str, task: str) -> List[str]:
    """Segment a model response according to task type."""
    task = task.lower()
    if task in ("dat", "cdat"):
        segments = _dat_response_to_words(text)
    elif task == "aut":
        segments = _aut_response_to_uses(text)
    elif task == "cwt":
        segments = _cwt_response_to_sentences(text)
    elif task == "pgt":
        segments = _pgt_response_to_words(text)
    else:
        # Generic fallback: split on newlines then sentences
        segments = _cwt_response_to_sentences(text)
    segments = [s for s in segments if s]
    return segments if segments else [text]


# ---------------------------------------------------------------------------
# DSI computation (sentence-transformer adaptation)
# ---------------------------------------------------------------------------

def _compute_dsi(segments: List[str], model) -> float:
    """
    Compute DSI as mean pairwise cosine distance between segment embeddings.

    Mirrors fast_dsi's core logic:
      1. Encode each segment → embedding vector.
      2. Form all unique (i, j) pairs (same as original's non-repeating token pairs,
         but at segment granularity since SentenceTransformer yields one vec/segment).
      3. DSI = mean(1 - cosine_similarity) across all pairs.

    Returns 0.0 for single-segment responses (undefined pairwise distance).
    """
    if len(segments) < 2:
        return 0.0

    # shape: (n_segments, hidden_dim)
    embeddings = model.encode(segments, convert_to_numpy=True, show_progress_bar=False)

    # Normalise to unit length for efficient cosine via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    # Mean pairwise cosine distance over all unique pairs
    distances = []
    for i, j in combinations(range(len(embeddings)), 2):
        cos_sim = float(np.dot(embeddings[i], embeddings[j]))
        distances.append(1.0 - cos_sim)

    return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# HELM metric class
# ---------------------------------------------------------------------------

class CreativityScoreMetric(Metric):
    """Divergent Semantic Integration (DSI) creativity score.

    Adapted from text-machine-lab/diverse-not-short (eval_utils.py::fast_dsi).

    DSI measures semantic diversity by computing the mean pairwise cosine distance
    between segment embeddings of a response.  Higher scores indicate that the
    response covers more semantically distant concepts — a proxy for creative
    divergent thinking.

    Args:
        model_name: SentenceTransformer model identifier.  Defaults to
            "all-MiniLM-L6-v2", which mirrors the embedding dimensionality of
            BERT-base used in the original paper.
        task: Segmentation scheme.  One of "dat"/"cdat" (word lists),
            "aut" (numbered uses), "cwt" (sentences), "pgt" (goal phrases).
            Defaults to "dat" for the CDAT benchmark.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        task: str = "dat",
    ):
        super().__init__()
        self.model_name = model_name
        self.task = task
        self._model: Optional[object] = None  # lazy-loaded
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from metrics.embedder_factory import get_embedder
                    self._model = get_embedder(self.model_name)
        return self._model

    def _extract_cue_word(self, input_text: str) -> str:
        """Extract the cue word from a CDAT prompt."""
        match = re.search(r"cue word:\s*(\w+)", input_text, re.IGNORECASE)
        return match.group(1).lower() if match else ""

    def _compute_appropriateness(self, words: List[str], cue_word: str, model) -> float:
        """Mean cosine similarity between each generated word and the cue word."""
        if not words or not cue_word:
            return 0.0
        texts = [cue_word] + words
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms
        cue_vec = embeddings[0]
        word_vecs = embeddings[1:]
        sims = [float(np.dot(cue_vec, w)) for w in word_vecs]
        return float(np.mean(sims))

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        if not completion:
            return [
                Stat(MetricName("creativity_score")).add(0.0),
                Stat(MetricName("novelty")).add(0.0),
                Stat(MetricName("appropriateness")).add(0.0),
            ]

        segments = _segment_response(completion, self.task)
        model = self._get_model()
        novelty = _compute_dsi(segments, model)

        input_text = request_state.instance.input.text
        cue_word = self._extract_cue_word(input_text)
        appropriateness = self._compute_appropriateness(segments, cue_word, model)

        # CDAT creativity score = novelty * appropriateness (combined measure)
        creativity_score = novelty * appropriateness if appropriateness > 0 else novelty

        return [
            Stat(MetricName("creativity_score")).add(creativity_score),
            Stat(MetricName("novelty")).add(novelty),
            Stat(MetricName("appropriateness")).add(appropriateness),
        ]
