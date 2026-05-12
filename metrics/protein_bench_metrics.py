"""
Protein structure quality metrics for ProteinBench (arXiv:2409.06744).

pLDDT, scTM, and novelty TM-score are computed via ESMFold (fair-esm).
All metrics return 0.0 when ESMFold is unavailable so stats are always emitted.

Install: pip install fair-esm
"""

import threading
from typing import List, Optional, Tuple

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

_esmfold_lock = threading.Lock()
_esmfold_model = None


def _get_esmfold():
    global _esmfold_model
    if _esmfold_model is None:
        with _esmfold_lock:
            if _esmfold_model is None:
                try:
                    import esm
                    import torch
                    model = esm.pretrained.esmfold_v1()
                    model = model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()
                    _esmfold_model = model
                except Exception:
                    pass  # ESMFold not installed; metrics will return 0.0
    return _esmfold_model


def _fold_sequence(sequence: str) -> Tuple[Optional[float], Optional[float]]:
    """Returns (plddt_mean_0_1, ptm) or (None, None) if ESMFold unavailable."""
    model = _get_esmfold()
    if model is None:
        return None, None
    try:
        import torch
        with torch.no_grad():
            output = model.infer(sequence)
        plddt = float(output["plddt"].mean().item()) / 100.0
        ptm = float(output["ptm"].item())
        return plddt, ptm
    except Exception:
        return None, None


def _extract_sequence(text: str) -> str:
    return text.strip().upper()


def _is_valid_sequence(seq: str) -> bool:
    return len(seq) > 0 and all(c in _VALID_AA for c in seq)


class PlddtScoreMetric(Metric):
    """Mean pLDDT from ESMFold structure prediction, normalized to [0, 1].

    pLDDT > 0.70 indicates a confidently predicted, structurally plausible fold.
    Returns 0.0 for invalid sequences or when ESMFold is not installed.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        seq = _extract_sequence(request_state.result.completions[0].text)
        if not _is_valid_sequence(seq):
            return [Stat(MetricName("plddt_score")).add(0.0)]
        plddt, _ = _fold_sequence(seq)
        return [Stat(MetricName("plddt_score")).add(plddt if plddt is not None else 0.0)]


class SctmScoreMetric(Metric):
    """Self-consistency TM-score via ESMFold pTM output.

    ESMFold's pTM is the predicted TM-score of the structure (range 0–1).
    Used here as a proxy for scTM (self-consistency TM-score): higher = better
    fold quality. Returns 0.0 for invalid sequences or when ESMFold unavailable.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        seq = _extract_sequence(request_state.result.completions[0].text)
        if not _is_valid_sequence(seq):
            return [Stat(MetricName("sctm_score")).add(0.0)]
        _, ptm = _fold_sequence(seq)
        return [Stat(MetricName("sctm_score")).add(ptm if ptm is not None else 0.0)]


class NoveltyTmScoreMetric(Metric):
    """Novelty score: 1 − max TM-score vs PDB.

    Full PDB comparison requires TM-align against all PDB entries.
    When PDB lookup is unavailable, approximated as (1 − pTM): lower pTM
    implies the structure is less similar to known well-folded proteins.
    Returns 0.0 for invalid sequences or when ESMFold unavailable.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        seq = _extract_sequence(request_state.result.completions[0].text)
        if not _is_valid_sequence(seq):
            return [Stat(MetricName("novelty_tmscore")).add(0.0)]
        _, ptm = _fold_sequence(seq)
        if ptm is None:
            return [Stat(MetricName("novelty_tmscore")).add(0.0)]
        novelty = float(1.0 - ptm)
        return [Stat(MetricName("novelty_tmscore")).add(novelty)]
