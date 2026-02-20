"""
Stage 3: Structural anomaly detection.

For each regime, scores each window against the regime baseline.
The baseline fingerprint IS the noise filter — templates that appear
consistently at expected rates are the normal heartbeat.

Anomalies are:
  - Missing templates (in baseline but absent from window)
  - New templates (not in baseline, appeared in window)
  - Rate anomalies (in both but count deviates >2x)
"""

import math
from typing import Dict, List, Tuple

from .constants import (
    ANOMALY_PERCENTILE,
    DEFAULT_NODE_IMPORTANCE,
    NEW_TEMPLATE_WEIGHT,
    NODE_IMPORTANCE,
    RATE_ANOMALY_THRESHOLD,
)
from .models import Anomaly, Regime, WindowFingerprint
from .utils import percentile


def score_window(
    window: WindowFingerprint,
    baseline: Dict[Tuple[str, str], int],
) -> Tuple[float, List[Anomaly]]:
    """
    Score a single window against the regime baseline.

    Returns (total_score, list_of_anomalies).
    """
    score = 0.0
    anomalies: List[Anomaly] = []
    baseline_keys = set(baseline.keys())
    window_keys = set(window.counts.keys())

    # 1. Missing templates (in baseline, absent from window)
    for key in baseline_keys - window_keys:
        node, template = key
        importance = NODE_IMPORTANCE.get(node, DEFAULT_NODE_IMPORTANCE)
        expected_count = baseline[key]
        contribution = importance * expected_count
        score += contribution

        anomalies.append(Anomaly(
            anomaly_type="missing_template",
            node=node,
            template=template,
            window_index=window.window_index,
            timestamp_ms=window.start_ms,
            severity_score=contribution,
            details={
                "expected_count": expected_count,
                "observed_count": 0,
            },
        ))

    # 2. New templates (in window, not in baseline)
    for key in window_keys - baseline_keys:
        node, template = key
        importance = NODE_IMPORTANCE.get(node, DEFAULT_NODE_IMPORTANCE)
        observed_count = window.counts[key]
        contribution = importance * observed_count * NEW_TEMPLATE_WEIGHT
        score += contribution

        anomalies.append(Anomaly(
            anomaly_type="new_template",
            node=node,
            template=template,
            window_index=window.window_index,
            timestamp_ms=window.start_ms,
            severity_score=contribution,
            details={
                "expected_count": 0,
                "observed_count": observed_count,
            },
        ))

    # 3. Rate anomalies (in both, but count deviates significantly)
    for key in baseline_keys & window_keys:
        node, template = key
        expected = baseline[key]
        observed = window.counts[key]

        if expected <= 0:
            continue

        ratio = observed / expected
        if ratio > RATE_ANOMALY_THRESHOLD or ratio < (1.0 / RATE_ANOMALY_THRESHOLD):
            importance = NODE_IMPORTANCE.get(node, DEFAULT_NODE_IMPORTANCE)
            contribution = importance * abs(math.log2(ratio))
            score += contribution

            anomalies.append(Anomaly(
                anomaly_type="rate_anomaly",
                node=node,
                template=template,
                window_index=window.window_index,
                timestamp_ms=window.start_ms,
                severity_score=contribution,
                details={
                    "expected_count": expected,
                    "observed_count": observed,
                    "ratio": round(ratio, 3),
                },
            ))

    return score, anomalies


def detect_structural_anomalies(
    regimes: List[Regime],
    windows: List[WindowFingerprint],
    anomaly_percentile: float = ANOMALY_PERCENTILE,
) -> List[Anomaly]:
    """
    Detect structural anomalies across all regimes.

    For each regime:
    1. Score every window against the regime baseline
    2. Flag windows above the anomaly_percentile threshold
    3. Collect anomalies from flagged windows

    Returns all detected anomalies.
    """
    all_anomalies: List[Anomaly] = []

    for regime in regimes:
        if not regime.baseline or not regime.window_indices:
            continue

        # Score every window in this regime
        window_scores: List[Tuple[float, List[Anomaly]]] = []
        for widx in regime.window_indices:
            w = windows[widx]
            score, anoms = score_window(w, regime.baseline)
            w.anomaly_score = score
            window_scores.append((score, anoms))

        # Compute threshold
        scores = [s for s, _ in window_scores]
        if not scores:
            continue

        threshold = percentile(scores, anomaly_percentile)
        regime.anomaly_density = sum(1 for s in scores if s > threshold) / len(scores)

        # Collect anomalies from windows above threshold
        for score, anoms in window_scores:
            if score > threshold and score > 0:
                all_anomalies.extend(anoms)

    return all_anomalies


def extract_regime_transitions(
    regimes: List[Regime],
    windows: List[WindowFingerprint],
) -> List[Anomaly]:
    """
    Extract anomalies at regime boundaries.

    Compares last 2 windows of the old regime with first 2 of the new regime.
    Changes are ranked by node importance.
    """
    transition_anomalies: List[Anomaly] = []

    for i in range(1, len(regimes)):
        prev_regime = regimes[i - 1]
        curr_regime = regimes[i]

        # Get boundary windows
        prev_indices = prev_regime.window_indices[-2:]
        curr_indices = curr_regime.window_indices[:2]

        prev_keys = set()
        curr_keys = set()
        for widx in prev_indices:
            prev_keys.update(windows[widx].counts.keys())
        for widx in curr_indices:
            curr_keys.update(windows[widx].counts.keys())

        # Templates that appeared
        for key in curr_keys - prev_keys:
            node, template = key
            importance = NODE_IMPORTANCE.get(node, DEFAULT_NODE_IMPORTANCE)
            transition_anomalies.append(Anomaly(
                anomaly_type="regime_transition_appeared",
                node=node,
                template=template,
                window_index=curr_indices[0] if curr_indices else 0,
                timestamp_ms=curr_regime.start_ms,
                severity_score=importance * NEW_TEMPLATE_WEIGHT,
                details={
                    "from_regime": prev_regime.label,
                    "to_regime": curr_regime.label,
                },
            ))

        # Templates that disappeared
        for key in prev_keys - curr_keys:
            node, template = key
            importance = NODE_IMPORTANCE.get(node, DEFAULT_NODE_IMPORTANCE)
            transition_anomalies.append(Anomaly(
                anomaly_type="regime_transition_disappeared",
                node=node,
                template=template,
                window_index=curr_indices[0] if curr_indices else 0,
                timestamp_ms=curr_regime.start_ms,
                severity_score=importance,
                details={
                    "from_regime": prev_regime.label,
                    "to_regime": curr_regime.label,
                },
            ))

    return transition_anomalies
