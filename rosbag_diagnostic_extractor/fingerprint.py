"""
Stage 2: Windowed fingerprinting + regime segmentation.

1. Node frequency analysis: classify each node as periodic/quasi-periodic/event-driven
2. Windowed fingerprinting: Counter[(node, template)] per time window
3. Regime segmentation via CUSUM change-point detection on window deviation scores
"""

import math
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from .constants import (
    CV_PERIODIC_THRESHOLD,
    CV_QUASI_PERIODIC_THRESHOLD,
    CUSUM_THRESHOLD_SIGMA,
    MIN_COUNT_FOR_PERIODIC,
    MIN_WINDOW_SIZE_MS,
    TRANSIENT_REGIME_MAX_WINDOWS,
    UNSTABLE_TEMPLATES,
    BASELINE_PRESENCE_THRESHOLD,
)
from .models import NodeProfile, ParsedLogRecord, Regime, WindowFingerprint
from .utils import cusum, jaccard_distance, count_divergence, median


# ---------------------------------------------------------------------------
# Node frequency analysis
# ---------------------------------------------------------------------------

def analyze_node_frequencies(
    records: List[ParsedLogRecord],
) -> Dict[str, NodeProfile]:
    """
    Classify each node by its firing pattern.

    Collects per-node timestamps, computes inter-message intervals,
    CV (coefficient of variation), and classifies:
      - periodic (CV < 0.3): heartbeat nodes
      - quasi_periodic (0.3 <= CV < 1.0): roughly regular
      - event_driven (CV >= 1.0 or count < 20): irregular
    """
    # Collect timestamps and templates per node
    node_timestamps: Dict[str, List[float]] = defaultdict(list)
    node_templates: Dict[str, Counter] = defaultdict(Counter)

    for rec in records:
        node_timestamps[rec.node].append(rec.timestamp_ms)
        node_templates[rec.node][rec.template] += 1

    profiles: Dict[str, NodeProfile] = {}

    for node, timestamps in node_timestamps.items():
        timestamps.sort()
        count = len(timestamps)

        if count < 2:
            profiles[node] = NodeProfile(
                node=node,
                total_count=count,
                median_interval_ms=0.0,
                cv=float("inf"),
                message_rate_hz=0.0,
                classification="event_driven",
                templates=node_templates[node],
            )
            continue

        # Compute inter-message intervals
        intervals = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, count)
            if timestamps[i] - timestamps[i - 1] > 0
        ]

        if not intervals:
            profiles[node] = NodeProfile(
                node=node,
                total_count=count,
                median_interval_ms=0.0,
                cv=float("inf"),
                message_rate_hz=0.0,
                classification="event_driven",
                templates=node_templates[node],
            )
            continue

        med_interval = median(intervals)
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_interval = math.sqrt(variance)

        cv = std_interval / mean_interval if mean_interval > 0 else float("inf")

        # Duration-based message rate
        duration_ms = timestamps[-1] - timestamps[0]
        rate_hz = (count / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

        # Classification
        if count < MIN_COUNT_FOR_PERIODIC:
            classification = "event_driven"
        elif cv < CV_PERIODIC_THRESHOLD:
            classification = "periodic"
        elif cv < CV_QUASI_PERIODIC_THRESHOLD:
            classification = "quasi_periodic"
        else:
            classification = "event_driven"

        profiles[node] = NodeProfile(
            node=node,
            total_count=count,
            median_interval_ms=med_interval,
            cv=round(cv, 4),
            message_rate_hz=round(rate_hz, 2),
            classification=classification,
            templates=node_templates[node],
        )

    return profiles


# ---------------------------------------------------------------------------
# Window size auto-detection
# ---------------------------------------------------------------------------

def auto_window_size(profiles: Dict[str, NodeProfile]) -> float:
    """
    Auto-detect window size from node frequency data.

    Window = max(longest periodic node interval, MIN_WINDOW_SIZE_MS).
    Ensures each window contains at least one full cycle of every periodic node.
    """
    periodic_intervals = [
        p.median_interval_ms
        for p in profiles.values()
        if p.classification == "periodic" and p.median_interval_ms > 0
    ]
    if periodic_intervals:
        return max(max(periodic_intervals), MIN_WINDOW_SIZE_MS)
    return MIN_WINDOW_SIZE_MS


# ---------------------------------------------------------------------------
# Windowed fingerprinting
# ---------------------------------------------------------------------------

def _is_unstable(template: str) -> bool:
    """Check if template matches structural instability filter."""
    return any(p.search(template) for p in UNSTABLE_TEMPLATES)


def build_fingerprints(
    records: List[ParsedLogRecord],
    window_size_ms: float,
    profiles: Dict[str, NodeProfile],
) -> List[WindowFingerprint]:
    """
    Build structural fingerprints for each time window.

    Each fingerprint is a Counter of (node, template) tuples.
    Counts are also normalized by expected rate for comparability.
    """
    if not records:
        return []

    start_ms = records[0].timestamp_ms
    end_ms = records[-1].timestamp_ms

    # Pre-compute expected counts per window for periodic/quasi-periodic nodes
    expected_per_window: Dict[Tuple[str, str], float] = {}
    for node, profile in profiles.items():
        if profile.classification in ("periodic", "quasi_periodic") and profile.median_interval_ms > 0:
            expected_rate = window_size_ms / profile.median_interval_ms
            for tmpl in profile.templates:
                # Proportion of this template among all templates for this node
                tmpl_fraction = profile.templates[tmpl] / profile.total_count
                expected_per_window[(node, tmpl)] = expected_rate * tmpl_fraction

    # Bin records into windows
    windows: List[WindowFingerprint] = []
    current_start = start_ms
    window_idx = 0
    rec_idx = 0

    while current_start < end_ms:
        current_end = current_start + window_size_ms
        counts: Counter = Counter()

        while rec_idx < len(records) and records[rec_idx].timestamp_ms < current_end:
            rec = records[rec_idx]
            if not _is_unstable(rec.template):
                counts[(rec.node, rec.template)] += 1
            rec_idx += 1

        # Normalize counts
        normalized: Dict[Tuple[str, str], float] = {}
        for key, count in counts.items():
            expected = expected_per_window.get(key, count)  # fallback: expected = observed
            if expected > 0:
                normalized[key] = count / expected
            else:
                normalized[key] = float(count)

        windows.append(WindowFingerprint(
            window_index=window_idx,
            start_ms=current_start,
            end_ms=current_end,
            counts=counts,
            normalized_counts=normalized,
        ))

        current_start = current_end
        window_idx += 1

    return windows


# ---------------------------------------------------------------------------
# Regime segmentation
# ---------------------------------------------------------------------------

def compute_deviation_scores(windows: List[WindowFingerprint]) -> List[float]:
    """
    Compute deviation score between consecutive window pairs.

    Score = jaccard_distance(keys) + count_divergence(normalized_counts).
    """
    if len(windows) < 2:
        return []

    scores: List[float] = []
    for i in range(1, len(windows)):
        keys_a = frozenset(windows[i - 1].counts.keys())
        keys_b = frozenset(windows[i].counts.keys())

        jd = jaccard_distance(keys_a, keys_b)
        cd = count_divergence(
            windows[i - 1].normalized_counts,
            windows[i].normalized_counts,
        )

        scores.append(jd + cd)

    return scores


def segment_regimes(
    windows: List[WindowFingerprint],
    cusum_threshold: float = CUSUM_THRESHOLD_SIGMA,
) -> List[Regime]:
    """
    Detect regime boundaries using CUSUM on window deviation scores.

    Returns list of Regime objects with boundaries and metadata.
    Short regimes (< TRANSIENT_REGIME_MAX_WINDOWS) between identical
    longer regimes are classified as transient anomalies.
    """
    if not windows:
        return []

    if len(windows) == 1:
        return [Regime(
            label="regime_0",
            start_ms=windows[0].start_ms,
            end_ms=windows[0].end_ms,
            window_indices=[0],
        )]

    # Compute deviation scores
    deviation_scores = compute_deviation_scores(windows)
    if not deviation_scores:
        return [Regime(
            label="regime_0",
            start_ms=windows[0].start_ms,
            end_ms=windows[-1].end_ms,
            window_indices=list(range(len(windows))),
        )]

    # CUSUM change-point detection
    change_points = cusum(deviation_scores, threshold=cusum_threshold)

    # Convert change point indices to regime boundaries
    # Change points are indices into deviation_scores (between windows i and i+1)
    boundaries = sorted(set([0] + [cp + 1 for cp in change_points] + [len(windows)]))

    regimes: List[Regime] = []
    for idx in range(len(boundaries) - 1):
        start_widx = boundaries[idx]
        end_widx = boundaries[idx + 1]
        window_indices = list(range(start_widx, end_widx))

        regimes.append(Regime(
            label=f"regime_{idx}",
            start_ms=windows[start_widx].start_ms,
            end_ms=windows[end_widx - 1].end_ms,
            window_indices=window_indices,
        ))

    # Merge transient regimes back into neighbors
    regimes = _merge_transient_regimes(regimes)

    return regimes


def _merge_transient_regimes(regimes: List[Regime]) -> List[Regime]:
    """
    Merge short regimes (< TRANSIENT_REGIME_MAX_WINDOWS) back into
    the preceding regime. These are transient anomalies, not true
    regime changes.
    """
    if len(regimes) <= 1:
        return regimes

    merged: List[Regime] = [regimes[0]]

    for regime in regimes[1:]:
        if len(regime.window_indices) < TRANSIENT_REGIME_MAX_WINDOWS:
            # Absorb into previous regime
            prev = merged[-1]
            prev.end_ms = regime.end_ms
            prev.window_indices.extend(regime.window_indices)
        else:
            merged.append(regime)

    # Re-label
    for i, r in enumerate(merged):
        r.label = f"regime_{i}"

    return merged


# ---------------------------------------------------------------------------
# Baseline fingerprint computation
# ---------------------------------------------------------------------------

def compute_regime_baselines(
    regimes: List[Regime],
    windows: List[WindowFingerprint],
) -> None:
    """
    Compute baseline fingerprint for each regime (in-place).

    Baseline template set = templates appearing in >60% of the regime's windows.
    Expected count = median count across the regime's windows for each template.
    """
    for regime in regimes:
        if not regime.window_indices:
            continue

        regime_windows = [windows[i] for i in regime.window_indices]
        n_windows = len(regime_windows)

        # Count presence of each (node, template) across windows
        presence: Counter = Counter()
        counts_per_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for w in regime_windows:
            seen_keys = set(w.counts.keys())
            for key in seen_keys:
                presence[key] += 1
                counts_per_key[key].append(w.counts[key])
            # Keys not in this window get 0
            for key in counts_per_key:
                if key not in seen_keys:
                    counts_per_key[key].append(0)

        # Baseline = templates in >60% of windows, with median count
        threshold = n_windows * BASELINE_PRESENCE_THRESHOLD
        baseline: Counter = Counter()
        for key, count in presence.items():
            if count >= threshold:
                baseline[key] = int(median(counts_per_key[key]))

        regime.baseline = baseline
