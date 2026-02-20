"""
Stage 4: Content anomaly detection.

For structurally expected messages (same template every cycle), detect
when the actual numeric values deviate from the norm.

Three types of value tracking:
  - Continuous values: EWMA mean + std, flag >3σ deviations
  - Discrete states: track transitions, flag any change
  - Monotonic counters: track deltas, flag drops to 0
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .constants import EWMA_ALPHA, VALUE_DEVIATION_SIGMA
from .models import Anomaly, ParsedLogRecord, Regime, WindowFingerprint
from .utils import EWMATracker


# ---------------------------------------------------------------------------
# Per-template value tracker
# ---------------------------------------------------------------------------

class TemplateValueTracker:
    """
    Tracks numeric values for a specific (node, template) pair.

    On creation, analyzes the value positions to determine:
    - Which positions are continuous (floating-point sensor values)
    - Which are discrete (small integer states)
    - Which are monotonic counters
    """

    def __init__(self, node: str, template: str, alpha: float = EWMA_ALPHA):
        self.node = node
        self.template = template
        self._trackers: Dict[int, EWMATracker] = {}  # position -> EWMA on value (or delta for counters)
        self._discrete_values: Dict[int, Optional[float]] = {}  # position -> last value
        self._distinct_counts: Dict[int, Set[float]] = defaultdict(set)
        self._prev_values: Dict[int, Optional[float]] = {}  # for counter delta tracking
        self._monotonic_up: Dict[int, int] = {}  # consecutive increasing count
        self._monotonic_down: Dict[int, int] = {}  # consecutive decreasing count
        self._alpha = alpha
        self._total_updates = 0
        self._CLASSIFY_AFTER = 20   # classify after this many updates
        self._DISCRETE_MAX_DISTINCT = 10    # max distinct values for discrete
        self._MONOTONIC_MIN_RUN = 15  # min consecutive monotonic to classify as counter
        self._classified: Dict[int, str] = {}  # position -> "continuous"/"discrete"/"counter"

    def update(self, values: List[float], timestamp_ms: float, window_index: int) -> List[Anomaly]:
        """
        Update trackers with new values. Returns any detected anomalies.
        """
        self._total_updates += 1
        anomalies: List[Anomaly] = []

        for pos, val in enumerate(values):
            # Initialize tracker if first time seeing this position (or after reset)
            if pos not in self._trackers:
                self._trackers[pos] = EWMATracker(alpha=self._alpha)
            if pos not in self._discrete_values:
                self._discrete_values[pos] = None
            if pos not in self._prev_values:
                self._prev_values[pos] = None
            if pos not in self._monotonic_up:
                self._monotonic_up[pos] = 0
                self._monotonic_down[pos] = 0

            # Track distinct values for classification
            self._distinct_counts[pos].add(val)

            # Track monotonicity
            prev_val = self._prev_values[pos]
            if prev_val is not None:
                if val > prev_val:
                    self._monotonic_up[pos] += 1
                    self._monotonic_down[pos] = 0
                elif val < prev_val:
                    self._monotonic_down[pos] += 1
                    self._monotonic_up[pos] = 0
                # equal: don't reset either counter
            self._prev_values[pos] = val

            # Classify this position if we have enough data
            if pos not in self._classified and self._total_updates >= self._CLASSIFY_AFTER:
                # Check for monotonic counter first
                if (self._monotonic_up[pos] >= self._MONOTONIC_MIN_RUN or
                        self._monotonic_down[pos] >= self._MONOTONIC_MIN_RUN):
                    self._classified[pos] = "counter"
                    # Reset tracker to track deltas instead of raw values
                    self._trackers[pos] = EWMATracker(alpha=self._alpha)
                elif len(self._distinct_counts[pos]) <= self._DISCRETE_MAX_DISTINCT:
                    self._classified[pos] = "discrete"
                else:
                    self._classified[pos] = "continuous"

            classification = self._classified.get(pos, "unclassified")

            if classification == "discrete":
                # Discrete state tracking — flag any transition
                prev = self._discrete_values[pos]
                if prev is not None and val != prev:
                    anomalies.append(Anomaly(
                        anomaly_type="state_transition",
                        node=self.node,
                        template=self.template,
                        window_index=window_index,
                        timestamp_ms=timestamp_ms,
                        severity_score=5.0,  # base score, adjusted by importance in pipeline
                        details={
                            "field_position": pos,
                            "old_value": prev,
                            "new_value": val,
                        },
                    ))
                self._discrete_values[pos] = val
            elif classification == "counter":
                # Monotonic counter — track deltas, flag stalls or resets
                if prev_val is not None:
                    delta = val - prev_val
                    tracker = self._trackers[pos]
                    if tracker.count > 10 and tracker.is_anomalous(delta, sigma=VALUE_DEVIATION_SIGMA):
                        anomalies.append(Anomaly(
                            anomaly_type="counter_anomaly",
                            node=self.node,
                            template=self.template,
                            window_index=window_index,
                            timestamp_ms=timestamp_ms,
                            severity_score=abs(delta - tracker.mean) / max(tracker.std, 1e-9),
                            details={
                                "field_position": pos,
                                "delta": delta,
                                "expected_delta_mean": round(tracker.mean, 6),
                                "expected_delta_std": round(tracker.std, 6),
                            },
                        ))
                    tracker.update(delta)
            elif classification == "continuous":
                # Continuous value tracking — EWMA + 3σ detection
                tracker = self._trackers[pos]
                if tracker.is_anomalous(val, sigma=VALUE_DEVIATION_SIGMA):
                    anomalies.append(Anomaly(
                        anomaly_type="value_deviation",
                        node=self.node,
                        template=self.template,
                        window_index=window_index,
                        timestamp_ms=timestamp_ms,
                        severity_score=abs(val - tracker.mean) / max(tracker.std, 1e-9),
                        details={
                            "field_position": pos,
                            "value": val,
                            "expected_mean": round(tracker.mean, 6),
                            "expected_std": round(tracker.std, 6),
                            "sigma": round(abs(val - tracker.mean) / max(tracker.std, 1e-9), 2),
                        },
                    ))
                tracker.update(val)
            else:
                # Unclassified (warmup period) — just update tracker, don't flag
                self._trackers[pos].update(val)

        return anomalies

    def reset(self) -> None:
        """Reset at regime boundaries."""
        for tracker in self._trackers.values():
            tracker.reset()
        self._discrete_values.clear()
        self._prev_values.clear()
        self._monotonic_up.clear()
        self._monotonic_down.clear()
        self._distinct_counts.clear()
        self._classified.clear()
        self._total_updates = 0


# ---------------------------------------------------------------------------
# Content anomaly detector
# ---------------------------------------------------------------------------

def detect_content_anomalies(
    records: List[ParsedLogRecord],
    regimes: List[Regime],
    windows: List[WindowFingerprint],
) -> List[Anomaly]:
    """
    Detect content anomalies across all records.

    For each (node, template) pair, tracks numeric values and flags deviations.
    Resets tracking at regime boundaries.

    Returns list of content anomalies.
    """
    all_anomalies: List[Anomaly] = []

    # Build regime boundary timestamps for reset detection
    regime_boundaries: List[float] = []
    for i in range(1, len(regimes)):
        regime_boundaries.append(regimes[i].start_ms)

    # Map timestamp to window index
    def _window_for_ts(ts_ms: float) -> int:
        if not windows:
            return 0
        for w in windows:
            if w.start_ms <= ts_ms < w.end_ms:
                return w.window_index
        return windows[-1].window_index if windows else 0

    # Per-(node, template) trackers
    trackers: Dict[Tuple[str, str], TemplateValueTracker] = {}
    current_regime_idx = 0

    for rec in records:
        # Check for regime boundary crossings
        while (current_regime_idx < len(regime_boundaries) and
               rec.timestamp_ms >= regime_boundaries[current_regime_idx]):
            # Reset all trackers at regime boundary
            for tracker in trackers.values():
                tracker.reset()
            current_regime_idx += 1

        # Skip records with no values
        if not rec.values:
            continue

        key = (rec.node, rec.template)
        if key not in trackers:
            trackers[key] = TemplateValueTracker(rec.node, rec.template)

        widx = _window_for_ts(rec.timestamp_ms)
        anomalies = trackers[key].update(rec.values, rec.timestamp_ms, widx)
        all_anomalies.extend(anomalies)

    return all_anomalies
