"""
Data models for the diagnostic extractor pipeline.

All stages communicate via these dataclasses. Kept minimal — only fields
that are actually used downstream.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stage 1: Parsed log records
# ---------------------------------------------------------------------------

@dataclass
class ParsedLogRecord:
    """Unified output format from both text log and bag parsers."""
    level: str                      # "INFO", "WARN", "ERROR", "FATAL", "DEBUG"
    timestamp_ms: float             # milliseconds from midnight (for fast arithmetic)
    node: str                       # node name with leading "/"
    raw_message: str                # full message text
    template: str                   # message with numbers replaced by "#"
    values: List[float]             # all numeric values extracted, in order
    line_number: int                # 1-indexed line number (or message index for bags)
    source_file: str = ""           # bag name or log filename
    absolute_timestamp: float = 0.0 # Unix epoch seconds (available for bags, 0 for text)


@dataclass
class NodeProfile:
    """Frequency classification for a single ROS node."""
    node: str
    total_count: int
    median_interval_ms: float       # median inter-message interval
    cv: float                       # coefficient of variation = std / mean
    message_rate_hz: float          # messages per second
    classification: str             # "periodic", "quasi_periodic", "event_driven"
    templates: Counter = field(default_factory=Counter)  # template -> count


# ---------------------------------------------------------------------------
# Stage 2: Fingerprints and regimes
# ---------------------------------------------------------------------------

@dataclass
class WindowFingerprint:
    """Structural fingerprint for one time window."""
    window_index: int
    start_ms: float
    end_ms: float
    counts: Counter                 # Counter[(node, template)] -> count
    normalized_counts: Dict[Tuple[str, str], float] = field(default_factory=dict)
    anomaly_score: float = 0.0


@dataclass
class Regime:
    """A contiguous time range where the log cycle pattern is stable."""
    label: str                      # "regime_0", "regime_1", ...
    start_ms: float
    end_ms: float
    window_indices: List[int]       # indices into the window list
    baseline: Optional[Counter] = None  # median template set + expected counts
    description: str = ""
    anomaly_density: float = 0.0    # fraction of windows that are anomalous


# ---------------------------------------------------------------------------
# Stage 3-4: Anomalies
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """A detected deviation — structural or content-based."""
    anomaly_type: str               # "missing_template", "new_template", "rate_anomaly",
                                    # "value_deviation", "state_transition", "counter_stall"
    node: str
    template: str
    window_index: int
    timestamp_ms: float
    severity_score: float           # importance-weighted score
    details: Dict = field(default_factory=dict)
    # details may contain: expected_count, observed_count, deviation_sigma,
    # old_value, new_value, field_name, etc.


# ---------------------------------------------------------------------------
# Stage 5: Causal chains
# ---------------------------------------------------------------------------

@dataclass
class CausalChain:
    """A group of correlated anomalies with an identified root cause."""
    chain_id: int
    root_cause: Anomaly             # the anomaly with highest importance + earliest time
    cascade: List[Anomaly]          # downstream anomalies, ordered by time
    severity_score: float           # composite severity
    affected_regimes: List[str]     # regime labels
    first_seen_ms: float
    last_seen_ms: float
    co_occurrence_jaccard: float    # average pairwise Jaccard in the chain
    validated_rule_match: Optional[str] = None  # name of matched rule, if any


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticReport:
    """Complete output from the diagnostic pipeline."""
    metadata: Dict                  # source_file, total_lines, time_range, duration
    node_profiles: Dict[str, NodeProfile]
    regimes: List[Regime]
    causal_chains: List[CausalChain]
    isolated_anomalies: List[Anomaly]  # anomalies not part of any chain
    sensor_summary: Dict            # per-extractor findings
    all_anomalies: List[Anomaly]    # full list for annotated log output
