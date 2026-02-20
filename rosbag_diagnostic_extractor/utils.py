"""
Utility functions and algorithms for the diagnostic extractor.

All algorithms are implemented directly — no external ML dependencies.
Only Python stdlib + numpy (for efficient array ops where needed).
"""

import math
import re
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Set, Tuple


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------

# Matches: integers, decimals, negative numbers, scientific notation, hex strings, UUIDs.
# Order matters — UUID/hex must be checked before plain integers to avoid partial matches.
_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_HEX_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
_NUMBER_PATTERN = re.compile(
    r"[-+]?"
    r"(?:\d+\.?\d*|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


def templatize(message: str) -> Tuple[str, List[float]]:
    """
    Replace all numeric values, UUIDs, and hex strings with '#' placeholders.
    Extract numeric values in order of appearance.

    Returns:
        (template, values) where template has '#' placeholders

    Examples:
        >>> templatize("[CHASSIS] IMU orientation(1802 1775 74)")
        ('[CHASSIS] IMU orientation(# # #)', [1802.0, 1775.0, 74.0])
        >>> templatize("localization_status  2")
        ('localization_status  #', [2.0])
        >>> templatize("Error code -5 at 0.134")
        ('Error code # at #', [-5.0, 0.134])
    """
    values: List[float] = []

    # Phase 1: Replace UUIDs and hex strings (non-numeric, just structural)
    result = _UUID_PATTERN.sub("#", message)
    result = _HEX_PATTERN.sub("#", result)

    # Phase 2: Replace numeric values and extract them
    def _replace_and_collect(match: re.Match) -> str:
        try:
            values.append(float(match.group(0)))
        except ValueError:
            pass
        return "#"

    result = _NUMBER_PATTERN.sub(_replace_and_collect, result)
    return result, values


# ---------------------------------------------------------------------------
# CUSUM change-point detection
# ---------------------------------------------------------------------------

def cusum(
    series: List[float],
    threshold: float = 3.0,
    drift: float = 0.0,
) -> List[int]:
    """
    Cumulative Sum (CUSUM) change-point detection.

    Detects points where the series shifts significantly above its running mean.

    Args:
        series: Time series of deviation scores
        threshold: Detection threshold in σ units
        drift: Allowable drift before accumulating (default 0 = raw CUSUM)

    Returns:
        List of indices where change points are detected
    """
    if len(series) < 3:
        return []

    # Compute baseline statistics
    mean = sum(series) / len(series)
    variance = sum((x - mean) ** 2 for x in series) / len(series)
    std = math.sqrt(variance) if variance > 0 else 1.0

    sigma_threshold = mean + threshold * std
    change_points: List[int] = []
    cumsum_pos = 0.0

    for i, value in enumerate(series):
        cumsum_pos = max(0.0, cumsum_pos + value - mean - drift)
        if cumsum_pos > sigma_threshold:
            change_points.append(i)
            cumsum_pos = 0.0  # Reset after detection

    return change_points


# ---------------------------------------------------------------------------
# EWMA tracker
# ---------------------------------------------------------------------------

class EWMATracker:
    """
    Exponentially Weighted Moving Average tracker with standard deviation.

    Tracks rolling mean and std for continuous value streams.
    Supports regime boundary resets.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean: float = 0.0
        self.var: float = 0.0
        self.count: int = 0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")

    def update(self, value: float) -> None:
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

        if self.count == 1:
            self.mean = value
            self.var = 0.0
            return

        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)

    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.var > 0 else 0.0

    def is_anomalous(self, value: float, sigma: float = 3.0) -> bool:
        """Check if value deviates beyond ±sigma standard deviations."""
        if self.count < 10 or self.std < 1e-9:
            return False
        return abs(value - self.mean) > sigma * self.std

    def reset(self) -> None:
        """Reset at regime boundaries."""
        self.mean = 0.0
        self.var = 0.0
        self.count = 0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6) if self.min_val != float("inf") else None,
            "max": round(self.max_val, 6) if self.max_val != float("-inf") else None,
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# Jaccard distance
# ---------------------------------------------------------------------------

def jaccard_distance(set_a: FrozenSet, set_b: FrozenSet) -> float:
    """
    Jaccard distance between two sets: 1 - |A ∩ B| / |A ∪ B|.
    Returns 1.0 if both sets are empty (maximally different by convention).
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union


def jaccard_similarity(set_a: FrozenSet, set_b: FrozenSet) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Connected components (BFS)
# ---------------------------------------------------------------------------

def connected_components(adjacency: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find connected components in an undirected graph via BFS.

    Args:
        adjacency: node_id -> set of neighbor node_ids

    Returns:
        List of sets, each set is a connected component
    """
    visited: Set[int] = set()
    components: List[Set[int]] = []

    for node in adjacency:
        if node in visited:
            continue
        # BFS from this node
        component: Set[int] = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if component:
            components.append(component)

    return components


# ---------------------------------------------------------------------------
# Normalized count divergence
# ---------------------------------------------------------------------------

def count_divergence(
    counts_a: Dict[Tuple[str, str], float],
    counts_b: Dict[Tuple[str, str], float],
) -> float:
    """
    Symmetric divergence between two normalized count dictionaries.
    Uses chi-squared-like distance: sum of (a - b)^2 / (a + b) for shared keys.
    """
    all_keys = set(counts_a.keys()) | set(counts_b.keys())
    if not all_keys:
        return 0.0

    total = 0.0
    for key in all_keys:
        a = counts_a.get(key, 0.0)
        b = counts_b.get(key, 0.0)
        denom = a + b
        if denom > 0:
            total += (a - b) ** 2 / denom
    return total


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def median(values: List[float]) -> float:
    """Compute median of a list. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentile(values: List[float], pct: float) -> float:
    """Compute percentile (0-100) of a list. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])
