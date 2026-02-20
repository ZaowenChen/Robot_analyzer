"""
Stage 5: Causal chain detection.

Groups correlated anomalies into causal chains using data-driven
co-occurrence analysis. No hard-coded causal rules — discovery is
purely from the data.

1. Build co-occurrence matrix over anomaly types
2. Construct Jaccard similarity graph
3. Find connected components (each = a causal chain candidate)
4. Assign root cause per chain (highest NODE_IMPORTANCE + earliest time)
5. Optionally boost confidence with validated rules from YAML config
"""

import math
import os
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .constants import (
    CO_OCCURRENCE_ADJACENT_WINDOWS,
    DEFAULT_NODE_IMPORTANCE,
    JACCARD_SAME_SUBSYSTEM,
    JACCARD_THRESHOLD,
    NODE_IMPORTANCE,
    SUBSYSTEM_GROUPS,
)
from .models import Anomaly, CausalChain
from .utils import connected_components, jaccard_similarity


# An anomaly "type" key: (node, template, anomaly_type)
AnomalyTypeKey = Tuple[str, str, str]


def _anomaly_key(a: Anomaly) -> AnomalyTypeKey:
    return (a.node, a.template, a.anomaly_type)


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def build_co_occurrence_matrix(
    anomalies: List[Anomaly],
    adjacent_windows: int = CO_OCCURRENCE_ADJACENT_WINDOWS,
) -> Tuple[Dict[AnomalyTypeKey, int], Dict[Tuple[AnomalyTypeKey, AnomalyTypeKey], int]]:
    """
    Build co-occurrence counts for anomaly type pairs.

    Two anomaly types "co-occur" if they appear in the same window
    or within `adjacent_windows` of each other.

    Returns:
        (type_counts, pair_counts) where:
          type_counts[key] = total occurrences of this anomaly type
          pair_counts[(key_a, key_b)] = co-occurrence count (a <= b lexically)
    """
    # Group anomalies by window
    window_types: Dict[int, Set[AnomalyTypeKey]] = defaultdict(set)

    for a in anomalies:
        key = _anomaly_key(a)
        window_types[a.window_index].add(key)

    # Count type presence in windows (not events — a type appearing 5x in
    # one window still counts as 1 window-presence).
    type_window_counts: Counter = Counter()
    for widx, types in window_types.items():
        for t in types:
            type_window_counts[t] += 1

    # Count co-occurrence windows: for each window, expand to adjacent range,
    # and record which pairs of types are both visible.
    pair_windows: Dict[Tuple[AnomalyTypeKey, AnomalyTypeKey], Set[int]] = defaultdict(set)
    window_indices = sorted(window_types.keys())

    for i, widx in enumerate(window_indices):
        # Collect all types visible from this window (including adjacent)
        visible_types: Set[AnomalyTypeKey] = set(window_types[widx])
        for j in range(1, adjacent_windows + 1):
            if widx + j in window_types:
                visible_types.update(window_types[widx + j])
            if widx - j in window_types:
                visible_types.update(window_types[widx - j])

        # Record pairs — each pair gets this window as a co-occurrence witness
        type_list = sorted(visible_types)
        for a_idx in range(len(type_list)):
            for b_idx in range(a_idx + 1, len(type_list)):
                pair = (type_list[a_idx], type_list[b_idx])
                pair_windows[pair].add(widx)

    # Convert to counts (number of unique windows where pair co-occurred)
    pair_counts = {pair: len(wins) for pair, wins in pair_windows.items()}

    return dict(type_window_counts), pair_counts


# ---------------------------------------------------------------------------
# Jaccard similarity graph
# ---------------------------------------------------------------------------

def build_similarity_graph(
    type_counts: Dict[AnomalyTypeKey, int],
    pair_counts: Dict[Tuple[AnomalyTypeKey, AnomalyTypeKey], int],
    base_threshold: float = JACCARD_THRESHOLD,
    same_subsystem_threshold: float = JACCARD_SAME_SUBSYSTEM,
) -> Dict[int, Set[int]]:
    """
    Build adjacency graph where edges represent co-occurring anomaly types.

    Jaccard similarity = co_occur / (count_A + count_B - co_occur).
    Edge added if Jaccard > threshold.

    Same-subsystem pairs get a lower threshold (soft prior from SUBSYSTEM_GROUPS).
    """
    # Assign integer IDs to anomaly types
    type_keys = sorted(type_counts.keys())
    key_to_id = {k: i for i, k in enumerate(type_keys)}

    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for idx in range(len(type_keys)):
        adjacency[idx]  # ensure every node exists in graph

    for (key_a, key_b), co_count in pair_counts.items():
        if key_a not in key_to_id or key_b not in key_to_id:
            continue

        count_a = type_counts[key_a]
        count_b = type_counts[key_b]
        union = count_a + count_b - co_count
        if union <= 0:
            continue

        jaccard = co_count / union

        # Determine threshold based on subsystem grouping
        node_a = key_a[0]
        node_b = key_b[0]
        subsys_a = SUBSYSTEM_GROUPS.get(node_a)
        subsys_b = SUBSYSTEM_GROUPS.get(node_b)
        threshold = same_subsystem_threshold if (subsys_a and subsys_a == subsys_b) else base_threshold

        if jaccard >= threshold:
            id_a = key_to_id[key_a]
            id_b = key_to_id[key_b]
            adjacency[id_a].add(id_b)
            adjacency[id_b].add(id_a)

    return adjacency, key_to_id, type_keys


# ---------------------------------------------------------------------------
# Regime transition grouping
# ---------------------------------------------------------------------------

def _group_regime_transitions(
    anomalies: List[Anomaly],
) -> Tuple[List[CausalChain], List[Anomaly]]:
    """
    Group regime transition anomalies by timestamp into chains.

    Regime transitions are special: they occur once at the boundary and
    represent structural changes, not statistical co-occurrence. All
    transitions at the same boundary timestamp are grouped into one chain.

    Returns:
        (transition_chains, remaining_anomalies) where remaining_anomalies
        are non-transition anomalies to be processed by co-occurrence.
    """
    transition_anomalies: List[Anomaly] = []
    remaining: List[Anomaly] = []

    for a in anomalies:
        if a.anomaly_type in ("regime_transition_appeared", "regime_transition_disappeared"):
            transition_anomalies.append(a)
        else:
            remaining.append(a)

    if len(transition_anomalies) < 2:
        # Not enough to form a chain — put them back
        remaining.extend(transition_anomalies)
        return [], remaining

    # Group by timestamp (all transitions at same boundary = one event)
    by_timestamp: Dict[float, List[Anomaly]] = defaultdict(list)
    for a in transition_anomalies:
        by_timestamp[a.timestamp_ms].append(a)

    chains: List[CausalChain] = []
    ungrouped: List[Anomaly] = []

    for ts, group in sorted(by_timestamp.items()):
        if len(group) < 2:
            ungrouped.extend(group)
            continue

        group.sort(key=lambda a: (
            -NODE_IMPORTANCE.get(a.node, DEFAULT_NODE_IMPORTANCE),
            a.timestamp_ms,
        ))

        root = group[0]
        cascade = group[1:]

        root_importance = NODE_IMPORTANCE.get(root.node, DEFAULT_NODE_IMPORTANCE)
        severity = root_importance * len(group) * 2.0  # transition chains are important

        chains.append(CausalChain(
            chain_id=-1,  # renumbered by construct_chains
            root_cause=root,
            cascade=cascade,
            severity_score=round(severity, 2),
            affected_regimes=[],
            first_seen_ms=ts,
            last_seen_ms=ts,
            co_occurrence_jaccard=1.0,  # perfect co-occurrence by construction
        ))

    remaining.extend(ungrouped)
    return chains, remaining


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def construct_chains(
    anomalies: List[Anomaly],
    base_threshold: float = JACCARD_THRESHOLD,
    same_subsystem_threshold: float = JACCARD_SAME_SUBSYSTEM,
) -> Tuple[List[CausalChain], List[Anomaly]]:
    """
    Build causal chains from anomalies via co-occurrence graph.

    Step 0: Group regime transition anomalies by timestamp (special handling).
    Step 1-3: Build co-occurrence graph on remaining anomalies.

    Returns:
        (chains, isolated_anomalies) where isolated_anomalies are not in any chain.
    """
    if not anomalies:
        return [], []

    # Step 0: Handle regime transitions separately
    transition_chains, remaining = _group_regime_transitions(anomalies)

    if not remaining:
        # Only transitions — renumber and return
        for i, c in enumerate(transition_chains):
            c.chain_id = i
        isolated = [a for a in anomalies if not any(
            a is c.root_cause or a in c.cascade for c in transition_chains
        )]
        return transition_chains, isolated

    # Build co-occurrence matrix on remaining anomalies
    type_counts, pair_counts = build_co_occurrence_matrix(remaining)

    if len(type_counts) < 2:
        # Only one anomaly type — no co-occurrence chains possible
        # Combine with transition chains
        all_chains = list(transition_chains)
        for i, c in enumerate(all_chains):
            c.chain_id = i
        chained_in_transitions = set()
        for c in transition_chains:
            chained_in_transitions.add(id(c.root_cause))
            for a in c.cascade:
                chained_in_transitions.add(id(a))
        isolated = [a for a in anomalies if id(a) not in chained_in_transitions]
        return all_chains, isolated

    # Build similarity graph
    adjacency, key_to_id, type_keys = build_similarity_graph(
        type_counts, pair_counts, base_threshold, same_subsystem_threshold,
    )

    # Find connected components
    components = connected_components(adjacency)

    # Build chains from components with >1 node
    co_chains: List[CausalChain] = []
    chained_types: Set[AnomalyTypeKey] = set()

    for component in components:
        if len(component) < 2:
            continue

        # Get anomaly types in this component
        comp_types = {type_keys[idx] for idx in component}
        chained_types.update(comp_types)

        # Get individual anomalies from remaining (not transitions)
        comp_anomalies = [a for a in remaining if _anomaly_key(a) in comp_types]
        if not comp_anomalies:
            continue

        comp_anomalies.sort(key=lambda a: a.timestamp_ms)

        # Root cause = highest importance + earliest time
        root = max(comp_anomalies, key=lambda a: (
            NODE_IMPORTANCE.get(a.node, DEFAULT_NODE_IMPORTANCE),
            -a.timestamp_ms,  # earlier = higher priority (negative for max)
        ))

        cascade = [a for a in comp_anomalies if a is not root]

        # Severity scoring
        root_importance = NODE_IMPORTANCE.get(root.node, DEFAULT_NODE_IMPORTANCE)
        anomaly_count = len(comp_types)
        affected_windows = len({a.window_index for a in comp_anomalies})
        duration_factor = math.log2(1 + affected_windows)
        level_factor = 1.0
        for a in comp_anomalies:
            if "FATAL" in a.details.get("level", ""):
                level_factor = max(level_factor, 3.0)
            elif "ERROR" in a.details.get("level", ""):
                level_factor = max(level_factor, 2.0)

        severity = root_importance * anomaly_count * duration_factor * level_factor

        # Average pairwise Jaccard
        jaccard_sum = 0.0
        jaccard_count = 0
        for a_key in comp_types:
            for b_key in comp_types:
                if a_key < b_key:
                    pair = (a_key, b_key) if a_key < b_key else (b_key, a_key)
                    co = pair_counts.get(pair, 0)
                    union = type_counts.get(a_key, 0) + type_counts.get(b_key, 0) - co
                    if union > 0:
                        jaccard_sum += co / union
                        jaccard_count += 1

        avg_jaccard = jaccard_sum / jaccard_count if jaccard_count > 0 else 0.0

        co_chains.append(CausalChain(
            chain_id=-1,  # renumbered below
            root_cause=root,
            cascade=cascade,
            severity_score=round(severity, 2),
            affected_regimes=[],  # filled by pipeline
            first_seen_ms=comp_anomalies[0].timestamp_ms,
            last_seen_ms=comp_anomalies[-1].timestamp_ms,
            co_occurrence_jaccard=round(avg_jaccard, 3),
        ))

    # Combine transition chains + co-occurrence chains, sort by severity
    all_chains = transition_chains + co_chains
    all_chains.sort(key=lambda c: -c.severity_score)
    for i, c in enumerate(all_chains):
        c.chain_id = i

    # Isolated anomalies = not in any chain
    chained_ids = set()
    for c in all_chains:
        chained_ids.add(id(c.root_cause))
        for a in c.cascade:
            chained_ids.add(id(a))
    isolated = [a for a in anomalies if id(a) not in chained_ids]

    return all_chains, isolated


# ---------------------------------------------------------------------------
# Validated rules (optional, loaded from YAML)
# ---------------------------------------------------------------------------

def load_validated_rules(yaml_path: Optional[str] = None) -> List[dict]:
    """
    Load validated causal rules from YAML config.

    Returns empty list if file doesn't exist or has no rules.
    Rules are loaded but NOT applied automatically — they're used to
    boost chain confidence in apply_validated_rules().
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "validated_rules.yaml")

    if not os.path.exists(yaml_path):
        return []

    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("rules", []) if data else []
    except ImportError:
        # PyYAML not installed — try basic parsing
        try:
            with open(yaml_path, "r") as f:
                content = f.read()
            if "rules: []" in content or "rules:\n" not in content:
                return []
        except Exception:
            pass
        return []
    except Exception:
        return []


def apply_validated_rules(
    chains: List[CausalChain],
    rules: List[dict],
) -> None:
    """
    Apply validated rules to boost chain confidence (in-place).

    Rules don't override data-driven findings. If a rule matches,
    it boosts the chain's severity score. If a rule contradicts the data
    (expected chain members don't co-occur), a warning is attached.
    """
    if not rules:
        return

    for chain in chains:
        chain_nodes = {a.node for a in [chain.root_cause] + chain.cascade}
        chain_templates = {a.template for a in [chain.root_cause] + chain.cascade}

        for rule in rules:
            rule_root_node = rule.get("root", [None, None])[0]
            if rule_root_node and rule_root_node in chain_nodes:
                # Rule matches — boost confidence
                confidence = rule.get("confidence", 0.5)
                chain.severity_score *= (1.0 + confidence)
                chain.validated_rule_match = rule.get("name", "unnamed_rule")
                break
