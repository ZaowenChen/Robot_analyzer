"""
Stage 6: Token-budget-aware report generation.

Produces three outputs:
  1. diagnostic_report.json — full structured data
  2. llm_prompt.txt — token-budget-constrained text for LLM consumption
  3. anomaly_log.txt — original log lines with anomaly annotations

Token budget allocation:
  - Header (5%): robot context, log overview
  - Regime Timeline (15%): chronological regimes
  - Causal Chains (50%): top chains by severity
  - Isolated Anomalies (15%): ungrouped anomalies
  - Sensor Baseline (10%): abnormal sensor findings
  - Event-Driven Messages (5%): rare event-driven node messages
"""

import json
from typing import Dict, List, Optional

from .constants import (
    BUDGET_CAUSAL_CHAINS,
    BUDGET_EVENT_DRIVEN,
    BUDGET_HEADER,
    BUDGET_ISOLATED_ANOMALIES,
    BUDGET_REGIME_TIMELINE,
    BUDGET_SENSOR_BASELINE,
    CHARS_PER_TOKEN,
    DEFAULT_TOKEN_BUDGET,
    LOC_STATUS,
    NODE_IMPORTANCE,
)
from .models import (
    Anomaly,
    CausalChain,
    DiagnosticReport,
    NodeProfile,
    Regime,
)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _anomaly_to_dict(a: Anomaly) -> dict:
    return {
        "anomaly_type": a.anomaly_type,
        "node": a.node,
        "template": a.template,
        "window_index": a.window_index,
        "timestamp_ms": a.timestamp_ms,
        "severity_score": round(a.severity_score, 2),
        "details": a.details,
    }


def _chain_to_dict(c: CausalChain) -> dict:
    return {
        "chain_id": c.chain_id,
        "severity_score": c.severity_score,
        "root_cause": _anomaly_to_dict(c.root_cause),
        "cascade": [_anomaly_to_dict(a) for a in c.cascade[:20]],
        "affected_regimes": c.affected_regimes,
        "first_seen_ms": c.first_seen_ms,
        "last_seen_ms": c.last_seen_ms,
        "co_occurrence_jaccard": c.co_occurrence_jaccard,
        "validated_rule_match": c.validated_rule_match,
    }


def generate_json_report(report: DiagnosticReport) -> dict:
    """Generate the full structured JSON report."""
    return {
        "metadata": report.metadata,
        "node_profiles": {
            node: {
                "classification": p.classification,
                "median_interval_ms": p.median_interval_ms,
                "cv": p.cv,
                "message_rate_hz": p.message_rate_hz,
                "total_messages": p.total_count,
            }
            for node, p in report.node_profiles.items()
        },
        "regimes": [
            {
                "label": r.label,
                "start_ms": r.start_ms,
                "end_ms": r.end_ms,
                "duration_ms": r.end_ms - r.start_ms,
                "window_count": len(r.window_indices),
                "anomaly_density": round(r.anomaly_density, 4),
                "description": r.description,
                "baseline_template_count": len(r.baseline) if r.baseline else 0,
            }
            for r in report.regimes
        ],
        "causal_chains": [_chain_to_dict(c) for c in report.causal_chains],
        "isolated_anomalies": [_anomaly_to_dict(a) for a in report.isolated_anomalies[:50]],
        "sensor_summary": report.sensor_summary,
        "total_anomalies": len(report.all_anomalies),
    }


# ---------------------------------------------------------------------------
# LLM prompt (token-budget constrained)
# ---------------------------------------------------------------------------

def _ms_to_timestr(ms: float) -> str:
    """Convert milliseconds-from-midnight to HH:MM:SS.mmm string."""
    total_sec = ms / 1000.0
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    seconds = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def generate_llm_prompt(
    report: DiagnosticReport,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    robot_context: str = "",
) -> str:
    """
    Generate token-budget-constrained LLM prompt text.

    Adapts detail level: fewer anomalies → more baseline info;
    more anomalies → compress baselines, expand chains.
    """
    char_budget = token_budget * CHARS_PER_TOKEN
    sections: List[str] = []

    # Adaptive allocation based on anomaly count
    n_anomalies = len(report.all_anomalies)
    n_chains = len(report.causal_chains)

    # If few anomalies, give more to baselines; if many, give more to chains
    chain_budget_pct = BUDGET_CAUSAL_CHAINS
    baseline_budget_pct = BUDGET_SENSOR_BASELINE
    if n_anomalies < 5:
        chain_budget_pct -= 0.15
        baseline_budget_pct += 0.15
    elif n_anomalies > 50:
        chain_budget_pct += 0.10
        baseline_budget_pct -= 0.05

    # --- Section 1: Header ---
    header_chars = int(char_budget * BUDGET_HEADER)
    header = f"=== ROBOT DIAGNOSTIC SUMMARY ===\n"
    if robot_context:
        header += f"Context: {robot_context}\n"
    header += (
        f"Source: {report.metadata.get('source_file', 'unknown')}\n"
        f"Duration: {report.metadata.get('duration_ms', 0)/1000:.1f}s | "
        f"Lines: {report.metadata.get('total_lines', 0)} | "
        f"Parsed: {report.metadata.get('parsed_lines', 0)}\n"
        f"Nodes: {len(report.node_profiles)} | "
        f"Regimes: {len(report.regimes)} | "
        f"Anomalies: {n_anomalies} | "
        f"Causal chains: {n_chains}\n"
    )
    sections.append(header[:header_chars])

    # --- Section 2: Regime Timeline ---
    regime_chars = int(char_budget * BUDGET_REGIME_TIMELINE)
    regime_lines = ["\n=== REGIME TIMELINE ==="]
    for r in report.regimes:
        duration_s = (r.end_ms - r.start_ms) / 1000.0
        desc = r.description or f"{len(r.window_indices)} windows"
        density_pct = r.anomaly_density * 100
        regime_lines.append(
            f"  [{_ms_to_timestr(r.start_ms)} - {_ms_to_timestr(r.end_ms)}] "
            f"{r.label} ({duration_s:.1f}s) anomaly_density={density_pct:.1f}% — {desc}"
        )
    regime_text = "\n".join(regime_lines)
    sections.append(regime_text[:regime_chars])

    # --- Section 3: Causal Chains ---
    chain_chars = int(char_budget * chain_budget_pct)
    chain_lines = ["\n=== CAUSAL CHAINS ==="]
    remaining = chain_chars - 25

    max_cascade_types = 10  # max unique anomaly types shown per chain

    for chain in report.causal_chains:
        root = chain.root_cause
        chain_header = (
            f"\nCHAIN [{chain.severity_score:.1f}]: {root.node} {root.anomaly_type}"
        )
        chain_detail = (
            f"  Root: {root.node} [{root.template[:60]}]\n"
            f"    first={_ms_to_timestr(chain.first_seen_ms)} "
            f"last={_ms_to_timestr(chain.last_seen_ms)} "
            f"jaccard={chain.co_occurrence_jaccard:.2f} "
            f"members={len(chain.cascade) + 1}"
        )

        # Count occurrences per (node, anomaly_type, template), keep first detail
        from collections import Counter as _Counter
        type_counter: dict = {}  # type_key -> (count, first_anomaly)
        for a in chain.cascade:
            type_key = (a.node, a.anomaly_type, a.template)
            if type_key not in type_counter:
                type_counter[type_key] = [0, a]
            type_counter[type_key][0] += 1

        # Sort by importance * count, show top N
        def _type_sort_key(item):
            (node, atype, tmpl), (count, _a) = item
            importance = NODE_IMPORTANCE.get(node, 0)
            return -(importance * count)

        cascade_lines = ["  Cascade:"]
        shown = 0
        for type_key, (count, first_a) in sorted(type_counter.items(), key=_type_sort_key):
            if shown >= max_cascade_types:
                omitted = len(type_counter) - max_cascade_types
                cascade_lines.append(f"    ... +{omitted} more anomaly types")
                break
            detail_str = ""
            if first_a.details:
                detail_parts = []
                for k, v in list(first_a.details.items())[:3]:
                    detail_parts.append(f"{k}={v}")
                detail_str = " (" + ", ".join(detail_parts) + ")"
            count_str = f" ×{count}" if count > 1 else ""
            cascade_lines.append(
                f"    → {first_a.node} {first_a.anomaly_type}{count_str}: [{first_a.template[:50]}]{detail_str}"
            )
            shown += 1

        block = chain_header + "\n" + chain_detail + "\n" + "\n".join(cascade_lines)
        if len(block) > remaining:
            # Try a compact version with fewer cascade entries
            if shown > 3:
                compact_cascade = cascade_lines[:4]  # header + top 3
                compact_cascade.append(f"    ... +{len(type_counter) - 3} more anomaly types")
                block = chain_header + "\n" + chain_detail + "\n" + "\n".join(compact_cascade)
            if len(block) > remaining:
                # Last resort: header only
                block = chain_header + "\n" + chain_detail
                if len(block) > remaining:
                    break
        chain_lines.append(block)
        remaining -= len(block)

    sections.append("\n".join(chain_lines))

    # --- Section 4: Isolated Anomalies ---
    iso_chars = int(char_budget * BUDGET_ISOLATED_ANOMALIES)
    iso_lines = ["\n=== ISOLATED ANOMALIES ==="]
    remaining = iso_chars - 30
    for a in sorted(report.isolated_anomalies, key=lambda x: -x.severity_score)[:15]:
        line = (
            f"  [{a.severity_score:.1f}] {a.node} {a.anomaly_type}: "
            f"[{a.template[:50]}] @ {_ms_to_timestr(a.timestamp_ms)}"
        )
        if len(line) > remaining:
            break
        iso_lines.append(line)
        remaining -= len(line)
    sections.append("\n".join(iso_lines))

    # --- Section 5: Sensor Baseline ---
    baseline_chars = int(char_budget * baseline_budget_pct)
    baseline_lines = ["\n=== SENSOR BASELINE ==="]
    for node, profile in sorted(
        report.node_profiles.items(),
        key=lambda x: -NODE_IMPORTANCE.get(x[0], 0),
    )[:10]:
        baseline_lines.append(
            f"  {node}: {profile.classification} "
            f"({profile.message_rate_hz}Hz, CV={profile.cv})"
        )
    baseline_text = "\n".join(baseline_lines)
    sections.append(baseline_text[:baseline_chars])

    # --- Section 6: Event-Driven Messages ---
    event_chars = int(char_budget * BUDGET_EVENT_DRIVEN)
    event_lines = ["\n=== EVENT-DRIVEN NODES ==="]
    for node, profile in report.node_profiles.items():
        if profile.classification == "event_driven" and profile.total_count > 0:
            top_templates = profile.templates.most_common(3)
            for tmpl, count in top_templates:
                event_lines.append(f"  {node} (×{count}): [{tmpl[:60]}]")
    event_text = "\n".join(event_lines)
    sections.append(event_text[:event_chars])

    # Combine and truncate
    full_text = "\n".join(sections)
    if len(full_text) > char_budget:
        full_text = full_text[:char_budget - 20] + "\n... [truncated]"

    return full_text


# ---------------------------------------------------------------------------
# Annotated anomaly log
# ---------------------------------------------------------------------------

def generate_anomaly_log(
    report: DiagnosticReport,
    records=None,
) -> str:
    """
    Generate annotated anomaly log — original lines with anomaly annotations.

    Shows only lines that are part of an anomaly, with context about
    which anomaly/chain they belong to.
    """
    lines: List[str] = []

    # Build anomaly lookup by (node, template)
    chain_lookup: Dict = {}
    for chain in report.causal_chains:
        for a in [chain.root_cause] + chain.cascade:
            key = (a.node, a.template, a.window_index)
            chain_lookup[key] = f"CHAIN[{chain.chain_id}] severity={chain.severity_score:.1f}"

    # Output anomalies grouped by time
    all_sorted = sorted(report.all_anomalies, key=lambda a: a.timestamp_ms)

    for a in all_sorted:
        chain_tag = chain_lookup.get(
            (a.node, a.template, a.window_index), "ISOLATED"
        )
        detail_str = ""
        if a.details:
            parts = [f"{k}={v}" for k, v in list(a.details.items())[:4]]
            detail_str = " | " + ", ".join(parts)

        lines.append(
            f"[{_ms_to_timestr(a.timestamp_ms)}] "
            f"{a.anomaly_type:25s} {a.node:35s} "
            f"[{a.template[:50]:50s}] "
            f"score={a.severity_score:6.1f} {chain_tag}{detail_str}"
        )

    return "\n".join(lines)
