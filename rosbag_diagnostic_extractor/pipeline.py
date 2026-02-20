"""
Pipeline orchestrator — wires all 6 stages together.

parse → fingerprint → structural anomaly → content anomaly → causal chains → report
"""

import os
import time
from typing import Dict, List, Optional

from .models import DiagnosticReport, ParsedLogRecord
from .parser import open_log_source, LogSource
from .fingerprint import (
    analyze_node_frequencies,
    auto_window_size,
    build_fingerprints,
    compute_regime_baselines,
    segment_regimes,
)
from .structural_anomaly import (
    detect_structural_anomalies,
    extract_regime_transitions,
)
from .content_anomaly import detect_content_anomalies
from .causal_chains import (
    construct_chains,
    load_validated_rules,
    apply_validated_rules,
)
from .report_generator import (
    generate_json_report,
    generate_llm_prompt,
    generate_anomaly_log,
)
from .constants import DEFAULT_TOKEN_BUDGET


def run_pipeline(
    input_path: str,
    *,
    output_dir: str = ".",
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    robot_context: str = "",
    verbose: bool = False,
    rules_path: Optional[str] = None,
    min_level: str = "INFO",
) -> DiagnosticReport:
    """
    Run the full 6-stage diagnostic extraction pipeline.

    Args:
        input_path: Path to .bag file or text log file
        output_dir: Directory for output files
        token_budget: Token budget for LLM prompt
        robot_context: Robot/site context string for report header
        verbose: Print progress details
        rules_path: Path to validated_rules.yaml (optional)
        min_level: Minimum log level ("INFO", "WARN", "ERROR")

    Returns:
        DiagnosticReport with all pipeline results
    """
    os.makedirs(output_dir, exist_ok=True)

    def log(msg: str):
        if verbose:
            print(msg)

    t_start = time.time()

    # ======================================================================
    # Stage 1: Parse + Node Frequency Discovery
    # ======================================================================
    log(f"\n--- Stage 1: Parse + Node Frequency Discovery ---")
    t0 = time.time()

    source = open_log_source(input_path, min_level=min_level)
    records: List[ParsedLogRecord] = list(source.records())
    metadata = source.get_metadata()

    if not records:
        log("  WARNING: No records parsed. Check input file and log level.")
        return _empty_report(metadata)

    # Compute time range
    metadata["time_range_ms"] = records[-1].timestamp_ms - records[0].timestamp_ms
    metadata["duration_ms"] = records[-1].timestamp_ms - records[0].timestamp_ms
    metadata["total_lines"] = metadata.get("total_lines", metadata.get("total_messages", len(records)))
    metadata["parsed_lines"] = metadata.get("parsed_lines", metadata.get("parsed_messages", len(records)))
    metadata["source_file"] = os.path.basename(input_path)

    profiles = analyze_node_frequencies(records)

    log(f"  Parsed {len(records)} records in {time.time()-t0:.2f}s")
    log(f"  Nodes: {len(profiles)}")
    if verbose:
        for node, p in sorted(profiles.items(), key=lambda x: -x[1].total_count)[:8]:
            log(f"    {node}: {p.classification} ({p.message_rate_hz}Hz, CV={p.cv})")

    # ======================================================================
    # Stage 2: Windowed Fingerprinting + Regime Segmentation
    # ======================================================================
    log(f"\n--- Stage 2: Fingerprinting + Regime Segmentation ---")
    t0 = time.time()

    window_size = auto_window_size(profiles)
    log(f"  Window size: {window_size:.0f}ms")

    windows = build_fingerprints(records, window_size, profiles)
    log(f"  Built {len(windows)} windows")

    regimes = segment_regimes(windows)
    compute_regime_baselines(regimes, windows)

    log(f"  Detected {len(regimes)} regimes in {time.time()-t0:.2f}s")
    if verbose:
        for r in regimes:
            n_baseline = len(r.baseline) if r.baseline else 0
            log(f"    {r.label}: {len(r.window_indices)} windows, {n_baseline} baseline templates")

    # ======================================================================
    # Stage 3: Structural Anomaly Detection
    # ======================================================================
    log(f"\n--- Stage 3: Structural Anomaly Detection ---")
    t0 = time.time()

    structural_anomalies = detect_structural_anomalies(regimes, windows)
    transition_anomalies = extract_regime_transitions(regimes, windows)

    all_structural = structural_anomalies + transition_anomalies
    log(f"  Found {len(structural_anomalies)} structural + {len(transition_anomalies)} transition anomalies in {time.time()-t0:.2f}s")

    # ======================================================================
    # Stage 4: Content Anomaly Detection
    # ======================================================================
    log(f"\n--- Stage 4: Content Anomaly Detection ---")
    t0 = time.time()

    content_anomalies = detect_content_anomalies(records, regimes, windows)
    log(f"  Found {len(content_anomalies)} content anomalies in {time.time()-t0:.2f}s")

    # ======================================================================
    # Stage 5: Causal Chain Detection
    # ======================================================================
    log(f"\n--- Stage 5: Causal Chain Detection ---")
    t0 = time.time()

    all_anomalies = all_structural + content_anomalies
    chains, isolated = construct_chains(all_anomalies)

    # Apply validated rules if available
    rules = load_validated_rules(rules_path)
    if rules:
        apply_validated_rules(chains, rules)
        log(f"  Applied {len(rules)} validated rules")

    log(f"  Built {len(chains)} causal chains, {len(isolated)} isolated anomalies in {time.time()-t0:.2f}s")
    if verbose:
        for c in chains[:5]:
            log(f"    CHAIN[{c.chain_id}] severity={c.severity_score:.1f}: "
                f"{c.root_cause.node} {c.root_cause.anomaly_type} "
                f"({len(c.cascade)} cascade)")

    # ======================================================================
    # Stage 6: Report Generation
    # ======================================================================
    log(f"\n--- Stage 6: Report Generation ---")

    report = DiagnosticReport(
        metadata=metadata,
        node_profiles=profiles,
        regimes=regimes,
        causal_chains=chains,
        isolated_anomalies=isolated,
        sensor_summary={},  # Could be enriched by extractors
        all_anomalies=all_anomalies,
    )

    # Generate outputs
    json_report = generate_json_report(report)
    json_path = os.path.join(output_dir, "diagnostic_report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    log(f"  JSON report: {json_path}")

    llm_text = generate_llm_prompt(report, token_budget, robot_context)
    prompt_path = os.path.join(output_dir, "llm_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(llm_text)
    est_tokens = len(llm_text) // 4
    log(f"  LLM prompt: {prompt_path} (~{est_tokens} tokens)")

    anomaly_text = generate_anomaly_log(report)
    anomaly_path = os.path.join(output_dir, "anomaly_log.txt")
    with open(anomaly_path, "w") as f:
        f.write(anomaly_text)
    log(f"  Anomaly log: {anomaly_path}")

    elapsed = time.time() - t_start
    log(f"\n=== Pipeline complete in {elapsed:.2f}s ===")
    log(f"  Total anomalies: {len(all_anomalies)}")
    log(f"  Causal chains: {len(chains)}")
    log(f"  LLM prompt: ~{est_tokens} tokens")

    return report


def _empty_report(metadata: dict) -> DiagnosticReport:
    """Return an empty report for when no records are found."""
    return DiagnosticReport(
        metadata=metadata,
        node_profiles={},
        regimes=[],
        causal_chains=[],
        isolated_anomalies=[],
        sensor_summary={},
        all_anomalies=[],
    )


import json
