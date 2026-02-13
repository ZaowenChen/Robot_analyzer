"""
Validation Script - Run the diagnostic agent against rosbag files.

This script:
1. Runs the Bridge tools directly to characterize each bag
2. Runs the full LangGraph agent for autonomous diagnosis
3. Reports findings and metrics (Grounding Ratio, etc.)

Can run in two modes:
- bridge_only: Just run the bridge tools to validate they work (no LLM needed)
- full: Run the complete LangGraph agent (requires API key)
"""

import json
import os
import sys
import time
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR) == "rosbag_analyzer" else THIS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from rosbag_analyzer.rosbag_bridge import ROSBagBridge
except ModuleNotFoundError:
    from rosbag_bridge import ROSBagBridge


def run_bridge_validation(bag_path: str, bag_name: str) -> dict:
    """
    Run all four bridge tools against a bag and produce a diagnostic report.
    This validates the bridge works correctly without requiring an LLM.
    """
    bridge = ROSBagBridge()
    report = {
        "bag_name": bag_name,
        "bag_path": bag_path,
        "timestamp": datetime.now().isoformat(),
        "tools_tested": {},
        "anomalies": [],
        "summary": "",
    }

    print(f"\n{'='*70}")
    print(f"BRIDGE VALIDATION: {bag_name}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Tool 1: get_bag_metadata
    # ------------------------------------------------------------------
    print("\n[1/4] get_bag_metadata...")
    t0 = time.time()
    metadata = bridge.get_bag_metadata(bag_path)
    elapsed = time.time() - t0
    report["tools_tested"]["get_bag_metadata"] = {
        "status": "OK",
        "elapsed_sec": round(elapsed, 2),
        "result_summary": {
            "duration": metadata["duration"],
            "num_topics": metadata["num_topics"],
            "total_messages": metadata["total_messages"],
        }
    }
    print(f"  Duration: {metadata['duration']}s")
    print(f"  Topics: {metadata['num_topics']}")
    print(f"  Messages: {metadata['total_messages']}")
    print(f"  Time: {elapsed:.2f}s")

    # Identify key diagnostic topics
    key_topics = {}
    for t in metadata["topics"]:
        if t["name"] in ["/odom", "/chassis_cmd_vel", "/cmd_vel",
                         "/unbiased_imu_PRY", "/localization/current_pose",
                         "/front_end_pose", "/scan", "/raw_scan"]:
            key_topics[t["name"]] = t

    print(f"\n  Key diagnostic topics:")
    for name, info in key_topics.items():
        print(f"    {name}: {info['message_count']} msgs @ {info['frequency']}Hz")

    # ------------------------------------------------------------------
    # Tool 2: get_topic_statistics (across multiple key topics)
    # ------------------------------------------------------------------
    print("\n[2/4] get_topic_statistics...")
    start_time = metadata["start_time"]
    end_time = metadata["end_time"]
    duration = metadata["duration"]

    topics_to_check = ["/odom", "/chassis_cmd_vel", "/unbiased_imu_PRY"]
    stats_results = {}

    for topic in topics_to_check:
        if topic not in key_topics:
            continue

        t0 = time.time()
        # Global stats
        global_stats = bridge.get_topic_statistics(bag_path, topic)
        elapsed_global = time.time() - t0

        # Windowed stats (30-second windows)
        t0 = time.time()
        windowed_stats = bridge.get_topic_statistics(
            bag_path, topic, window_size=30.0
        )
        elapsed_windowed = time.time() - t0

        stats_results[topic] = {
            "global": global_stats,
            "windowed": windowed_stats,
        }

        print(f"\n  {topic}:")
        print(f"    Global stats ({elapsed_global:.2f}s):")
        if global_stats and "fields" in global_stats[0]:
            for field, stats in global_stats[0]["fields"].items():
                flag = ""
                if stats["std"] < 1e-6 and stats["count"] > 10:
                    flag = " ** FROZEN **"
                    report["anomalies"].append({
                        "type": "frozen_sensor",
                        "topic": topic,
                        "field": field,
                        "std": stats["std"],
                        "mean": stats["mean"],
                        "count": stats["count"],
                    })
                print(f"      {field}: mean={stats['mean']:.4f} std={stats['std']:.6f} "
                      f"[{stats['min']:.4f}, {stats['max']:.4f}]{flag}")

        print(f"    Windowed stats ({elapsed_windowed:.2f}s): {len(windowed_stats)} windows")

        # Check each window for frozen fields
        for w in windowed_stats:
            for field, stats in w.get("fields", {}).items():
                if stats["std"] < 1e-6 and stats["count"] > 5:
                    window_info = f"[{w['window_start']:.1f}, {w['window_end']:.1f}]"
                    # Check if this is a field that SHOULD have variation
                    # (not things like z-position which are legitimately 0)
                    if field not in ["position_z", "twist_linear_y", "twist_linear_z",
                                   "twist_angular_x", "twist_angular_y",
                                   "orientation_x", "orientation_y",
                                   "angular_x", "angular_y", "angular_z",
                                   "linear_y", "linear_z"]:
                        report["anomalies"].append({
                            "type": "frozen_window",
                            "topic": topic,
                            "field": field,
                            "window": window_info,
                            "std": stats["std"],
                            "mean": stats["mean"],
                        })

    report["tools_tested"]["get_topic_statistics"] = {
        "status": "OK",
        "topics_checked": list(stats_results.keys()),
    }

    # ------------------------------------------------------------------
    # Tool 3: check_topic_frequency
    # ------------------------------------------------------------------
    print("\n[3/4] check_topic_frequency...")
    freq_results = {}
    for topic in topics_to_check:
        if topic not in key_topics:
            continue
        t0 = time.time()
        freq = bridge.check_topic_frequency(bag_path, topic, resolution=5.0)
        elapsed = time.time() - t0

        freq_results[topic] = freq
        print(f"\n  {topic}:")
        print(f"    Mean: {freq.get('mean_hz', 0):.2f} Hz")
        print(f"    Std: {freq.get('std_hz', 0):.2f} Hz")
        print(f"    Range: [{freq.get('min_hz', 0):.2f}, {freq.get('max_hz', 0):.2f}] Hz")
        print(f"    Time: {elapsed:.2f}s")

        # Check for significant dropouts
        mean_hz = freq.get("mean_hz", 0)
        series = freq.get("frequency_series", [])
        dropouts = [s for s in series if s["hz"] < mean_hz * 0.3 and mean_hz > 0]
        if dropouts:
            print(f"    ** {len(dropouts)} DROPOUT(s) detected **")
            for d in dropouts[:5]:
                print(f"       t={d['time']:.1f}: {d['hz']:.2f} Hz")
                report["anomalies"].append({
                    "type": "frequency_dropout",
                    "topic": topic,
                    "time": d["time"],
                    "hz": d["hz"],
                    "expected_hz": mean_hz,
                })

    report["tools_tested"]["check_topic_frequency"] = {
        "status": "OK",
        "topics_checked": list(freq_results.keys()),
    }

    # ------------------------------------------------------------------
    # Tool 4: sample_messages
    # ------------------------------------------------------------------
    print("\n[4/4] sample_messages...")
    for topic in ["/chassis_cmd_vel", "/odom"]:
        if topic not in key_topics:
            continue
        t0 = time.time()
        # Sample at start
        samples_start = bridge.sample_messages(bag_path, topic, timestamp=start_time, count=3)
        # Sample at middle
        mid_time = start_time + duration / 2
        samples_mid = bridge.sample_messages(bag_path, topic, timestamp=mid_time, count=3)
        # Sample at end
        samples_end = bridge.sample_messages(bag_path, topic, timestamp=end_time, count=3)
        elapsed = time.time() - t0
        print(f"\n  {topic}: sampled at start/mid/end ({elapsed:.2f}s)")
        for label, samples in [("start", samples_start), ("mid", samples_mid), ("end", samples_end)]:
            if samples.get("messages"):
                msg = samples["messages"][0]
                print(f"    {label} (t={msg['timestamp']:.2f}): {json.dumps(msg['data'], default=str)[:150]}")

    report["tools_tested"]["sample_messages"] = {"status": "OK"}

    # ------------------------------------------------------------------
    # Anomaly Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"ANOMALY SUMMARY: {bag_name}")
    print(f"{'='*70}")

    if report["anomalies"]:
        # Group by type
        by_type = {}
        for a in report["anomalies"]:
            by_type.setdefault(a["type"], []).append(a)
        for atype, items in by_type.items():
            print(f"\n  {atype.upper()} ({len(items)}):")
            for item in items[:10]:
                details = {k: v for k, v in item.items() if k != "type"}
                print(f"    {json.dumps(details)}")
    else:
        print("  No anomalies detected.")

    report["summary"] = (
        f"Found {len(report['anomalies'])} anomalies in {bag_name} "
        f"(duration={metadata['duration']}s, {metadata['num_topics']} topics)"
    )
    print(f"\n  {report['summary']}")

    return report


def _has_llm_api_key() -> bool:
    """Check if an LLM API key is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Hallucination Rate (Section 3.5.3 of the Diagnostic Agent Plan)
# ---------------------------------------------------------------------------
_FACT_CHECK_PROMPT = """\
You are a strict Fact-Checker for robotic diagnostic reports.

You are given:
1. A DIAGNOSIS produced by a diagnostic agent.
2. An EVIDENCE LOCKER – a list of factual observations extracted by
   programmatic tools from a ROS bag.

Your task:
- Extract every distinct factual claim from the DIAGNOSIS (a "claim" is any
  statement about sensor values, frequencies, anomalies, timestamps, topic
  names, or statistical results).
- For each claim, determine whether it is SUPPORTED or UNSUPPORTED by the
  evidence locker.  A claim is SUPPORTED only if there is a matching or
  clearly entailed item in the evidence.
- Respond with EXACTLY this JSON format – no commentary before or after:

{{
  "claims": [
    {{"claim": "<text of claim>", "supported": true}},
    {{"claim": "<text of claim>", "supported": false}}
  ]
}}
"""


def compute_hallucination_rate(
    diagnosis: str,
    evidence: list,
    llm=None,
) -> dict:
    """Compute the Hallucination Rate H_r for a diagnosis.

    H_r = (number of unsupported claims) / (total claims).

    Uses an LLM call to decompose the diagnosis into claims and check each
    against the evidence locker.  Returns a dict with ``hallucination_rate``,
    ``total_claims``, ``unsupported_claims``, and the raw ``claims`` list.
    """
    if not diagnosis or not evidence:
        return {
            "hallucination_rate": 0.0,
            "total_claims": 0,
            "unsupported_claims": 0,
            "claims": [],
        }

    if llm is None:
        # Ensure the system CA bundle is available for proxy environments
        _ca_bundle = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.exists(_ca_bundle):
            os.environ.setdefault("SSL_CERT_FILE", _ca_bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_bundle)
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
        except Exception:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o", temperature=0)

    from langchain_core.messages import SystemMessage, HumanMessage

    evidence_str = "\n".join(f"  - {e}" for e in evidence)
    user_msg = (
        f"DIAGNOSIS:\n{diagnosis}\n\n"
        f"EVIDENCE LOCKER:\n{evidence_str}"
    )

    response = llm.invoke([
        SystemMessage(content=_FACT_CHECK_PROMPT),
        HumanMessage(content=user_msg),
    ])

    # Parse the JSON response
    import re
    text = response.content if hasattr(response, "content") else str(response)

    # Try to extract JSON from the response
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            claims = parsed.get("claims", [])
        except json.JSONDecodeError:
            claims = []
    else:
        claims = []

    total = len(claims)
    unsupported = sum(1 for c in claims if not c.get("supported", True))
    rate = unsupported / total if total > 0 else 0.0

    return {
        "hallucination_rate": round(rate, 4),
        "total_claims": total,
        "unsupported_claims": unsupported,
        "claims": claims,
    }


def run_full_agent_validation(bag_path: str, bag_name: str) -> dict:
    """Run the full LangGraph agent for autonomous diagnosis.

    Falls back to the rule-based DiagnosticAnalyzer when no LLM API key
    is available, so ``validate.py full`` always produces results.
    """
    if not _has_llm_api_key():
        print(f"\n{'='*70}")
        print(f"NOTE: No LLM API key found (ANTHROPIC_API_KEY / OPENAI_API_KEY).")
        print(f"Falling back to rule-based DiagnosticAnalyzer for: {bag_name}")
        print(f"{'='*70}")
        try:
            from rosbag_analyzer.analyze import DiagnosticAnalyzer
        except ModuleNotFoundError:
            from analyze import DiagnosticAnalyzer

        analyzer = DiagnosticAnalyzer()
        report = analyzer.analyze(bag_path)
        return report

    try:
        from rosbag_analyzer.agent import run_diagnostic
    except ModuleNotFoundError:
        from agent import run_diagnostic

    print(f"\n{'='*70}")
    print(f"AGENT VALIDATION: {bag_name}")
    print(f"{'='*70}")

    query = (
        "Perform a comprehensive diagnostic analysis of this ROS bag. "
        "Check all key sensor topics (odometry, IMU, command velocities, laser scans) "
        "for anomalies including: frozen sensors, frequency dropouts, "
        "command/feedback mismatches, and any other issues. "
        "Use windowed statistics to localize any anomalies in time. "
        "Provide a detailed diagnosis with specific evidence."
    )

    result = run_diagnostic(bag_path, query, verbose=True)

    # Compute Grounding Ratio
    grounding_ratio = result["num_tool_calls"] / max(result["steps"], 1)

    # ---- Hallucination Rate (Section 3.5.3) ----
    print(f"\n  Computing Hallucination Rate (H_r)...")
    hr_result = compute_hallucination_rate(
        result["diagnosis"],
        result["evidence"],
    )

    print(f"\n{'='*70}")
    print(f"AGENT RESULTS: {bag_name}")
    print(f"{'='*70}")
    print(f"  Steps: {result['steps']}")
    print(f"  Tool Calls: {result['num_tool_calls']}")
    print(f"  Grounding Ratio: {grounding_ratio:.2f}")
    print(f"  Evidence items: {len(result['evidence'])}")
    print(f"  Hallucination Rate (H_r): {hr_result['hallucination_rate']:.2%}"
          f"  ({hr_result['unsupported_claims']}/{hr_result['total_claims']} unsupported)")
    print(f"\n  Evidence Locker:")
    for e in result["evidence"]:
        print(f"    - {e}")
    if hr_result["claims"]:
        unsupported = [c for c in hr_result["claims"] if not c.get("supported")]
        if unsupported:
            print(f"\n  Unsupported Claims:")
            for c in unsupported:
                print(f"    - {c['claim']}")
    print(f"\n  Diagnosis (first 1000 chars):")
    print(f"    {result['diagnosis'][:1000]}")

    return {
        "bag_name": bag_name,
        "steps": result["steps"],
        "tool_calls": result["num_tool_calls"],
        "grounding_ratio": grounding_ratio,
        "hallucination_rate": hr_result["hallucination_rate"],
        "hallucination_detail": hr_result,
        "evidence": result["evidence"],
        "diagnosis": result["diagnosis"],
    }


def main():
    """Main validation entry point."""
    bag_dir = PROJECT_ROOT
    if not os.path.isdir(bag_dir):
        bag_dir = "/sessions/busy-nice-hamilton/mnt/robotic_test"

    bag_files = sorted([
        f for f in os.listdir(bag_dir)
        if f.endswith('.bag')
    ])

    if not bag_files:
        print("No .bag files found!")
        sys.exit(1)

    print(f"Found {len(bag_files)} bag file(s):")
    for f in bag_files:
        print(f"  {f}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "bridge_only"

    all_reports = []

    for bag_file in bag_files:
        bag_path = os.path.join(bag_dir, bag_file)

        if mode in ("bridge_only", "both"):
            report = run_bridge_validation(bag_path, bag_file)
            all_reports.append(report)

        if mode in ("full", "both"):
            agent_report = run_full_agent_validation(bag_path, bag_file)
            all_reports.append(agent_report)

    # Save reports
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n\nReport saved to: {report_path}")

    return all_reports


if __name__ == "__main__":
    main()
