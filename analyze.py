"""
Standalone Diagnostic Analyzer

This performs a comprehensive, rule-based diagnostic analysis of rosbag files
using the Bridge tools. It implements the same diagnostic logic that the
LangGraph agent would use, but without requiring an LLM API key.

This serves as:
1. A validation that all Bridge tools work correctly
2. A baseline diagnostic against which the LLM agent can be compared
3. A standalone tool for immediate bag analysis

The analyzer follows the hierarchical zooming workflow from the project plan:
  Level 1 (Global) -> Level 2 (Regional) -> Level 3 (Local)
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosbag_analyzer.rosbag_bridge import ROSBagBridge


# Fields that are expected to be zero on a 2D differential-drive robot
EXPECTED_ZERO_FIELDS = {
    "/odom": {"orientation_x", "orientation_y", "position_z",
              "twist_angular_x", "twist_angular_y",
              "twist_linear_y", "twist_linear_z"},
    "/chassis_cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
    "/cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
}


class DiagnosticAnalyzer:
    """
    Rule-based diagnostic analyzer implementing the project plan's
    hypothesis-testing workflow using the ROSBag Bridge tools.
    """

    def __init__(self):
        self.bridge = ROSBagBridge()
        self.evidence = []
        self.anomalies = []
        self.warnings = []

    def add_evidence(self, finding: str):
        self.evidence.append(finding)
        print(f"    [EVIDENCE] {finding}")

    def add_anomaly(self, severity: str, category: str, description: str, details: dict = None):
        anomaly = {
            "severity": severity,
            "category": category,
            "description": description,
            "details": details or {},
        }
        self.anomalies.append(anomaly)
        icon = {"CRITICAL": "!!!", "WARNING": "!!", "INFO": "!"}[severity]
        print(f"    [{icon} {severity}] {category}: {description}")

    def analyze(self, bag_path: str) -> dict:
        """Run complete diagnostic analysis on a bag file."""
        bag_name = os.path.basename(bag_path)
        self.evidence = []
        self.anomalies = []

        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        print(f"Bag: {bag_name}")
        print(f"{'='*70}")

        # ---- Phase 1: Reconnaissance ----
        print(f"\n--- Phase 1: Reconnaissance (get_bag_metadata) ---")
        t0 = time.time()
        metadata = self.bridge.get_bag_metadata(bag_path)
        elapsed = time.time() - t0
        print(f"  Duration: {metadata['duration']:.2f}s")
        print(f"  Time range: {metadata['start_time']:.4f} to {metadata['end_time']:.4f}")
        print(f"  Topics: {metadata['num_topics']}, Messages: {metadata['total_messages']}")
        print(f"  (took {elapsed:.2f}s)")

        start_time = metadata["start_time"]
        end_time = metadata["end_time"]
        duration = metadata["duration"]

        # Categorize topics
        topic_map = {t["name"]: t for t in metadata["topics"]}
        self.add_evidence(f"Bag: {bag_name}, duration={duration:.1f}s, {metadata['num_topics']} topics")

        # ---- Phase 2: Global Statistics (Level 1) ----
        print(f"\n--- Phase 2: Global Statistics (Level 1 - Whole Bag) ---")
        key_topics = ["/odom", "/chassis_cmd_vel", "/unbiased_imu_PRY",
                      "/localization/current_pose", "/front_end_pose"]
        available_topics = [t for t in key_topics if t in topic_map]

        global_stats = {}
        for topic in available_topics:
            print(f"\n  Analyzing {topic}...")
            stats = self.bridge.get_topic_statistics(bag_path, topic)
            global_stats[topic] = stats

            if stats and "fields" in stats[0]:
                for field, s in stats[0]["fields"].items():
                    expected_zeros = EXPECTED_ZERO_FIELDS.get(topic, set())
                    is_expected_zero = field in expected_zeros

                    if s["std"] < 1e-6 and s["count"] > 10:
                        if is_expected_zero:
                            pass  # Normal for 2D robot
                        elif s["mean"] == 0.0:
                            self.add_anomaly("WARNING", "ZERO_FIELD",
                                f"{topic}.{field} is constantly 0.0 ({s['count']} msgs)",
                                {"topic": topic, "field": field, "mean": s["mean"]})
                        else:
                            self.add_anomaly("CRITICAL", "FROZEN_SENSOR",
                                f"{topic}.{field} frozen at {s['mean']:.6f} (std=0, {s['count']} msgs)",
                                {"topic": topic, "field": field, "mean": s["mean"], "std": s["std"]})
                    else:
                        self.add_evidence(
                            f"{topic}.{field}: mean={s['mean']:.4f} std={s['std']:.4f} "
                            f"range=[{s['min']:.4f}, {s['max']:.4f}]")

        # ---- Phase 3: Regional Statistics (Level 2 - Windowed) ----
        print(f"\n--- Phase 3: Regional Statistics (Level 2 - 30s Windows) ---")
        windowed_stats = {}
        for topic in available_topics:
            print(f"\n  Windowed analysis: {topic}...")
            wstats = self.bridge.get_topic_statistics(bag_path, topic, window_size=30.0)
            windowed_stats[topic] = wstats

            # Look for transitions: fields that change from normal to frozen or vice versa
            for field in (wstats[0].get("fields", {}).keys() if wstats else []):
                expected_zeros = EXPECTED_ZERO_FIELDS.get(topic, set())
                if field in expected_zeros:
                    continue

                stds = [(w["window_start"], w["fields"].get(field, {}).get("std", 0),
                         w["fields"].get(field, {}).get("mean", 0))
                        for w in wstats if field in w.get("fields", {})]

                # Detect freeze onset: std transitions from >0 to ~0
                for i in range(1, len(stds)):
                    prev_std = stds[i-1][1]
                    curr_std = stds[i][1]
                    if prev_std > 0.001 and curr_std < 1e-6:
                        self.add_anomaly("WARNING", "FREEZE_ONSET",
                            f"{topic}.{field} froze at t~{stds[i][0]:.1f} "
                            f"(std went from {prev_std:.6f} to {curr_std:.6f}, "
                            f"value={stds[i][2]:.6f})",
                            {"topic": topic, "field": field, "time": stds[i][0]})

                    # Detect resume: std transitions from ~0 to >0
                    if prev_std < 1e-6 and curr_std > 0.001 and stds[i-1][2] != 0:
                        self.add_anomaly("INFO", "SENSOR_RESUME",
                            f"{topic}.{field} resumed at t~{stds[i][0]:.1f}",
                            {"topic": topic, "field": field, "time": stds[i][0]})

        # ---- Phase 4: Frequency Analysis ----
        print(f"\n--- Phase 4: Frequency Analysis ---")
        freq_results = {}
        for topic in available_topics:
            if topic not in topic_map:
                continue
            print(f"\n  Checking frequency: {topic}...")
            freq = self.bridge.check_topic_frequency(bag_path, topic, resolution=5.0)
            freq_results[topic] = freq

            mean_hz = freq.get("mean_hz", 0)
            std_hz = freq.get("std_hz", 0)
            self.add_evidence(f"{topic} frequency: mean={mean_hz:.1f}Hz std={std_hz:.2f}Hz")

            # Detect dropouts
            series = freq.get("frequency_series", [])
            for entry in series:
                if mean_hz > 0 and entry["hz"] < mean_hz * 0.3:
                    # Exclude the very last bin (bag ending)
                    if entry["time"] < end_time - 10:
                        self.add_anomaly("CRITICAL", "FREQUENCY_DROPOUT",
                            f"{topic} dropped to {entry['hz']:.1f}Hz at t={entry['time']:.1f} "
                            f"(expected ~{mean_hz:.1f}Hz)",
                            {"topic": topic, "time": entry["time"], "hz": entry["hz"]})

            # Overall frequency stability
            if mean_hz > 0 and std_hz > mean_hz * 0.2:
                self.add_anomaly("WARNING", "UNSTABLE_FREQUENCY",
                    f"{topic} has unstable frequency: std={std_hz:.2f} vs mean={mean_hz:.1f}",
                    {"topic": topic, "mean_hz": mean_hz, "std_hz": std_hz})

        # ---- Phase 5: Cross-Topic Consistency ----
        print(f"\n--- Phase 5: Cross-Topic Consistency Check ---")
        self._check_cmd_vel_vs_odom(bag_path, global_stats, windowed_stats, topic_map)

        # ---- Phase 6: IMU Analysis ----
        if "/unbiased_imu_PRY" in available_topics:
            print(f"\n--- Phase 6: IMU Analysis ---")
            self._check_imu_health(bag_path, global_stats, windowed_stats)

        # ---- Phase 7: Localization Consistency ----
        if "/localization/current_pose" in available_topics:
            print(f"\n--- Phase 7: Localization Consistency ---")
            self._check_localization(bag_path, global_stats, windowed_stats)

        # ---- Generate Report ----
        report = self._generate_report(bag_name, metadata)
        return report

    def _check_cmd_vel_vs_odom(self, bag_path, global_stats, windowed_stats, topic_map):
        """Check if commanded velocities match actual odometry."""
        if "/chassis_cmd_vel" not in global_stats or "/odom" not in global_stats:
            return

        cmd_stats = global_stats["/chassis_cmd_vel"]
        odom_stats = global_stats["/odom"]

        if not cmd_stats or not odom_stats:
            return

        cmd_fields = cmd_stats[0].get("fields", {})
        odom_fields = odom_stats[0].get("fields", {})

        # Check: Commands sent but no motion
        cmd_linear = cmd_fields.get("linear_x", {})
        odom_twist = odom_fields.get("twist_linear_x", {})

        if cmd_linear.get("max", 0) > 0.01 and odom_twist.get("max", 0) < 0.001:
            self.add_anomaly("CRITICAL", "CMD_ODOM_MISMATCH",
                "Commands were sent (cmd_vel.linear_x > 0) but odom shows no motion",
                {"cmd_max": cmd_linear.get("max"), "odom_max": odom_twist.get("max")})

        # Check: No commands but robot moving (unexpected motion)
        if cmd_linear.get("max", 0) < 0.001 and odom_twist.get("max", 0) > 0.1:
            self.add_anomaly("WARNING", "UNEXPECTED_MOTION",
                "No commands sent but odom shows motion",
                {"cmd_max": cmd_linear.get("max"), "odom_max": odom_twist.get("max")})

        # Check robot activity state
        if cmd_linear.get("std", 0) < 1e-6 and cmd_linear.get("mean", 0) == 0:
            self.add_evidence("Robot appears IDLE (no velocity commands sent)")
        else:
            self.add_evidence(
                f"Robot was ACTIVE (cmd_vel linear_x: mean={cmd_linear.get('mean', 0):.4f} "
                f"max={cmd_linear.get('max', 0):.4f})")

        # Windowed comparison
        if "/chassis_cmd_vel" in windowed_stats and "/odom" in windowed_stats:
            cmd_windows = windowed_stats["/chassis_cmd_vel"]
            odom_windows = windowed_stats["/odom"]

            for i, (cw, ow) in enumerate(zip(cmd_windows, odom_windows)):
                cmd_lx = cw.get("fields", {}).get("linear_x", {})
                odom_tlx = ow.get("fields", {}).get("twist_linear_x", {})

                # Command sent but no motion in this window
                if cmd_lx.get("mean", 0) > 0.05 and odom_tlx.get("std", 0) < 1e-6:
                    self.add_anomaly("WARNING", "STALL_DETECTED",
                        f"Commands active but odom frozen in window "
                        f"[{cw['window_start']:.1f}, {cw['window_end']:.1f}]",
                        {"window_start": cw["window_start"],
                         "cmd_mean": cmd_lx.get("mean"),
                         "odom_std": odom_tlx.get("std")})

    def _check_imu_health(self, bag_path, global_stats, windowed_stats):
        """Check IMU for frozen axes or anomalous behavior."""
        if "/unbiased_imu_PRY" not in global_stats:
            return

        imu_stats = global_stats["/unbiased_imu_PRY"]
        if not imu_stats:
            return

        fields = imu_stats[0].get("fields", {})

        for axis in ["x", "y", "z"]:
            if axis in fields:
                s = fields[axis]
                if s["std"] < 1e-6 and s["count"] > 10:
                    self.add_anomaly("CRITICAL", "IMU_FROZEN",
                        f"IMU {axis}-axis frozen at {s['mean']:.6f} (std=0)",
                        {"axis": axis, "mean": s["mean"]})
                else:
                    self.add_evidence(
                        f"IMU {axis}: mean={s['mean']:.4f} std={s['std']:.4f} "
                        f"(healthy - has natural noise)")

        # Check for abnormal Z-axis behavior (heading)
        z_stats = fields.get("z", {})
        if z_stats and z_stats.get("std", 0) > 100:
            self.add_anomaly("WARNING", "IMU_HIGH_VARIANCE",
                f"IMU Z-axis has very high variance (std={z_stats['std']:.2f})",
                {"std": z_stats["std"]})

    def _check_localization(self, bag_path, global_stats, windowed_stats):
        """Check localization consistency."""
        if "/localization/current_pose" not in global_stats:
            return

        loc_stats = global_stats["/localization/current_pose"]
        if not loc_stats:
            return

        fields = loc_stats[0].get("fields", {})

        pos_x = fields.get("position_x", {})
        pos_y = fields.get("position_y", {})

        if pos_x and pos_y:
            total_range_x = pos_x.get("max", 0) - pos_x.get("min", 0)
            total_range_y = pos_y.get("max", 0) - pos_y.get("min", 0)
            self.add_evidence(
                f"Localization range: X=[{pos_x.get('min', 0):.2f}, {pos_x.get('max', 0):.2f}] "
                f"Y=[{pos_y.get('min', 0):.2f}, {pos_y.get('max', 0):.2f}]")

            # Check if localization is stuck
            if pos_x.get("std", 0) < 0.001 and pos_y.get("std", 0) < 0.001:
                self.add_anomaly("WARNING", "LOCALIZATION_STUCK",
                    "Localization position is not changing (robot may be stationary or localization is frozen)",
                    {"pos_x_std": pos_x.get("std"), "pos_y_std": pos_y.get("std")})

        # Compare localization with odom if both available
        if "/odom" in global_stats and global_stats["/odom"]:
            odom_fields = global_stats["/odom"][0].get("fields", {})
            odom_px = odom_fields.get("position_x", {})
            loc_px = fields.get("position_x", {})

            if odom_px and loc_px:
                odom_range = odom_px.get("max", 0) - odom_px.get("min", 0)
                loc_range = loc_px.get("max", 0) - loc_px.get("min", 0)

                if odom_range > 0.5 and loc_range > 0.5:
                    # Both show movement - compare scales
                    ratio = loc_range / odom_range if odom_range > 0 else float('inf')
                    if abs(ratio - 1.0) > 0.5:
                        self.add_anomaly("WARNING", "ODOM_LOC_DIVERGENCE",
                            f"Odometry and localization ranges differ significantly "
                            f"(odom: {odom_range:.2f}m, loc: {loc_range:.2f}m, ratio: {ratio:.2f})",
                            {"odom_range": odom_range, "loc_range": loc_range})

    def _generate_report(self, bag_name: str, metadata: dict) -> dict:
        """Generate the final diagnostic report."""
        # Count anomalies by severity
        by_severity = defaultdict(list)
        for a in self.anomalies:
            by_severity[a["severity"]].append(a)

        # Generate diagnosis summary
        critical_count = len(by_severity.get("CRITICAL", []))
        warning_count = len(by_severity.get("WARNING", []))
        info_count = len(by_severity.get("INFO", []))

        if critical_count > 0:
            health = "UNHEALTHY"
        elif warning_count > 2:
            health = "DEGRADED"
        elif warning_count > 0:
            health = "MARGINAL"
        else:
            health = "HEALTHY"

        diagnosis_lines = [
            f"Bag: {bag_name}",
            f"Overall Health: {health}",
            f"Duration: {metadata['duration']:.1f}s",
            f"Critical Issues: {critical_count}",
            f"Warnings: {warning_count}",
            f"Info: {info_count}",
            "",
        ]

        if by_severity.get("CRITICAL"):
            diagnosis_lines.append("CRITICAL ISSUES:")
            for a in by_severity["CRITICAL"]:
                diagnosis_lines.append(f"  - [{a['category']}] {a['description']}")

        if by_severity.get("WARNING"):
            diagnosis_lines.append("\nWARNINGS:")
            for a in by_severity["WARNING"]:
                diagnosis_lines.append(f"  - [{a['category']}] {a['description']}")

        diagnosis_lines.append(f"\nEvidence ({len(self.evidence)} items):")
        for e in self.evidence:
            diagnosis_lines.append(f"  - {e}")

        diagnosis = "\n".join(diagnosis_lines)

        print(f"\n{'='*70}")
        print(f"FINAL DIAGNOSIS: {bag_name}")
        print(f"{'='*70}")
        print(diagnosis)

        return {
            "bag_name": bag_name,
            "health_status": health,
            "metadata": {
                "duration": metadata["duration"],
                "num_topics": metadata["num_topics"],
                "total_messages": metadata["total_messages"],
            },
            "anomalies": self.anomalies,
            "evidence": self.evidence,
            "diagnosis": diagnosis,
            "metrics": {
                "critical_count": critical_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "evidence_count": len(self.evidence),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Run rule-based diagnostics on .bag files.")
    default_bag_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--bag-dir",
        default=default_bag_dir,
        help=f"Directory containing .bag files (default: {default_bag_dir})",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional output path for diagnostic_report.json",
    )
    args = parser.parse_args()
    bag_dir = os.path.abspath(args.bag_dir)

    if not os.path.isdir(bag_dir):
        print(f"Bag directory does not exist: {bag_dir}")
        return

    bag_files = sorted([
        f for f in os.listdir(bag_dir)
        if f.endswith('.bag')
    ])

    if not bag_files:
        print("No .bag files found!")
        return

    all_reports = []
    for bag_file in bag_files:
        bag_path = os.path.join(bag_dir, bag_file)
        analyzer = DiagnosticAnalyzer()
        report = analyzer.analyze(bag_path)
        all_reports.append(report)

    # Save comprehensive report
    report_path = args.report_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diagnostic_report.json",
    )
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n\nFull report saved to: {report_path}")

    # Print cross-bag comparison
    print(f"\n{'='*70}")
    print("CROSS-BAG COMPARISON")
    print(f"{'='*70}")
    for r in all_reports:
        print(f"\n  {r['bag_name']}: {r['health_status']}")
        print(f"    Duration: {r['metadata']['duration']:.1f}s")
        print(f"    Critical: {r['metrics']['critical_count']}")
        print(f"    Warnings: {r['metrics']['warning_count']}")
        print(f"    Evidence: {r['metrics']['evidence_count']} items")


if __name__ == "__main__":
    main()
