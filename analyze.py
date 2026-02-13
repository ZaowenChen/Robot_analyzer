"""
Standalone Diagnostic Analyzer (v2.0 — Mission-Aware)

This performs a comprehensive, rule-based diagnostic analysis of rosbag files
using the Bridge tools. It implements the same diagnostic logic that the
LangGraph agent would use, but without requiring an LLM API key.

This serves as:
1. A validation that all Bridge tools work correctly
2. A baseline diagnostic against which the LLM agent can be compared
3. A standalone tool for immediate bag analysis

The analyzer follows the hierarchical zooming workflow from the project plan:
  Level 1 (Global) -> Level 2 (Regional) -> Level 3 (Local)

v2.0 additions:
  - Mission-aware multi-bag virtual timeline (--mode mission)
  - /rosout log extraction and correlation with sensor anomalies
  - Absolute timestamps (CST, UTC+8)
  - Cross-bag anomaly detection (persistent freezes, hardware faults)
  - CSV timeline export (--csv)
  - Backward-compatible legacy mode (--mode legacy)
"""

import csv
import json
import os
import sys
import time
import argparse
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR) == "rosbag_analyzer" else THIS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from rosbag_analyzer.rosbag_bridge import ROSBagBridge, LOG_LEVELS
except ModuleNotFoundError:
    from rosbag_bridge import ROSBagBridge, LOG_LEVELS

from rosbags.serde import deserialize_cdr, ros1_to_cdr


# ---------- Constants ----------

# China Standard Time (UTC+8)
CST = timezone(timedelta(hours=8))

# Maximum gap (seconds) between end of one bag and start of next
# to consider them part of the same mission
MISSION_GAP_THRESHOLD = 300  # 5 minutes

# Fields that are expected to be zero on a 2D differential-drive robot
EXPECTED_ZERO_FIELDS = {
    "/odom": {"orientation_x", "orientation_y", "position_z",
              "twist_angular_x", "twist_angular_y",
              "twist_linear_y", "twist_linear_z"},
    "/chassis_cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
    "/cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
    # DiagnosticStatus topics: 'level' is expected to be 0 (OK) when healthy
    "/device/health_status": {"level"},
    "/device/odom_status": {"level", "unicycle_angle_offset"},
    "/device/imu_data": {"level", "stamp_sec", "stamp_nsec"},
    "/device/scrubber_status": {"level"},
    "/device/scrubber_motor_limit": {"level"},
}

# Hardware topics for Gaussian robot diagnostics
HARDWARE_TOPICS = [
    "/device/health_status",
    "/device/odom_status",
    "/device/imu_data",
    "/device/scrubber_status",
    "/device/scrubber_motor_limit",
    "/raw_scan",
    "/scan_rear",
    "/ir_sticker3", "/ir_sticker6", "/ir_sticker7",
    "/protector",
    "/localization/status",
    "/navigation/status",
]

# Health flags in /device/health_status where "false" (0.0) indicates a fault.
# These are critical hardware subsystems to monitor.
HEALTH_FLAG_NAMES = {
    "rear_rolling_brush_motor", "front_rolling_brush_motor",
    "imu_board", "ultrasonic_board", "motor_driver",
    "battery_disconnection", "mcu_disconnection", "mcu_delay",
    "laser_disconnection", "router_disconnection", "tablet_disconnection",
    "odom_left_delta", "odom_right_delta", "odom_delta_speed", "odom_track_delta",
    "motor_driver_emergency", "imu_roll_pitch_abnormal", "imu_overturn",
}


# ---------- Helpers ----------

def format_absolute_time(unix_sec: float) -> str:
    """Convert Unix epoch seconds to human-readable datetime string (CST)."""
    dt = datetime.fromtimestamp(unix_sec, tz=CST)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecond precision


# ======================================================================
# DiagnosticAnalyzer — per-bag analysis (11 phases + /rosout)
# ======================================================================

class DiagnosticAnalyzer:
    """
    Rule-based diagnostic analyzer implementing the project plan's
    hypothesis-testing workflow using the ROSBag Bridge tools.
    """

    def __init__(self):
        self.bridge = ROSBagBridge()
        self.evidence = []
        self.anomalies = []
        self.log_events = []
        self.warnings = []

    def add_evidence(self, finding: str):
        self.evidence.append(finding)
        print(f"    [EVIDENCE] {finding}")

    def add_anomaly(self, severity: str, category: str, description: str,
                    details: dict = None, timestamp: float = None):
        anomaly = {
            "severity": severity,
            "category": category,
            "description": description,
            "details": details or {},
        }
        if timestamp is not None:
            anomaly["timestamp"] = timestamp
            anomaly["timestamp_str"] = format_absolute_time(timestamp)
        self.anomalies.append(anomaly)
        icon = {"CRITICAL": "!!!", "WARNING": "!!", "INFO": "!"}[severity]
        print(f"    [{icon} {severity}] {category}: {description}")

    # ------------------------------------------------------------------
    # Phase 1.5: /rosout log extraction
    # ------------------------------------------------------------------
    def _scan_rosout_logs(self, bag_path: str) -> list:
        """Scan /rosout and /rosout_agg for WARN/ERROR/FATAL messages."""
        log_events = []
        try:
            with self.bridge._open_cached(bag_path) as reader:
                log_conns = [c for c in reader.connections
                             if c.topic in ['/rosout', '/rosout_agg']]
                if not log_conns:
                    return log_events

                for conn, ts_ns, rawdata in reader.messages(connections=log_conns):
                    try:
                        msg = deserialize_cdr(
                            ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
                        if hasattr(msg, 'level') and msg.level >= 4:  # WARN+
                            log_events.append({
                                "timestamp": ts_ns / 1e9,
                                "timestamp_str": format_absolute_time(ts_ns / 1e9),
                                "level": int(msg.level),
                                "level_str": LOG_LEVELS.get(msg.level, f"LVL{msg.level}"),
                                "node": str(getattr(msg, 'name', '')),
                                "message": str(getattr(msg, 'msg', '')),
                                "topic": conn.topic,
                            })
                    except Exception:
                        continue
        except Exception as e:
            print(f"    [WARN] Could not scan /rosout: {e}")

        log_events.sort(key=lambda ev: ev["timestamp"])
        return log_events

    def analyze(self, bag_path: str) -> dict:
        """Run complete diagnostic analysis on a bag file."""
        bag_name = os.path.basename(bag_path)
        self.evidence = []
        self.anomalies = []
        self.log_events = []

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
        print(f"  Time range: {format_absolute_time(metadata['start_time'])} to {format_absolute_time(metadata['end_time'])}")
        print(f"  Topics: {metadata['num_topics']}, Messages: {metadata['total_messages']}")
        print(f"  (took {elapsed:.2f}s)")

        start_time = metadata["start_time"]
        end_time = metadata["end_time"]
        duration = metadata["duration"]

        # Categorize topics
        topic_map = {t["name"]: t for t in metadata["topics"]}
        self.add_evidence(f"Bag: {bag_name}, duration={duration:.1f}s, {metadata['num_topics']} topics")

        # ---- Phase 1.5: /rosout Log Extraction ----
        print(f"\n--- Phase 1.5: /rosout Log Extraction ---")
        t0 = time.time()
        self.log_events = self._scan_rosout_logs(bag_path)
        elapsed = time.time() - t0
        print(f"  Found {len(self.log_events)} WARN/ERROR/FATAL log messages (took {elapsed:.2f}s)")
        if self.log_events:
            for ev in self.log_events[:5]:
                print(f"    [{ev['timestamp_str']}] {ev['level_str']:<5} [{ev['node']}]: "
                      f"{ev['message'][:100]}")
            if len(self.log_events) > 5:
                print(f"    ... and {len(self.log_events) - 5} more")

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
                        ts = stds[i][0]
                        self.add_anomaly("WARNING", "FREEZE_ONSET",
                            f"{topic}.{field} froze at {format_absolute_time(ts)} "
                            f"(std went from {prev_std:.6f} to {curr_std:.6f}, "
                            f"value={stds[i][2]:.6f})",
                            {"topic": topic, "field": field, "time": ts},
                            timestamp=ts)

                    # Detect resume: std transitions from ~0 to >0
                    if prev_std < 1e-6 and curr_std > 0.001 and stds[i-1][2] != 0:
                        ts = stds[i][0]
                        self.add_anomaly("INFO", "SENSOR_RESUME",
                            f"{topic}.{field} resumed at {format_absolute_time(ts)}",
                            {"topic": topic, "field": field, "time": ts},
                            timestamp=ts)

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
                        ts = entry["time"]
                        self.add_anomaly("CRITICAL", "FREQUENCY_DROPOUT",
                            f"{topic} dropped to {entry['hz']:.1f}Hz at {format_absolute_time(ts)} "
                            f"(expected ~{mean_hz:.1f}Hz)",
                            {"topic": topic, "time": ts, "hz": entry["hz"]},
                            timestamp=ts)

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

        # ---- Phase 8: Hardware Health (DiagnosticStatus topics) ----
        hw_available = [t for t in HARDWARE_TOPICS if t in topic_map]
        if hw_available:
            print(f"\n--- Phase 8: Hardware Health ({len(hw_available)} topics) ---")
            self._check_hardware_health(bag_path, topic_map)

        # ---- Phase 9: Lidar Health ----
        lidar_topics = [t for t in ["/raw_scan", "/scan_rear"] if t in topic_map]
        if lidar_topics:
            print(f"\n--- Phase 9: Lidar Health ---")
            self._check_lidar_health(bag_path, lidar_topics, end_time)

        # ---- Phase 10: Scrubber Health ----
        if "/device/scrubber_status" in topic_map:
            print(f"\n--- Phase 10: Scrubber Health ---")
            self._check_scrubber_health(bag_path, topic_map)

        # ---- Phase 11: Safety Systems ----
        safety_topics = [t for t in ["/protector", "/ir_sticker3", "/ir_sticker6",
                                      "/ir_sticker7", "/localization/status",
                                      "/navigation/status"] if t in topic_map]
        if safety_topics:
            print(f"\n--- Phase 11: Safety Systems ---")
            self._check_safety_systems(bag_path, topic_map)

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
                    ts = cw["window_start"]
                    self.add_anomaly("WARNING", "STALL_DETECTED",
                        f"Commands active but odom frozen in window "
                        f"[{format_absolute_time(ts)}, {format_absolute_time(cw['window_end'])}]",
                        {"window_start": ts,
                         "cmd_mean": cmd_lx.get("mean"),
                         "odom_std": odom_tlx.get("std")},
                        timestamp=ts)

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

    # ------------------------------------------------------------------
    # Phase 8: Hardware Health (DiagnosticStatus topics)
    # ------------------------------------------------------------------
    def _check_hardware_health(self, bag_path, topic_map):
        """Analyse /device/* DiagnosticStatus topics for hardware faults."""

        # --- 8a: /device/health_status — boolean health flags ---
        if "/device/health_status" in topic_map:
            print(f"\n  Checking /device/health_status...")
            stats = self.bridge.get_topic_statistics(bag_path, "/device/health_status")
            if stats and "fields" in stats[0]:
                for field, s in stats[0]["fields"].items():
                    if field in EXPECTED_ZERO_FIELDS.get("/device/health_status", set()):
                        continue
                    if field not in HEALTH_FLAG_NAMES:
                        continue
                    # value=0.0 means "false" = fault for boolean health flags
                    if s["mean"] < 0.5 and s["count"] > 5:
                        self.add_anomaly("CRITICAL", "HW_FAULT",
                            f"/device/health_status.{field} reports FAULT "
                            f"(mean={s['mean']:.2f}, {s['count']} msgs)",
                            {"topic": "/device/health_status", "field": field,
                             "mean": s["mean"]})
                    else:
                        self.add_evidence(f"HW health {field}: OK")

            # Windowed: detect fault onset/recovery
            wstats = self.bridge.get_topic_statistics(
                bag_path, "/device/health_status", window_size=30.0)
            if wstats:
                for field in HEALTH_FLAG_NAMES:
                    vals = [(w["window_start"],
                             w["fields"].get(field, {}).get("mean", 1.0))
                            for w in wstats if field in w.get("fields", {})]
                    for i in range(1, len(vals)):
                        prev_val, curr_val = vals[i-1][1], vals[i][1]
                        ts = vals[i][0]
                        if prev_val > 0.5 and curr_val < 0.5:
                            self.add_anomaly("CRITICAL", "HW_FAULT_ONSET",
                                f"{field} went to FAULT at {format_absolute_time(ts)}",
                                {"field": field, "time": ts},
                                timestamp=ts)
                        elif prev_val < 0.5 and curr_val > 0.5:
                            self.add_anomaly("INFO", "HW_FAULT_RECOVERED",
                                f"{field} recovered at {format_absolute_time(ts)}",
                                {"field": field, "time": ts},
                                timestamp=ts)

        # --- 8b: /device/odom_status — wheel encoder diagnostics ---
        if "/device/odom_status" in topic_map:
            print(f"\n  Checking /device/odom_status...")
            stats = self.bridge.get_topic_statistics(bag_path, "/device/odom_status")
            if stats and "fields" in stats[0]:
                for field, s in stats[0]["fields"].items():
                    if field in EXPECTED_ZERO_FIELDS.get("/device/odom_status", set()):
                        continue
                    # For odom error flags: "true" (1.0) = no error; "false" (0.0) = error
                    if field.endswith("_error") or field == "is_gliding":
                        if s["min"] < 0.5:
                            self.add_anomaly("WARNING", "ODOM_HW_ERROR",
                                f"/device/odom_status.{field} reported error "
                                f"(min={s['min']:.0f}, mean={s['mean']:.2f})",
                                {"field": field, "mean": s["mean"]})

        # --- 8c: /device/imu_data — raw IMU via DiagnosticStatus ---
        if "/device/imu_data" in topic_map:
            print(f"\n  Checking /device/imu_data...")
            stats = self.bridge.get_topic_statistics(bag_path, "/device/imu_data")
            if stats and "fields" in stats[0]:
                for field, s in stats[0]["fields"].items():
                    if field in EXPECTED_ZERO_FIELDS.get("/device/imu_data", set()):
                        continue
                    if s["std"] < 1e-6 and s["count"] > 10:
                        if s["mean"] == 0.0 and field.startswith("magnetic"):
                            # Magnetometer may be disabled — info only
                            self.add_anomaly("INFO", "IMU_HW_FIELD_ZERO",
                                f"/device/imu_data.{field} always 0 (sensor may be disabled)",
                                {"field": field})
                        elif s["mean"] != 0.0:
                            self.add_anomaly("WARNING", "IMU_HW_FROZEN",
                                f"/device/imu_data.{field} frozen at {s['mean']:.1f}",
                                {"field": field, "mean": s["mean"]})
                    else:
                        self.add_evidence(
                            f"IMU HW {field}: mean={s['mean']:.1f} std={s['std']:.2f}")

    # ------------------------------------------------------------------
    # Phase 9: Lidar Health
    # ------------------------------------------------------------------
    def _check_lidar_health(self, bag_path, lidar_topics, end_time):
        """Check laser scanners for point count drops and frequency issues."""
        for topic in lidar_topics:
            print(f"\n  Checking {topic}...")

            # Global stats
            stats = self.bridge.get_topic_statistics(bag_path, topic)
            if not stats or "fields" not in stats[0]:
                continue
            fields = stats[0]["fields"]

            valid = fields.get("num_valid_points", {})
            total = fields.get("num_total_points", {})
            if valid:
                self.add_evidence(
                    f"{topic} valid points: mean={valid.get('mean', 0):.0f} "
                    f"min={valid.get('min', 0):.0f} max={valid.get('max', 0):.0f}")

                # Severe drop in valid points
                if valid.get("mean", 0) > 50 and valid.get("min", 0) < valid.get("mean", 0) * 0.3:
                    self.add_anomaly("WARNING", "LIDAR_POINT_DROP",
                        f"{topic} valid points dropped to {valid['min']:.0f} "
                        f"(mean={valid['mean']:.0f})",
                        {"topic": topic, "min_points": valid["min"],
                         "mean_points": valid["mean"]})

            # Windowed: detect point count degradation over time
            wstats = self.bridge.get_topic_statistics(bag_path, topic, window_size=30.0)
            if wstats:
                for i in range(1, len(wstats)):
                    prev_valid = wstats[i-1].get("fields", {}).get("num_valid_points", {})
                    curr_valid = wstats[i].get("fields", {}).get("num_valid_points", {})
                    prev_mean = prev_valid.get("mean", 0)
                    curr_mean = curr_valid.get("mean", 0)
                    if prev_mean > 50 and curr_mean < prev_mean * 0.3:
                        ts = wstats[i]["window_start"]
                        self.add_anomaly("WARNING", "LIDAR_DEGRADATION",
                            f"{topic} point count dropped from {prev_mean:.0f} to "
                            f"{curr_mean:.0f} at {format_absolute_time(ts)}",
                            {"topic": topic, "time": ts},
                            timestamp=ts)

            # Frequency check
            freq = self.bridge.check_topic_frequency(bag_path, topic, resolution=5.0)
            mean_hz = freq.get("mean_hz", 0)
            self.add_evidence(f"{topic} frequency: {mean_hz:.1f} Hz")
            series = freq.get("frequency_series", [])
            for entry in series:
                if mean_hz > 0 and entry["hz"] < mean_hz * 0.3:
                    if entry["time"] < end_time - 10:
                        ts = entry["time"]
                        self.add_anomaly("CRITICAL", "LIDAR_FREQ_DROPOUT",
                            f"{topic} dropped to {entry['hz']:.1f}Hz "
                            f"at {format_absolute_time(ts)} (expected ~{mean_hz:.1f}Hz)",
                            {"topic": topic, "time": ts, "hz": entry["hz"]},
                            timestamp=ts)

    # ------------------------------------------------------------------
    # Phase 10: Scrubber Health
    # ------------------------------------------------------------------
    def _check_scrubber_health(self, bag_path, topic_map):
        """Check cleaning subsystem: brushes, water, valves."""
        if "/device/scrubber_status" in topic_map:
            print(f"\n  Checking /device/scrubber_status...")
            stats = self.bridge.get_topic_statistics(bag_path, "/device/scrubber_status")
            if stats and "fields" in stats[0]:
                fields = stats[0]["fields"]

                # Water level (0 = empty, higher = more)
                water = fields.get("water_level", {})
                if water and water.get("min", 99) == 0:
                    self.add_anomaly("WARNING", "SCRUBBER_NO_WATER",
                        f"Water level reached 0 during operation",
                        {"min_level": water["min"], "mean_level": water.get("mean", 0)})

                # Brush motor state
                brush = fields.get("rolling_brush_motor", {})
                if brush:
                    self.add_evidence(
                        f"Brush motor: mean={brush.get('mean', 0):.2f} "
                        f"(1.0=on, 0.0=off)")

                # Log key scrubber metrics
                for key in ["brush_spin_level", "detergent_level", "valve"]:
                    s = fields.get(key, {})
                    if s:
                        self.add_evidence(
                            f"Scrubber {key}: mean={s.get('mean', 0):.2f}")

        if "/device/scrubber_motor_limit" in topic_map:
            print(f"\n  Checking /device/scrubber_motor_limit...")
            stats = self.bridge.get_topic_statistics(bag_path, "/device/scrubber_motor_limit")
            if stats and "fields" in stats[0]:
                fields = stats[0]["fields"]
                vel = fields.get("velocity_speed", {})
                if vel:
                    self.add_evidence(
                        f"Scrubber motor velocity: mean={vel.get('mean', 0):.4f} "
                        f"max={vel.get('max', 0):.4f}")

    # ------------------------------------------------------------------
    # Phase 11: Safety Systems
    # ------------------------------------------------------------------
    def _check_safety_systems(self, bag_path, topic_map):
        """Check protector, IR stickers, localization/navigation status."""

        # --- 11a: /protector — safety bumper string ---
        if "/protector" in topic_map:
            print(f"\n  Checking /protector...")
            # /protector is std_msgs/String with data like "000000"
            # Any '1' means a protector is triggered
            samples = self.bridge.sample_messages(bag_path, "/protector", count=50)
            triggered_count = 0
            total_count = 0
            for m in samples.get("messages", []):
                total_count += 1
                val = m.get("data", {}).get("data", "")
                if isinstance(val, str) and "1" in val:
                    triggered_count += 1

            if triggered_count > 0:
                self.add_anomaly("WARNING", "PROTECTOR_TRIGGERED",
                    f"Protector triggered in {triggered_count}/{total_count} sampled messages",
                    {"triggered": triggered_count, "sampled": total_count})
            else:
                self.add_evidence(f"Protector: clear ({total_count} samples)")

        # --- 11b: IR stickers (cliff/proximity sensors) ---
        for topic in ["/ir_sticker3", "/ir_sticker6", "/ir_sticker7"]:
            if topic not in topic_map:
                continue
            stats = self.bridge.get_topic_statistics(bag_path, topic)
            if not stats or "fields" not in stats[0]:
                continue
            data = stats[0]["fields"].get("data", {})
            if data:
                self.add_evidence(
                    f"{topic}: mean={data.get('mean', 0):.2f} "
                    f"range=[{data.get('min', 0):.2f}, {data.get('max', 0):.2f}]")
                # All-zero IR may mean sensor disconnected
                if data.get("std", 0) < 1e-6 and data.get("mean", 0) == 0 and data.get("count", 0) > 10:
                    self.add_anomaly("WARNING", "IR_SENSOR_DEAD",
                        f"{topic} always 0.0 — sensor may be disconnected",
                        {"topic": topic})

        # --- 11c: Localization & navigation status ---
        for topic, label in [("/localization/status", "Localization"),
                              ("/navigation/status", "Navigation")]:
            if topic not in topic_map:
                continue
            stats = self.bridge.get_topic_statistics(bag_path, topic)
            if not stats or "fields" not in stats[0]:
                continue
            data = stats[0]["fields"].get("data", {})
            if data:
                self.add_evidence(
                    f"{label} status: mean={data.get('mean', 0):.1f} "
                    f"range=[{data.get('min', 0):.0f}, {data.get('max', 0):.0f}]")

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
            f"Time: {format_absolute_time(metadata['start_time'])} to {format_absolute_time(metadata['end_time'])}",
            f"Critical Issues: {critical_count}",
            f"Warnings: {warning_count}",
            f"Info: {info_count}",
            f"Log Events (WARN+): {len(self.log_events)}",
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
                "start_time": metadata["start_time"],
                "end_time": metadata["end_time"],
                "start_time_str": format_absolute_time(metadata["start_time"]),
                "end_time_str": format_absolute_time(metadata["end_time"]),
            },
            "anomalies": self.anomalies,
            "log_events": self.log_events,
            "evidence": self.evidence,
            "diagnosis": diagnosis,
            "metrics": {
                "critical_count": critical_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "evidence_count": len(self.evidence),
                "log_event_count": len(self.log_events),
            },
        }


# ======================================================================
# MissionOrchestrator — multi-bag virtual timeline
# ======================================================================

@dataclass
class BagInfo:
    """Lightweight metadata for sorting/grouping bags."""
    path: str
    name: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class Mission:
    """A group of temporally contiguous bags from one recording session."""
    mission_id: int
    bags: List[BagInfo]
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self):
        if self.bags:
            self.start_time = self.bags[0].start_time
            self.end_time = self.bags[-1].end_time


class MissionOrchestrator:
    """Orchestrates multi-bag analysis with virtual timeline."""

    def __init__(self, bag_dir: str, gap_threshold: float = MISSION_GAP_THRESHOLD):
        self.bag_dir = bag_dir
        self.gap_threshold = gap_threshold
        self.bridge = ROSBagBridge()

    def discover_and_group(self) -> List[Mission]:
        """Discover bags, sort by time, group into missions."""
        bag_files = sorted(f for f in os.listdir(self.bag_dir) if f.endswith('.bag'))
        if not bag_files:
            return []

        # Collect metadata for all bags
        bag_infos = []
        for bf in bag_files:
            path = os.path.join(self.bag_dir, bf)
            try:
                meta = self.bridge.get_bag_metadata(path)
                bag_infos.append(BagInfo(
                    path=path,
                    name=bf,
                    start_time=meta["start_time"],
                    end_time=meta["end_time"],
                    duration=meta["duration"],
                ))
            except Exception as e:
                print(f"  [WARN] Could not read metadata for {bf}: {e}")
                continue

        if not bag_infos:
            return []

        # Sort by start_time
        bag_infos.sort(key=lambda b: b.start_time)

        # Group into missions based on time gaps
        missions = []
        current_group = [bag_infos[0]]

        for i in range(1, len(bag_infos)):
            prev_end = current_group[-1].end_time
            curr_start = bag_infos[i].start_time
            gap = curr_start - prev_end

            if gap <= self.gap_threshold:
                current_group.append(bag_infos[i])
            else:
                missions.append(Mission(
                    mission_id=len(missions) + 1,
                    bags=current_group,
                ))
                current_group = [bag_infos[i]]

        # Don't forget the last group
        missions.append(Mission(
            mission_id=len(missions) + 1,
            bags=current_group,
        ))

        return missions

    def analyze_mission(self, mission: Mission) -> dict:
        """Analyze all bags in a mission with cross-bag awareness."""
        per_bag_reports = []
        all_timeline_events = []

        for bag_info in mission.bags:
            analyzer = DiagnosticAnalyzer()
            report = analyzer.analyze(bag_info.path)
            per_bag_reports.append(report)

            # Collect timeline events from anomalies
            for anomaly in report["anomalies"]:
                ts = anomaly.get("timestamp")
                if ts is None:
                    # Bag-wide anomalies: use bag start_time
                    ts = bag_info.start_time
                all_timeline_events.append({
                    "timestamp": ts,
                    "timestamp_str": format_absolute_time(ts),
                    "bag": bag_info.name,
                    "type": "anomaly",
                    "severity": anomaly["severity"],
                    "category": anomaly["category"],
                    "description": anomaly["description"],
                    "details": anomaly.get("details", {}),
                })

            # Collect log events
            for log_ev in report.get("log_events", []):
                all_timeline_events.append({
                    "timestamp": log_ev["timestamp"],
                    "timestamp_str": log_ev["timestamp_str"],
                    "bag": bag_info.name,
                    "type": "log",
                    "severity": log_ev["level_str"],
                    "category": f"LOG_{log_ev['level_str']}",
                    "description": f"[{log_ev['node']}]: {log_ev['message']}",
                    "details": {
                        "node": log_ev["node"],
                        "level": log_ev["level"],
                    },
                })

        # Sort timeline by timestamp
        all_timeline_events.sort(key=lambda e: e["timestamp"])

        # Suppress boundary false positives
        all_timeline_events = self._suppress_boundary_artifacts(
            mission, all_timeline_events)

        # Detect cross-bag anomalies
        cross_bag_anomalies = self._detect_cross_bag_anomalies(
            mission, per_bag_reports)

        # Correlate logs with sensor anomalies
        correlations = self._correlate_logs_and_anomalies(all_timeline_events)

        # Compute overall mission health
        overall_health = self._compute_mission_health(
            per_bag_reports, cross_bag_anomalies)

        return {
            "mission_id": mission.mission_id,
            "start_time": format_absolute_time(mission.start_time),
            "end_time": format_absolute_time(mission.end_time),
            "duration_sec": round(mission.end_time - mission.start_time, 1),
            "num_bags": len(mission.bags),
            "bags": [b.name for b in mission.bags],
            "overall_health": overall_health,
            "timeline": all_timeline_events,
            "per_bag_summaries": [
                {
                    "bag_name": r["bag_name"],
                    "health_status": r["health_status"],
                    "duration": r["metadata"]["duration"],
                    "start_time": r["metadata"].get("start_time_str", ""),
                    "end_time": r["metadata"].get("end_time_str", ""),
                    "critical_count": r["metrics"]["critical_count"],
                    "warning_count": r["metrics"]["warning_count"],
                    "log_event_count": r["metrics"].get("log_event_count", 0),
                }
                for r in per_bag_reports
            ],
            "cross_bag_anomalies": cross_bag_anomalies,
            "correlations": correlations,
        }

    def _suppress_boundary_artifacts(self, mission: Mission,
                                      events: list) -> list:
        """Remove false frequency dropouts at bag boundaries."""
        if len(mission.bags) < 2:
            return events

        BOUNDARY_MARGIN = 3.0  # seconds
        boundary_windows = []

        # Windows around each bag-to-bag boundary
        for i in range(len(mission.bags) - 1):
            boundary_time = mission.bags[i].end_time
            boundary_windows.append(
                (boundary_time - BOUNDARY_MARGIN,
                 boundary_time + BOUNDARY_MARGIN))

        # Also suppress at the very start/end of the mission
        boundary_windows.append(
            (mission.bags[0].start_time,
             mission.bags[0].start_time + BOUNDARY_MARGIN))
        boundary_windows.append(
            (mission.bags[-1].end_time - BOUNDARY_MARGIN,
             mission.bags[-1].end_time))

        def is_in_boundary(ts):
            for bw_start, bw_end in boundary_windows:
                if bw_start <= ts <= bw_end:
                    return True
            return False

        filtered = []
        suppressed_count = 0
        for event in events:
            if (event["type"] == "anomaly" and
                event["category"] in ("FREQUENCY_DROPOUT", "LIDAR_FREQ_DROPOUT") and
                is_in_boundary(event["timestamp"])):
                suppressed_count += 1
                continue
            filtered.append(event)

        if suppressed_count > 0:
            print(f"  [Boundary filter] Suppressed {suppressed_count} "
                  f"frequency dropout(s) at bag boundaries")

        return filtered

    def _detect_cross_bag_anomalies(self, mission: Mission,
                                     per_bag_reports: list) -> list:
        """Detect anomalies that span bag boundaries."""
        cross_bag = []

        if len(per_bag_reports) < 2:
            return cross_bag

        for i in range(1, len(per_bag_reports)):
            prev_report = per_bag_reports[i - 1]
            curr_report = per_bag_reports[i]
            prev_bag = mission.bags[i - 1].name
            curr_bag = mission.bags[i].name

            # --- Persistent frozen sensors ---
            prev_frozen = {
                (a["details"].get("topic"), a["details"].get("field"))
                for a in prev_report["anomalies"]
                if a["category"] in ("FROZEN_SENSOR", "FREEZE_ONSET")
            }
            curr_frozen = {
                (a["details"].get("topic"), a["details"].get("field"))
                for a in curr_report["anomalies"]
                if a["category"] in ("FROZEN_SENSOR", "FREEZE_ONSET")
            }

            persistent = prev_frozen & curr_frozen
            for topic, field_name in persistent:
                if topic is None:
                    continue

                # Compare frozen values
                prev_val = next(
                    (a["details"].get("mean") for a in prev_report["anomalies"]
                     if a["details"].get("topic") == topic and
                     a["details"].get("field") == field_name),
                    None)
                curr_val = next(
                    (a["details"].get("mean") for a in curr_report["anomalies"]
                     if a["details"].get("topic") == topic and
                     a["details"].get("field") == field_name),
                    None)

                same_value = True
                if prev_val is not None and curr_val is not None:
                    same_value = abs(prev_val - curr_val) < 1e-4

                cross_bag.append({
                    "type": "PERSISTENT_FREEZE",
                    "severity": "CRITICAL",
                    "description": (
                        f"{topic}.{field_name} frozen across bag boundary "
                        f"({prev_bag} -> {curr_bag})"
                        + (f", same value={prev_val:.6f}" if same_value and prev_val is not None else "")
                    ),
                    "bags": [prev_bag, curr_bag],
                    "topic": topic,
                    "field": field_name,
                    "same_value": same_value,
                })

            # --- Persistent hardware faults ---
            prev_hw = {
                a["details"].get("field")
                for a in prev_report["anomalies"]
                if a["category"] in ("HW_FAULT", "HW_FAULT_ONSET")
            }
            curr_hw = {
                a["details"].get("field")
                for a in curr_report["anomalies"]
                if a["category"] in ("HW_FAULT", "HW_FAULT_ONSET")
            }

            persistent_hw = prev_hw & curr_hw
            for hw_field in persistent_hw:
                if hw_field is None:
                    continue
                cross_bag.append({
                    "type": "PERSISTENT_HW_FAULT",
                    "severity": "CRITICAL",
                    "description": (
                        f"Hardware fault '{hw_field}' persists across "
                        f"{prev_bag} -> {curr_bag}"
                    ),
                    "bags": [prev_bag, curr_bag],
                    "field": hw_field,
                })

            # --- Cross-bag recovery ---
            prev_not_curr = prev_frozen - curr_frozen
            for topic, field_name in prev_not_curr:
                if topic is None:
                    continue
                has_resume = any(
                    a["category"] == "SENSOR_RESUME" and
                    a["details"].get("topic") == topic and
                    a["details"].get("field") == field_name
                    for a in curr_report["anomalies"]
                )
                if has_resume:
                    cross_bag.append({
                        "type": "CROSS_BAG_RECOVERY",
                        "severity": "INFO",
                        "description": (
                            f"{topic}.{field_name} frozen in {prev_bag}, "
                            f"resumed in {curr_bag}"
                        ),
                        "bags": [prev_bag, curr_bag],
                        "topic": topic,
                        "field": field_name,
                    })

        return cross_bag

    def _correlate_logs_and_anomalies(self, timeline_events: list,
                                       window_sec: float = 2.0) -> list:
        """Find temporal correlations between log events and sensor anomalies.

        Two-pointer sweep for O(N+M) efficiency over sorted lists.
        """
        logs = [e for e in timeline_events if e["type"] == "log"]
        anomalies = [e for e in timeline_events if e["type"] == "anomaly"]

        if not logs or not anomalies:
            return []

        correlations = []
        log_idx = 0

        for anomaly in anomalies:
            a_ts = anomaly["timestamp"]
            matched_logs = []

            # Rewind if needed
            while log_idx > 0 and logs[log_idx]["timestamp"] > a_ts - window_sec:
                log_idx -= 1

            # Scan forward through logs within the window
            j = log_idx
            while j < len(logs):
                l_ts = logs[j]["timestamp"]
                if l_ts > a_ts + window_sec:
                    break
                if l_ts >= a_ts - window_sec:
                    matched_logs.append({
                        "log_timestamp": logs[j]["timestamp_str"],
                        "log_node": logs[j].get("details", {}).get("node", ""),
                        "log_message": logs[j]["description"],
                        "time_delta_sec": round(l_ts - a_ts, 3),
                    })
                j += 1

            if matched_logs:
                correlations.append({
                    "anomaly_timestamp": anomaly["timestamp_str"],
                    "anomaly_category": anomaly["category"],
                    "anomaly_description": anomaly["description"],
                    "anomaly_bag": anomaly["bag"],
                    "correlated_logs": matched_logs,
                })

        return correlations

    def _compute_mission_health(self, per_bag_reports: list,
                                 cross_bag_anomalies: list) -> str:
        """Compute overall mission health."""
        total_critical = sum(r["metrics"]["critical_count"] for r in per_bag_reports)
        total_warnings = sum(r["metrics"]["warning_count"] for r in per_bag_reports)
        cross_critical = sum(1 for a in cross_bag_anomalies
                            if a.get("severity") == "CRITICAL")

        if total_critical > 0 or cross_critical > 0:
            return "UNHEALTHY"
        elif total_warnings > 5:
            return "DEGRADED"
        elif total_warnings > 0:
            return "MARGINAL"
        else:
            return "HEALTHY"


# ======================================================================
# CSV Export
# ======================================================================

def export_timeline_csv(mission_report: dict, output_path: str):
    """Export the timeline from a mission report to CSV."""
    fieldnames = ["timestamp", "bag", "type", "severity", "category",
                  "description", "details"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()

        for event in mission_report.get("timeline", []):
            row = {
                "timestamp": event.get("timestamp_str", ""),
                "bag": event.get("bag", ""),
                "type": event.get("type", ""),
                "severity": event.get("severity", ""),
                "category": event.get("category", ""),
                "description": event.get("description", ""),
                "details": json.dumps(event.get("details", {})),
            }
            writer.writerow(row)

    print(f"CSV timeline exported to: {output_path}")


# ======================================================================
# CLI Entry Points
# ======================================================================

def _run_legacy_mode(bag_dir: str, report_path: str = None):
    """Original per-bag analysis — backward compatible."""
    bag_files = sorted(f for f in os.listdir(bag_dir) if f.endswith('.bag'))
    if not bag_files:
        print("No .bag files found!")
        return

    all_reports = []
    for bag_file in bag_files:
        bag_path = os.path.join(bag_dir, bag_file)
        analyzer = DiagnosticAnalyzer()
        report = analyzer.analyze(bag_path)
        all_reports.append(report)

    report_path = report_path or os.path.join(
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


def _run_mission_mode(bag_dir: str, report_path: str = None,
                      csv_path: str = None, gap_threshold: float = 300.0):
    """Mission-centric multi-bag analysis with virtual timeline."""
    orchestrator = MissionOrchestrator(bag_dir, gap_threshold=gap_threshold)

    # Phase A: Discover and group
    print(f"\n{'='*70}")
    print("MISSION DISCOVERY")
    print(f"{'='*70}")
    missions = orchestrator.discover_and_group()

    if not missions:
        print("No bag files found or all unreadable!")
        return

    for mission in missions:
        print(f"\n  Mission {mission.mission_id}:")
        print(f"    Time range: {format_absolute_time(mission.start_time)} to "
              f"{format_absolute_time(mission.end_time)}")
        print(f"    Duration: {mission.end_time - mission.start_time:.1f}s")
        print(f"    Bags: {len(mission.bags)}")
        for bag in mission.bags:
            print(f"      - {bag.name} ({bag.duration:.1f}s)")

    # Phase B: Analyze each mission
    mission_reports = []
    for mission in missions:
        print(f"\n{'='*70}")
        print(f"ANALYZING MISSION {mission.mission_id} "
              f"({len(mission.bags)} bags)")
        print(f"{'='*70}")
        report = orchestrator.analyze_mission(mission)
        mission_reports.append(report)

    # Phase C: Output
    full_report = {
        "analyzer_version": "2.0",
        "analysis_mode": "mission",
        "generated_at": format_absolute_time(time.time()),
        "missions": mission_reports,
    }

    report_path = report_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diagnostic_report.json",
    )
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n\nMission report saved to: {report_path}")

    # CSV export if requested
    if csv_path:
        for mr in mission_reports:
            if len(mission_reports) > 1:
                base, ext = os.path.splitext(csv_path)
                path = f"{base}_mission{mr['mission_id']}{ext}"
            else:
                path = csv_path
            export_timeline_csv(mr, path)

    # Print mission summaries
    print(f"\n{'='*70}")
    print("MISSION SUMMARIES")
    print(f"{'='*70}")
    for mr in mission_reports:
        print(f"\n  Mission {mr['mission_id']}: {mr['overall_health']}")
        print(f"    Time: {mr['start_time']} to {mr['end_time']}")
        print(f"    Duration: {mr['duration_sec']:.1f}s")
        print(f"    Bags: {mr['num_bags']}")
        print(f"    Timeline events: {len(mr['timeline'])}")
        print(f"    Cross-bag anomalies: {len(mr['cross_bag_anomalies'])}")
        print(f"    Log-sensor correlations: {len(mr['correlations'])}")
        for s in mr["per_bag_summaries"]:
            print(f"      {s['bag_name']}: {s['health_status']} "
                  f"(C:{s['critical_count']} W:{s['warning_count']} "
                  f"L:{s['log_event_count']})")


def main():
    parser = argparse.ArgumentParser(
        description="Run rule-based diagnostics on .bag files.")
    default_bag_dir = PROJECT_ROOT
    parser.add_argument(
        "--bag-dir",
        default=default_bag_dir,
        help=f"Directory containing .bag files (default: {default_bag_dir})",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Output path for diagnostic report JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "mission"],
        default="mission",
        help="Analysis mode: 'legacy' for per-bag (backward compat), "
             "'mission' for multi-bag virtual timeline (default: mission)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path for CSV timeline export (mission mode only)",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=300.0,
        help="Max seconds between bags to group into same mission (default: 300)",
    )
    args = parser.parse_args()
    bag_dir = os.path.abspath(args.bag_dir)

    if not os.path.isdir(bag_dir):
        print(f"Bag directory does not exist: {bag_dir}")
        return

    if args.mode == "legacy":
        _run_legacy_mode(bag_dir, args.report_path)
    else:
        _run_mission_mode(bag_dir, args.report_path, args.csv,
                         args.gap_threshold)


if __name__ == "__main__":
    main()
