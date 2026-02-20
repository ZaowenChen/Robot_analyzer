"""
DiagnosticAnalyzer -- per-bag log-first diagnostic analysis (v3.0).

Implements a 7-phase pipeline:
  Phase 0: Log Extraction (PRIMARY)
  Phase 1: Reconnaissance
  Phase 2: Sensor Statistics (Global + Windowed)
  Phase 3: Frequency Analysis
  Phase 4: Hardware Health
  Phase 5: Log-Sensor Correlation
  Phase 6: Incident Clustering
"""

import os
import time
from collections import defaultdict
from typing import List, Optional

from bridge import ROSBagBridge
from core.constants import (
    EXPECTED_ZERO_FIELDS,
    HEALTH_FLAG_NAMES,
    HARDWARE_TOPICS,
    MOTION_RELATED_FIELDS,
)
from core.utils import format_absolute_time, LOG_LEVELS
from core.models import LogEvent, Incident
from logs import LogExtractor, StateTimelineBuilder, extract_log_timeline
from analysis.incident_builder import IncidentBuilder


class DiagnosticAnalyzer:
    """
    Log-first diagnostic analyzer implementing the v3.0 pipeline:
      Phase 0: Log Extraction (PRIMARY)
      Phase 1: Reconnaissance
      Phase 2: Sensor Statistics (Global + Windowed)
      Phase 3: Frequency Analysis
      Phase 4: Hardware Health
      Phase 5: Log-Sensor Correlation
      Phase 6: Incident Clustering
    """

    def __init__(self):
        self.bridge = ROSBagBridge()
        self.evidence = []
        self.anomalies = []
        self.log_events: List[LogEvent] = []
        self.state_builder: Optional[StateTimelineBuilder] = None
        self.incidents: List[Incident] = []
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

    def analyze(self, bag_path: str, mode: str = "incident") -> dict:
        """Run complete diagnostic analysis on a bag file.

        Args:
            bag_path: Path to the .bag file
            mode: "incident" (v3.0) or "mission"/"legacy" (v2.0 compat)
        """
        bag_name = os.path.basename(bag_path)
        self.evidence = []
        self.anomalies = []
        self.log_events = []
        self.state_builder = StateTimelineBuilder()
        self.incidents = []

        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC ANALYSIS (v3.0 — Log-First)")
        print(f"Bag: {bag_name}")
        print(f"{'='*70}")

        # ---- Phase 0: Log Extraction (PRIMARY) ----
        print(f"\n--- Phase 0: Log Extraction (PRIMARY — INFO+) ---")
        t0 = time.time()
        self.log_events, self.state_builder = extract_log_timeline(
            bag_path, self.bridge, min_level=2, bag_name=bag_name)
        elapsed = time.time() - t0

        extractor = LogExtractor(self.bridge)
        log_summary = extractor.get_event_summary(self.log_events)
        print(f"  Extracted {log_summary['total_events']} log events "
              f"(pattern match rate: {log_summary['matched_pct']}%) "
              f"(took {elapsed:.2f}s)")

        # Print event type breakdown
        if log_summary["by_event_type"]:
            for etype, count in list(log_summary["by_event_type"].items())[:8]:
                print(f"    {etype:<30} {count:>5}")
            remaining = len(log_summary["by_event_type"]) - 8
            if remaining > 0:
                print(f"    ... and {remaining} more types")

        # Print state transitions
        state_summary = self.state_builder.get_summary()
        if state_summary:
            print(f"\n  State transitions detected:")
            for key, info in state_summary.items():
                print(f"    {key}: {info['total_transitions']} transitions, "
                      f"current={info['current_value']}")

        # ---- Phase 1: Reconnaissance ----
        print(f"\n--- Phase 1: Reconnaissance (get_bag_metadata) ---")
        t0 = time.time()
        metadata = self.bridge.get_bag_metadata(bag_path)
        elapsed = time.time() - t0
        print(f"  Duration: {metadata['duration']:.2f}s")
        print(f"  Time range: {format_absolute_time(metadata['start_time'])} to "
              f"{format_absolute_time(metadata['end_time'])}")
        print(f"  Topics: {metadata['num_topics']}, Messages: {metadata['total_messages']}")
        print(f"  (took {elapsed:.2f}s)")

        start_time = metadata["start_time"]
        end_time = metadata["end_time"]
        duration = metadata["duration"]

        topic_map = {t["name"]: t for t in metadata["topics"]}
        self.add_evidence(f"Bag: {bag_name}, duration={duration:.1f}s, "
                         f"{metadata['num_topics']} topics, "
                         f"{log_summary['total_events']} log events")

        # ---- Phase 2: Sensor Statistics (Global + Windowed) ----
        print(f"\n--- Phase 2: Sensor Statistics ---")
        key_topics = ["/odom", "/chassis_cmd_vel", "/unbiased_imu_PRY",
                      "/localization/current_pose", "/front_end_pose"]
        available_topics = [t for t in key_topics if t in topic_map]

        global_stats = {}
        for topic in available_topics:
            print(f"\n  Analyzing {topic}...")
            stats = self.bridge.get_topic_statistics(bag_path, topic)
            global_stats[topic] = stats

            if stats and "fields" in stats[0]:
                for field_name, s in stats[0]["fields"].items():
                    expected_zeros = EXPECTED_ZERO_FIELDS.get(topic, set())
                    is_expected_zero = field_name in expected_zeros

                    if s["std"] < 1e-6 and s["count"] > 10:
                        if is_expected_zero:
                            pass
                        elif s["mean"] == 0.0:
                            self.add_anomaly("WARNING", "ZERO_FIELD",
                                f"{topic}.{field_name} is constantly 0.0 ({s['count']} msgs)",
                                {"topic": topic, "field": field_name, "mean": s["mean"]})
                        else:
                            self.add_anomaly("CRITICAL", "FROZEN_SENSOR",
                                f"{topic}.{field_name} frozen at {s['mean']:.6f} (std=0, {s['count']} msgs)",
                                {"topic": topic, "field": field_name, "mean": s["mean"], "std": s["std"]})
                    else:
                        self.add_evidence(
                            f"{topic}.{field_name}: mean={s['mean']:.4f} std={s['std']:.4f} "
                            f"range=[{s['min']:.4f}, {s['max']:.4f}]")

        # Windowed analysis for freeze onset/resume
        windowed_stats = {}
        for topic in available_topics:
            wstats = self.bridge.get_topic_statistics(bag_path, topic, window_size=30.0)
            windowed_stats[topic] = wstats

            for field_name in (wstats[0].get("fields", {}).keys() if wstats else []):
                expected_zeros = EXPECTED_ZERO_FIELDS.get(topic, set())
                if field_name in expected_zeros:
                    continue

                stds = [(w["window_start"], w["fields"].get(field_name, {}).get("std", 0),
                         w["fields"].get(field_name, {}).get("mean", 0))
                        for w in wstats if field_name in w.get("fields", {})]

                for i in range(1, len(stds)):
                    prev_std = stds[i-1][1]
                    curr_std = stds[i][1]
                    if prev_std > 0.001 and curr_std < 1e-6:
                        ts = stds[i][0]
                        self.add_anomaly("WARNING", "FREEZE_ONSET",
                            f"{topic}.{field_name} froze at {format_absolute_time(ts)} "
                            f"(std went from {prev_std:.6f} to {curr_std:.6f}, "
                            f"value={stds[i][2]:.6f})",
                            {"topic": topic, "field": field_name, "time": ts},
                            timestamp=ts)

                    if prev_std < 1e-6 and curr_std > 0.001 and stds[i-1][2] != 0:
                        ts = stds[i][0]
                        self.add_anomaly("INFO", "SENSOR_RESUME",
                            f"{topic}.{field_name} resumed at {format_absolute_time(ts)}",
                            {"topic": topic, "field": field_name, "time": ts},
                            timestamp=ts)

        # ---- Phase 3: Frequency Analysis ----
        print(f"\n--- Phase 3: Frequency Analysis ---")
        freq_results = {}
        for topic in available_topics:
            if topic not in topic_map:
                continue
            freq = self.bridge.check_topic_frequency(bag_path, topic, resolution=5.0)
            freq_results[topic] = freq

            mean_hz = freq.get("mean_hz", 0)
            std_hz = freq.get("std_hz", 0)
            self.add_evidence(f"{topic} frequency: mean={mean_hz:.1f}Hz std={std_hz:.2f}Hz")

            series = freq.get("frequency_series", [])
            for entry in series:
                if mean_hz > 0 and entry["hz"] < mean_hz * 0.3:
                    if entry["time"] < end_time - 10:
                        ts = entry["time"]
                        self.add_anomaly("CRITICAL", "FREQUENCY_DROPOUT",
                            f"{topic} dropped to {entry['hz']:.1f}Hz at {format_absolute_time(ts)} "
                            f"(expected ~{mean_hz:.1f}Hz)",
                            {"topic": topic, "time": ts, "hz": entry["hz"]},
                            timestamp=ts)

            if mean_hz > 0 and std_hz > mean_hz * 0.2:
                self.add_anomaly("WARNING", "UNSTABLE_FREQUENCY",
                    f"{topic} has unstable frequency: std={std_hz:.2f} vs mean={mean_hz:.1f}",
                    {"topic": topic, "mean_hz": mean_hz, "std_hz": std_hz})

        # ---- Phase 4: Hardware Health ----
        print(f"\n--- Phase 4: Hardware Health ---")
        self._check_cmd_vel_vs_odom(bag_path, global_stats, windowed_stats, topic_map)

        if "/unbiased_imu_PRY" in available_topics:
            self._check_imu_health(bag_path, global_stats, windowed_stats)

        if "/localization/current_pose" in available_topics:
            self._check_localization(bag_path, global_stats, windowed_stats)

        hw_available = [t for t in HARDWARE_TOPICS if t in topic_map]
        if hw_available:
            self._check_hardware_health(bag_path, topic_map)

        lidar_topics = [t for t in ["/raw_scan", "/scan_rear"] if t in topic_map]
        if lidar_topics:
            self._check_lidar_health(bag_path, lidar_topics, end_time)

        if "/device/scrubber_status" in topic_map:
            self._check_scrubber_health(bag_path, topic_map)

        safety_topics = [t for t in ["/protector", "/ir_sticker3", "/ir_sticker6",
                                      "/ir_sticker7", "/localization/status",
                                      "/navigation/status"] if t in topic_map]
        if safety_topics:
            self._check_safety_systems(bag_path, topic_map)

        # ---- Phase 5 & 6: Incident Clustering (incident mode only) ----
        if mode == "incident":
            print(f"\n--- Phase 5-6: Incident Clustering ---")
            builder = IncidentBuilder(
                log_events=self.log_events,
                state_builder=self.state_builder,
                anomalies=self.anomalies,
                bag_name=bag_name,
            )
            self.incidents = builder.build()
            active = [i for i in self.incidents if not i.suppressed]
            suppressed = [i for i in self.incidents if i.suppressed]
            print(f"  Created {len(active)} incidents "
                  f"({len(suppressed)} suppressed) from "
                  f"{len(self.anomalies)} raw anomalies")

        # ---- Generate Report ----
        report = self._generate_report(bag_name, metadata, mode)
        return report

    # ------------------------------------------------------------------
    # Phase 4 sub-methods (kept from v2.0)
    # ------------------------------------------------------------------
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
        cmd_linear = cmd_fields.get("linear_x", {})
        odom_twist = odom_fields.get("twist_linear_x", {})

        if cmd_linear.get("max", 0) > 0.01 and odom_twist.get("max", 0) < 0.001:
            self.add_anomaly("CRITICAL", "CMD_ODOM_MISMATCH",
                "Commands were sent (cmd_vel.linear_x > 0) but odom shows no motion",
                {"cmd_max": cmd_linear.get("max"), "odom_max": odom_twist.get("max")})

        if cmd_linear.get("max", 0) < 0.001 and odom_twist.get("max", 0) > 0.1:
            self.add_anomaly("WARNING", "UNEXPECTED_MOTION",
                "No commands sent but odom shows motion",
                {"cmd_max": cmd_linear.get("max"), "odom_max": odom_twist.get("max")})

        if cmd_linear.get("std", 0) < 1e-6 and cmd_linear.get("mean", 0) == 0:
            self.add_evidence("Robot appears IDLE (no velocity commands sent)")
        else:
            self.add_evidence(
                f"Robot was ACTIVE (cmd_vel linear_x: mean={cmd_linear.get('mean', 0):.4f} "
                f"max={cmd_linear.get('max', 0):.4f})")

        if "/chassis_cmd_vel" in windowed_stats and "/odom" in windowed_stats:
            cmd_windows = windowed_stats["/chassis_cmd_vel"]
            odom_windows = windowed_stats["/odom"]
            for i, (cw, ow) in enumerate(zip(cmd_windows, odom_windows)):
                cmd_lx = cw.get("fields", {}).get("linear_x", {})
                odom_tlx = ow.get("fields", {}).get("twist_linear_x", {})
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
                        f"IMU {axis}: mean={s['mean']:.4f} std={s['std']:.4f}")
        z_stats = fields.get("z", {})
        if z_stats and z_stats.get("std", 0) > 100:
            self.add_anomaly("WARNING", "IMU_HIGH_VARIANCE",
                f"IMU Z-axis has very high variance (std={z_stats['std']:.2f})",
                {"std": z_stats["std"]})

    def _check_localization(self, bag_path, global_stats, windowed_stats):
        if "/localization/current_pose" not in global_stats:
            return
        loc_stats = global_stats["/localization/current_pose"]
        if not loc_stats:
            return
        fields = loc_stats[0].get("fields", {})
        pos_x = fields.get("position_x", {})
        pos_y = fields.get("position_y", {})
        if pos_x and pos_y:
            self.add_evidence(
                f"Localization range: X=[{pos_x.get('min', 0):.2f}, {pos_x.get('max', 0):.2f}] "
                f"Y=[{pos_y.get('min', 0):.2f}, {pos_y.get('max', 0):.2f}]")
            if pos_x.get("std", 0) < 0.001 and pos_y.get("std", 0) < 0.001:
                self.add_anomaly("WARNING", "LOCALIZATION_STUCK",
                    "Localization position is not changing",
                    {"pos_x_std": pos_x.get("std"), "pos_y_std": pos_y.get("std")})

        if "/odom" in global_stats and global_stats["/odom"]:
            odom_fields = global_stats["/odom"][0].get("fields", {})
            odom_px = odom_fields.get("position_x", {})
            loc_px = fields.get("position_x", {})
            if odom_px and loc_px:
                odom_range = odom_px.get("max", 0) - odom_px.get("min", 0)
                loc_range = loc_px.get("max", 0) - loc_px.get("min", 0)
                if odom_range > 0.5 and loc_range > 0.5:
                    ratio = loc_range / odom_range if odom_range > 0 else float('inf')
                    if abs(ratio - 1.0) > 0.5:
                        self.add_anomaly("WARNING", "ODOM_LOC_DIVERGENCE",
                            f"Odometry and localization ranges differ significantly "
                            f"(odom: {odom_range:.2f}m, loc: {loc_range:.2f}m)",
                            {"odom_range": odom_range, "loc_range": loc_range})

    def _check_hardware_health(self, bag_path, topic_map):
        if "/device/health_status" in topic_map:
            stats = self.bridge.get_topic_statistics(bag_path, "/device/health_status")
            if stats and "fields" in stats[0]:
                for field_name, s in stats[0]["fields"].items():
                    if field_name in EXPECTED_ZERO_FIELDS.get("/device/health_status", set()):
                        continue
                    if field_name not in HEALTH_FLAG_NAMES:
                        continue
                    if s["mean"] < 0.5 and s["count"] > 5:
                        self.add_anomaly("CRITICAL", "HW_FAULT",
                            f"/device/health_status.{field_name} reports FAULT "
                            f"(mean={s['mean']:.2f}, {s['count']} msgs)",
                            {"topic": "/device/health_status", "field": field_name,
                             "mean": s["mean"]})
                    else:
                        self.add_evidence(f"HW health {field_name}: OK")

            wstats = self.bridge.get_topic_statistics(
                bag_path, "/device/health_status", window_size=30.0)
            if wstats:
                for field_name in HEALTH_FLAG_NAMES:
                    vals = [(w["window_start"],
                             w["fields"].get(field_name, {}).get("mean", 1.0))
                            for w in wstats if field_name in w.get("fields", {})]
                    for i in range(1, len(vals)):
                        prev_val, curr_val = vals[i-1][1], vals[i][1]
                        ts = vals[i][0]
                        if prev_val > 0.5 and curr_val < 0.5:
                            self.add_anomaly("CRITICAL", "HW_FAULT_ONSET",
                                f"{field_name} went to FAULT at {format_absolute_time(ts)}",
                                {"field": field_name, "time": ts},
                                timestamp=ts)
                        elif prev_val < 0.5 and curr_val > 0.5:
                            self.add_anomaly("INFO", "HW_FAULT_RECOVERED",
                                f"{field_name} recovered at {format_absolute_time(ts)}",
                                {"field": field_name, "time": ts},
                                timestamp=ts)

        if "/device/odom_status" in topic_map:
            stats = self.bridge.get_topic_statistics(bag_path, "/device/odom_status")
            if stats and "fields" in stats[0]:
                for field_name, s in stats[0]["fields"].items():
                    if field_name in EXPECTED_ZERO_FIELDS.get("/device/odom_status", set()):
                        continue
                    if field_name.endswith("_error") or field_name == "is_gliding":
                        if s["min"] < 0.5:
                            self.add_anomaly("WARNING", "ODOM_HW_ERROR",
                                f"/device/odom_status.{field_name} reported error "
                                f"(min={s['min']:.0f}, mean={s['mean']:.2f})",
                                {"field": field_name, "mean": s["mean"]})

        if "/device/imu_data" in topic_map:
            stats = self.bridge.get_topic_statistics(bag_path, "/device/imu_data")
            if stats and "fields" in stats[0]:
                for field_name, s in stats[0]["fields"].items():
                    if field_name in EXPECTED_ZERO_FIELDS.get("/device/imu_data", set()):
                        continue
                    if s["std"] < 1e-6 and s["count"] > 10:
                        if s["mean"] == 0.0 and field_name.startswith("magnetic"):
                            self.add_anomaly("INFO", "IMU_HW_FIELD_ZERO",
                                f"/device/imu_data.{field_name} always 0 (sensor may be disabled)",
                                {"field": field_name})
                        elif s["mean"] != 0.0:
                            self.add_anomaly("WARNING", "IMU_HW_FROZEN",
                                f"/device/imu_data.{field_name} frozen at {s['mean']:.1f}",
                                {"field": field_name, "mean": s["mean"]})
                    else:
                        self.add_evidence(
                            f"IMU HW {field_name}: mean={s['mean']:.1f} std={s['std']:.2f}")

    def _check_lidar_health(self, bag_path, lidar_topics, end_time):
        for topic in lidar_topics:
            stats = self.bridge.get_topic_statistics(bag_path, topic)
            if not stats or "fields" not in stats[0]:
                continue
            fields = stats[0]["fields"]
            valid = fields.get("num_valid_points", {})
            if valid:
                self.add_evidence(
                    f"{topic} valid points: mean={valid.get('mean', 0):.0f} "
                    f"min={valid.get('min', 0):.0f} max={valid.get('max', 0):.0f}")
                if valid.get("mean", 0) > 50 and valid.get("min", 0) < valid.get("mean", 0) * 0.3:
                    self.add_anomaly("WARNING", "LIDAR_POINT_DROP",
                        f"{topic} valid points dropped to {valid['min']:.0f} "
                        f"(mean={valid['mean']:.0f})",
                        {"topic": topic, "min_points": valid["min"],
                         "mean_points": valid["mean"]})

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

    def _check_scrubber_health(self, bag_path, topic_map):
        if "/device/scrubber_status" in topic_map:
            stats = self.bridge.get_topic_statistics(bag_path, "/device/scrubber_status")
            if stats and "fields" in stats[0]:
                fields = stats[0]["fields"]
                water = fields.get("water_level", {})
                if water and water.get("min", 99) == 0:
                    self.add_anomaly("WARNING", "SCRUBBER_NO_WATER",
                        f"Water level reached 0 during operation",
                        {"min_level": water["min"], "mean_level": water.get("mean", 0)})
                brush = fields.get("rolling_brush_motor", {})
                if brush:
                    self.add_evidence(
                        f"Brush motor: mean={brush.get('mean', 0):.2f} (1.0=on, 0.0=off)")
                for key in ["brush_spin_level", "detergent_level", "valve"]:
                    s = fields.get(key, {})
                    if s:
                        self.add_evidence(f"Scrubber {key}: mean={s.get('mean', 0):.2f}")

        if "/device/scrubber_motor_limit" in topic_map:
            stats = self.bridge.get_topic_statistics(bag_path, "/device/scrubber_motor_limit")
            if stats and "fields" in stats[0]:
                vel = stats[0]["fields"].get("velocity_speed", {})
                if vel:
                    self.add_evidence(
                        f"Scrubber motor velocity: mean={vel.get('mean', 0):.4f} "
                        f"max={vel.get('max', 0):.4f}")

    def _check_safety_systems(self, bag_path, topic_map):
        if "/protector" in topic_map:
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
                if data.get("std", 0) < 1e-6 and data.get("mean", 0) == 0 and data.get("count", 0) > 10:
                    self.add_anomaly("WARNING", "IR_SENSOR_DEAD",
                        f"{topic} always 0.0 — sensor may be disconnected",
                        {"topic": topic})

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

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def _generate_report(self, bag_name: str, metadata: dict, mode: str) -> dict:
        by_severity = defaultdict(list)
        for a in self.anomalies:
            by_severity[a["severity"]].append(a)

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

        # In incident mode, adjust health based on non-suppressed incidents
        if mode == "incident" and self.incidents:
            active_incidents = [i for i in self.incidents if not i.suppressed]
            inc_critical = sum(1 for i in active_incidents if i.severity == "CRITICAL")
            inc_warning = sum(1 for i in active_incidents if i.severity == "WARNING")
            if inc_critical > 0:
                health = "UNHEALTHY"
            elif inc_warning > 2:
                health = "DEGRADED"
            elif inc_warning > 0:
                health = "MARGINAL"
            else:
                health = "HEALTHY"

        diagnosis_lines = [
            f"Bag: {bag_name}",
            f"Overall Health: {health}",
            f"Duration: {metadata['duration']:.1f}s",
            f"Time: {format_absolute_time(metadata['start_time'])} to "
            f"{format_absolute_time(metadata['end_time'])}",
            f"Log Events: {len(self.log_events)}",
            f"Raw Anomalies: {len(self.anomalies)} "
            f"(C:{critical_count} W:{warning_count} I:{info_count})",
        ]

        if mode == "incident" and self.incidents:
            active = [i for i in self.incidents if not i.suppressed]
            suppressed = [i for i in self.incidents if i.suppressed]
            diagnosis_lines.append(
                f"Incidents: {len(active)} active, {len(suppressed)} suppressed")

        diagnosis = "\n".join(diagnosis_lines)
        print(f"\n{'='*70}")
        print(f"DIAGNOSIS: {bag_name} — {health}")
        print(f"{'='*70}")
        print(diagnosis)

        report = {
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
            "log_events": [ev.to_dict() for ev in self.log_events
                           if ev.level >= 4],  # Only WARN+ in raw output
            "evidence": self.evidence,
            "diagnosis": diagnosis,
            "metrics": {
                "critical_count": critical_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "evidence_count": len(self.evidence),
                "log_event_count": len(self.log_events),
                "log_events_by_type": LogExtractor(self.bridge).get_event_summary(
                    self.log_events) if self.log_events else {},
            },
        }

        if mode == "incident":
            active_incidents = [i for i in self.incidents if not i.suppressed]
            suppressed_incidents = [i for i in self.incidents if i.suppressed]
            report["incidents"] = [i.to_dict() for i in active_incidents]
            report["suppressed_anomalies"] = [
                {
                    "incident_id": i.incident_id,
                    "original_category": i.category,
                    "description": i.title,
                    "suppression_reason": i.suppression_reason,
                    "timestamp": i.time_start_str,
                }
                for i in suppressed_incidents
            ]
            report["state_timeline_summary"] = (
                self.state_builder.get_summary() if self.state_builder else {})

        return report
