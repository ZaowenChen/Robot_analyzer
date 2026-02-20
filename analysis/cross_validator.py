"""
CrossValidator -- Timestamp-Aligned Log-Sensor Correlation Engine.

This module bridges noisy /rosout logs and raw sensor data. It builds
a unified timeline where every log event is paired with a sensor
snapshot from the same time window, enabling:

  1. Noise filtering: Discard repetitive/uninformative log messages
  2. Cross-validation: Match log claims against sensor reality
  3. Evidence packets: Structured bundles for LLM-based root cause analysis

Architecture:
  Layer 1 -- LogDenoiser (in logs/denoiser.py)
  Layer 2 -- SensorSnapshotBuilder: Captures sensor state at any timestamp
             using windowed statistics from the bridge.
  Layer 3 -- CrossValidator: Aligns log events with sensor snapshots,
             detects agreements/disagreements, and emits EvidencePackets.
"""

import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from bridge import ROSBagBridge, open_bag, WelfordAccumulator
from core.constants import (
    SENSOR_SNAPSHOT_TOPICS,
    CORRELATION_WINDOW_SEC,
    SNAPSHOT_HALF_WINDOW_SEC,
)
from core.utils import format_absolute_time, CST
from core.models import LogEvent, SensorSnapshot, EvidencePacket, DenoisedLogEvent
from logs import (
    LogExtractor,
    StateTimelineBuilder,
    LogDenoiser,
    extract_log_timeline,
    PATTERN_REGISTRY,
)


# ---------------------------------------------------------------------------
# Layer 2: Sensor Snapshot Builder
# ---------------------------------------------------------------------------

class SensorSnapshotBuilder:
    """
    Builds sensor snapshots at specific timestamps.

    Uses a two-phase approach for performance:
    1. Pre-compute windowed stats for the entire bag at a fixed resolution
       (e.g., every 2 seconds) with a SINGLE pass per topic.
    2. For any query timestamp, look up the nearest pre-computed window.

    This turns O(N * T) topic scans into O(T) scans (one per topic),
    making cross-validation feasible on large bags.
    """

    def __init__(self, bridge: ROSBagBridge,
                 resolution: float = 2.0):
        self.bridge = bridge
        self.resolution = resolution
        # Cache: bag_path -> {topic -> [{window_start, fields}]}
        self._precomputed: Dict[str, Dict[str, List[dict]]] = {}
        self._topic_cache: Dict[str, Set[str]] = {}

    def _get_available_topics(self, bag_path: str) -> Set[str]:
        """Get the set of topics available in a bag (cached)."""
        key = os.path.realpath(bag_path)
        if key not in self._topic_cache:
            meta = self.bridge.get_bag_metadata(bag_path)
            self._topic_cache[key] = {t["name"] for t in meta["topics"]}
        return self._topic_cache[key]

    def precompute(self, bag_path: str,
                   topics: Optional[List[str]] = None):
        """
        Pre-compute windowed stats for an entire bag (single pass per topic).

        This is the expensive step -- but it's done once per bag, not once
        per log event.
        """
        key = os.path.realpath(bag_path)
        if key in self._precomputed:
            return  # Already done

        available = self._get_available_topics(bag_path)
        target_topics = topics or SENSOR_SNAPSHOT_TOPICS
        topics_to_query = [t for t in target_topics if t in available]

        bag_stats: Dict[str, List[dict]] = {}
        for topic in topics_to_query:
            try:
                stats = self.bridge.get_topic_statistics(
                    bag_path, topic,
                    window_size=self.resolution,
                )
                if stats and isinstance(stats, list):
                    bag_stats[topic] = stats
            except Exception:
                continue

        self._precomputed[key] = bag_stats

    def build_snapshot(self, bag_path: str, timestamp: float,
                       state_builder: Optional[StateTimelineBuilder] = None,
                       topics: Optional[List[str]] = None,
                       ) -> SensorSnapshot:
        """
        Build a sensor snapshot at the given timestamp using pre-computed data.

        Falls back to live query if not pre-computed.
        """
        key = os.path.realpath(bag_path)

        # Ensure pre-computed
        if key not in self._precomputed:
            self.precompute(bag_path, topics)

        bag_stats = self._precomputed.get(key, {})
        topic_stats: Dict[str, Dict[str, dict]] = {}

        for topic, windows in bag_stats.items():
            # Find nearest window to timestamp
            best = None
            best_dist = float('inf')
            for w in windows:
                w_mid = (w["window_start"] + w["window_end"]) / 2.0
                dist = abs(w_mid - timestamp)
                if dist < best_dist:
                    best_dist = dist
                    best = w
            if best and "fields" in best and best_dist < self.resolution * 2:
                topic_stats[topic] = best["fields"]

        robot_state = {}
        if state_builder:
            robot_state = state_builder.get_state_at(timestamp)

        return SensorSnapshot(
            timestamp=timestamp,
            timestamp_str=format_absolute_time(timestamp),
            topic_stats=topic_stats,
            robot_state=robot_state,
            bag_name=os.path.basename(bag_path),
        )


# ---------------------------------------------------------------------------
# Cross-validation rules
# ---------------------------------------------------------------------------

# Cross-validation rules: define how to check log claims against sensor data
VALIDATION_RULES = [
    {
        "name": "path_planning_failed_check_velocity",
        "log_pattern": re.compile(r"(make.*plan.*fail|DijkstraSearch.*fail|MakePlan.*fail)", re.I),
        "check": lambda snapshot: _check_zero_velocity(snapshot),
        "description": "Path planning failed -> expect zero velocity (robot stuck)",
    },
    {
        "name": "still_flag_check_odom",
        "log_pattern": re.compile(r"still_flag[:\s=]*1", re.I),
        "check": lambda snapshot: _check_odom_frozen(snapshot),
        "description": "still_flag=1 -> expect frozen odometry twist",
    },
    {
        "name": "still_flag_moving_check_odom",
        "log_pattern": re.compile(r"still_flag[:\s=]*0", re.I),
        "check": lambda snapshot: _check_odom_active(snapshot),
        "description": "still_flag=0 -> expect active odometry twist",
    },
    {
        "name": "localization_lost_check_pose",
        "log_pattern": re.compile(r"(is_localized:\s*0|locali[sz]ation.*lost|Tf tranform failed)", re.I),
        "check": lambda snapshot: _check_pose_frozen(snapshot),
        "description": "Localization lost -> expect frozen/stale pose",
    },
    {
        "name": "imu_calibrating_check_imu",
        "log_pattern": re.compile(r"imu.*calibrat", re.I),
        "check": lambda snapshot: _check_imu_frozen(snapshot),
        "description": "IMU calibrating -> expect frozen IMU during calibration",
    },
    {
        "name": "not_moving_check_velocity",
        "log_pattern": re.compile(r"not move time:\s*([\d.]+)", re.I),
        "check": lambda snapshot: _check_zero_velocity(snapshot),
        "description": "Not-moving timer active -> expect zero velocity",
    },
    {
        "name": "cmd_not_safe_check_cmd_vel",
        "log_pattern": re.compile(r"cmd not safe", re.I),
        "check": lambda snapshot: _check_zero_cmd_vel(snapshot),
        "description": "Command not safe -> expect suppressed cmd_vel",
    },
]


def _check_zero_velocity(snapshot: Optional[SensorSnapshot]) -> Tuple[str, str]:
    """Check if velocity is near zero."""
    if snapshot is None:
        return "NO_SENSOR_DATA", "No sensor data available"
    odom = snapshot.topic_stats.get("/odom", {})
    twist_lx = odom.get("twist_linear_x", {})
    twist_az = odom.get("twist_angular_z", {})
    if not twist_lx:
        return "NO_SENSOR_DATA", "No /odom twist data in window"

    lx_mean = abs(twist_lx.get("mean", 0))
    az_mean = abs(twist_az.get("mean", 0))
    if lx_mean < 0.01 and az_mean < 0.01:
        return "CONFIRMED", f"Velocity near zero (linear={lx_mean:.4f}, angular={az_mean:.4f})"
    else:
        return "CONTRADICTED", f"Velocity NOT zero (linear={lx_mean:.4f}, angular={az_mean:.4f})"


def _check_odom_frozen(snapshot: Optional[SensorSnapshot]) -> Tuple[str, str]:
    """Check if odometry twist is frozen (std ~0)."""
    if snapshot is None:
        return "NO_SENSOR_DATA", "No sensor data available"
    odom = snapshot.topic_stats.get("/odom", {})
    twist_lx = odom.get("twist_linear_x", {})
    if not twist_lx or twist_lx.get("count", 0) < 2:
        return "NO_SENSOR_DATA", "Insufficient /odom data in window"

    std = twist_lx.get("std", 0)
    if std < 1e-4:
        return "CONFIRMED", f"Odom twist frozen (std={std:.6f}) — consistent with still_flag=1"
    else:
        return "CONTRADICTED", f"Odom twist NOT frozen (std={std:.6f}) — robot may be moving despite still_flag=1"


def _check_odom_active(snapshot: Optional[SensorSnapshot]) -> Tuple[str, str]:
    """Check if odometry twist is active (non-zero std)."""
    if snapshot is None:
        return "NO_SENSOR_DATA", "No sensor data available"
    odom = snapshot.topic_stats.get("/odom", {})
    twist_lx = odom.get("twist_linear_x", {})
    if not twist_lx or twist_lx.get("count", 0) < 2:
        return "NO_SENSOR_DATA", "Insufficient /odom data in window"

    std = twist_lx.get("std", 0)
    mean = abs(twist_lx.get("mean", 0))
    if std > 1e-4 or mean > 0.01:
        return "CONFIRMED", f"Odom active (mean={mean:.4f}, std={std:.6f}) — consistent with still_flag=0"
    else:
        return "CONTRADICTED", f"Odom appears frozen (mean={mean:.4f}, std={std:.6f}) — robot not moving despite still_flag=0"


def _check_pose_frozen(snapshot: Optional[SensorSnapshot]) -> Tuple[str, str]:
    """Check if localization pose is frozen."""
    if snapshot is None:
        return "NO_SENSOR_DATA", "No sensor data available"
    pose = snapshot.topic_stats.get("/localization/current_pose", {})
    pos_x = pose.get("position_x", {})
    if not pos_x or pos_x.get("count", 0) < 2:
        return "NO_SENSOR_DATA", "Insufficient localization data in window"

    std_x = pos_x.get("std", 0)
    std_y = pose.get("position_y", {}).get("std", 0)
    if std_x < 1e-4 and std_y < 1e-4:
        return "CONFIRMED", f"Pose frozen (std_x={std_x:.6f}, std_y={std_y:.6f})"
    else:
        return "CONTRADICTED", f"Pose changing (std_x={std_x:.6f}, std_y={std_y:.6f})"


def _check_imu_frozen(snapshot: Optional[SensorSnapshot]) -> Tuple[str, str]:
    """Check if IMU data is frozen."""
    if snapshot is None:
        return "NO_SENSOR_DATA", "No sensor data available"
    imu = snapshot.topic_stats.get("/unbiased_imu_PRY", {})
    if not imu:
        imu = snapshot.topic_stats.get("/device/imu_data", {})
    if not imu:
        return "NO_SENSOR_DATA", "No IMU data in window"

    # Check any axis
    for axis in ["z", "x", "y", "yaw_angle", "pitch_angle", "roll_angle"]:
        stats = imu.get(axis, {})
        if stats and stats.get("count", 0) >= 2:
            if stats.get("std", 0) < 1e-4:
                return "CONFIRMED", f"IMU {axis} frozen (std={stats['std']:.6f})"
            else:
                return "CONTRADICTED", f"IMU {axis} active (std={stats['std']:.6f})"

    return "NO_SENSOR_DATA", "No usable IMU fields in window"


def _check_zero_cmd_vel(snapshot: Optional[SensorSnapshot]) -> Tuple[str, str]:
    """Check if cmd_vel is zero."""
    if snapshot is None:
        return "NO_SENSOR_DATA", "No sensor data available"
    cmd = snapshot.topic_stats.get("/chassis_cmd_vel", {})
    if not cmd:
        cmd = snapshot.topic_stats.get("/cmd_vel", {})
    lx = cmd.get("linear_x", {})
    if not lx:
        return "NO_SENSOR_DATA", "No cmd_vel data in window"

    mean = abs(lx.get("mean", 0))
    if mean < 0.01:
        return "CONFIRMED", f"cmd_vel suppressed (mean={mean:.4f})"
    else:
        return "CONTRADICTED", f"cmd_vel NOT suppressed (mean={mean:.4f})"


# ---------------------------------------------------------------------------
# Main CrossValidator
# ---------------------------------------------------------------------------

class CrossValidator:
    """
    Timestamp-aligned log-sensor cross-validation engine.

    Pipeline:
    1. Extract /rosout logs from all bags (LogExtractor)
    2. Build state timeline (StateTimelineBuilder)
    3. Denoise logs (LogDenoiser -> 80% reduction)
    4. For each significant log event, build a sensor snapshot
    5. Apply validation rules to cross-check log claims vs sensor data
    6. Emit EvidencePackets
    """

    def __init__(self, bag_paths: List[str],
                 snapshot_topics: Optional[List[str]] = None,
                 snapshot_half_window: float = SNAPSHOT_HALF_WINDOW_SEC,
                 sensor_snapshot_sample_rate: int = 1):
        """
        Args:
            bag_paths: List of bag file paths to analyze
            snapshot_topics: Override sensor topics to sample
            snapshot_half_window: Half-window size for sensor snapshots
            sensor_snapshot_sample_rate: Only snapshot every Nth event (1=all)
        """
        self.bag_paths = sorted(bag_paths)
        self.bridge = ROSBagBridge()
        self.denoiser = LogDenoiser()
        self.snapshot_builder = SensorSnapshotBuilder(
            self.bridge, resolution=snapshot_half_window * 2)
        self.snapshot_topics = snapshot_topics or SENSOR_SNAPSHOT_TOPICS
        self.sample_rate = max(1, sensor_snapshot_sample_rate)

        # Results
        self.all_log_events: List[LogEvent] = []
        self.state_builder = StateTimelineBuilder()
        self.denoised_events: List[DenoisedLogEvent] = []
        self.evidence_packets: List[EvidencePacket] = []
        self.denoise_stats: dict = {}
        self.run_stats: dict = {}

    def run(self, skip_sensor_snapshots: bool = False) -> List[EvidencePacket]:
        """
        Run the full cross-validation pipeline.

        Args:
            skip_sensor_snapshots: If True, skip the slow sensor snapshot
                step (useful for log-only analysis).

        Returns:
            List of EvidencePacket objects
        """
        t_start = time.time()

        # Step 1: Extract logs from all bags
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATOR — Timestamp-Aligned Log-Sensor Correlation")
        print(f"{'='*70}")
        print(f"Bags: {len(self.bag_paths)}")

        print(f"\n--- Step 1: Log Extraction ---")
        all_events = []
        for bag_path in self.bag_paths:
            bag_name = os.path.basename(bag_path)
            print(f"  Extracting from {bag_name}...")
            events, state = extract_log_timeline(
                bag_path, self.bridge, min_level=2, bag_name=bag_name)
            all_events.extend(events)
            # Merge state builders
            self.state_builder.process_events(events)

        all_events.sort(key=lambda e: e.timestamp)
        self.all_log_events = all_events
        print(f"  Total raw log events: {len(all_events)}")

        # Step 2: Denoise
        print(f"\n--- Step 2: Noise Filtering ---")
        self.denoised_events = self.denoiser.denoise(all_events)
        self.denoise_stats = self.denoiser.get_stats()
        print(f"  Input:  {self.denoise_stats['total_input']} messages")
        print(f"  Noise filtered: {self.denoise_stats['noise_filtered']} "
              f"({self.denoise_stats.get('noise_reduction_pct', 0):.1f}%)")
        print(f"  Deduplicated: {self.denoise_stats['deduped']} "
              f"({self.denoise_stats.get('dedup_reduction_pct', 0):.1f}%)")
        print(f"  Output: {len(self.denoised_events)} events")

        # Step 3: Pre-compute sensor stats (one pass per bag per topic)
        if not skip_sensor_snapshots:
            print(f"\n--- Step 3a: Pre-computing Sensor Stats ---")
            t_precompute = time.time()
            for bag_path in self.bag_paths:
                bag_name = os.path.basename(bag_path)
                print(f"  Pre-computing {bag_name}...")
                self.snapshot_builder.precompute(bag_path, self.snapshot_topics)
            print(f"  Pre-compute took {time.time() - t_precompute:.1f}s")

        # Step 4: Build evidence packets
        print(f"\n--- Step {'3b' if not skip_sensor_snapshots else '3'}: "
              f"Cross-Validation ---")
        self._build_evidence_packets(skip_sensor_snapshots)

        # Stats
        elapsed = time.time() - t_start
        verdicts = Counter(p.verdict for p in self.evidence_packets)
        self.run_stats = {
            "total_bags": len(self.bag_paths),
            "raw_log_events": len(all_events),
            "denoised_events": len(self.denoised_events),
            "evidence_packets": len(self.evidence_packets),
            "verdicts": dict(verdicts),
            "elapsed_sec": round(elapsed, 1),
        }

        print(f"\n--- Results ---")
        print(f"  Evidence packets: {len(self.evidence_packets)}")
        for verdict, count in sorted(verdicts.items()):
            print(f"    {verdict}: {count}")
        print(f"  Total time: {elapsed:.1f}s")

        return self.evidence_packets

    def _build_evidence_packets(self, skip_sensors: bool):
        """Build evidence packets from denoised events."""
        self.evidence_packets = []
        packet_counter = 0
        snapshot_counter = 0

        # Determine which events are "significant" enough for sensor snapshots
        # ERROR/WARN always get snapshots; INFO only if they match a validation rule
        for i, denoised in enumerate(self.denoised_events):
            ev = denoised.event
            packet_counter += 1

            # Classify the event
            category, severity = self._classify_event(ev, denoised)

            # Build log event dict
            log_dict = {
                "timestamp_str": ev.timestamp_str,
                "node": ev.node,
                "level": ev.level_str,
                "message": ev.raw_message[:500],
                "event_type": ev.event_type,
                "parsed_data": ev.parsed_data,
            }

            packet = EvidencePacket(
                packet_id=f"EP-{packet_counter:04d}",
                timestamp=ev.timestamp,
                timestamp_str=ev.timestamp_str,
                bag_name=ev.bag_name,
                log_event=log_dict,
                log_dedup_count=denoised.occurrence_count,
                robot_state=self.state_builder.get_state_at(ev.timestamp),
                category=category,
                severity=severity,
            )

            # Decide if we need a sensor snapshot -- only for events that
            # actually trigger a validation rule or are ERROR/FATAL level.
            # This keeps the expensive bridge queries to O(100) not O(10K).
            has_rule_match = any(
                r["log_pattern"].search(ev.raw_message) for r in VALIDATION_RULES)
            needs_snapshot = (
                not skip_sensors and
                (ev.level >= 8 or has_rule_match) and
                (snapshot_counter % self.sample_rate == 0)
            )

            if needs_snapshot:
                snapshot_counter += 1
                snapshot = self.snapshot_builder.build_snapshot(
                    self._find_bag_for_timestamp(ev.timestamp, ev.bag_name),
                    ev.timestamp,
                    self.state_builder,
                    self.snapshot_topics,
                )
                packet.sensor_snapshot = snapshot.to_dict()

                # Apply validation rules
                self._apply_validation_rules(packet, ev, snapshot)
            else:
                packet.verdict = "UNCHECKED"

            # Generate summary
            packet.summary = self._generate_summary(packet, ev, denoised)

            self.evidence_packets.append(packet)

    def _classify_event(self, ev: LogEvent, denoised: DenoisedLogEvent
                        ) -> Tuple[str, str]:
        """Classify a log event into category and severity."""
        # Severity from log level
        if ev.level >= 8:
            severity = "WARNING"  # ERROR/FATAL in log -> WARNING in our system
            if any(kw in ev.raw_message.lower() for kw in
                   ["fault", "disconnect", "emergency", "critical"]):
                severity = "CRITICAL"
        elif ev.level >= 4:
            severity = "WARNING"
        else:
            severity = "INFO"

        # Category from event type
        if ev.event_type != "UNMATCHED":
            category = ev.event_type.upper()
        elif ev.level >= 8:
            category = "ERROR"
        elif "fail" in ev.raw_message.lower():
            category = "FAILURE"
        elif "stuck" in ev.raw_message.lower() or "deadlock" in ev.raw_message.lower():
            category = "NAVIGATION_STUCK"
        else:
            category = "INFO"

        # Escalate if this is a high-frequency repeater
        if denoised.occurrence_count > 50 and severity == "INFO":
            severity = "WARNING"
            category = "HIGH_FREQUENCY_EVENT"

        return category, severity

    def _apply_validation_rules(self, packet: EvidencePacket,
                                 ev: LogEvent,
                                 snapshot: SensorSnapshot):
        """Apply cross-validation rules to check log claims against sensor data."""
        msg = ev.raw_message

        for rule in VALIDATION_RULES:
            if rule["log_pattern"].search(msg):
                verdict, detail = rule["check"](snapshot)
                packet.cross_validation_details.append({
                    "rule": rule["name"],
                    "description": rule["description"],
                    "verdict": verdict,
                    "detail": detail,
                })

        # Determine overall verdict from rule results
        if not packet.cross_validation_details:
            packet.verdict = "UNCHECKED"
            packet.confidence = 0.0
        else:
            verdicts = [d["verdict"] for d in packet.cross_validation_details]
            if "CONTRADICTED" in verdicts:
                packet.verdict = "CONTRADICTED"
                packet.confidence = 0.9
            elif all(v == "CONFIRMED" for v in verdicts):
                packet.verdict = "CONFIRMED"
                packet.confidence = 0.8
            elif "CONFIRMED" in verdicts:
                packet.verdict = "CONFIRMED"
                packet.confidence = 0.6
            else:
                packet.verdict = "NO_SENSOR_DATA"
                packet.confidence = 0.0

    def _generate_summary(self, packet: EvidencePacket, ev: LogEvent,
                          denoised: DenoisedLogEvent) -> str:
        """Generate a human-readable summary for the evidence packet."""
        parts = []

        # Log part
        msg_short = ev.raw_message[:120]
        if denoised.occurrence_count > 1:
            parts.append(f"[{ev.level_str}] [{ev.node}] {msg_short} "
                        f"(repeated {denoised.occurrence_count}x over "
                        f"{denoised.last_timestamp - denoised.first_timestamp:.1f}s)")
        else:
            parts.append(f"[{ev.level_str}] [{ev.node}] {msg_short}")

        # Verdict part
        if packet.verdict == "CONFIRMED":
            details = [d["detail"] for d in packet.cross_validation_details
                       if d["verdict"] == "CONFIRMED"]
            if details:
                parts.append(f"  -> CONFIRMED by sensor: {details[0]}")
        elif packet.verdict == "CONTRADICTED":
            details = [d["detail"] for d in packet.cross_validation_details
                       if d["verdict"] == "CONTRADICTED"]
            if details:
                parts.append(f"  -> CONTRADICTED by sensor: {details[0]}")

        # State context
        if packet.robot_state:
            state_str = ", ".join(f"{k}={v}" for k, v in packet.robot_state.items())
            parts.append(f"  -> State: {state_str}")

        return "\n".join(parts)

    def _find_bag_for_timestamp(self, timestamp: float, bag_name: str) -> str:
        """Find the bag file path that contains a given timestamp."""
        # Fast path: use the bag_name hint
        for path in self.bag_paths:
            if os.path.basename(path) == bag_name:
                return path
        # Fallback: return first bag
        return self.bag_paths[0]

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def export_json(self, output_path: str):
        """Export full cross-validation report as JSON."""
        report = {
            "cross_validator_version": "1.0",
            "generated_at": format_absolute_time(time.time()),
            "stats": self.run_stats,
            "denoise_stats": self.denoise_stats,
            "state_timeline_summary": self.state_builder.get_summary(),
            "evidence_packets": [p.to_dict() for p in self.evidence_packets],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {output_path}")

    def get_contradictions(self) -> List[EvidencePacket]:
        """Get all evidence packets where sensor data contradicts log claims."""
        return [p for p in self.evidence_packets if p.verdict == "CONTRADICTED"]

    def get_confirmed(self) -> List[EvidencePacket]:
        """Get all evidence packets where sensor data confirms log claims."""
        return [p for p in self.evidence_packets if p.verdict == "CONFIRMED"]

    def get_by_severity(self, severity: str) -> List[EvidencePacket]:
        """Get evidence packets filtered by severity."""
        return [p for p in self.evidence_packets if p.severity == severity]

    def get_summary_for_llm(self, max_packets: int = 50) -> str:
        """
        Generate a concise text summary suitable for LLM consumption.

        Prioritizes:
        1. CONTRADICTED findings (log-sensor disagreements)
        2. CRITICAL severity events
        3. CONFIRMED findings with high dedup counts
        """
        lines = [
            f"Cross-Validation Report: {len(self.bag_paths)} bags, "
            f"{self.run_stats.get('raw_log_events', 0)} raw logs -> "
            f"{self.run_stats.get('denoised_events', 0)} denoised -> "
            f"{self.run_stats.get('evidence_packets', 0)} evidence packets",
            "",
        ]

        # Verdicts summary
        verdicts = self.run_stats.get("verdicts", {})
        lines.append("Verdicts: " + ", ".join(
            f"{k}={v}" for k, v in sorted(verdicts.items())))
        lines.append("")

        # State timeline
        state_summary = self.state_builder.get_summary()
        if state_summary:
            lines.append("State Timeline:")
            for key, info in state_summary.items():
                lines.append(f"  {key}: {info['total_transitions']} transitions, "
                            f"current={info['current_value']}")
            lines.append("")

        # Prioritized packets
        contradictions = self.get_contradictions()
        criticals = [p for p in self.evidence_packets if p.severity == "CRITICAL"]
        high_dedup = [p for p in self.evidence_packets
                      if p.log_dedup_count > 10 and p.verdict == "CONFIRMED"]

        selected = []
        for p in contradictions:
            if p not in selected:
                selected.append(p)
        for p in criticals:
            if p not in selected:
                selected.append(p)
        for p in high_dedup:
            if p not in selected:
                selected.append(p)

        # Fill remaining slots with WARNING events
        if len(selected) < max_packets:
            warnings = [p for p in self.evidence_packets
                        if p.severity == "WARNING" and p not in selected]
            selected.extend(warnings[:max_packets - len(selected)])

        # Truncate
        selected = selected[:max_packets]

        if contradictions:
            lines.append(f"=== CONTRADICTIONS ({len(contradictions)}) ===")
            for p in contradictions[:10]:
                lines.append(f"  [{p.packet_id}] {p.summary}")
                lines.append("")

        if criticals:
            lines.append(f"=== CRITICAL EVENTS ({len(criticals)}) ===")
            for p in criticals[:15]:
                if p not in contradictions:
                    lines.append(f"  [{p.packet_id}] {p.summary}")
                    lines.append("")

        if high_dedup:
            lines.append(f"=== HIGH-FREQUENCY CONFIRMED ({len(high_dedup)}) ===")
            for p in high_dedup[:10]:
                lines.append(f"  [{p.packet_id}] {p.summary}")
                lines.append("")

        return "\n".join(lines)
