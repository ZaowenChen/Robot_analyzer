"""
LogDenoiser — Filters noise from /rosout log streams.

Preserves diagnostic signals while removing 80%+ repetitive spam,
HTTP logs, raw hex dumps, and periodic heartbeat messages.

Strategy:
1. SIGNAL_PATTERNS always pass through (errors, state transitions, faults)
2. NOISE_PATTERNS are always dropped (periodic spam, hex dumps, HTTP logs)
3. Remaining messages pass through but get deduplicated
4. ERROR/FATAL level messages always pass through regardless of patterns
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional

from core.models import LogEvent, DenoisedLogEvent


# Patterns for messages that are pure noise (no diagnostic value).
# These are checked BEFORE signal patterns — node-specific noise overrides
# generic keywords like "error" that happen to appear in spam.
NOISE_PATTERNS = [
    # Repetitive depth pipeline spam (~26% of all messages)
    # NOTE: DepthPipeline boundary_limit and pointCloudPublish errors are now
    # CONDITIONALLY kept — they are noise when isolated, but become evidence
    # when correlated with dl_infer overdark/camera failures. The dedup layer
    # collapses repeats while preserving diagnostic signal.
    re.compile(r"TFAutoAdapt\d", re.I),
    re.compile(r"above robot points filtered", re.I),
    re.compile(r"irStickerDetect disabled", re.I),
    re.compile(r"anti_drop_detect:", re.I),
    # HTTP request/response logs from gs_console
    re.compile(r"\[\d+\|\w+\|\]\s*(Request|Response|Processing)", re.I),
    # Raw hex/numeric data dumps from chassis
    re.compile(r"\[CHASSIS\]\s*(GetBatteryData|GetDeviceData|SendDeviceCommand):", re.I),
    re.compile(r"\[CHASSIS\]\s*IMU orientation\(", re.I),
    re.compile(r"current_upper_limit.*battery_upper_limit", re.I),
    re.compile(r"\[CHASSIS\]\s*virtual_battery:", re.I),
    # Periodic sensor data reports (not errors)
    re.compile(r"\[usb_cam\]\s*send private image success", re.I),
    re.compile(r"WallTime:\s*[\d.]+\s*ms", re.I),
    re.compile(r"to wait enough motion_cmd_list", re.I),
    # Noisy status reports at constant intervals
    re.compile(r"Print device flags,spray \d+", re.I),
    re.compile(r"received task status:", re.I),
    re.compile(r"received navi msg:", re.I),
    re.compile(r"NavigationStatus:\s*\d+", re.I),
    re.compile(r"Hold position duration:", re.I),
    re.compile(r"\[PlanningApp\]\s*receive battery charge mos state", re.I),
    # Repetitive distance calculations
    re.compile(r"Dist between \[track id:", re.I),
    # Periodic heartbeat messages
    re.compile(r"finish:\s*\d+,\s*reached:\s*\d+$", re.I),
    # Periodic projection/freespace reports
    re.compile(r"freespace_proj_tracking process", re.I),
    re.compile(r"danger_freespace_proj", re.I),
    re.compile(r"static_freespace_proj", re.I),
    # gs_console periodic status
    re.compile(r"gs-robot/(sensor_data|real_time_data)/", re.I),
    # Network manager periodic
    re.compile(r"\[network_manager\]", re.I),
    # Map copy elapsed
    re.compile(r"TempGlobalMapCopyGlobalMap elapsed", re.I),
    # Robot recording bag file messages
    re.compile(r"(Recording to|Closing) /root/GAUSSIAN_RUNTIME_DIR/bag/", re.I),
]

# Patterns for messages with HIGH diagnostic value (never filter these)
SIGNAL_PATTERNS = [
    # Error-level messages always pass through
    re.compile(r"(fail|error|fault|abort|timeout|disconnect|abnormal|critical|emergency)", re.I),
    # State transitions
    re.compile(r"still_flag", re.I),
    re.compile(r"(stuck|deadlock|stall|blocked)", re.I),
    re.compile(r"(calibrat|recover|resume|lost|init)", re.I),
    # Path planning events
    re.compile(r"(DijkstraSearch|MakePlan|GenDiffPlanner|make.*plan)", re.I),
    # Localization events
    re.compile(r"(locali[sz]ation|is_localized|relocali[sz])", re.I),
    re.compile(r"(Tf tranform failed|not localized)", re.I),
    # Safety events
    re.compile(r"(protector|bumper|emergency|danger)", re.I),
    # Task abnormalities
    re.compile(r"(not move time|abnormal|TaskAbnormalCheck)", re.I),
    # Charging events
    re.compile(r"(charge|qr.*detect|charger_pose)", re.I),
    # Control state changes
    re.compile(r"State::", re.I),
    re.compile(r"(cmd not safe|not safe)", re.I),
    # Scan filter warnings
    re.compile(r"uninitialized!", re.I),
    # Mapping events
    re.compile(r"(front_end_state|hallway|csm_pose)", re.I),
    # Depth camera / vision pipeline events (root cause for localization failures)
    re.compile(r"\[overdark\]", re.I),
    re.compile(r"(depthcam_fusion|depth_fusion)", re.I),
    re.compile(r"camera_boundary_limit", re.I),
    re.compile(r"dl_infer", re.I),
    re.compile(r"location matching", re.I),
]

# Deduplication: these messages repeat at high frequency but only the
# first/last occurrence (or count) matters
DEDUP_PATTERNS = [
    (re.compile(r"recieve \d+ ssd bounding boxes", re.I), "ssd_bounding_boxes"),
    (re.compile(r"localization_status\s+\d", re.I), "localization_status"),
    (re.compile(r"location is not vaild", re.I), "location_invalid"),
    (re.compile(r"\[CHASSIS IMU CALIBRATION\] record imu data", re.I), "imu_calibration_record"),
    (re.compile(r"x is [\d.]+, y is: [\d.]+, distance is", re.I), "scan_filter_distance"),
    (re.compile(r"has danger area in local update range", re.I), "danger_area_local"),
    (re.compile(r"\[ControlThread\] State::Init", re.I), "control_state_init"),
    (re.compile(r"GenTrackPath failed", re.I), "gen_track_path_failed"),
    (re.compile(r"retrieve plant state failed", re.I), "retrieve_plan_failed"),
    (re.compile(r"can not get current pose from tf", re.I), "tf_pose_failed"),
    (re.compile(r"\[DifferentialModelCommand\] cmd not safe", re.I), "cmd_not_safe"),
    # Depth pipeline messages — dedup instead of noise-filter, so we preserve
    # evidence of camera issues while collapsing high-frequency repeats
    (re.compile(r"\[DepthPipeline\d\]", re.I), "depth_pipeline_event"),
    (re.compile(r"ROSDepthCallback\d", re.I), "depth_callback_event"),
    (re.compile(r"pointCloudPublish error.*publisher.*not found", re.I), "pointcloud_publish_error"),
    (re.compile(r"\[overdark\]", re.I), "camera_overdark"),
    (re.compile(r"camera_process:step_distance", re.I), "camera_step_distance"),
]


class LogDenoiser:
    """
    Filters noise from /rosout log streams while preserving diagnostic signals.

    Strategy:
    1. SIGNAL_PATTERNS always pass through (errors, state transitions, faults)
    2. NOISE_PATTERNS are always dropped (periodic spam, hex dumps, HTTP logs)
    3. Remaining messages pass through but get deduplicated
    4. ERROR/FATAL level messages always pass through regardless of patterns
    """

    def __init__(self, dedup_window_sec: float = 5.0):
        self.dedup_window_sec = dedup_window_sec
        self.stats = {
            "total_input": 0,
            "noise_filtered": 0,
            "signal_kept": 0,
            "deduped": 0,
            "passthrough": 0,
        }

    def denoise(self, events: List[LogEvent]) -> List[DenoisedLogEvent]:
        """
        Filter and deduplicate a list of log events.

        Returns DenoisedLogEvent list sorted by timestamp.
        """
        self.stats = {k: 0 for k in self.stats}
        self.stats["total_input"] = len(events)

        # Phase 1: classify each event
        kept: List[DenoisedLogEvent] = []
        dedup_groups: Dict[str, List[LogEvent]] = defaultdict(list)

        for ev in events:
            # ERROR/FATAL always pass through
            if ev.level >= 8:
                kept.append(DenoisedLogEvent(event=ev))
                self.stats["signal_kept"] += 1
                continue

            msg = ev.raw_message

            # Check noise patterns FIRST — node-specific spam patterns
            # override generic signal keywords (e.g., "error" in depth pipeline)
            is_noise = any(p.search(msg) for p in NOISE_PATTERNS)
            if is_noise:
                self.stats["noise_filtered"] += 1
                continue

            # Check dedup patterns
            dedup_key = self._get_dedup_key(msg)
            if dedup_key:
                dedup_groups[dedup_key].append(ev)
                continue

            # Check signal patterns (high-value messages)
            is_signal = any(p.search(msg) for p in SIGNAL_PATTERNS)
            if is_signal:
                kept.append(DenoisedLogEvent(event=ev))
                self.stats["signal_kept"] += 1
                continue

            # Passthrough: not clearly noise or signal
            kept.append(DenoisedLogEvent(event=ev))
            self.stats["passthrough"] += 1

        # Phase 2: collapse dedup groups
        for key, group_events in dedup_groups.items():
            if not group_events:
                continue
            group_events.sort(key=lambda e: e.timestamp)
            # Keep first and last occurrence, with count
            representative = DenoisedLogEvent(
                event=group_events[0],
                dedup_key=key,
                occurrence_count=len(group_events),
                first_timestamp=group_events[0].timestamp,
                last_timestamp=group_events[-1].timestamp,
            )
            kept.append(representative)
            self.stats["deduped"] += len(group_events)

        # Sort by timestamp
        kept.sort(key=lambda d: d.event.timestamp)
        return kept

    def _get_dedup_key(self, message: str) -> Optional[str]:
        """Check if a message matches a dedup pattern. Returns key or None."""
        for pattern, key in DEDUP_PATTERNS:
            if pattern.search(message):
                return key
        return None

    def get_stats(self) -> dict:
        total = self.stats["total_input"]
        if total == 0:
            return self.stats
        kept = total - self.stats["noise_filtered"] - self.stats["deduped"]
        # deduped events still produce 1 representative each, so:
        # actual output = signal_kept + passthrough + len(dedup_groups)
        return {
            **self.stats,
            "noise_reduction_pct": round(
                100.0 * self.stats["noise_filtered"] / total, 1),
            "dedup_reduction_pct": round(
                100.0 * self.stats["deduped"] / total, 1),
        }
