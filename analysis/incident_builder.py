"""
IncidentBuilder -- Clusters raw anomalies + log events into meaningful Incidents.

Rules:
1. Temporal binning (events within INCIDENT_TEMPORAL_BIN_SEC)
2. Causal chains (log event type -> sensor anomaly category)
3. State-aware suppression (still_flag=1 -> suppress motion freezes)
4. Deduplication (multiple FREEZE_ONSET on same topic -> single incident)
"""

from collections import defaultdict
from typing import List, Tuple

from core.constants import (
    MOTION_RELATED_FIELDS,
    CAUSAL_CHAIN_WINDOW_SEC,
    INCIDENT_TEMPORAL_BIN_SEC,
)
from core.utils import format_absolute_time
from core.models import LogEvent, Incident
from logs import StateTimelineBuilder


class IncidentBuilder:
    """
    Clusters raw anomalies + log events into meaningful Incidents.

    Rules:
    1. Temporal binning (events within INCIDENT_TEMPORAL_BIN_SEC)
    2. Causal chains (log event type -> sensor anomaly category)
    3. State-aware suppression (still_flag=1 -> suppress motion freezes)
    4. Deduplication (multiple FREEZE_ONSET on same topic -> single incident)
    """

    # Causal chain mappings: log event_type -> expected sensor anomaly categories
    CAUSAL_CHAINS = {
        "imu_calibration": {
            "sensor_categories": {"IMU_FROZEN", "IMU_HW_FROZEN"},
            "title_template": "IMU Calibration Event",
            "category": "IMU",
            "root_cause_template": "IMU entered calibration mode, causing temporary sensor freeze",
            "recommended_actions": [
                "Check IMU calibration trigger conditions",
                "Verify IMU thermal stability",
            ],
        },
        "motion_state_change": {
            "sensor_categories": {"FROZEN_SENSOR", "ZERO_FIELD", "LOCALIZATION_STUCK"},
            "title_template": "Robot State Transition",
            "category": "MOTION",
            "root_cause_template": "Robot motion state changed (still_flag transition)",
            "recommended_actions": [],
        },
        "nav_stuck": {
            "sensor_categories": {"STALL_DETECTED", "CMD_ODOM_MISMATCH"},
            "title_template": "Navigation Deadlock",
            "category": "NAVIGATION",
            "root_cause_template": "Navigation module reported stuck/deadlock condition",
            "recommended_actions": [
                "Check navigation parameters and recovery behavior",
                "Inspect environment for obstacles at reported location",
            ],
        },
        "pointcloud_error": {
            "sensor_categories": {"LIDAR_POINT_DROP", "LIDAR_FREQ_DROPOUT", "LIDAR_DEGRADATION"},
            "title_template": "LiDAR Communication Error",
            "category": "LIDAR",
            "root_cause_template": "LiDAR point cloud processing error detected in logs",
            "recommended_actions": [
                "Check LiDAR USB/ethernet connection",
                "Inspect LiDAR lens for obstruction",
            ],
        },
        "location_state_change": {
            "sensor_categories": {"LOCALIZATION_STUCK", "ODOM_LOC_DIVERGENCE"},
            "title_template": "Localization State Change",
            "category": "LOCALIZATION",
            "root_cause_template": "Localization system state changed",
            "recommended_actions": [
                "Verify map quality in the affected area",
                "Check for environmental changes since mapping",
            ],
        },
        "safety_event": {
            "sensor_categories": {"PROTECTOR_TRIGGERED"},
            "title_template": "Safety System Activation",
            "category": "SAFETY",
            "root_cause_template": "Safety system triggered (protector/bumper/emergency)",
            "recommended_actions": [
                "Inspect physical environment for obstacles",
                "Check protector sensor calibration",
            ],
        },
        "imu_error": {
            "sensor_categories": {"IMU_FROZEN", "IMU_HW_FROZEN", "IMU_HIGH_VARIANCE"},
            "title_template": "IMU Hardware Error",
            "category": "IMU",
            "root_cause_template": "IMU hardware error reported in logs",
            "recommended_actions": [
                "Check IMU board connection",
                "Verify IMU firmware version",
            ],
        },
        "depth_camera_error": {
            "sensor_categories": {"LOCALIZATION_STUCK", "FROZEN_SENSOR"},
            "title_template": "Depth Camera Failure (overdark/offline)",
            "category": "DEPTH_CAMERA",
            "root_cause_template": "Depth camera reported overdark/offline state — "
                "likely lens obstruction, hardware failure, or USB disconnect. "
                "This cascades to depthcam_fusion TF failure -> localization loss -> navigation stuck.",
            "recommended_actions": [
                "Inspect depth camera lens for dirt/obstruction",
                "Check depth camera USB/cable connection",
                "Verify camera power supply and LED status",
                "Check dl_infer node for overdark duration (Time field)",
            ],
        },
        "depthcam_fusion_failure": {
            "sensor_categories": {"LOCALIZATION_STUCK", "FROZEN_SENSOR"},
            "title_template": "Depth Camera Fusion Failure",
            "category": "DEPTH_CAMERA",
            "root_cause_template": "Depth camera fusion module cannot compute TF transforms — "
                "likely caused by upstream depth camera failure (check dl_infer overdark). "
                "Downstream effect: localization degrades or fails.",
            "recommended_actions": [
                "Check depth camera status (see DEPTH_CAMERA_ERROR incidents)",
                "Verify TF tree completeness (missing camera frames?)",
                "Check if front_end_state shows hallway (narrow space amplifies impact)",
            ],
        },
        "dl_infer_event": {
            "sensor_categories": {"LOCALIZATION_STUCK", "FROZEN_SENSOR"},
            "title_template": "Vision/DL Inference Pipeline Issue",
            "category": "DEPTH_CAMERA",
            "root_cause_template": "DL inference pipeline reported camera issue (overdark/timeout). "
                "Vision-based perception degraded, affecting obstacle detection and localization.",
            "recommended_actions": [
                "Check camera hardware and connection",
                "Inspect lighting conditions at robot location",
                "Review dl_infer overdark Time value for duration of failure",
            ],
        },
        "localization_status_change": {
            "sensor_categories": {"LOCALIZATION_STUCK", "ODOM_LOC_DIVERGENCE"},
            "title_template": "Localization Status Changed",
            "category": "LOCALIZATION",
            "root_cause_template": "Localization status code changed (0=not localized, 1=OK, 2=error). "
                "Check upstream causes: depth camera, LiDAR, mapping state.",
            "recommended_actions": [
                "Check for concurrent depth camera or LiDAR errors",
                "Verify map quality at robot's location",
                "Check front_end_state (hallway = challenging environment)",
            ],
        },
        "mapping_state_change": {
            "sensor_categories": {"LOCALIZATION_STUCK"},
            "title_template": "Mapping State Transition",
            "category": "LOCALIZATION",
            "root_cause_template": "Front-end mapping state changed (e.g., hallway mode). "
                "Hallway = narrow space with limited LiDAR features, increasing localization difficulty.",
            "recommended_actions": [
                "Normal in narrow corridors — but monitor localization quality",
                "If combined with depth camera failure, expect localization loss",
            ],
        },
    }

    # Fields where freeze is expected when robot is stationary
    STATIONARY_SUPPRESS_FIELDS = MOTION_RELATED_FIELDS

    def __init__(self, log_events: List[LogEvent],
                 state_builder: StateTimelineBuilder,
                 anomalies: List[dict],
                 bag_name: str):
        self.log_events = log_events
        self.state_builder = state_builder
        self.anomalies = anomalies
        self.bag_name = bag_name
        self._incident_counter = 0

    def _next_id(self) -> str:
        self._incident_counter += 1
        return f"INC-{self._incident_counter:03d}"

    def build(self) -> List[Incident]:
        """Build incidents from anomalies and log events."""
        incidents = []

        # Step 1: State-aware suppression
        suppressed, remaining = self._apply_state_suppression(self.anomalies)
        incidents.extend(suppressed)

        # Step 2: Causal chain matching (log events -> sensor anomalies)
        chained, unchained = self._apply_causal_chains(remaining)
        incidents.extend(chained)

        # Step 3: Temporal clustering of remaining anomalies
        clustered = self._cluster_remaining(unchained)
        incidents.extend(clustered)

        # Sort by time
        incidents.sort(key=lambda i: i.time_start)

        # Assign IDs
        for inc in incidents:
            inc.incident_id = self._next_id()

        return incidents

    def _apply_state_suppression(self, anomalies: List[dict]
                                  ) -> Tuple[List[Incident], List[dict]]:
        """
        Suppress anomalies that are explained by robot state.

        E.g., FROZEN_SENSOR on /odom.twist_linear_x while still_flag=1
        is normal (robot is stationary).
        """
        suppressed_incidents = []
        remaining = []

        for anomaly in anomalies:
            topic = anomaly.get("details", {}).get("topic", "")
            field = anomaly.get("details", {}).get("field", "")
            ts = anomaly.get("timestamp")
            category = anomaly.get("category", "")

            should_suppress = False
            reason = ""

            # Check if this is a motion-related freeze while stationary
            if category in ("FROZEN_SENSOR", "FREEZE_ONSET", "ZERO_FIELD",
                            "LOCALIZATION_STUCK"):
                if (topic, field) in self.STATIONARY_SUPPRESS_FIELDS:
                    if ts and self.state_builder.is_state_active(
                            ts, "motion_state", "still_flag", "1"):
                        should_suppress = True
                        reason = "Robot stationary per still_flag=1"
                    elif ts is None:
                        # Global anomaly — check if robot was mostly stationary
                        still_periods = self.state_builder.get_active_periods(
                            "motion_state", "still_flag", "1")
                        if still_periods:
                            total_still = sum(
                                min(p[1], float('inf')) - p[0]
                                for p in still_periods
                                if p[1] != float('inf')
                            )
                            if total_still > 0:
                                should_suppress = True
                                reason = (f"Robot was stationary for "
                                         f"{total_still:.0f}s (still_flag=1)")

            if should_suppress:
                inc = Incident(
                    title=f"{category}: {topic}.{field}",
                    severity=anomaly.get("severity", "INFO"),
                    category=category,
                    time_start=ts or 0,
                    time_end=ts or 0,
                    time_start_str=anomaly.get("timestamp_str", ""),
                    time_end_str=anomaly.get("timestamp_str", ""),
                    root_cause=reason,
                    sensor_evidence=[{
                        "category": category,
                        "description": anomaly.get("description", ""),
                        "details": anomaly.get("details", {}),
                    }],
                    state_context=self.state_builder.get_state_at(ts) if ts else {},
                    bags=[self.bag_name],
                    raw_anomaly_count=1,
                    suppressed=True,
                    suppression_reason=reason,
                )
                suppressed_incidents.append(inc)
            else:
                remaining.append(anomaly)

        return suppressed_incidents, remaining

    def _apply_causal_chains(self, anomalies: List[dict]
                              ) -> Tuple[List[Incident], List[dict]]:
        """
        Match log events to sensor anomalies via causal chain rules.

        For each causal chain pattern:
        1. Find log events of matching type
        2. Find sensor anomalies of matching category within time window
        3. Create incident linking them
        """
        used_anomaly_indices = set()
        chained_incidents = []

        # Get matched log events (not UNMATCHED)
        matched_logs = [ev for ev in self.log_events
                        if ev.event_type != "UNMATCHED"]

        for log_event in matched_logs:
            chain = self.CAUSAL_CHAINS.get(log_event.event_type)
            if chain is None:
                continue

            # Find anomalies that match this causal chain within time window
            matching_anomalies = []
            for idx, anomaly in enumerate(anomalies):
                if idx in used_anomaly_indices:
                    continue
                if anomaly["category"] not in chain["sensor_categories"]:
                    continue
                a_ts = anomaly.get("timestamp")
                if a_ts is None:
                    continue
                if abs(a_ts - log_event.timestamp) <= CAUSAL_CHAIN_WINDOW_SEC:
                    matching_anomalies.append((idx, anomaly))

            if not matching_anomalies:
                continue

            # Mark anomalies as used
            for idx, _ in matching_anomalies:
                used_anomaly_indices.add(idx)

            all_timestamps = [log_event.timestamp] + [
                a.get("timestamp", 0) for _, a in matching_anomalies
            ]
            t_start = min(all_timestamps)
            t_end = max(all_timestamps)

            # Determine severity from the worst anomaly
            severities = [a.get("severity", "INFO") for _, a in matching_anomalies]
            severity_order = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
            worst = max(severities, key=lambda s: severity_order.get(s, 0))

            # Build log evidence
            log_evidence = [{
                "timestamp_str": log_event.timestamp_str,
                "node": log_event.node,
                "level": log_event.level_str,
                "message": log_event.raw_message[:200],
                "event_type": log_event.event_type,
            }]

            # Also find other log events near this time
            for other_log in matched_logs:
                if other_log is log_event:
                    continue
                if abs(other_log.timestamp - log_event.timestamp) <= CAUSAL_CHAIN_WINDOW_SEC:
                    log_evidence.append({
                        "timestamp_str": other_log.timestamp_str,
                        "node": other_log.node,
                        "level": other_log.level_str,
                        "message": other_log.raw_message[:200],
                        "event_type": other_log.event_type,
                    })

            # Build sensor evidence
            sensor_evidence = [{
                "category": a["category"],
                "description": a.get("description", ""),
                "details": a.get("details", {}),
            } for _, a in matching_anomalies]

            state_context = self.state_builder.get_state_at(log_event.timestamp)

            inc = Incident(
                title=chain["title_template"],
                severity=worst,
                category=chain["category"],
                time_start=t_start,
                time_end=t_end,
                time_start_str=format_absolute_time(t_start),
                time_end_str=format_absolute_time(t_end),
                root_cause=chain["root_cause_template"],
                log_evidence=log_evidence,
                sensor_evidence=sensor_evidence,
                state_context=state_context,
                recommended_actions=list(chain["recommended_actions"]),
                bags=[self.bag_name],
                raw_anomaly_count=len(matching_anomalies),
            )
            chained_incidents.append(inc)

        # Return unchained anomalies
        unchained = [a for idx, a in enumerate(anomalies)
                     if idx not in used_anomaly_indices]

        return chained_incidents, unchained

    def _cluster_remaining(self, anomalies: List[dict]) -> List[Incident]:
        """
        Cluster remaining (unchained) anomalies by temporal proximity
        and category.
        """
        if not anomalies:
            return []

        incidents = []

        # Group by category first, then temporal bin
        by_category = defaultdict(list)
        for anomaly in anomalies:
            by_category[anomaly["category"]].append(anomaly)

        for category, cat_anomalies in by_category.items():
            # Sort by timestamp (those without timestamp go to one group)
            timed = [(a, a.get("timestamp", 0)) for a in cat_anomalies
                     if a.get("timestamp") is not None]
            untimed = [a for a in cat_anomalies
                       if a.get("timestamp") is None]

            # Temporal binning for timed anomalies
            timed.sort(key=lambda x: x[1])
            bins = []
            current_bin = []

            for anomaly, ts in timed:
                if current_bin:
                    _, last_ts = current_bin[-1]
                    if ts - last_ts > INCIDENT_TEMPORAL_BIN_SEC:
                        bins.append(current_bin)
                        current_bin = []
                current_bin.append((anomaly, ts))

            if current_bin:
                bins.append(current_bin)

            # Create incidents from temporal bins
            for bin_items in bins:
                anomalies_in_bin = [a for a, _ in bin_items]
                timestamps = [ts for _, ts in bin_items]
                t_start = min(timestamps)
                t_end = max(timestamps)

                # Dedup: same topic+different fields -> single incident
                topics = set()
                for a in anomalies_in_bin:
                    t = a.get("details", {}).get("topic", "")
                    if t:
                        topics.add(t)

                severities = [a.get("severity", "INFO") for a in anomalies_in_bin]
                severity_order = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
                worst = max(severities, key=lambda s: severity_order.get(s, 0))

                # Build description
                topic_str = ", ".join(sorted(topics)) if topics else "multiple"
                if len(anomalies_in_bin) == 1:
                    title = anomalies_in_bin[0].get("description", category)[:80]
                else:
                    title = f"{category} on {topic_str} ({len(anomalies_in_bin)} anomalies)"

                # Get state context
                state_context = self.state_builder.get_state_at(t_start)

                # Find correlated log events
                log_evidence = []
                for ev in self.log_events:
                    if ev.event_type != "UNMATCHED" and \
                       abs(ev.timestamp - t_start) <= CAUSAL_CHAIN_WINDOW_SEC:
                        log_evidence.append({
                            "timestamp_str": ev.timestamp_str,
                            "node": ev.node,
                            "level": ev.level_str,
                            "message": ev.raw_message[:200],
                            "event_type": ev.event_type,
                        })

                sensor_evidence = [{
                    "category": a["category"],
                    "description": a.get("description", ""),
                    "details": a.get("details", {}),
                } for a in anomalies_in_bin]

                # Root cause
                if log_evidence:
                    root_cause = f"Sensor anomaly correlated with log events: " \
                                 f"{log_evidence[0]['event_type']}"
                elif state_context:
                    root_cause = f"Sensor anomaly detected (state: " \
                                 f"{', '.join(f'{k}={v}' for k, v in state_context.items())})"
                else:
                    root_cause = "Sensor anomaly detected, no correlated log events found"

                inc = Incident(
                    title=title,
                    severity=worst,
                    category=category,
                    time_start=t_start,
                    time_end=t_end,
                    time_start_str=format_absolute_time(t_start),
                    time_end_str=format_absolute_time(t_end),
                    root_cause=root_cause,
                    log_evidence=log_evidence[:5],  # Limit evidence items
                    sensor_evidence=sensor_evidence,
                    state_context=state_context,
                    recommended_actions=self._get_default_actions(category),
                    bags=[self.bag_name],
                    raw_anomaly_count=len(anomalies_in_bin),
                )
                incidents.append(inc)

            # Handle untimed anomalies (bag-wide findings)
            if untimed:
                severities = [a.get("severity", "INFO") for a in untimed]
                severity_order = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
                worst = max(severities, key=lambda s: severity_order.get(s, 0))

                sensor_evidence = [{
                    "category": a["category"],
                    "description": a.get("description", ""),
                    "details": a.get("details", {}),
                } for a in untimed]

                inc = Incident(
                    title=f"{category} (bag-wide, {len(untimed)} anomalies)",
                    severity=worst,
                    category=category,
                    root_cause="Bag-wide sensor anomaly, no specific timestamp",
                    sensor_evidence=sensor_evidence,
                    bags=[self.bag_name],
                    raw_anomaly_count=len(untimed),
                )
                incidents.append(inc)

        return incidents

    @staticmethod
    def _get_default_actions(category: str) -> List[str]:
        """Get default recommended actions for a category."""
        defaults = {
            "FROZEN_SENSOR": ["Check sensor connections", "Verify sensor power"],
            "FREQUENCY_DROPOUT": ["Check communication bus", "Verify driver status"],
            "HW_FAULT": ["Inspect hardware component", "Check wiring"],
            "HW_FAULT_ONSET": ["Inspect hardware component", "Check wiring"],
            "LIDAR_FREQ_DROPOUT": ["Check LiDAR connection", "Clean LiDAR lens"],
            "LIDAR_POINT_DROP": ["Clean LiDAR lens", "Check for obstructions"],
            "LIDAR_DEGRADATION": ["Inspect LiDAR hardware", "Check for interference"],
            "IMU_FROZEN": ["Check IMU board", "Verify IMU calibration"],
            "CMD_ODOM_MISMATCH": ["Check motor drivers", "Inspect wheel encoders"],
            "STALL_DETECTED": ["Check for physical obstruction", "Inspect drive system"],
            "PROTECTOR_TRIGGERED": ["Inspect robot surroundings", "Check protector sensors"],
            "IR_SENSOR_DEAD": ["Check IR sensor connection", "Replace sensor if needed"],
        }
        return defaults.get(category, ["Investigate further"])
