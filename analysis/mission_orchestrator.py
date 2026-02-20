"""
MissionOrchestrator -- multi-bag virtual timeline analysis.

Groups temporally contiguous bags into missions and runs diagnostic
analysis across the full timeline, detecting cross-bag persistent
incidents and boundary artifacts.

Also includes CSV export helpers for incident and timeline reports.
"""

import csv
import json
import os
from typing import List

from core.constants import MISSION_GAP_THRESHOLD
from core.utils import CST, format_absolute_time
from core.models import BagInfo, Mission, Incident
from analysis.diagnostic_analyzer import DiagnosticAnalyzer


class MissionOrchestrator:
    """Orchestrates multi-bag analysis with virtual timeline."""

    def __init__(self, bag_dir: str, gap_threshold: float = MISSION_GAP_THRESHOLD):
        self.bag_dir = bag_dir
        self.gap_threshold = gap_threshold
        from bridge import ROSBagBridge
        self.bridge = ROSBagBridge()

    def discover_and_group(self) -> List[Mission]:
        """Discover bags, sort by time, group into missions."""
        bag_files = sorted(f for f in os.listdir(self.bag_dir) if f.endswith('.bag'))
        if not bag_files:
            return []

        bag_infos = []
        for bf in bag_files:
            path = os.path.join(self.bag_dir, bf)
            try:
                meta = self.bridge.get_bag_metadata(path)
                bag_infos.append(BagInfo(
                    path=path, name=bf,
                    start_time=meta["start_time"],
                    end_time=meta["end_time"],
                    duration=meta["duration"],
                ))
            except Exception as e:
                print(f"  [WARN] Could not read metadata for {bf}: {e}")
                continue

        if not bag_infos:
            return []

        bag_infos.sort(key=lambda b: b.start_time)
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

        missions.append(Mission(
            mission_id=len(missions) + 1,
            bags=current_group,
        ))
        return missions

    def analyze_mission(self, mission: Mission, mode: str = "incident") -> dict:
        """Analyze all bags in a mission."""
        per_bag_reports = []
        all_incidents: List[Incident] = []
        all_suppressed: List[dict] = []
        all_state_summaries = {}

        for bag_info in mission.bags:
            analyzer = DiagnosticAnalyzer()
            report = analyzer.analyze(bag_info.path, mode=mode)
            per_bag_reports.append(report)

            if mode == "incident":
                # Collect incidents
                for inc_dict in report.get("incidents", []):
                    inc = self._dict_to_incident(inc_dict)
                    all_incidents.append(inc)
                all_suppressed.extend(report.get("suppressed_anomalies", []))
                # Merge state summaries
                bag_state = report.get("state_timeline_summary", {})
                for key, info in bag_state.items():
                    if key not in all_state_summaries:
                        all_state_summaries[key] = info
                    else:
                        # Merge transition counts
                        existing = all_state_summaries[key]
                        existing["total_transitions"] += info.get("total_transitions", 0)

        if mode == "incident":
            return self._build_incident_mission_report(
                mission, per_bag_reports, all_incidents,
                all_suppressed, all_state_summaries)
        else:
            return self._build_v2_mission_report(mission, per_bag_reports)

    def _build_incident_mission_report(self, mission, per_bag_reports,
                                        all_incidents, all_suppressed,
                                        all_state_summaries) -> dict:
        """Build v3.0 incident-oriented mission report."""
        # Sort incidents by time
        all_incidents.sort(key=lambda i: i.time_start)

        # Suppress boundary artifacts
        all_incidents = self._suppress_boundary_incidents(mission, all_incidents)

        # Detect cross-bag persistent incidents
        cross_bag_incidents = self._detect_persistent_incidents(
            mission, per_bag_reports)
        all_incidents.extend(cross_bag_incidents)

        # Re-sort after adding cross-bag
        all_incidents.sort(key=lambda i: i.time_start)

        # Renumber incidents
        for idx, inc in enumerate(all_incidents, 1):
            inc.incident_id = f"INC-{idx:03d}"

        # Compute health
        active = [i for i in all_incidents if not i.suppressed]
        n_critical = sum(1 for i in active if i.severity == "CRITICAL")
        n_warning = sum(1 for i in active if i.severity == "WARNING")
        n_info = sum(1 for i in active if i.severity == "INFO")
        raw_total = sum(r["metrics"].get("critical_count", 0) +
                        r["metrics"].get("warning_count", 0) +
                        r["metrics"].get("info_count", 0)
                        for r in per_bag_reports)
        log_total = sum(r["metrics"].get("log_event_count", 0)
                        for r in per_bag_reports)

        if n_critical > 0:
            health = "UNHEALTHY"
        elif n_warning > 3:
            health = "DEGRADED"
        elif n_warning > 0:
            health = "MARGINAL"
        else:
            health = "HEALTHY"

        return {
            "mission_id": mission.mission_id,
            "start_time": format_absolute_time(mission.start_time),
            "end_time": format_absolute_time(mission.end_time),
            "duration_sec": round(mission.end_time - mission.start_time, 1),
            "num_bags": len(mission.bags),
            "bags": [b.name for b in mission.bags],
            "overall_health": health,
            "summary": {
                "total_incidents": len(active),
                "critical": n_critical,
                "warning": n_warning,
                "info": n_info,
                "suppressed": len(all_suppressed) + sum(
                    1 for i in all_incidents if i.suppressed),
                "raw_anomaly_count": raw_total,
                "log_events_parsed": log_total,
                "state_transitions": sum(
                    s.get("total_transitions", 0)
                    for s in all_state_summaries.values()),
            },
            "incidents": [i.to_dict() for i in all_incidents if not i.suppressed],
            "suppressed_anomalies": all_suppressed + [
                {
                    "incident_id": i.incident_id,
                    "original_category": i.category,
                    "description": i.title,
                    "suppression_reason": i.suppression_reason,
                    "timestamp": i.time_start_str,
                }
                for i in all_incidents if i.suppressed
            ],
            "state_timeline_summary": all_state_summaries,
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
                    "incident_count": len(r.get("incidents", [])),
                    "suppressed_count": len(r.get("suppressed_anomalies", [])),
                }
                for r in per_bag_reports
            ],
        }

    def _build_v2_mission_report(self, mission, per_bag_reports) -> dict:
        """Build v2.0 compatible mission report (timeline-based)."""
        all_timeline_events = []

        for idx, bag_info in enumerate(mission.bags):
            report = per_bag_reports[idx]
            for anomaly in report["anomalies"]:
                ts = anomaly.get("timestamp", bag_info.start_time)
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
            for log_ev in report.get("log_events", []):
                all_timeline_events.append({
                    "timestamp": log_ev["timestamp"],
                    "timestamp_str": log_ev["timestamp_str"],
                    "bag": bag_info.name,
                    "type": "log",
                    "severity": log_ev["level_str"],
                    "category": f"LOG_{log_ev['level_str']}",
                    "description": f"[{log_ev['node']}]: {log_ev['raw_message'][:200]}",
                    "details": {"node": log_ev["node"], "level": log_ev["level"]},
                })

        all_timeline_events.sort(key=lambda e: e["timestamp"])
        all_timeline_events = self._suppress_boundary_timeline(
            mission, all_timeline_events)

        cross_bag_anomalies = self._detect_cross_bag_anomalies(
            mission, per_bag_reports)
        correlations = self._correlate_logs_and_anomalies(all_timeline_events)
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

    @staticmethod
    def _dict_to_incident(d: dict) -> Incident:
        """Convert an incident dict back to an Incident object."""
        return Incident(
            incident_id=d.get("incident_id", ""),
            title=d.get("title", ""),
            severity=d.get("severity", "INFO"),
            category=d.get("category", ""),
            time_start=d.get("time_start", 0) if isinstance(d.get("time_start"), (int, float)) else 0,
            time_end=d.get("time_end", 0) if isinstance(d.get("time_end"), (int, float)) else 0,
            time_start_str=d.get("time_start", ""),
            time_end_str=d.get("time_end", ""),
            root_cause=d.get("root_cause", ""),
            log_evidence=d.get("log_evidence", []),
            sensor_evidence=d.get("sensor_evidence", []),
            state_context=d.get("state_context", {}),
            recommended_actions=d.get("recommended_actions", []),
            bags=d.get("bags", []),
            raw_anomaly_count=d.get("raw_anomaly_count", 0),
            is_cross_bag=d.get("is_cross_bag", False),
            suppressed=d.get("suppressed", False),
            suppression_reason=d.get("suppression_reason", ""),
        )

    def _suppress_boundary_incidents(self, mission: Mission,
                                      incidents: List[Incident]) -> List[Incident]:
        """Suppress frequency dropout incidents near bag boundaries."""
        if len(mission.bags) < 2:
            return incidents

        BOUNDARY_MARGIN = 3.0
        boundary_windows = []
        for i in range(len(mission.bags) - 1):
            bt = mission.bags[i].end_time
            boundary_windows.append((bt - BOUNDARY_MARGIN, bt + BOUNDARY_MARGIN))
        boundary_windows.append(
            (mission.bags[0].start_time,
             mission.bags[0].start_time + BOUNDARY_MARGIN))
        boundary_windows.append(
            (mission.bags[-1].end_time - BOUNDARY_MARGIN,
             mission.bags[-1].end_time))

        def in_boundary(ts):
            return any(s <= ts <= e for s, e in boundary_windows)

        suppressed = 0
        for inc in incidents:
            if inc.category in ("FREQUENCY_DROPOUT", "LIDAR_FREQ_DROPOUT") and \
               inc.time_start > 0 and in_boundary(inc.time_start):
                inc.suppressed = True
                inc.suppression_reason = "Frequency dropout at bag boundary (expected artifact)"
                suppressed += 1

        if suppressed > 0:
            print(f"  [Boundary filter] Suppressed {suppressed} "
                  f"frequency dropout(s) at bag boundaries")
        return incidents

    def _detect_persistent_incidents(self, mission: Mission,
                                      per_bag_reports: list) -> List[Incident]:
        """Detect incidents that persist across 3+ consecutive bags."""
        if len(per_bag_reports) < 3:
            return []

        persistent = []

        # Track categories per bag
        bag_categories = []
        for report in per_bag_reports:
            cats = set()
            for inc in report.get("incidents", []):
                cats.add(inc.get("category", ""))
            bag_categories.append(cats)

        # Find categories in 3+ consecutive bags
        for cat in set().union(*bag_categories):
            consecutive = 0
            max_consecutive = 0
            start_bag_idx = 0
            current_start = 0

            for idx, cats in enumerate(bag_categories):
                if cat in cats:
                    if consecutive == 0:
                        current_start = idx
                    consecutive += 1
                    if consecutive > max_consecutive:
                        max_consecutive = consecutive
                        start_bag_idx = current_start
                else:
                    consecutive = 0

            if max_consecutive >= 3:
                affected_bags = [
                    mission.bags[i].name
                    for i in range(start_bag_idx, start_bag_idx + max_consecutive)
                ]
                persistent.append(Incident(
                    title=f"Persistent {cat} across {max_consecutive} consecutive bags",
                    severity="CRITICAL",
                    category=cat,
                    time_start=mission.bags[start_bag_idx].start_time,
                    time_end=mission.bags[start_bag_idx + max_consecutive - 1].end_time,
                    time_start_str=format_absolute_time(
                        mission.bags[start_bag_idx].start_time),
                    time_end_str=format_absolute_time(
                        mission.bags[start_bag_idx + max_consecutive - 1].end_time),
                    root_cause=f"Issue persists across {max_consecutive} bag files, "
                               f"indicating a sustained hardware or software problem",
                    recommended_actions=[
                        f"Persistent issue across {max_consecutive} bags — "
                        f"hardware inspection recommended",
                        "Check system logs for recurring errors",
                    ],
                    bags=affected_bags,
                    raw_anomaly_count=0,
                    is_cross_bag=True,
                ))

        return persistent

    # --- v2.0 compatibility methods ---

    def _suppress_boundary_timeline(self, mission, events):
        if len(mission.bags) < 2:
            return events
        BOUNDARY_MARGIN = 3.0
        boundary_windows = []
        for i in range(len(mission.bags) - 1):
            bt = mission.bags[i].end_time
            boundary_windows.append((bt - BOUNDARY_MARGIN, bt + BOUNDARY_MARGIN))
        boundary_windows.append(
            (mission.bags[0].start_time,
             mission.bags[0].start_time + BOUNDARY_MARGIN))
        boundary_windows.append(
            (mission.bags[-1].end_time - BOUNDARY_MARGIN,
             mission.bags[-1].end_time))

        def in_boundary(ts):
            return any(s <= ts <= e for s, e in boundary_windows)

        filtered = []
        for event in events:
            if (event["type"] == "anomaly" and
                event["category"] in ("FREQUENCY_DROPOUT", "LIDAR_FREQ_DROPOUT") and
                in_boundary(event["timestamp"])):
                continue
            filtered.append(event)
        return filtered

    def _detect_cross_bag_anomalies(self, mission, per_bag_reports):
        cross_bag = []
        if len(per_bag_reports) < 2:
            return cross_bag
        for i in range(1, len(per_bag_reports)):
            prev_report = per_bag_reports[i - 1]
            curr_report = per_bag_reports[i]
            prev_bag = mission.bags[i - 1].name
            curr_bag = mission.bags[i].name

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
            for topic, field_name in (prev_frozen & curr_frozen):
                if topic is None:
                    continue
                cross_bag.append({
                    "type": "PERSISTENT_FREEZE",
                    "severity": "CRITICAL",
                    "description": f"{topic}.{field_name} frozen across "
                                   f"{prev_bag} -> {curr_bag}",
                    "bags": [prev_bag, curr_bag],
                    "topic": topic, "field": field_name,
                })

            prev_hw = {a["details"].get("field")
                       for a in prev_report["anomalies"]
                       if a["category"] in ("HW_FAULT", "HW_FAULT_ONSET")}
            curr_hw = {a["details"].get("field")
                       for a in curr_report["anomalies"]
                       if a["category"] in ("HW_FAULT", "HW_FAULT_ONSET")}
            for hw_field in (prev_hw & curr_hw):
                if hw_field is None:
                    continue
                cross_bag.append({
                    "type": "PERSISTENT_HW_FAULT",
                    "severity": "CRITICAL",
                    "description": f"Hardware fault '{hw_field}' persists "
                                   f"{prev_bag} -> {curr_bag}",
                    "bags": [prev_bag, curr_bag], "field": hw_field,
                })
        return cross_bag

    def _correlate_logs_and_anomalies(self, timeline_events, window_sec=2.0):
        logs = [e for e in timeline_events if e["type"] == "log"]
        anomalies = [e for e in timeline_events if e["type"] == "anomaly"]
        if not logs or not anomalies:
            return []
        correlations = []
        log_idx = 0
        for anomaly in anomalies:
            a_ts = anomaly["timestamp"]
            matched_logs = []
            while log_idx > 0 and logs[log_idx]["timestamp"] > a_ts - window_sec:
                log_idx -= 1
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

    def _compute_mission_health(self, per_bag_reports, cross_bag_anomalies):
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
    """Export the timeline/incidents from a mission report to CSV."""

    # Detect mode by checking for incidents vs timeline
    if "incidents" in mission_report:
        _export_incident_csv(mission_report, output_path)
    elif "timeline" in mission_report:
        _export_v2_timeline_csv(mission_report, output_path)


def _export_incident_csv(mission_report: dict, output_path: str):
    """Export incident-oriented CSV."""
    fieldnames = ["timestamp", "incident_id", "severity", "category",
                  "title", "type", "description", "bag", "details"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()

        for inc in mission_report.get("incidents", []):
            # Write header row for incident
            writer.writerow({
                "timestamp": inc.get("time_start", ""),
                "incident_id": inc.get("incident_id", ""),
                "severity": inc.get("severity", ""),
                "category": inc.get("category", ""),
                "title": inc.get("title", ""),
                "type": "incident",
                "description": inc.get("root_cause", ""),
                "bag": ", ".join(inc.get("bags", [])),
                "details": json.dumps(inc.get("state_context", {})),
            })

            # Write log evidence rows
            for ev in inc.get("log_evidence", []):
                writer.writerow({
                    "timestamp": ev.get("timestamp_str", ""),
                    "incident_id": inc.get("incident_id", ""),
                    "severity": ev.get("level", ""),
                    "category": inc.get("category", ""),
                    "title": "",
                    "type": "log_evidence",
                    "description": ev.get("message", ""),
                    "bag": "",
                    "details": json.dumps({"node": ev.get("node", ""),
                                          "event_type": ev.get("event_type", "")}),
                })

            # Write sensor evidence rows
            for ev in inc.get("sensor_evidence", []):
                writer.writerow({
                    "timestamp": "",
                    "incident_id": inc.get("incident_id", ""),
                    "severity": "",
                    "category": ev.get("category", ""),
                    "title": "",
                    "type": "sensor_evidence",
                    "description": ev.get("description", ""),
                    "bag": "",
                    "details": json.dumps(ev.get("details", {})),
                })

        # Suppressed anomalies
        for s in mission_report.get("suppressed_anomalies", []):
            writer.writerow({
                "timestamp": s.get("timestamp", ""),
                "incident_id": "SUPPRESSED",
                "severity": "",
                "category": s.get("original_category", ""),
                "title": s.get("description", ""),
                "type": "suppressed",
                "description": s.get("suppression_reason", ""),
                "bag": "",
                "details": "",
            })

    print(f"Incident CSV exported to: {output_path}")


def _export_v2_timeline_csv(mission_report: dict, output_path: str):
    """Export v2.0 compatible timeline CSV."""
    fieldnames = ["timestamp", "bag", "type", "severity", "category",
                  "description", "details"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for event in mission_report.get("timeline", []):
            writer.writerow({
                "timestamp": event.get("timestamp_str", ""),
                "bag": event.get("bag", ""),
                "type": event.get("type", ""),
                "severity": event.get("severity", ""),
                "category": event.get("category", ""),
                "description": event.get("description", ""),
                "details": json.dumps(event.get("details", {})),
            })
    print(f"CSV timeline exported to: {output_path}")
