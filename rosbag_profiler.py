#!/usr/bin/env python3
"""
Enhanced ROSBag Profiler & Anomaly Slicer

Runs two complementary passes on every bag:
  Pass 1 – Log & Diagnostic scan  (/rosout, DiagnosticStatus)
  Pass 2 – Sensor-data analysis   (frozen fields, frequency drops,
            cmd/odom mismatch) via the ROSBagBridge statistical tools.

Pass 2 runs regardless of whether Pass 1 found anything, so *silent*
failures (frozen sensors without any WARN/ERROR log) are caught.

Batch mode (--batch-dir) scans an entire folder of bags and prints a
single summary table so you can spot the problematic files at a glance.
"""

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict

from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbag_bridge import open_bag, msg_to_dict, ROSBagBridge

LOG_LEVELS = {1: "DEBUG", 2: "INFO", 4: "WARN", 8: "ERROR", 16: "FATAL"}
DIAG_LEVELS = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}

# Fields that are legitimately zero on a 2-D differential-drive robot
EXPECTED_ZERO_FIELDS = {
    "/odom": {"orientation_x", "orientation_y", "position_z",
              "twist_angular_x", "twist_angular_y",
              "twist_linear_y", "twist_linear_z"},
    "/chassis_cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
    "/cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
}

# Topics worth checking for sensor-level anomalies
KEY_SENSOR_TOPICS = [
    "/odom", "/chassis_cmd_vel", "/cmd_vel",
    "/unbiased_imu_PRY", "/localization/current_pose",
]


def format_time(ts_sec):
    return datetime.datetime.fromtimestamp(ts_sec).strftime('%H:%M:%S.%f')[:-3]


# -----------------------------------------------------------------------
# Single-bag profiling
# -----------------------------------------------------------------------
def profile_and_slice_bag(bag_path, bridge=None):
    """Profile a single bag.  Returns a summary dict."""
    if bridge is None:
        bridge = ROSBagBridge()

    bag_name = os.path.basename(bag_path)
    result = {
        "bag_name": bag_name,
        "bag_path": bag_path,
        "duration": 0,
        "message_count": 0,
        "log_events": 0,
        "sensor_anomalies": [],
        "health": "UNKNOWN",
    }

    print(f"\n{'='*80}")
    print(f"ROSBAG PROFILER & ANOMALY SLICER: {bag_name}")
    print(f"{'='*80}\n")

    try:
        # ==============================================================
        # STEP 1 – Metadata
        # ==============================================================
        metadata = bridge.get_bag_metadata(bag_path)
        duration = metadata["duration"]
        start_time = metadata["start_time"]
        end_time = metadata["end_time"]
        result["duration"] = duration
        result["message_count"] = metadata["total_messages"]
        result["num_topics"] = metadata["num_topics"]

        print(f"[1] METADATA: Duration: {duration:.1f}s | "
              f"Msgs: {metadata['total_messages']:,} | "
              f"Topics: {metadata['num_topics']}")

        topic_map = {t["name"]: t for t in metadata["topics"]}

        # ==============================================================
        # STEP 2 – Log & hardware diagnostic scan  (original Pass 1)
        # ==============================================================
        print(f"\n[2] SCANNING LOGS & HARDWARE DIAGNOSTICS...")

        with bridge._open_cached(bag_path) as reader:
            anomaly_conns = []
            for c in reader.connections:
                if c.topic in ['/rosout', '/rosout_agg']:
                    anomaly_conns.append(c)
                elif 'DiagnosticStatus' in c.msgtype:
                    anomaly_conns.append(c)

            abnormal_events = []
            if anomaly_conns:
                for conn, ts_ns, rawdata in reader.messages(connections=anomaly_conns):
                    try:
                        msg = deserialize_cdr(
                            ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)

                        if conn.topic in ['/rosout', '/rosout_agg']:
                            if hasattr(msg, 'level') and msg.level >= 4:
                                abnormal_events.append((
                                    ts_ns / 1e9,
                                    f"SYS_LOG [{LOG_LEVELS.get(msg.level, 'ERR')}]",
                                    getattr(msg, 'name', ''),
                                    getattr(msg, 'msg', ''),
                                ))
                        elif 'DiagnosticStatus' in conn.msgtype:
                            if hasattr(msg, 'level') and msg.level > 0:
                                hw_name = (getattr(msg, 'name', '') or conn.topic)
                                err_msg = (getattr(msg, 'message', '') or "Hardware Fault")
                                abnormal_events.append((
                                    ts_ns / 1e9,
                                    f"HW_DIAG [{DIAG_LEVELS.get(msg.level, 'ERR')}]",
                                    hw_name,
                                    err_msg,
                                ))
                    except Exception:
                        continue

        result["log_events"] = len(abnormal_events)

        if not abnormal_events:
            print("  No abnormal logs or hardware faults found.")
            print("  Continuing to sensor-data analysis to check for SILENT failures...")
        else:
            print(f"  Found {len(abnormal_events)} abnormal log/diagnostic events.")
            # Print the first few
            for ev in abnormal_events[:8]:
                print(f"    [{format_time(ev[0])}] {ev[1]:<20} [{ev[2]}]: {ev[3][:120]}")
            if len(abnormal_events) > 8:
                print(f"    ... and {len(abnormal_events)-8} more")

        # ==============================================================
        # STEP 3 – Sensor-data anomaly detection  (NEW)
        # ==============================================================
        print(f"\n[3] SENSOR DATA ANALYSIS (frozen fields, frequency, cmd/odom)...")

        available_topics = [t for t in KEY_SENSOR_TOPICS if t in topic_map]
        sensor_anomalies = []

        # -- 3a. Global statistics per topic --
        for topic in available_topics:
            stats = bridge.get_topic_statistics(bag_path, topic)
            if not stats or "fields" not in stats[0]:
                continue

            expected_zeros = EXPECTED_ZERO_FIELDS.get(topic, set())
            for field, s in stats[0]["fields"].items():
                if field in expected_zeros:
                    continue
                if s["std"] < 1e-6 and s["count"] > 10:
                    if s["mean"] == 0.0:
                        sensor_anomalies.append({
                            "type": "ZERO_FIELD",
                            "topic": topic,
                            "field": field,
                            "mean": s["mean"],
                            "count": s["count"],
                        })
                    else:
                        sensor_anomalies.append({
                            "type": "FROZEN_SENSOR",
                            "topic": topic,
                            "field": field,
                            "mean": s["mean"],
                            "std": s["std"],
                            "count": s["count"],
                        })

        # -- 3b. Windowed analysis: detect freeze onset / resume --
        for topic in available_topics:
            wstats = bridge.get_topic_statistics(
                bag_path, topic, window_size=30.0)
            if not wstats:
                continue

            expected_zeros = EXPECTED_ZERO_FIELDS.get(topic, set())
            for field in wstats[0].get("fields", {}).keys():
                if field in expected_zeros:
                    continue
                stds = [
                    (w["window_start"],
                     w["fields"].get(field, {}).get("std", 0),
                     w["fields"].get(field, {}).get("mean", 0))
                    for w in wstats if field in w.get("fields", {})
                ]
                for i in range(1, len(stds)):
                    prev_std, curr_std = stds[i-1][1], stds[i][1]
                    if prev_std > 0.001 and curr_std < 1e-6:
                        sensor_anomalies.append({
                            "type": "FREEZE_ONSET",
                            "topic": topic,
                            "field": field,
                            "time": stds[i][0],
                            "value_at_freeze": stds[i][2],
                        })

        # -- 3c. Frequency analysis --
        for topic in available_topics:
            freq = bridge.check_topic_frequency(bag_path, topic, resolution=5.0)
            if "error" in freq:
                continue
            mean_hz = freq.get("mean_hz", 0)
            series = freq.get("frequency_series", [])
            for entry in series:
                if mean_hz > 0 and entry["hz"] < mean_hz * 0.3:
                    if entry["time"] < end_time - 10:
                        sensor_anomalies.append({
                            "type": "FREQ_DROPOUT",
                            "topic": topic,
                            "time": entry["time"],
                            "hz": entry["hz"],
                            "expected_hz": round(mean_hz, 1),
                        })
            if mean_hz > 0 and freq.get("std_hz", 0) > mean_hz * 0.3:
                sensor_anomalies.append({
                    "type": "UNSTABLE_FREQ",
                    "topic": topic,
                    "mean_hz": mean_hz,
                    "std_hz": freq["std_hz"],
                })

        # -- 3d. Command / odometry mismatch --
        if "/chassis_cmd_vel" in topic_map and "/odom" in topic_map:
            cmd_stats = bridge.get_topic_statistics(bag_path, "/chassis_cmd_vel")
            odom_stats = bridge.get_topic_statistics(bag_path, "/odom")
            if cmd_stats and odom_stats:
                cmd_f = cmd_stats[0].get("fields", {})
                odom_f = odom_stats[0].get("fields", {})
                cmd_lx = cmd_f.get("linear_x", {})
                odom_tlx = odom_f.get("twist_linear_x", {})
                if cmd_lx.get("max", 0) > 0.01 and odom_tlx.get("max", 0) < 0.001:
                    sensor_anomalies.append({
                        "type": "CMD_ODOM_MISMATCH",
                        "detail": "Commands sent but odometry shows no motion",
                        "cmd_max": cmd_lx.get("max"),
                        "odom_max": odom_tlx.get("max"),
                    })

        result["sensor_anomalies"] = sensor_anomalies

        # -- Print sensor findings --
        if sensor_anomalies:
            by_type = defaultdict(list)
            for a in sensor_anomalies:
                by_type[a["type"]].append(a)

            for atype, items in sorted(by_type.items()):
                print(f"\n  {atype} ({len(items)}):")
                for item in items[:6]:
                    detail_parts = []
                    if "topic" in item:
                        detail_parts.append(item["topic"])
                    if "field" in item:
                        detail_parts.append(f".{item['field']}")
                    if "mean" in item:
                        detail_parts.append(f" mean={item['mean']:.4f}")
                    if "time" in item:
                        detail_parts.append(f" t={item['time']:.1f}")
                    if "hz" in item:
                        detail_parts.append(f" hz={item['hz']:.1f}")
                    if "value_at_freeze" in item:
                        detail_parts.append(f" val={item['value_at_freeze']:.4f}")
                    print(f"    {''.join(detail_parts)}")
                if len(items) > 6:
                    print(f"    ... and {len(items)-6} more")
        else:
            print("  No sensor-data anomalies detected.")

        # ==============================================================
        # STEP 4 – Context Slicer  (original Pass 2 — runs if anomalies)
        # ==============================================================
        all_anomaly_times = [ev[0] for ev in abnormal_events]
        for a in sensor_anomalies:
            if "time" in a:
                all_anomaly_times.append(a["time"])

        if all_anomaly_times and duration > 0:
            first_time = min(all_anomaly_times)
            slice_start = first_time - 2.0
            slice_end = first_time + 2.0

            print(f"\n[4] DATA SLICE around first anomaly ({format_time(first_time)})")

            context_topics = ['/chassis_cmd_vel', '/odom',
                              '/eco_control/path_follow_state']
            with bridge._open_cached(bag_path) as reader:
                context_conns = [c for c in reader.connections
                                 if c.topic in context_topics]
                timeline = []

                for ev in abnormal_events[:5]:
                    timeline.append({
                        "time": ev[0], "type": ev[1],
                        "data": f"[{ev[2]}]: {ev[3][:120]}"})

                if context_conns:
                    for conn, ts_ns, rawdata in reader.messages(
                            connections=context_conns):
                        ts_sec = ts_ns / 1e9
                        if slice_start <= ts_sec <= slice_end:
                            try:
                                msg = deserialize_cdr(
                                    ros1_to_cdr(rawdata, conn.msgtype),
                                    conn.msgtype)
                                data_dict = msg_to_dict(msg, conn.msgtype)

                                if conn.topic == '/odom':
                                    pose = data_dict.get('pose', {}).get('pose', {})
                                    twist = data_dict.get('twist', {}).get('twist', {})
                                    data_summary = (
                                        f"x={pose.get('position',{}).get('x',0):.2f}, "
                                        f"v={twist.get('linear',{}).get('x',0):.2f}")
                                elif conn.topic == '/chassis_cmd_vel':
                                    data_summary = (
                                        f"v_cmd={data_dict.get('linear',{}).get('x',0):.2f}, "
                                        f"w_cmd={data_dict.get('angular',{}).get('z',0):.2f}")
                                else:
                                    data_summary = str({
                                        k: v for k, v in data_dict.items()
                                        if k in ['state', 'status', 'data']
                                    })
                                timeline.append({
                                    "time": ts_sec,
                                    "type": f"DATA: {conn.topic}",
                                    "data": data_summary,
                                })
                            except Exception:
                                continue

                timeline.sort(key=lambda x: x["time"])
                print("-" * 80)
                for event in timeline:
                    prefix = ("  ** " if ("LOG" in event["type"]
                                          or "DIAG" in event["type"])
                              else "  |  ")
                    print(f"{prefix}[{format_time(event['time'])}] "
                          f"{event['type']:<25} : {event['data']}")
                print("-" * 80)

        # ==============================================================
        # STEP 5 – Health verdict
        # ==============================================================
        critical_types = {"FROZEN_SENSOR", "CMD_ODOM_MISMATCH", "FREQ_DROPOUT"}
        n_critical = sum(1 for a in sensor_anomalies
                         if a["type"] in critical_types)
        n_warn = sum(1 for a in sensor_anomalies
                     if a["type"] not in critical_types)
        n_log = len(abnormal_events)

        if n_critical > 0 or n_log > 20:
            health = "PROBLEMATIC"
        elif n_warn > 3 or n_log > 5:
            health = "DEGRADED"
        elif n_warn > 0 or n_log > 0:
            health = "MARGINAL"
        else:
            health = "HEALTHY"

        result["health"] = health

        print(f"\n[VERDICT] {bag_name}: {health}  "
              f"(log_events={n_log}, "
              f"sensor_critical={n_critical}, "
              f"sensor_warnings={n_warn})")

    except Exception as e:
        import traceback
        print(f"[!] Error processing bag: {e}")
        traceback.print_exc()
        result["health"] = "ERROR"
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Batch mode
# -----------------------------------------------------------------------
def batch_scan(bag_dir, report_path=None):
    """Scan every .bag in a directory and print a summary table."""
    bag_files = sorted(f for f in os.listdir(bag_dir) if f.endswith('.bag'))
    if not bag_files:
        print("No .bag files found!")
        return []

    print(f"Found {len(bag_files)} bag(s) in {bag_dir}\n")

    bridge = ROSBagBridge()          # single bridge instance → reader cache
    results = []

    for bag_file in bag_files:
        bag_path = os.path.join(bag_dir, bag_file)
        r = profile_and_slice_bag(bag_path, bridge=bridge)
        results.append(r)

    # ----- Summary table -----
    print(f"\n\n{'='*100}")
    print("BATCH SUMMARY")
    print(f"{'='*100}")
    print(f"{'Bag':<42} {'Dur(s)':>7} {'Msgs':>9} "
          f"{'Logs':>5} {'Sensor':>7} {'Health':<13}")
    print("-" * 100)
    for r in results:
        n_sens = len(r.get("sensor_anomalies", []))
        print(f"{r['bag_name']:<42} {r['duration']:>7.1f} "
              f"{r['message_count']:>9,} {r['log_events']:>5} "
              f"{n_sens:>7} {r['health']:<13}")
    print("-" * 100)

    problematic = [r for r in results if r["health"] in ("PROBLEMATIC", "ERROR")]
    degraded = [r for r in results if r["health"] == "DEGRADED"]
    healthy = [r for r in results
               if r["health"] in ("HEALTHY", "MARGINAL")]

    print(f"\nPROBLEMATIC: {len(problematic)}  |  DEGRADED: {len(degraded)}  |  "
          f"HEALTHY/MARGINAL: {len(healthy)}")

    if problematic:
        print("\nProblematic bags:")
        for r in problematic:
            types = defaultdict(int)
            for a in r.get("sensor_anomalies", []):
                types[a["type"]] += 1
            summary = ", ".join(f"{t}:{n}" for t, n in sorted(types.items()))
            print(f"  {r['bag_name']}  ({summary or 'log errors only'})")

    # Save JSON report
    if report_path is None:
        report_path = os.path.join(bag_dir, "profiler_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")

    return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile ROS bags for log anomalies AND sensor-data issues.")
    parser.add_argument(
        "bag_path", nargs="?", default=None,
        help="Path to a single .bag file")
    parser.add_argument(
        "--batch-dir", "-d", default=None,
        help="Directory of .bag files to scan in batch mode")
    parser.add_argument(
        "--report", "-r", default=None,
        help="Path for JSON report output")
    args = parser.parse_args()

    if args.batch_dir:
        batch_scan(args.batch_dir, report_path=args.report)
    elif args.bag_path:
        profile_and_slice_bag(args.bag_path)
    else:
        parser.print_help()
