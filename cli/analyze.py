"""
CLI entry point for the ROSBag Diagnostic Analyzer.

Supports four analysis modes:
  --mode incident   (DEFAULT) Log-first incident pipeline with clustering
  --mode crossval   Log-sensor cross-validation
  --mode mission    Multi-bag virtual timeline with absolute timestamps
  --mode legacy     Per-bag independent analysis

Usage:
  python -m cli.analyze --bag-dir ./bags --mode incident
  python -m cli.analyze --mode crossval --bag-dir ./bags
  python -m cli.analyze --mode mission --bag-dir ./bags --csv timeline.csv
"""

import argparse
import json
import os
import time

from analysis.diagnostic_analyzer import DiagnosticAnalyzer
from analysis.mission_orchestrator import MissionOrchestrator
from analysis.cross_validator import CrossValidator
from analysis.mission_orchestrator import export_timeline_csv
from core.utils import format_absolute_time

# Project root: one level up from the cli/ directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_legacy_mode(bag_dir: str, report_path: str = None):
    """Original per-bag analysis -- backward compatible."""
    bag_files = sorted(f for f in os.listdir(bag_dir) if f.endswith('.bag'))
    if not bag_files:
        print("No .bag files found!")
        return

    all_reports = []
    for bag_file in bag_files:
        bag_path = os.path.join(bag_dir, bag_file)
        analyzer = DiagnosticAnalyzer()
        report = analyzer.analyze(bag_path, mode="legacy")
        all_reports.append(report)

    report_path = report_path or os.path.join(PROJECT_ROOT, "diagnostic_report.json")
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n\nFull report saved to: {report_path}")

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
                      csv_path: str = None, gap_threshold: float = 300.0,
                      mode: str = "incident"):
    """Mission-centric multi-bag analysis."""
    orchestrator = MissionOrchestrator(bag_dir, gap_threshold=gap_threshold)

    print(f"\n{'='*70}")
    print(f"MISSION DISCOVERY (mode={mode})")
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

    mission_reports = []
    for mission in missions:
        print(f"\n{'='*70}")
        print(f"ANALYZING MISSION {mission.mission_id} "
              f"({len(mission.bags)} bags, mode={mode})")
        print(f"{'='*70}")
        report = orchestrator.analyze_mission(mission, mode=mode)
        mission_reports.append(report)

    version = "3.0" if mode == "incident" else "2.0"
    full_report = {
        "analyzer_version": version,
        "analysis_mode": mode,
        "generated_at": format_absolute_time(time.time()),
        "missions": mission_reports,
    }

    report_path = report_path or os.path.join(PROJECT_ROOT, "diagnostic_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n\nReport saved to: {report_path}")

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

        if mode == "incident":
            summary = mr.get("summary", {})
            print(f"    Incidents: {summary.get('total_incidents', 0)} "
                  f"(C:{summary.get('critical', 0)} "
                  f"W:{summary.get('warning', 0)} "
                  f"I:{summary.get('info', 0)})")
            print(f"    Suppressed: {summary.get('suppressed', 0)}")
            print(f"    Raw anomalies: {summary.get('raw_anomaly_count', 0)}")
            print(f"    Log events parsed: {summary.get('log_events_parsed', 0)}")
            print(f"    State transitions: {summary.get('state_transitions', 0)}")
        else:
            print(f"    Timeline events: {len(mr.get('timeline', []))}")
            print(f"    Cross-bag anomalies: {len(mr.get('cross_bag_anomalies', []))}")
            print(f"    Log-sensor correlations: {len(mr.get('correlations', []))}")

        for s in mr.get("per_bag_summaries", []):
            extra = ""
            if mode == "incident":
                extra = f" Inc:{s.get('incident_count', 0)}"
                extra += f" Sup:{s.get('suppressed_count', 0)}"
            print(f"      {s['bag_name']}: {s['health_status']} "
                  f"(C:{s['critical_count']} W:{s['warning_count']} "
                  f"L:{s['log_event_count']}{extra})")


def _run_crossval_mode(bag_dir: str, report_path: str = None):
    """Cross-validation mode: timestamp-aligned log-sensor correlation."""
    import glob

    bag_paths = sorted(glob.glob(os.path.join(bag_dir, "**/*.bag"), recursive=True))
    if not bag_paths:
        print(f"No .bag files found in {bag_dir}")
        return

    cv = CrossValidator(bag_paths=bag_paths)
    cv.run(skip_sensor_snapshots=False)

    report_path = report_path or os.path.join(PROJECT_ROOT, "cross_validation_report.json")
    cv.export_json(report_path)

    # Print LLM-ready summary
    print(f"\n{'='*70}")
    print("LLM-READY SUMMARY")
    print(f"{'='*70}")
    print(cv.get_summary_for_llm())


def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostics on .bag files (v3.0 -- Log-First).")
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
        choices=["legacy", "mission", "incident", "crossval"],
        default="incident",
        help="Analysis mode: 'incident' (v3.0 log-first, DEFAULT), "
             "'crossval' (v4.0 log-sensor cross-validation), "
             "'mission' (v2.0 timeline), 'legacy' (v2.0 per-bag)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path for CSV export",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=300.0,
        help="Max seconds between bags to group into same mission (default: 300)",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="Generate an interactive HTML report at the given path",
    )
    parser.add_argument(
        "--foxglove-dir",
        default=None,
        help="Generate problem-specific Foxglove layouts in the given directory",
    )
    args = parser.parse_args()
    bag_dir = os.path.abspath(args.bag_dir)

    if not os.path.isdir(bag_dir):
        print(f"Bag directory does not exist: {bag_dir}")
        return

    if args.mode == "legacy":
        _run_legacy_mode(bag_dir, args.report_path)
    elif args.mode == "crossval":
        _run_crossval_mode(bag_dir, args.report_path)
    else:
        _run_mission_mode(bag_dir, args.report_path, args.csv,
                         args.gap_threshold, mode=args.mode)

    # Determine the report path that was actually used
    report_path = args.report_path
    if report_path is None:
        report_path = "diagnostic_report.json"

    # Generate Foxglove layouts
    foxglove_layouts = None
    if args.foxglove_dir:
        try:
            from reporting.foxglove_layouts import generate_foxglove_layouts
            print(f"\n--- Generating Foxglove Layouts ---")
            foxglove_layouts = generate_foxglove_layouts(
                diagnostic_report=report_path,
                output_dir=args.foxglove_dir,
            )
            print(f"  {len(foxglove_layouts)} layout(s) generated in {args.foxglove_dir}/")
        except ImportError:
            print("Warning: reporting.foxglove_layouts not found, skipping layout generation")

    # Generate HTML report
    if args.html:
        try:
            from reporting.html_report import generate_html_report
            print(f"\n--- Generating HTML Report ---")
            generate_html_report(
                diagnostic_report=report_path,
                foxglove_layouts=foxglove_layouts,
                output_path=args.html,
            )
        except ImportError:
            print("Warning: reporting.html_report not found, skipping HTML generation")


if __name__ == "__main__":
    main()
