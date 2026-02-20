"""
CLI entry point for the ROSBag Diagnostic Extractor.

Usage:
    python -m rosbag_diagnostic_extractor analyze <logfile> [options]
    python -m rosbag_diagnostic_extractor batch <directory> [options]
"""

import argparse
import glob
import os
import sys

from .pipeline import run_pipeline
from .constants import DEFAULT_TOKEN_BUDGET


def main():
    parser = argparse.ArgumentParser(
        prog="rosbag_diagnostic_extractor",
        description="Compress robot logs into LLM-ready diagnostic signal",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze command ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a single log file or bag file",
    )
    analyze_parser.add_argument(
        "input",
        help="Path to .bag file or text log file",
    )
    analyze_parser.add_argument(
        "--output-dir", "-o",
        default="./diagnostics",
        help="Output directory (default: ./diagnostics)",
    )
    analyze_parser.add_argument(
        "--token-budget", "-t",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=f"Token budget for LLM prompt (default: {DEFAULT_TOKEN_BUDGET})",
    )
    analyze_parser.add_argument(
        "--robot-context", "-c",
        default="",
        help="Robot/site context string (e.g., 'Gaussian S50, Dakota-Valley')",
    )
    analyze_parser.add_argument(
        "--min-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
        help="Minimum log level to process (default: INFO)",
    )
    analyze_parser.add_argument(
        "--rules",
        default=None,
        help="Path to validated_rules.yaml",
    )
    analyze_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    # --- batch command ---
    batch_parser = subparsers.add_parser(
        "batch",
        help="Analyze multiple files in a directory",
    )
    batch_parser.add_argument(
        "directory",
        help="Directory containing .bag or .log files",
    )
    batch_parser.add_argument(
        "--output-dir", "-o",
        default="./diagnostics",
        help="Output directory (default: ./diagnostics)",
    )
    batch_parser.add_argument(
        "--token-budget", "-t",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=f"Token budget for LLM prompt (default: {DEFAULT_TOKEN_BUDGET})",
    )
    batch_parser.add_argument(
        "--robot-context", "-c",
        default="",
        help="Robot/site context string",
    )
    batch_parser.add_argument(
        "--min-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
        help="Minimum log level to process (default: INFO)",
    )
    batch_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        run_pipeline(
            args.input,
            output_dir=args.output_dir,
            token_budget=args.token_budget,
            robot_context=args.robot_context,
            verbose=args.verbose,
            rules_path=args.rules,
            min_level=args.min_level,
        )

    elif args.command == "batch":
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
            sys.exit(1)

        files = (
            glob.glob(os.path.join(args.directory, "*.bag"))
            + glob.glob(os.path.join(args.directory, "*.log"))
            + glob.glob(os.path.join(args.directory, "*.txt"))
        )

        if not files:
            print(f"No .bag, .log, or .txt files found in {args.directory}")
            sys.exit(1)

        print(f"Found {len(files)} files to process")
        for i, filepath in enumerate(sorted(files), 1):
            basename = os.path.splitext(os.path.basename(filepath))[0]
            file_output_dir = os.path.join(args.output_dir, basename)
            print(f"\n[{i}/{len(files)}] Processing {os.path.basename(filepath)}...")
            try:
                run_pipeline(
                    filepath,
                    output_dir=file_output_dir,
                    token_budget=args.token_budget,
                    robot_context=args.robot_context,
                    verbose=args.verbose,
                    min_level=args.min_level,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print(f"\nBatch complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
