"""
Shared utilities — timezone, log levels, timestamp formatting.

Single source of truth for constants that were previously duplicated
across analyze.py, log_parser.py, cross_validator.py, report_generator.py,
and rosbag_profiler.py.
"""

from datetime import datetime, timezone, timedelta

# China Standard Time (UTC+8) — matches the Gaussian robot's clock
CST = timezone(timedelta(hours=8))

# ROS log level integer → string mapping
LOG_LEVELS = {1: "DEBUG", 2: "INFO", 4: "WARN", 8: "ERROR", 16: "FATAL"}


def format_absolute_time(unix_sec: float) -> str:
    """Convert Unix epoch seconds to human-readable datetime string (CST)."""
    dt = datetime.fromtimestamp(unix_sec, tz=CST)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecond precision
