"""
Stage 1: Log parsing with dual input support (text logs + binary bags).

Both sources produce the same ParsedLogRecord stream via the LogSource protocol.
Template extraction happens at parse time for both.
"""

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterator, Optional, Protocol

from .models import ParsedLogRecord
from .utils import templatize
from .constants import LOG_LEVELS, UNSTABLE_TEMPLATES

# China Standard Time (UTC+8) — matches robot's clock
CST = timezone(timedelta(hours=8))


# ---------------------------------------------------------------------------
# LogSource protocol
# ---------------------------------------------------------------------------

class LogSource(Protocol):
    """Protocol for streaming parsed log records from any source."""

    def records(self) -> Iterator[ParsedLogRecord]:
        """Yield parsed log records. Must be a generator for memory efficiency."""
        ...

    def get_metadata(self) -> dict:
        """Return source-specific metadata."""
        ...


# ---------------------------------------------------------------------------
# Text log parser
# ---------------------------------------------------------------------------

# Regex for Gaussian robot text log format:
#   [ LEVEL][H:MM:SS.mmm AM/PM TZ][/node_name]: message
# Also handles 24h format (no AM/PM).
_LOG_LINE_RE = re.compile(
    r"^\s*\[\s*(\w+)\s*\]"                  # [LEVEL]
    r"\[(\d{1,2}:\d{2}:\d{2}\.\d{3})"       # [H:MM:SS.mmm
    r"(?:\s*(AM|PM))?"                        # optional AM/PM
    r"(?:\s+\w+)?\]"                          # optional TZ]
    r"\[(/[^\]]+)\]"                          # [/node_name]
    r":\s*(.*)",                               # : message
    re.DOTALL,
)

# Level name -> minimum level filtering
_LEVEL_PRIORITY = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4}


def _parse_timestamp_ms(time_str: str, ampm: Optional[str]) -> float:
    """
    Parse timestamp string to milliseconds from midnight.
    Handles both 12h (with AM/PM) and 24h formats.
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_ms = parts[2].split(".")
    seconds = int(sec_ms[0])
    millis = int(sec_ms[1]) if len(sec_ms) > 1 else 0

    # Convert 12h to 24h
    if ampm:
        ampm = ampm.upper()
        if ampm == "PM" and hours != 12:
            hours += 12
        elif ampm == "AM" and hours == 12:
            hours = 0

    return (hours * 3600 + minutes * 60 + seconds) * 1000.0 + millis


def _is_unstable_template(template: str) -> bool:
    """Check if a template matches the structural instability filter."""
    return any(p.search(template) for p in UNSTABLE_TEMPLATES)


class TextLogSource:
    """
    Reads plain text log files exported from Foxglove or rosbag tools.

    Handles:
    - 12h/24h timestamps
    - Midnight rollover detection
    - Malformed lines (skip with warning count)
    - Streaming via generator (O(1) memory)
    """

    def __init__(
        self,
        log_path: str,
        *,
        min_level: str = "INFO",
    ):
        self.log_path = log_path
        self.min_level = _LEVEL_PRIORITY.get(min_level.upper(), 1)
        self._metadata: Dict = {
            "source_type": "text_log",
            "path": log_path,
            "total_lines": 0,
            "parsed_lines": 0,
            "skipped_lines": 0,
        }

    def records(self) -> Iterator[ParsedLogRecord]:
        """Generator that streams parsed records from the text log file."""
        prev_ts_ms = 0.0
        day_offset_ms = 0.0
        source_file = os.path.basename(self.log_path)

        with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                self._metadata["total_lines"] += 1
                line = line.rstrip("\n\r")
                if not line:
                    continue

                match = _LOG_LINE_RE.match(line)
                if not match:
                    self._metadata["skipped_lines"] += 1
                    continue

                level_str = match.group(1).upper()
                level_priority = _LEVEL_PRIORITY.get(level_str, 1)
                if level_priority < self.min_level:
                    continue

                time_str = match.group(2)
                ampm = match.group(3)
                node = match.group(4)
                message = match.group(5)

                ts_ms = _parse_timestamp_ms(time_str, ampm)

                # Midnight rollover detection
                if ts_ms < prev_ts_ms - 3_600_000:  # dropped by >1 hour
                    day_offset_ms += 86_400_000
                ts_ms += day_offset_ms
                prev_ts_ms = ts_ms - day_offset_ms  # compare raw, not offset

                # Template extraction
                template, values = templatize(message)

                self._metadata["parsed_lines"] += 1

                yield ParsedLogRecord(
                    level=level_str,
                    timestamp_ms=ts_ms,
                    node=node,
                    raw_message=message,
                    template=template,
                    values=values,
                    line_number=line_num,
                    source_file=source_file,
                )

    def get_metadata(self) -> dict:
        return dict(self._metadata)


# ---------------------------------------------------------------------------
# Binary bag parser
# ---------------------------------------------------------------------------

class BagLogSource:
    """
    Reads /rosout messages from binary ROS .bag files.

    Wraps existing rosbag_bridge infrastructure for reader caching
    and truncated bag support.
    """

    # Map string level names to rosgraph_msgs/msg/Log integer levels
    _LEVEL_STR_TO_INT = {"DEBUG": 1, "INFO": 2, "WARN": 4, "ERROR": 8, "FATAL": 16}

    def __init__(
        self,
        bag_path: str,
        *,
        min_level=2,  # 2 = INFO; accepts int or string ("INFO", "WARN", etc.)
    ):
        self.bag_path = bag_path
        # Accept both int and string min_level
        if isinstance(min_level, str):
            self.min_level = self._LEVEL_STR_TO_INT.get(min_level.upper(), 2)
        else:
            self.min_level = int(min_level)
        self._metadata: Dict = {
            "source_type": "rosbag",
            "path": bag_path,
            "total_messages": 0,
            "parsed_messages": 0,
        }

    def records(self) -> Iterator[ParsedLogRecord]:
        """Generator that streams parsed records from the bag file."""
        # Lazy import to avoid dependency when using text-only mode
        try:
            from rosbags.rosbag1 import Reader
            from rosbags.serde import deserialize_cdr, ros1_to_cdr
        except ImportError:
            raise ImportError(
                "rosbags library required for bag mode. "
                "Install with: pip install rosbags"
            )

        # Import bridge for truncated bag support
        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        try:
            from bridge import ROSBagBridge
        except ImportError:
            ROSBagBridge = None

        bag_name = os.path.basename(self.bag_path)
        _LOG_LEVEL_NAMES = {1: "DEBUG", 2: "INFO", 4: "WARN", 8: "ERROR", 16: "FATAL"}

        # Register Log message type if needed
        try:
            from rosbags.typesys import get_types_from_msg, register_types, types as _ts
            _LOG_TYPE = "rosgraph_msgs/msg/Log"
            if _LOG_TYPE not in _ts.FIELDDEFS:
                _LOG_MSGDEF = (
                    "byte DEBUG=1\nbyte INFO=2\nbyte WARN=4\n"
                    "byte ERROR=8\nbyte FATAL=16\n"
                    "Header header\nbyte level\nstring name\nstring msg\n"
                    "string file\nstring function\nuint32 line\nstring[] topics"
                )
                _log_types = get_types_from_msg(_LOG_MSGDEF, _LOG_TYPE)
                register_types(_log_types)
        except Exception:
            pass

        msg_index = 0
        midnight_epoch = None

        try:
            with Reader(self.bag_path) as reader:
                # Compute midnight for ms-from-midnight
                if reader.start_time:
                    start_sec = reader.start_time / 1e9
                    dt = datetime.fromtimestamp(start_sec, tz=CST)
                    midnight_epoch = dt.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ).timestamp()

                    self._metadata["start_time"] = start_sec
                    self._metadata["end_time"] = reader.end_time / 1e9 if reader.end_time else start_sec
                    self._metadata["duration_sec"] = (reader.end_time - reader.start_time) / 1e9 if reader.end_time else 0

                log_conns = [
                    c for c in reader.connections
                    if c.topic in ["/rosout", "/rosout_agg"]
                ]
                if not log_conns:
                    return

                for conn, ts_ns, rawdata in reader.messages(connections=log_conns):
                    msg_index += 1
                    self._metadata["total_messages"] += 1

                    try:
                        msg = deserialize_cdr(
                            ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype
                        )
                    except Exception:
                        # Deserialization failure — skip this message
                        continue

                    level_int = int(getattr(msg, "level", 0))
                    if level_int < self.min_level:
                        continue

                    raw_message = str(getattr(msg, "msg", ""))
                    node = str(getattr(msg, "name", ""))
                    if not node.startswith("/"):
                        node = "/" + node
                    ts_sec = ts_ns / 1e9

                    # Compute ms from midnight
                    if midnight_epoch is not None:
                        ts_ms = (ts_sec - midnight_epoch) * 1000.0
                    else:
                        ts_ms = ts_sec * 1000.0

                    # Template extraction
                    template, values = templatize(raw_message)

                    self._metadata["parsed_messages"] += 1

                    yield ParsedLogRecord(
                        level=_LOG_LEVEL_NAMES.get(level_int, f"LVL{level_int}"),
                        timestamp_ms=ts_ms,
                        node=node,
                        raw_message=raw_message,
                        template=template,
                        values=values,
                        line_number=msg_index,
                        source_file=bag_name,
                        absolute_timestamp=ts_sec,
                    )

        except Exception as e:
            print(f"[WARN] Could not read bag {bag_name}: {e}")

    def get_metadata(self) -> dict:
        return dict(self._metadata)


# ---------------------------------------------------------------------------
# Auto-detection factory
# ---------------------------------------------------------------------------

def open_log_source(path: str, **kwargs) -> LogSource:
    """
    Auto-detect input format and return the appropriate LogSource.

    .bag files → BagLogSource
    .log/.txt/other → TextLogSource
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bag":
        return BagLogSource(path, **kwargs)
    else:
        return TextLogSource(path, **kwargs)
