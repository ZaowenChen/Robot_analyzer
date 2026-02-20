"""
LogExtractor — Reads /rosout and classifies messages (v3.0)

Reads ALL /rosout messages (INFO+) from ROS bag files, matches them against
a registry of regex patterns specific to Gaussian/Gausium cleaning robots,
and produces classified LogEvent objects.

This module is the PRIMARY data source for the log-first diagnostic pipeline.
Sensor data (analyzed by the bridge tools) serves as VERIFICATION of what
the logs report.

Usage:
    from logs import LogExtractor, extract_log_timeline
    from bridge import ROSBagBridge

    events, state_builder = extract_log_timeline(bag_path, ROSBagBridge())
"""

import os
from collections import defaultdict
from typing import List, Tuple

from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types, types as _typesys

from core.models import LogEvent, LogPattern
from core.utils import CST, LOG_LEVELS, format_absolute_time
from logs.patterns import PATTERN_REGISTRY
from logs.state_timeline import StateTimelineBuilder


# ---------------------------------------------------------------------------
# Register rosgraph_msgs/msg/Log if not already in the type system
# (rosbags 0.9.15 does not include it by default)
# ---------------------------------------------------------------------------

_LOG_TYPE_NAME = "rosgraph_msgs/msg/Log"

if _LOG_TYPE_NAME not in _typesys.FIELDDEFS:
    _LOG_MSGDEF = """
byte DEBUG=1
byte INFO=2
byte WARN=4
byte ERROR=8
byte FATAL=16
Header header
byte level
string name
string msg
string file
string function
uint32 line
string[] topics
"""
    try:
        _log_types = get_types_from_msg(
            _LOG_MSGDEF.strip(), _LOG_TYPE_NAME)
        register_types(_log_types)
    except Exception as _e:
        print(f"[WARN] Could not register rosgraph_msgs/msg/Log: {_e}")


# ---------------------------------------------------------------------------
# LogExtractor — Reads /rosout and classifies messages
# ---------------------------------------------------------------------------

class LogExtractor:
    """
    Extracts and classifies log messages from /rosout topics in a ROS bag.

    Unlike the v2.0 approach (WARN+ only), this reads ALL messages at
    min_level (default INFO+) because INFO-level state transitions are
    the primary diagnostic signal for Gaussian robots.
    """

    def __init__(self, bridge, min_level: int = 2):
        """
        Args:
            bridge: ROSBagBridge instance (for cached reader access)
            min_level: Minimum log level to extract.
                       2=INFO+, 4=WARN+ (legacy), 1=DEBUG+ (verbose)
        """
        self.bridge = bridge
        self.min_level = min_level

    def extract(self, bag_path: str, bag_name: str = None) -> List[LogEvent]:
        """
        Extract all log events from a bag file.

        Args:
            bag_path: Path to the .bag file
            bag_name: Display name (defaults to basename)

        Returns:
            Sorted list of LogEvent objects
        """
        if bag_name is None:
            bag_name = os.path.basename(bag_path)

        events: List[LogEvent] = []

        try:
            with self.bridge._open_cached(bag_path) as reader:
                log_conns = [c for c in reader.connections
                             if c.topic in ['/rosout', '/rosout_agg']]
                if not log_conns:
                    return events

                for conn, ts_ns, rawdata in reader.messages(connections=log_conns):
                    try:
                        msg = deserialize_cdr(
                            ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)

                        level = int(getattr(msg, 'level', 0))
                        if level < self.min_level:
                            continue

                        raw_message = str(getattr(msg, 'msg', ''))
                        node = str(getattr(msg, 'name', ''))
                        ts_sec = ts_ns / 1e9

                        # Classify against pattern registry
                        event_type, parsed_data = self._classify(raw_message)

                        events.append(LogEvent(
                            timestamp=ts_sec,
                            timestamp_str=format_absolute_time(ts_sec),
                            node=node,
                            level=level,
                            level_str=LOG_LEVELS.get(level, f"LVL{level}"),
                            raw_message=raw_message,
                            event_type=event_type,
                            parsed_data=parsed_data,
                            bag_name=bag_name,
                        ))
                    except Exception:
                        continue

        except Exception as e:
            print(f"    [WARN] Could not extract logs from {bag_name}: {e}")

        # Sort by timestamp
        events.sort(key=lambda ev: ev.timestamp)
        return events

    def _classify(self, message: str) -> Tuple[str, dict]:
        """
        Match a log message against the pattern registry (first-match-wins).

        Returns:
            (event_type, parsed_data) tuple
        """
        for pattern in PATTERN_REGISTRY:
            match = pattern.regex.search(message)
            if match:
                parsed_data = {
                    "pattern": pattern.name,
                    "matched_text": match.group(0),
                }
                # Capture named groups if any
                if match.groupdict():
                    parsed_data.update(match.groupdict())
                # Capture the value group
                try:
                    parsed_data["value"] = match.group(pattern.value_group)
                except (IndexError, AttributeError):
                    pass
                # Include all captured groups
                if match.groups():
                    parsed_data["groups"] = list(match.groups())

                return pattern.event_type, parsed_data

        return "UNMATCHED", {}

    def get_event_summary(self, events: List[LogEvent]) -> dict:
        """Generate summary statistics for extracted events."""
        by_type = defaultdict(int)
        by_level = defaultdict(int)
        by_node = defaultdict(int)

        for ev in events:
            by_type[ev.event_type] += 1
            by_level[ev.level_str] += 1
            by_node[ev.node] += 1

        return {
            "total_events": len(events),
            "by_event_type": dict(sorted(by_type.items(),
                                         key=lambda x: -x[1])),
            "by_level": dict(sorted(by_level.items(),
                                    key=lambda x: -x[1])),
            "by_node": dict(sorted(by_node.items(),
                                   key=lambda x: -x[1])[:10]),  # top 10
            "matched_pct": round(
                100 * (1 - by_type.get("UNMATCHED", 0) / max(len(events), 1)),
                1),
        }


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

def extract_log_timeline(bag_path: str, bridge,
                         min_level: int = 2,
                         bag_name: str = None,
                         ) -> Tuple[List[LogEvent], StateTimelineBuilder]:
    """
    One-call extraction: parse /rosout and build state timeline.

    Args:
        bag_path: Path to the .bag file
        bridge: ROSBagBridge instance
        min_level: Minimum log level (2=INFO+, 4=WARN+)
        bag_name: Display name (defaults to basename)

    Returns:
        (events, state_builder) tuple
    """
    extractor = LogExtractor(bridge, min_level=min_level)
    events = extractor.extract(bag_path, bag_name=bag_name)

    state_builder = StateTimelineBuilder()
    state_builder.process_events(events)

    return events, state_builder
