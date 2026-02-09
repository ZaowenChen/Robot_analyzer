"""
ROSBag Bridge - The Compute Layer / MCP Server Equivalent

This module implements the four core tools described in the project plan:
  1. get_bag_metadata    - Initial reconnaissance of bag contents
  2. get_topic_statistics - Statistical summarization with windowing
  3. check_topic_frequency - Detect silent failures / dropouts
  4. sample_messages      - Raw message inspection

It uses Welford's online algorithm for single-pass variance computation
(O(1) memory) and sliding window aggregation for hierarchical zooming.
"""

import json
import math
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr


# ---------------------------------------------------------------------------
# Welford's Online Variance Algorithm
# ---------------------------------------------------------------------------
@dataclass
class WelfordAccumulator:
    """Single-pass mean/variance computation with O(1) memory."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, value: float):
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6) if self.min_val != float('inf') else None,
            "max": round(self.max_val, 6) if self.max_val != float('-inf') else None,
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# Message Field Extractors
# ---------------------------------------------------------------------------

# Known message types and their extractable scalar fields
FIELD_EXTRACTORS = {
    "geometry_msgs/msg/Twist": lambda msg: {
        "linear_x": msg.linear.x,
        "linear_y": msg.linear.y,
        "linear_z": msg.linear.z,
        "angular_x": msg.angular.x,
        "angular_y": msg.angular.y,
        "angular_z": msg.angular.z,
    },
    "geometry_msgs/msg/TwistWithCovariance": lambda msg: {
        "linear_x": msg.twist.linear.x,
        "linear_y": msg.twist.linear.y,
        "linear_z": msg.twist.linear.z,
        "angular_x": msg.twist.angular.x,
        "angular_y": msg.twist.angular.y,
        "angular_z": msg.twist.angular.z,
    },
    "geometry_msgs/msg/Vector3Stamped": lambda msg: {
        "x": msg.vector.x,
        "y": msg.vector.y,
        "z": msg.vector.z,
    },
    "geometry_msgs/msg/PoseWithCovarianceStamped": lambda msg: {
        "position_x": msg.pose.pose.position.x,
        "position_y": msg.pose.pose.position.y,
        "position_z": msg.pose.pose.position.z,
        "orientation_x": msg.pose.pose.orientation.x,
        "orientation_y": msg.pose.pose.orientation.y,
        "orientation_z": msg.pose.pose.orientation.z,
        "orientation_w": msg.pose.pose.orientation.w,
    },
    "nav_msgs/msg/Odometry": lambda msg: {
        "position_x": msg.pose.pose.position.x,
        "position_y": msg.pose.pose.position.y,
        "position_z": msg.pose.pose.position.z,
        "orientation_x": msg.pose.pose.orientation.x,
        "orientation_y": msg.pose.pose.orientation.y,
        "orientation_z": msg.pose.pose.orientation.z,
        "orientation_w": msg.pose.pose.orientation.w,
        "twist_linear_x": msg.twist.twist.linear.x,
        "twist_linear_y": msg.twist.twist.linear.y,
        "twist_linear_z": msg.twist.twist.linear.z,
        "twist_angular_x": msg.twist.twist.angular.x,
        "twist_angular_y": msg.twist.twist.angular.y,
        "twist_angular_z": msg.twist.twist.angular.z,
    },
    "sensor_msgs/msg/LaserScan": lambda msg: {
        "range_min_val": float(np.nanmin(msg.ranges)) if len(msg.ranges) > 0 else 0.0,
        "range_max_val": float(np.nanmax(msg.ranges[msg.ranges < msg.range_max])) if len(msg.ranges[msg.ranges < msg.range_max]) > 0 else 0.0,
        "range_mean": float(np.nanmean(msg.ranges[msg.ranges < msg.range_max])) if len(msg.ranges[msg.ranges < msg.range_max]) > 0 else 0.0,
        "num_valid_points": int(np.sum((msg.ranges > msg.range_min) & (msg.ranges < msg.range_max))),
        "num_total_points": len(msg.ranges),
    },
    "std_msgs/msg/Float32": lambda msg: {
        "data": float(msg.data),
    },
    "std_msgs/msg/Int64": lambda msg: {
        "data": int(msg.data),
    },
    "std_msgs/msg/UInt32": lambda msg: {
        "data": int(msg.data),
    },
    "std_msgs/msg/UInt8": lambda msg: {
        "data": int(msg.data),
    },
    "std_msgs/msg/Int8": lambda msg: {
        "data": int(msg.data),
    },
    "std_msgs/msg/Bool": lambda msg: {
        "data": int(msg.data),
    },
}


def extract_fields(msg, msgtype: str) -> Optional[Dict[str, float]]:
    """Extract numeric fields from a deserialized message."""
    if msgtype in FIELD_EXTRACTORS:
        try:
            return FIELD_EXTRACTORS[msgtype](msg)
        except Exception:
            return None
    # Fallback: try to extract any numeric attributes
    try:
        fields = {}
        for attr in dir(msg):
            if attr.startswith('_'):
                continue
            val = getattr(msg, attr)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                fields[attr] = float(val)
        return fields if fields else None
    except Exception:
        return None


def msg_to_dict(msg, msgtype: str) -> dict:
    """Convert a deserialized message to a JSON-serializable dict."""
    result = {}
    for attr in dir(msg):
        if attr.startswith('_'):
            continue
        val = getattr(msg, attr)
        result[attr] = _convert_value(val)
    return result


def _convert_value(val) -> Any:
    """Recursively convert message values to JSON-serializable types."""
    if isinstance(val, (bool, int, float, str, type(None))):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return str(val)
        return val
    if isinstance(val, np.ndarray):
        arr = val.tolist()
        # Truncate large arrays
        if len(arr) > 20:
            return {"first_10": arr[:10], "last_10": arr[-10:], "length": len(arr)}
        return arr
    if isinstance(val, (list, tuple)):
        if len(val) > 20:
            return {"first_10": [_convert_value(v) for v in val[:10]], "length": len(val)}
        return [_convert_value(v) for v in val]
    if isinstance(val, bytes):
        return f"<bytes length={len(val)}>"
    if hasattr(val, '__dict__') or hasattr(val, '__slots__'):
        try:
            sub = {}
            for attr in dir(val):
                if attr.startswith('_'):
                    continue
                sub[attr] = _convert_value(getattr(val, attr))
            return sub
        except Exception:
            return str(val)
    return str(val)


# ---------------------------------------------------------------------------
# ROSBag Bridge Class
# ---------------------------------------------------------------------------

class ROSBagBridge:
    """
    The Bridge / MCP Server for ROSBag data.
    Provides four tools for the LLM diagnostic agent.
    """

    def __init__(self):
        self._type_registry = set()
        # Types that are known to fail deserialization (custom types)
        self._skip_types = set()

    def _try_deserialize(self, rawdata: bytes, msgtype: str):
        """Attempt to deserialize a ROS1 message, handling custom types gracefully."""
        if msgtype in self._skip_types:
            return None
        try:
            return deserialize_cdr(ros1_to_cdr(rawdata, msgtype), msgtype)
        except (KeyError, Exception) as e:
            self._skip_types.add(msgtype)
            return None

    # -----------------------------------------------------------------------
    # Tool 1: get_bag_metadata
    # -----------------------------------------------------------------------
    def get_bag_metadata(self, bag_path: str) -> dict:
        """
        Initial reconnaissance. Returns bag duration, time range, and
        topic inventory with message counts and estimated frequencies.
        """
        with Reader(bag_path) as reader:
            duration = reader.duration / 1e9  # nanoseconds to seconds
            start_time = reader.start_time / 1e9
            end_time = reader.end_time / 1e9

            topics = []
            for name, topic in sorted(reader.topics.items()):
                freq = topic.msgcount / duration if duration > 0 else 0
                topics.append({
                    "name": name,
                    "type": topic.msgtype,
                    "message_count": topic.msgcount,
                    "frequency": round(freq, 2),
                })

            return {
                "duration": round(duration, 2),
                "start_time": round(start_time, 4),
                "end_time": round(end_time, 4),
                "total_messages": reader.message_count,
                "num_topics": len(reader.topics),
                "topics": topics,
            }

    # -----------------------------------------------------------------------
    # Tool 2: get_topic_statistics (The Workhorse)
    # -----------------------------------------------------------------------
    def get_topic_statistics(
        self,
        bag_path: str,
        topic_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        window_size: Optional[float] = None,
    ) -> list:
        """
        Compute descriptive statistics for all scalar fields in a topic.

        If window_size is provided, divides the time range into chunks and
        computes stats per window (Sliding Window Aggregation).
        Otherwise returns stats for the entire range.

        Uses Welford's algorithm for O(1) memory.
        """
        with Reader(bag_path) as reader:
            bag_start = reader.start_time / 1e9
            bag_end = reader.end_time / 1e9

            if start_time is None:
                start_time = bag_start
            if end_time is None:
                end_time = bag_end

            # Find connections for this topic
            connections = [c for c in reader.connections if c.topic == topic_name]
            if not connections:
                return [{"error": f"Topic '{topic_name}' not found in bag"}]

            msgtype = connections[0].msgtype

            # Determine windows
            if window_size and window_size > 0:
                windows = []
                t = start_time
                while t < end_time:
                    w_end = min(t + window_size, end_time)
                    windows.append((t, w_end))
                    t = w_end
            else:
                windows = [(start_time, end_time)]

            # Initialize accumulators per window
            window_accumulators = [defaultdict(WelfordAccumulator) for _ in windows]
            window_counts = [0 for _ in windows]

            # Single pass through messages
            start_ns = int(start_time * 1e9)
            end_ns = int(end_time * 1e9)

            for conn, timestamp, rawdata in reader.messages(connections=connections):
                ts_sec = timestamp / 1e9
                if ts_sec < start_time or ts_sec > end_time:
                    continue

                msg = self._try_deserialize(rawdata, msgtype)
                if msg is None:
                    continue

                fields = extract_fields(msg, msgtype)
                if fields is None:
                    continue

                # Find which window this belongs to
                for i, (w_start, w_end) in enumerate(windows):
                    if w_start <= ts_sec <= w_end:
                        window_counts[i] += 1
                        for field_name, value in fields.items():
                            if isinstance(value, (int, float)):
                                window_accumulators[i][field_name].update(float(value))
                        break

            # Build results
            results = []
            for i, (w_start, w_end) in enumerate(windows):
                window_result = {
                    "window_start": round(w_start, 4),
                    "window_end": round(w_end, 4),
                    "count": window_counts[i],
                    "fields": {},
                }
                for field_name, acc in sorted(window_accumulators[i].items()):
                    window_result["fields"][field_name] = acc.to_dict()
                results.append(window_result)

            return results

    # -----------------------------------------------------------------------
    # Tool 3: check_topic_frequency
    # -----------------------------------------------------------------------
    def check_topic_frequency(
        self,
        bag_path: str,
        topic_name: str,
        resolution: float = 1.0,
    ) -> dict:
        """
        Compute a time-series of message frequency (Hz) at the given resolution.
        Useful for detecting silent failures like packet loss or driver crashes.
        """
        with Reader(bag_path) as reader:
            connections = [c for c in reader.connections if c.topic == topic_name]
            if not connections:
                return {"error": f"Topic '{topic_name}' not found"}

            bag_start = reader.start_time / 1e9
            bag_end = reader.end_time / 1e9

            # Collect timestamps
            timestamps = []
            for conn, timestamp, rawdata in reader.messages(connections=connections):
                timestamps.append(timestamp / 1e9)

            if not timestamps:
                return {"error": f"No messages found for '{topic_name}'"}

            # Bin into resolution-sized windows
            t_start = timestamps[0]
            t_end = timestamps[-1]
            num_bins = max(1, int(math.ceil((t_end - t_start) / resolution)))

            bins = [0] * num_bins
            for ts in timestamps:
                idx = min(int((ts - t_start) / resolution), num_bins - 1)
                bins[idx] += 1

            frequency_series = []
            for i, count in enumerate(bins):
                bin_start = t_start + i * resolution
                hz = count / resolution
                frequency_series.append({
                    "time": round(bin_start, 4),
                    "hz": round(hz, 2),
                    "count": count,
                })

            # Compute summary statistics
            hz_values = [b["hz"] for b in frequency_series]
            return {
                "topic": topic_name,
                "resolution_sec": resolution,
                "total_messages": len(timestamps),
                "duration_sec": round(t_end - t_start, 2),
                "mean_hz": round(np.mean(hz_values), 2),
                "std_hz": round(np.std(hz_values), 2),
                "min_hz": round(min(hz_values), 2),
                "max_hz": round(max(hz_values), 2),
                "frequency_series": frequency_series,
            }

    # -----------------------------------------------------------------------
    # Tool 4: sample_messages
    # -----------------------------------------------------------------------
    def sample_messages(
        self,
        bag_path: str,
        topic_name: str,
        timestamp: Optional[float] = None,
        count: int = 5,
    ) -> dict:
        """
        Return raw JSON deserialization of messages for qualitative inspection.
        If timestamp is given, returns messages closest to that time.
        """
        with Reader(bag_path) as reader:
            connections = [c for c in reader.connections if c.topic == topic_name]
            if not connections:
                return {"error": f"Topic '{topic_name}' not found"}

            msgtype = connections[0].msgtype
            messages = []

            if timestamp is not None:
                # Collect all messages and find the closest ones
                all_msgs = []
                for conn, ts, rawdata in reader.messages(connections=connections):
                    all_msgs.append((ts / 1e9, rawdata, msgtype))

                # Sort by distance to target timestamp
                all_msgs.sort(key=lambda x: abs(x[0] - timestamp))
                candidates = all_msgs[:count]
                candidates.sort(key=lambda x: x[0])  # Sort by time

                for ts_sec, rawdata, mt in candidates:
                    msg = self._try_deserialize(rawdata, mt)
                    if msg is not None:
                        messages.append({
                            "timestamp": round(ts_sec, 6),
                            "data": msg_to_dict(msg, mt),
                        })
                    else:
                        messages.append({
                            "timestamp": round(ts_sec, 6),
                            "data": f"<could not deserialize {mt}>",
                        })
            else:
                # Return first N messages
                collected = 0
                for conn, ts, rawdata in reader.messages(connections=connections):
                    if collected >= count:
                        break
                    msg = self._try_deserialize(rawdata, conn.msgtype)
                    if msg is not None:
                        messages.append({
                            "timestamp": round(ts / 1e9, 6),
                            "data": msg_to_dict(msg, conn.msgtype),
                        })
                    else:
                        messages.append({
                            "timestamp": round(ts / 1e9, 6),
                            "data": f"<could not deserialize {conn.msgtype}>",
                        })
                    collected += 1

            return {
                "topic": topic_name,
                "type": msgtype,
                "num_messages": len(messages),
                "messages": messages,
            }


# ---------------------------------------------------------------------------
# Convenience test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    bridge = ROSBagBridge()
    if len(sys.argv) < 2:
        print("Usage: python rosbag_bridge.py <bag_path>")
        sys.exit(1)

    bag_path = sys.argv[1]

    print("\n=== BAG METADATA ===")
    meta = bridge.get_bag_metadata(bag_path)
    print(json.dumps(meta, indent=2)[:3000])

    # Test statistics on /odom
    print("\n=== TOPIC STATISTICS: /odom (first 30s) ===")
    stats = bridge.get_topic_statistics(bag_path, "/odom",
                                         start_time=meta["start_time"],
                                         end_time=meta["start_time"] + 30,
                                         window_size=10.0)
    print(json.dumps(stats, indent=2)[:3000])

    # Test frequency check
    print("\n=== FREQUENCY CHECK: /odom ===")
    freq = bridge.check_topic_frequency(bag_path, "/odom", resolution=5.0)
    print(json.dumps({k: v for k, v in freq.items() if k != "frequency_series"}, indent=2))
    print(f"  (series has {len(freq.get('frequency_series', []))} entries)")

    # Test sample messages
    print("\n=== SAMPLE MESSAGES: /chassis_cmd_vel ===")
    samples = bridge.sample_messages(bag_path, "/chassis_cmd_vel", count=3)
    print(json.dumps(samples, indent=2)[:2000])
