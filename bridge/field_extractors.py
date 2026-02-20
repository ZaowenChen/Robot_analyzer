"""
Message field extractors — extract numeric scalars from deserialized ROS messages.

Contains FIELD_EXTRACTORS dict, extract_fields(), msg_to_dict(), and helpers.
"""

import math
from typing import Any, Dict, Optional

import numpy as np

from core.utils import LOG_LEVELS


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


def _extract_diagnostic_status(msg) -> Dict[str, float]:
    """Extract numeric fields from a DiagnosticStatus message.

    DiagnosticStatus uses key-value pairs rather than typed fields.
    We extract:
      - ``level`` (0=OK, 1=WARN, 2=ERROR, 3=STALE)
      - Each key-value pair whose value is numeric -> float
      - Each key-value pair whose value is "true"/"false" -> 1.0/0.0
    This lets Welford's pipeline track transitions (e.g. a health flag
    flipping from healthy to fault).
    """
    fields: Dict[str, float] = {"level": float(msg.level)}
    for kv in msg.values:
        key = kv.key
        val = kv.value
        # Boolean strings
        if val.lower() == "true":
            fields[key] = 1.0
        elif val.lower() == "false":
            fields[key] = 0.0
        else:
            # Try numeric
            try:
                fields[key] = float(val)
            except (ValueError, TypeError):
                pass
    return fields


FIELD_EXTRACTORS["diagnostic_msgs/msg/DiagnosticStatus"] = _extract_diagnostic_status


# ---------- /rosout log message support ----------

def _extract_log_message(msg) -> dict:
    """Extract structured data from a rosgraph_msgs/msg/Log message.

    Unlike FIELD_EXTRACTORS (which return only numeric scalars for Welford),
    this returns both numeric and string fields for log analysis.
    """
    return {
        "level": int(msg.level),
        "name": str(getattr(msg, 'name', '')),
        "msg": str(getattr(msg, 'msg', '')),
        "file": str(getattr(msg, 'file', '')),
        "function": str(getattr(msg, 'function', '')),
        "line": int(getattr(msg, 'line', 0)),
    }


# Register /rosout extractors
FIELD_EXTRACTORS["rosgraph_msgs/msg/Log"] = _extract_log_message
FIELD_EXTRACTORS["rcl_interfaces/msg/Log"] = _extract_log_message


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
