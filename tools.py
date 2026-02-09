"""
LangChain Tool definitions that wrap the ROSBag Bridge.

These tools are what the LLM agent can call. They map directly to
the four Bridge functions but are wrapped with LangChain's @tool
decorator for integration with the agent framework.
"""

import json
from typing import Optional

from langchain_core.tools import tool

try:
    from .rosbag_bridge import ROSBagBridge
except ImportError:
    from rosbag_bridge import ROSBagBridge

# Singleton bridge instance
_bridge = ROSBagBridge()


@tool
def get_bag_metadata(bag_path: str) -> str:
    """Get metadata about a ROS bag file including duration, topics, message counts, and frequencies.

    Use this first to understand what data is available in the bag.

    Args:
        bag_path: Path to the .bag file

    Returns:
        JSON string with bag metadata including duration, start/end times, and topic list
    """
    result = _bridge.get_bag_metadata(bag_path)
    # Truncate topic list if too long to fit in context
    if len(result.get("topics", [])) > 30:
        # Return summary of key topics
        key_prefixes = ["/odom", "/cmd_vel", "/chassis_cmd_vel", "/imu", "/scan",
                       "/localization", "/device", "/front_end", "/rosout",
                       "/unbiased_imu", "/pointcloud", "/tf", "/speed",
                       "/navigation", "/protector", "/v5_"]
        key_topics = [t for t in result["topics"]
                     if any(t["name"].startswith(p) or t["name"] == p for p in key_prefixes)]
        other_topics = [t for t in result["topics"] if t not in key_topics]
        result["key_topics"] = key_topics
        result["other_topics_summary"] = {
            "count": len(other_topics),
            "names": [t["name"] for t in other_topics],
        }
        del result["topics"]
    return json.dumps(result, indent=2)


@tool
def get_topic_statistics(
    bag_path: str,
    topic_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    window_size: Optional[float] = None,
) -> str:
    """Compute descriptive statistics (mean, std, min, max) for all numeric fields in a topic.

    This is the primary diagnostic tool. Use it to:
    - Detect frozen sensors (std_dev = 0.0 on a sensor that should have noise)
    - Find anomalous values (unexpected min/max)
    - Compare commanded vs actual velocities

    Use window_size for time-series analysis (e.g., window_size=10.0 for 10-second windows).

    Args:
        bag_path: Path to the .bag file
        topic_name: Name of the ROS topic (e.g., "/odom", "/cmd_vel")
        start_time: Start of time range (unix timestamp). Defaults to bag start.
        end_time: End of time range (unix timestamp). Defaults to bag end.
        window_size: If provided, divide range into windows of this size (seconds).

    Returns:
        JSON list of statistical summaries per window with mean/std/min/max for each field
    """
    result = _bridge.get_topic_statistics(bag_path, topic_name, start_time, end_time, window_size)
    return json.dumps(result, indent=2)


@tool
def check_topic_frequency(
    bag_path: str,
    topic_name: str,
    resolution: float = 1.0,
) -> str:
    """Check the publishing frequency of a topic over time.

    Use this to detect silent failures like:
    - Sensor driver crashes (frequency drops to 0)
    - Network packet loss (intermittent frequency drops)
    - Sensor going offline then recovering

    Args:
        bag_path: Path to the .bag file
        topic_name: Name of the ROS topic
        resolution: Time bin size in seconds (default 1.0)

    Returns:
        JSON with frequency time-series and summary statistics (mean, std, min, max Hz)
    """
    result = _bridge.check_topic_frequency(bag_path, topic_name, resolution)
    # Truncate the series if too long
    series = result.get("frequency_series", [])
    if len(series) > 60:
        # Return summary + notable points (drops)
        mean_hz = result.get("mean_hz", 0)
        threshold = mean_hz * 0.5  # Flag anything below 50% of mean
        anomalous = [s for s in series if s["hz"] < threshold]
        result["anomalous_bins"] = anomalous[:20]
        result["frequency_series_first_10"] = series[:10]
        result["frequency_series_last_10"] = series[-10:]
        result["total_bins"] = len(series)
        del result["frequency_series"]
    return json.dumps(result, indent=2)


@tool
def sample_messages(
    bag_path: str,
    topic_name: str,
    timestamp: Optional[float] = None,
    count: int = 5,
) -> str:
    """Get raw message data from a topic for qualitative inspection.

    Use this to:
    - Inspect actual message content at specific timestamps
    - Check string/enum fields that statistics can't capture
    - Verify specific values around anomalous time periods

    Args:
        bag_path: Path to the .bag file
        topic_name: Name of the ROS topic
        timestamp: If provided, returns messages closest to this time
        count: Number of messages to return (default 5)

    Returns:
        JSON with raw message data
    """
    result = _bridge.sample_messages(bag_path, topic_name, timestamp, count)
    return json.dumps(result, indent=2, default=str)


# List of all tools for binding to the LLM
ALL_TOOLS = [get_bag_metadata, get_topic_statistics, check_topic_frequency, sample_messages]
