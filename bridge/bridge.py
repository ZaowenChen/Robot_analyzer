"""
ROSBag Bridge Class — the 4-tool API for the LLM diagnostic agent.

Provides:
  1. get_bag_metadata    - Initial reconnaissance of bag contents
  2. get_topic_statistics - Statistical summarization with windowing
  3. check_topic_frequency - Detect silent failures / dropouts
  4. sample_messages      - Raw message inspection
"""

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rosbags.serde import deserialize_cdr, ros1_to_cdr

from bridge.truncated_reader import TruncatedBagReader, _is_bag_truncated, open_bag
from bridge.welford import WelfordAccumulator
from bridge.field_extractors import FIELD_EXTRACTORS, extract_fields, msg_to_dict
from core.utils import LOG_LEVELS


class ROSBagBridge:
    """
    The Bridge / MCP Server for ROSBag data.
    Provides four tools for the LLM diagnostic agent.
    """

    def __init__(self):
        self._type_registry = set()
        # Types that are known to fail deserialization (custom types)
        self._skip_types = set()
        # Cache for open readers -- avoids re-scanning truncated bags on
        # every tool call.  Keyed by the *resolved* absolute path so that
        # relative and absolute references hit the same entry.
        self._reader_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Reader cache helpers
    # ------------------------------------------------------------------
    def _get_reader(self, bag_path: str):
        """Return a (possibly cached) reader for *bag_path*.

        For ``TruncatedBagReader`` instances the expensive byte-scan only
        happens once; subsequent calls reuse the cached object.  Normal
        ``Reader`` instances from *rosbags* are also cached so the index
        parsing is not repeated.
        """
        key = os.path.realpath(bag_path)
        if key not in self._reader_cache:
            reader = open_bag(bag_path)
            self._reader_cache[key] = reader
        return self._reader_cache[key]

    class _CachedReaderCtx:
        """Thin context-manager wrapper that does NOT close the reader."""
        def __init__(self, reader):
            self.reader = reader
        def __enter__(self):
            return self.reader
        def __exit__(self, *args):
            pass  # keep alive in cache -- do not close
        # Forward attribute access so callers can use the wrapper directly
        def __getattr__(self, name):
            return getattr(self.reader, name)

    def _open_cached(self, bag_path: str):
        """Return a context-manager over the cached reader.

        Unlike ``open_bag()`` this will NOT close the reader on exit so
        it stays available for the next tool call.
        """
        return self._CachedReaderCtx(self._get_reader(bag_path))

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
        with self._open_cached(bag_path) as reader:
            duration = reader.duration / 1e9  # nanoseconds to seconds
            start_time = reader.start_time / 1e9
            end_time = reader.end_time / 1e9

            topics = []
            topic_items = reader.topics
            if isinstance(topic_items, dict):
                # TruncatedBagReader returns a plain dict
                for name in sorted(topic_items.keys()):
                    info = topic_items[name]
                    msgcount = info['msgcount'] if isinstance(info, dict) else info.msgcount
                    msgtype = info['msgtype'] if isinstance(info, dict) else info.msgtype
                    freq = msgcount / duration if duration > 0 else 0
                    topics.append({
                        "name": name,
                        "type": msgtype,
                        "message_count": msgcount,
                        "frequency": round(freq, 2),
                    })
            else:
                for name, topic in sorted(topic_items.items()):
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
        with self._open_cached(bag_path) as reader:
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
        with self._open_cached(bag_path) as reader:
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
        with self._open_cached(bag_path) as reader:
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
