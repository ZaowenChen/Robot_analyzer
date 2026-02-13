"""
ROSBag Bridge - The Compute Layer / MCP Server Equivalent

This module implements the four core tools described in the project plan:
  1. get_bag_metadata    - Initial reconnaissance of bag contents
  2. get_topic_statistics - Statistical summarization with windowing
  3. check_topic_frequency - Detect silent failures / dropouts
  4. sample_messages      - Raw message inspection

It uses Welford's online algorithm for single-pass variance computation
(O(1) memory) and sliding window aggregation for hierarchical zooming.

Supports truncated ROS1 bags (bags whose recording was interrupted before
the index section was written) via sequential chunk scanning.
"""

import json
import math
import os
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.rosbag1.reader import (
    ReaderError,
    Header,
    RecordType,
    Connection,
    ConnectionExtRosbag1,
    IndexData,
    normalize_msgtype,
    read_bytes,
    read_uint32,
)
from rosbags.serde import deserialize_cdr, ros1_to_cdr


# ---------------------------------------------------------------------------
# Truncated Bag Support
# ---------------------------------------------------------------------------

def _is_bag_truncated(bag_path: str) -> bool:
    """Check if a ROS1 bag file is truncated (index beyond file end)."""
    with open(bag_path, 'rb') as f:
        magic = f.readline().decode()
        if '#ROSBAG V2.0' not in magic:
            return False
        header = Header.read(f, RecordType.BAGHEADER)
        index_pos = header.get_uint64('index_pos')
        f.seek(0, 2)
        file_size = f.tell()
        return index_pos == 0 or index_pos >= file_size


class TruncatedBagReader:
    """
    Sequential reader for truncated ROS1 bags.

    When a bag's recording is interrupted (e.g., robot power-off), the
    index section at the end is missing or incomplete.  This reader scans
    the file sequentially, extracting connections and messages from the
    chunk payloads without needing the index.
    """

    def __init__(self, path: str):
        self.path = path
        self.connections: List[Connection] = []
        self.topics: dict = {}
        self.start_time: int = 0
        self.end_time: int = 0
        self.duration: int = 0
        self.message_count: int = 0
        self._conn_map: Dict[int, Connection] = {}

    def open(self) -> 'TruncatedBagReader':
        """Scan the bag sequentially to build connection list and time range."""
        import bz2
        import lz4.frame

        conn_set: Dict[int, Connection] = {}
        min_ts = float('inf')
        max_ts = 0
        total_msgs = 0
        topic_msg_counts: Dict[str, int] = defaultdict(int)

        with open(self.path, 'rb') as bio:
            # Read magic
            magic = bio.readline()
            if b'#ROSBAG V2.0' not in magic:
                raise ReaderError('Not a ROS1 bag v2.0 file')

            # Read bag header
            bag_header = Header.read(bio, RecordType.BAGHEADER)
            # Skip the padding after bag header
            data_len = read_uint32(bio)
            bio.read(data_len)

            # Sequentially scan records
            while True:
                pos = bio.tell()
                try:
                    header_len_data = bio.read(4)
                    if len(header_len_data) < 4:
                        break
                    header_len = struct.unpack('<I', header_len_data)[0]
                    header_data = bio.read(header_len)
                    if len(header_data) < header_len:
                        break
                    data_len_data = bio.read(4)
                    if len(data_len_data) < 4:
                        break
                    data_len = struct.unpack('<I', data_len_data)[0]
                except struct.error:
                    break

                # Parse the op code from header
                op = None
                offset = 0
                while offset < len(header_data):
                    if offset + 4 > len(header_data):
                        break
                    field_len = struct.unpack('<I', header_data[offset:offset + 4])[0]
                    offset += 4
                    if offset + field_len > len(header_data):
                        break
                    field_data = header_data[offset:offset + field_len]
                    offset += field_len
                    if b'=' in field_data:
                        key, val = field_data.split(b'=', 1)
                        if key == b'op':
                            op = struct.unpack('B', val)[0]
                            break

                if op == 0x05:  # Chunk record
                    # Read compression type and size from header
                    compression = None
                    chunk_size = 0
                    offset = 0
                    while offset < len(header_data):
                        if offset + 4 > len(header_data):
                            break
                        field_len = struct.unpack('<I', header_data[offset:offset + 4])[0]
                        offset += 4
                        if offset + field_len > len(header_data):
                            break
                        field_data = header_data[offset:offset + field_len]
                        offset += field_len
                        if b'=' in field_data:
                            key, val = field_data.split(b'=', 1)
                            if key == b'compression':
                                compression = val.decode()
                            elif key == b'size':
                                chunk_size = struct.unpack('<I', val)[0]

                    # Read chunk data
                    chunk_data = bio.read(data_len)
                    if len(chunk_data) < data_len:
                        break

                    # Decompress
                    try:
                        if compression == 'lz4':
                            chunk_data = lz4.frame.decompress(chunk_data)
                        elif compression == 'bz2':
                            chunk_data = bz2.decompress(chunk_data)
                        # 'none' means no compression
                    except Exception:
                        continue

                    # Parse records inside the chunk
                    coff = 0
                    while coff < len(chunk_data):
                        if coff + 4 > len(chunk_data):
                            break
                        rec_header_len = struct.unpack('<I', chunk_data[coff:coff + 4])[0]
                        coff += 4
                        if coff + rec_header_len > len(chunk_data):
                            break
                        rec_header_data = chunk_data[coff:coff + rec_header_len]
                        coff += rec_header_len
                        if coff + 4 > len(chunk_data):
                            break
                        rec_data_len = struct.unpack('<I', chunk_data[coff:coff + 4])[0]
                        coff += 4
                        if coff + rec_data_len > len(chunk_data):
                            break
                        rec_data = chunk_data[coff:coff + rec_data_len]
                        coff += rec_data_len

                        # Parse record header fields
                        rec_op = None
                        rec_conn = None
                        rec_time = None
                        rec_topic = None

                        roff = 0
                        while roff < len(rec_header_data):
                            if roff + 4 > len(rec_header_data):
                                break
                            flen = struct.unpack('<I', rec_header_data[roff:roff + 4])[0]
                            roff += 4
                            if roff + flen > len(rec_header_data):
                                break
                            fdata = rec_header_data[roff:roff + flen]
                            roff += flen
                            if b'=' in fdata:
                                k, v = fdata.split(b'=', 1)
                                if k == b'op':
                                    rec_op = struct.unpack('B', v)[0]
                                elif k == b'conn':
                                    rec_conn = struct.unpack('<I', v)[0]
                                elif k == b'time':
                                    rec_time = struct.unpack('<II', v)
                                elif k == b'topic':
                                    rec_topic = v.decode()

                        if rec_op == 0x07 and rec_conn is not None:
                            # Connection record: topic is in header,
                            # but type/md5/msgdef are in DATA section
                            if rec_conn not in conn_set and rec_topic:
                                # Parse the data section for connection info
                                rec_msgtype = None
                                rec_md5 = None
                                rec_msgdef = None
                                rec_callerid = None
                                rec_latching = None

                                doff = 0
                                while doff < len(rec_data):
                                    if doff + 4 > len(rec_data):
                                        break
                                    dflen = struct.unpack('<I', rec_data[doff:doff + 4])[0]
                                    doff += 4
                                    if doff + dflen > len(rec_data):
                                        break
                                    dfield = rec_data[doff:doff + dflen]
                                    doff += dflen
                                    if b'=' in dfield:
                                        dk, dv = dfield.split(b'=', 1)
                                        if dk == b'type':
                                            rec_msgtype = dv.decode()
                                        elif dk == b'md5sum':
                                            rec_md5 = dv.decode()
                                        elif dk == b'message_definition':
                                            rec_msgdef = dv.decode()
                                        elif dk == b'callerid':
                                            rec_callerid = dv.decode()
                                        elif dk == b'latching':
                                            rec_latching = dv.decode()

                                if rec_msgtype:
                                    # Normalize msgtype from ROS1 to rosbags format
                                    normalized_type = rec_msgtype.replace('/', '/msg/', 1) if '/msg/' not in rec_msgtype else rec_msgtype
                                    ext = ConnectionExtRosbag1(
                                        callerid=rec_callerid or '',
                                        latching=int(rec_latching == '1') if rec_latching else 0,
                                    )
                                    conn = Connection(
                                        id=rec_conn,
                                        topic=rec_topic,
                                        msgtype=normalized_type,
                                        msgdef=rec_msgdef or '',
                                        digest=rec_md5 or '',
                                        msgcount=0,
                                        ext=ext,
                                        owner=None,
                                    )
                                    conn_set[rec_conn] = conn

                        elif rec_op == 0x02 and rec_conn is not None and rec_time is not None:
                            # Message data record
                            # rec_time = (secs, nsecs) in little-endian
                            ts_ns = rec_time[0] * 1_000_000_000 + rec_time[1]
                            total_msgs += 1
                            if ts_ns < min_ts:
                                min_ts = ts_ns
                            if ts_ns > max_ts:
                                max_ts = ts_ns
                            if rec_conn in conn_set:
                                topic_msg_counts[conn_set[rec_conn].topic] += 1

                elif op == 0x06:  # Index Data - skip
                    bio.read(data_len)
                elif op == 0x07:  # Connection - outside chunk, skip
                    bio.read(data_len)
                elif op == 0x03:  # Chunk Info - skip
                    bio.read(data_len)
                else:
                    # Unknown or other record type - skip data
                    skip_data = bio.read(data_len)
                    if len(skip_data) < data_len:
                        break

        # Update connection message counts
        for conn_id, conn in conn_set.items():
            topic = conn.topic
            count = topic_msg_counts.get(topic, 0)
            conn_set[conn_id] = Connection(
                id=conn.id,
                topic=conn.topic,
                msgtype=conn.msgtype,
                msgdef=conn.msgdef,
                digest=conn.digest,
                msgcount=count,
                ext=conn.ext,
                owner=None,
            )

        self._conn_map = conn_set
        self.connections = list(conn_set.values())
        self.start_time = min_ts if min_ts != float('inf') else 0
        self.end_time = max_ts
        self.duration = max_ts - min_ts if max_ts > min_ts else 0
        self.message_count = total_msgs

        # Build topics dict
        topic_info: Dict[str, dict] = {}
        for conn in self.connections:
            if conn.topic not in topic_info:
                topic_info[conn.topic] = {
                    'msgtype': conn.msgtype,
                    'msgcount': conn.msgcount,
                }
            else:
                topic_info[conn.topic]['msgcount'] += conn.msgcount
        self.topics = topic_info

        return self

    def messages(self, connections=None):
        """
        Yield (connection, timestamp_ns, rawdata) for each message.

        If connections is provided, only yield messages for those connections.
        """
        import bz2
        import lz4.frame

        if connections is not None:
            conn_ids = {c.id for c in connections}
        else:
            conn_ids = None

        with open(self.path, 'rb') as bio:
            # Skip magic
            bio.readline()
            # Skip bag header
            Header.read(bio, RecordType.BAGHEADER)
            data_len = read_uint32(bio)
            bio.read(data_len)

            while True:
                try:
                    header_len_data = bio.read(4)
                    if len(header_len_data) < 4:
                        break
                    header_len = struct.unpack('<I', header_len_data)[0]
                    header_data = bio.read(header_len)
                    if len(header_data) < header_len:
                        break
                    data_len_data = bio.read(4)
                    if len(data_len_data) < 4:
                        break
                    data_len = struct.unpack('<I', data_len_data)[0]
                except struct.error:
                    break

                # Parse op from header
                op = None
                offset = 0
                while offset < len(header_data):
                    if offset + 4 > len(header_data):
                        break
                    field_len = struct.unpack('<I', header_data[offset:offset + 4])[0]
                    offset += 4
                    if offset + field_len > len(header_data):
                        break
                    field_data = header_data[offset:offset + field_len]
                    offset += field_len
                    if b'=' in field_data:
                        key, val = field_data.split(b'=', 1)
                        if key == b'op':
                            op = struct.unpack('B', val)[0]
                            break

                if op != 0x05:  # Not a chunk
                    skip_data = bio.read(data_len)
                    if len(skip_data) < data_len:
                        break
                    continue

                # Parse chunk header for compression
                compression = None
                offset = 0
                while offset < len(header_data):
                    if offset + 4 > len(header_data):
                        break
                    field_len = struct.unpack('<I', header_data[offset:offset + 4])[0]
                    offset += 4
                    if offset + field_len > len(header_data):
                        break
                    field_data = header_data[offset:offset + field_len]
                    offset += field_len
                    if b'=' in field_data:
                        key, val = field_data.split(b'=', 1)
                        if key == b'compression':
                            compression = val.decode()

                chunk_data = bio.read(data_len)
                if len(chunk_data) < data_len:
                    break

                try:
                    if compression == 'lz4':
                        chunk_data = lz4.frame.decompress(chunk_data)
                    elif compression == 'bz2':
                        import bz2
                        chunk_data = bz2.decompress(chunk_data)
                except Exception:
                    continue

                # Parse records inside chunk
                coff = 0
                while coff < len(chunk_data):
                    if coff + 4 > len(chunk_data):
                        break
                    rec_header_len = struct.unpack('<I', chunk_data[coff:coff + 4])[0]
                    coff += 4
                    if coff + rec_header_len > len(chunk_data):
                        break
                    rec_header_data = chunk_data[coff:coff + rec_header_len]
                    coff += rec_header_len
                    if coff + 4 > len(chunk_data):
                        break
                    rec_data_len = struct.unpack('<I', chunk_data[coff:coff + 4])[0]
                    coff += 4
                    if coff + rec_data_len > len(chunk_data):
                        break
                    rec_data = chunk_data[coff:coff + rec_data_len]
                    coff += rec_data_len

                    # Parse header fields
                    rec_op = None
                    rec_conn = None
                    rec_time = None
                    roff = 0
                    while roff < len(rec_header_data):
                        if roff + 4 > len(rec_header_data):
                            break
                        flen = struct.unpack('<I', rec_header_data[roff:roff + 4])[0]
                        roff += 4
                        if roff + flen > len(rec_header_data):
                            break
                        fdata = rec_header_data[roff:roff + flen]
                        roff += flen
                        if b'=' in fdata:
                            k, v = fdata.split(b'=', 1)
                            if k == b'op':
                                rec_op = struct.unpack('B', v)[0]
                            elif k == b'conn':
                                rec_conn = struct.unpack('<I', v)[0]
                            elif k == b'time':
                                rec_time = struct.unpack('<II', v)

                    if rec_op == 0x02 and rec_conn is not None and rec_time is not None:
                        if conn_ids is not None and rec_conn not in conn_ids:
                            continue
                        # rec_time = (secs, nsecs) in little-endian
                        ts_ns = rec_time[0] * 1_000_000_000 + rec_time[1]
                        conn = self._conn_map.get(rec_conn)
                        if conn is not None:
                            yield conn, ts_ns, rec_data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_bag(bag_path: str):
    """
    Open a ROS1 bag, falling back to sequential reading for truncated bags.

    Returns a context manager that provides:
      - .connections, .topics, .start_time, .end_time, .duration, .message_count
      - .messages(connections=...) iterator
    """
    try:
        reader = Reader(bag_path)
        reader.open()
        return reader
    except (ReaderError, Exception) as e:
        err_msg = str(e)
        if 'damaged' in err_msg or 'reindex' in err_msg or 'Header could not be read' in err_msg:
            print(f"  [INFO] Bag index damaged/missing, using sequential reader for: {os.path.basename(bag_path)}")
            reader = TruncatedBagReader(bag_path)
            reader.open()
            return reader
        raise


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


def _extract_diagnostic_status(msg) -> Dict[str, float]:
    """Extract numeric fields from a DiagnosticStatus message.

    DiagnosticStatus uses key-value pairs rather than typed fields.
    We extract:
      - ``level`` (0=OK, 1=WARN, 2=ERROR, 3=STALE)
      - Each key-value pair whose value is numeric → float
      - Each key-value pair whose value is "true"/"false" → 1.0/0.0
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
        # Cache for open readers – avoids re-scanning truncated bags on
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
            pass  # keep alive in cache – do not close
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
