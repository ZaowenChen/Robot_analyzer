"""
Truncated Bag Support — sequential reader for ROS1 bags with missing indices.

When a bag's recording is interrupted (e.g., robot power-off), the index section
at the end is missing or incomplete.  This module provides `TruncatedBagReader`
which scans the file sequentially, and `open_bag()` which auto-detects and falls
back to sequential reading when needed.
"""

import os
import struct
from collections import defaultdict
from typing import Dict, List

import bz2
import lz4.frame

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
                                        md5sum=rec_md5 or '',
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
                md5sum=conn.md5sum,
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
