"""
Shared data models — dataclasses used across multiple packages.

All robot-specific data structures live here to avoid circular imports
and ensure consistent serialization.
"""

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Log parsing models (originally in log_parser.py)
# ---------------------------------------------------------------------------

@dataclass
class LogEvent:
    """A single parsed log message from /rosout."""
    timestamp: float          # Unix epoch seconds
    timestamp_str: str        # "YYYY-MM-DD HH:MM:SS.mmm" (CST)
    node: str                 # ROS node name (e.g., "/gs_nav")
    level: int                # 1=DEBUG, 2=INFO, 4=WARN, 8=ERROR, 16=FATAL
    level_str: str            # "INFO", "WARN", etc.
    raw_message: str          # Original log text
    event_type: str           # Pattern name or "UNMATCHED"
    parsed_data: dict         # Regex group captures
    bag_name: str             # Source bag file

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "node": self.node,
            "level": self.level,
            "level_str": self.level_str,
            "raw_message": self.raw_message,
            "event_type": self.event_type,
            "parsed_data": self.parsed_data,
            "bag_name": self.bag_name,
        }


@dataclass
class StateTransition:
    """Records a change in robot state inferred from log messages."""
    timestamp: float
    timestamp_str: str
    state_key: str            # e.g., "motion_state"
    field: str                # e.g., "still_flag"
    old_value: str
    new_value: str
    bag_name: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "state_key": self.state_key,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "bag_name": self.bag_name,
        }


@dataclass
class LogPattern:
    """A regex pattern for classifying log messages."""
    name: str                 # Pattern identifier (e.g., "MOTION_STATE")
    regex: re.Pattern         # Compiled regex
    event_type: str           # Classification label
    state_key: Optional[str] = None   # State key for timeline tracking
    state_field: Optional[str] = None # Field name within the state key
    value_group: int = 1      # Regex group index for extracted value


# ---------------------------------------------------------------------------
# Cross-validator models (originally in cross_validator.py)
# ---------------------------------------------------------------------------

@dataclass
class DenoisedLogEvent:
    """A log event that survived noise filtering, with dedup metadata."""
    event: LogEvent
    dedup_key: Optional[str] = None   # None = unique, str = dedup group
    occurrence_count: int = 1          # How many times this repeated
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0


@dataclass
class SensorSnapshot:
    """Sensor state captured at a specific moment in time."""
    timestamp: float
    timestamp_str: str
    # topic_name -> {field_name -> {mean, std, min, max, count}}
    topic_stats: Dict[str, Dict[str, dict]]
    # topic_name -> robot state dict
    robot_state: Dict[str, str]
    # Which bag this came from
    bag_name: str

    def get_field(self, topic: str, field: str) -> Optional[dict]:
        """Get stats for a specific topic.field."""
        return self.topic_stats.get(topic, {}).get(field)

    def is_frozen(self, topic: str, field: str, threshold: float = 1e-6) -> Optional[bool]:
        """Check if a field appears frozen (std near zero)."""
        stats = self.get_field(topic, field)
        if stats is None or stats.get("count", 0) < 3:
            return None  # Not enough data
        return stats.get("std", 0) < threshold

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "topic_stats": self.topic_stats,
            "robot_state": self.robot_state,
            "bag_name": self.bag_name,
        }


@dataclass
class EvidencePacket:
    """
    A structured bundle pairing a log event with sensor context.

    This is the unit of analysis: "The log said X, the sensor showed Y,
    therefore the verdict is Z."
    """
    packet_id: str
    timestamp: float
    timestamp_str: str
    bag_name: str

    # The log event (denoised)
    log_event: dict                   # Simplified log event dict
    log_dedup_count: int = 1          # How many times this message repeated

    # The sensor snapshot at this moment
    sensor_snapshot: Optional[dict] = None

    # Robot state context
    robot_state: Dict[str, str] = dataclass_field(default_factory=dict)

    # Cross-validation results
    verdict: str = "UNCHECKED"        # CONFIRMED, CONTRADICTED, NO_SENSOR_DATA, UNCHECKED
    confidence: float = 0.0           # 0.0 to 1.0
    cross_validation_details: List[dict] = dataclass_field(default_factory=list)

    # Summary for LLM consumption
    summary: str = ""
    category: str = ""                # ERROR, STATE_CHANGE, ANOMALY, INFO
    severity: str = "INFO"            # CRITICAL, WARNING, INFO

    def to_dict(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "timestamp": self.timestamp_str,
            "bag_name": self.bag_name,
            "category": self.category,
            "severity": self.severity,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "log_event": self.log_event,
            "log_dedup_count": self.log_dedup_count,
            "sensor_snapshot": self.sensor_snapshot,
            "robot_state": self.robot_state,
            "cross_validation_details": self.cross_validation_details,
        }


# ---------------------------------------------------------------------------
# Analysis models (originally in analyze.py)
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """A meaningful diagnostic finding, clustering multiple raw anomalies."""
    incident_id: str = ""              # "INC-001"
    title: str = ""                    # Human-readable summary
    severity: str = "INFO"             # CRITICAL/WARNING/INFO
    category: str = ""                 # "IMU", "NAVIGATION", "SENSOR_FREEZE", etc.
    time_start: float = 0.0
    time_end: float = 0.0
    time_start_str: str = ""
    time_end_str: str = ""
    root_cause: str = ""               # Best-guess explanation
    log_evidence: List[dict] = dataclass_field(default_factory=list)
    sensor_evidence: List[dict] = dataclass_field(default_factory=list)
    state_context: dict = dataclass_field(default_factory=dict)
    recommended_actions: List[str] = dataclass_field(default_factory=list)
    bags: List[str] = dataclass_field(default_factory=list)
    raw_anomaly_count: int = 0
    is_cross_bag: bool = False
    suppressed: bool = False
    suppression_reason: str = ""

    def to_dict(self) -> dict:
        d = {
            "incident_id": self.incident_id,
            "title": self.title,
            "severity": self.severity,
            "category": self.category,
            "time_start": self.time_start_str,
            "time_end": self.time_end_str,
            "duration_sec": round(self.time_end - self.time_start, 1) if self.time_end > self.time_start else 0,
            "root_cause": self.root_cause,
            "log_evidence": self.log_evidence,
            "sensor_evidence": self.sensor_evidence,
            "state_context": self.state_context,
            "recommended_actions": self.recommended_actions,
            "bags": self.bags,
            "raw_anomaly_count": self.raw_anomaly_count,
            "is_cross_bag": self.is_cross_bag,
        }
        if self.suppressed:
            d["suppressed"] = True
            d["suppression_reason"] = self.suppression_reason
        return d


@dataclass
class BagInfo:
    """Lightweight metadata for sorting/grouping bags."""
    path: str
    name: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class Mission:
    """A group of temporally contiguous bags from one recording session."""
    mission_id: int
    bags: List[BagInfo]
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self):
        if self.bags:
            self.start_time = self.bags[0].start_time
            self.end_time = self.bags[-1].end_time
