"""
Pattern Registry — Gaussian/Gausium Robot Log Patterns

Contains all regex patterns used to classify /rosout log messages.
Each pattern maps a log message to an event_type and optionally to
a state_key for timeline tracking.
"""

import re
from typing import List, Optional

from core.models import LogPattern


# ---------------------------------------------------------------------------
# Pattern Registry — Gaussian/Gausium Robot Log Patterns
# ---------------------------------------------------------------------------

PATTERN_REGISTRY: List[LogPattern] = []


def register_pattern(name: str, pattern: str, event_type: str,
                     state_key: Optional[str] = None,
                     state_field: Optional[str] = None,
                     value_group: int = 1,
                     flags: int = re.IGNORECASE) -> LogPattern:
    """Register a new log pattern. Returns the created LogPattern."""
    lp = LogPattern(
        name=name,
        regex=re.compile(pattern, flags),
        event_type=event_type,
        state_key=state_key,
        state_field=state_field,
        value_group=value_group,
    )
    PATTERN_REGISTRY.append(lp)
    return lp


# --- 1. Motion State ---
register_pattern(
    "MOTION_STATE",
    r"still_flag[:\s=]*(\d)",
    "motion_state_change",
    state_key="motion_state",
    state_field="still_flag",
    value_group=1,
)

# --- 2. Navigation Stuck/Deadlock ---
register_pattern(
    "NAV_STUCK",
    r"(navigation|nav)[^.]{0,50}?(stuck|deadlock|timeout|abort|fail)",
    "nav_stuck",
    state_key="nav_state",
    state_field="stuck",
    value_group=2,
)

# --- 3. IMU Calibration ---
register_pattern(
    "IMU_CALIBRATION",
    r"imu[^.]{0,50}?(calibrat\w*|bias|offset|compensat\w*)",
    "imu_calibration",
    state_key="imu_state",
    state_field="calibrating",
    value_group=1,
)

# --- 4. IMU Error ---
register_pattern(
    "IMU_ERROR",
    r"imu[^.]{0,50}?(error|fault|abnormal|reset)",
    "imu_error",
    value_group=1,
)

# --- 5. Point Cloud / LiDAR Error ---
register_pattern(
    "POINTCLOUD_ERROR",
    r"(point_?cloud|pcl|lidar)[^.]{0,50}?(error|fail\w*|timeout|lost)",
    "pointcloud_error",
    value_group=2,
)

# --- 6. Safety State ---
register_pattern(
    "SAFETY_STATE",
    r"(protector|bumper|emergency)[^.]{0,50}?(trigger\w*|activat\w*|stop\w*)",
    "safety_event",
    state_key="safety_state",
    state_field="active",
    value_group=2,
)

# --- 7. IR Sticker Events ---
register_pattern(
    "IR_STICKER_STATE",
    r"ir_sticker[^.]{0,50}?(trigger\w*|detect\w*|block\w*)",
    "ir_sticker_event",
    value_group=1,
)

# --- 8. Localization State ---
register_pattern(
    "LOCATION_STATE",
    r"(locali[sz]ation|loc_state)[^.]{0,50}?(lost|recover\w*|init\w*|success|fail\w*)",
    "location_state_change",
    state_key="location_state",
    state_field="status",
    value_group=2,
)

# --- 9. Battery State ---
register_pattern(
    "BATTERY_STATE",
    r"batter\w*[^.]{0,50}?(low|critical|charg\w*|discharg\w*|level\s*[:=]\s*\d+)",
    "battery_event",
    state_key="battery_state",
    state_field="status",
    value_group=1,
)

# --- 10. Navigation Tracking ---
register_pattern(
    "NAV_TRACKING",
    r"(path_follow\w*|tracking|waypoint)[^.]{0,50}?(start\w*|stop\w*|complet\w*|fail\w*)",
    "nav_tracking",
    state_key="nav_state",
    state_field="tracking",
    value_group=2,
)

# --- 11. Position Report ---
register_pattern(
    "POSITION_REPORT",
    r"pos\w*[:\s=]*\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)",
    "position_report",
    value_group=1,
)

# --- 12. Obstacle Detection ---
register_pattern(
    "OBSTACLE_DETECT",
    r"obstacle[^.]{0,50}?(detect\w*|avoid\w*|clear\w*|block\w*)",
    "obstacle_event",
    value_group=1,
)

# --- 13. Virtual Obstacle Wait ---
register_pattern(
    "VIRTUAL_OBSTACLE_WAIT",
    r"(virtual[^.]{0,30}?obstacle|waiting[^.]{0,30}?obstacle|obstacle[^.]{0,30}?wait)",
    "virtual_obstacle_wait",
    value_group=1,
)

# --- 14. Depth Camera / Vision Pipeline ---
register_pattern(
    "DEPTH_CAMERA_ERROR",
    r"\[overdark\].*Time:\s*([\d.]+)",
    "depth_camera_error",
    value_group=1,
)

# --- 15. Depth Camera Fusion / TF Failure ---
register_pattern(
    "DEPTHCAM_FUSION_FAIL",
    r"(depthcam_fusion|depth_fusion)[^.]{0,80}?(Tf tranform failed|not localized|is_localized:\s*0)",
    "depthcam_fusion_failure",
    state_key="depthcam_state",
    state_field="status",
    value_group=2,
)

# --- 16. DL Inference / Vision Pipeline ---
register_pattern(
    "DL_INFER_EVENT",
    r"dl_infer[^.]{0,50}?(overdark|timeout|fail\w*|error|camera_process)",
    "dl_infer_event",
    value_group=1,
)

# --- 17. Localization Status Code ---
register_pattern(
    "LOCALIZATION_STATUS",
    r"localization_status\s+(\d+)",
    "localization_status_change",
    state_key="localization_state",
    state_field="status_code",
    value_group=1,
)

# --- 18. Front-End Mapping State ---
register_pattern(
    "MAPPING_STATE",
    r"front_end_state is (\w+)",
    "mapping_state_change",
    state_key="mapping_state",
    state_field="front_end_state",
    value_group=1,
)
