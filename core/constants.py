"""
Gaussian robot domain knowledge — the single source of truth.

All robot-specific constants consolidated from analyze.py, rosbag_profiler.py,
and cross_validator.py. Previously duplicated across 2-3 files each.
"""

# Fields that are expected to be zero on a 2D differential-drive robot.
# Used for anomaly detection: nonzero values in these fields indicate faults.
EXPECTED_ZERO_FIELDS = {
    "/odom": {"orientation_x", "orientation_y", "position_z",
              "twist_angular_x", "twist_angular_y",
              "twist_linear_y", "twist_linear_z"},
    "/chassis_cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
    "/cmd_vel": {"angular_x", "angular_y", "linear_y", "linear_z"},
    # DiagnosticStatus topics: 'level' is expected to be 0 (OK) when healthy
    "/device/health_status": {"level"},
    "/device/odom_status": {"level", "unicycle_angle_offset"},
    "/device/imu_data": {"level", "stamp_sec", "stamp_nsec"},
    "/device/scrubber_status": {"level"},
    "/device/scrubber_motor_limit": {"level"},
    # Pose topics: Z and roll/pitch expected zero on flat ground
    "/localization/current_pose": {"position_z", "orientation_x", "orientation_y"},
    "/front_end_pose": {"position_z", "orientation_x", "orientation_y"},
}

# Hardware topics for Gaussian robot diagnostics
HARDWARE_TOPICS = [
    "/device/health_status",
    "/device/odom_status",
    "/device/imu_data",
    "/device/scrubber_status",
    "/device/scrubber_motor_limit",
    "/raw_scan",
    "/scan_rear",
    "/ir_sticker3", "/ir_sticker6", "/ir_sticker7",
    "/protector",
    "/localization/status",
    "/navigation/status",
]

# Health flags in /device/health_status where "false" (0.0) indicates a fault.
HEALTH_FLAG_NAMES = {
    "rear_rolling_brush_motor", "front_rolling_brush_motor",
    "imu_board", "ultrasonic_board", "motor_driver",
    "battery_disconnection", "mcu_disconnection", "mcu_delay",
    "laser_disconnection", "router_disconnection", "tablet_disconnection",
    "odom_left_delta", "odom_right_delta", "odom_delta_speed", "odom_track_delta",
    "motor_driver_emergency", "imu_roll_pitch_abnormal", "imu_overturn",
}

# Odom/cmd_vel twist fields that freeze when robot is stationary.
# Used for state-aware suppression of false positives.
MOTION_RELATED_FIELDS = {
    ("/odom", "twist_linear_x"),
    ("/odom", "twist_angular_z"),
    ("/chassis_cmd_vel", "linear_x"),
    ("/chassis_cmd_vel", "angular_z"),
    ("/cmd_vel", "linear_x"),
    ("/cmd_vel", "angular_z"),
}

# Topics to pull sensor snapshots from (fast-changing, high diagnostic value)
SENSOR_SNAPSHOT_TOPICS = [
    "/odom",
    "/chassis_cmd_vel",
    "/cmd_vel",
    "/unbiased_imu_PRY",
    "/localization/current_pose",
    "/front_end_pose",
    "/device/health_status",
    "/device/imu_data",
    "/device/odom_status",
    "/device/scrubber_status",
]

# Topics worth checking for sensor-level anomalies (profiler)
KEY_SENSOR_TOPICS = [
    "/odom", "/chassis_cmd_vel", "/cmd_vel",
    "/unbiased_imu_PRY", "/localization/current_pose",
]

# DiagnosticStatus level mapping
DIAG_LEVELS = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}


# ---------------------------------------------------------------------------
# Timing thresholds
# ---------------------------------------------------------------------------

# Maximum gap (seconds) between end of one bag and start of next
# to consider them part of the same mission
MISSION_GAP_THRESHOLD = 300  # 5 minutes

# Incident clustering: temporal window for grouping events
INCIDENT_TEMPORAL_BIN_SEC = 10.0

# Causal chain: max time between log event and sensor anomaly
CAUSAL_CHAIN_WINDOW_SEC = 5.0

# Correlation window: how close (in seconds) a log event and sensor anomaly
# must be to be considered related
CORRELATION_WINDOW_SEC = 2.0

# Sensor snapshot window: time range around a log event to sample sensor stats
SNAPSHOT_HALF_WINDOW_SEC = 1.0
