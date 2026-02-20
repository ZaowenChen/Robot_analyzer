"""
Foxglove Layout Generator — Auto-generate problem-specific Foxglove layouts

Takes diagnostic findings (from cross_validator.py or analyze.py) and generates
Foxglove Studio layout JSON files that technicians can import directly.

Each layout is tailored to the specific problem category detected, mirroring
the manual layouts your team already uses (运动控制, 电池, 异常停车, etc.)

Usage:
    from reporting.foxglove_layouts import generate_foxglove_layouts

    layouts = generate_foxglove_layouts(
        cross_validation_report="cross_validation_report.json",
        output_dir="./foxglove_layouts/",
    )
    # Returns: {"motion_control": "foxglove_layouts/motion_control.json", ...}
"""

import json
import os
import random
import string
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Panel ID generator (Foxglove uses "PanelType!random_id" format)
# ---------------------------------------------------------------------------

def _panel_id(panel_type: str) -> str:
    """Generate a Foxglove-style panel ID like 'Plot!3bgcrks'."""
    chars = string.ascii_lowercase + string.digits
    suffix = "".join(random.choices(chars, k=7))
    return f"{panel_type}!{suffix}"


# ---------------------------------------------------------------------------
# Panel builders — each returns (panel_id, config_dict)
# ---------------------------------------------------------------------------

def _rosout_panel(search_terms: List[str] = None,
                  name_filter: Dict[str, bool] = None) -> Tuple[str, dict]:
    """Build a RosOut log panel config."""
    pid = _panel_id("RosOut")
    config = {
        "searchTerms": search_terms or [],
        "minLogLevel": 1,
        "reverseOrder": False,
    }
    if name_filter:
        config["nameFilter"] = {k: {"visible": v} for k, v in name_filter.items()}
    return pid, config


def _raw_messages_panel(topic_path: str = "/device/all_device_status") -> Tuple[str, dict]:
    """Build a RawMessages panel for inspecting a specific topic."""
    pid = _panel_id("RawMessages")
    config = {
        "diffEnabled": False,
        "diffMethod": "custom",
        "diffTopicPath": "",
        "showFullMessageForDiff": False,
        "topicPath": topic_path,
        "fontSize": 12,
    }
    return pid, config


def _3d_panel(visible_topics: List[str] = None,
              follow_mode: str = "follow-pose") -> Tuple[str, dict]:
    """Build a 3D scene panel."""
    pid = _panel_id("3D")
    topics = {}
    if visible_topics:
        for t in visible_topics:
            topic_config = {"visible": True}
            # Point clouds get colormap rendering
            if any(kw in t for kw in ["cloud", "scan", "obstacle", "pointcloud"]):
                topic_config.update({
                    "colorField": "intensity",
                    "colorMode": "colormap",
                    "colorMap": "turbo",
                })
            # Camera depth images get frameLocked
            if "depth/image_raw" in t:
                topic_config.update({
                    "frameLocked": True,
                    "distance": 1,
                    "planarProjectionFactor": 0,
                    "color": "#ffffff",
                })
                # Auto-infer camera_info topic
                info_topic = t.replace("/image_raw", "/camera_info")
                topic_config["cameraInfoTopic"] = info_topic
            topics[t] = topic_config

    config = {
        "cameraState": {
            "perspective": True,
            "distance": 15.0,
            "phi": 0.00006,
            "thetaOffset": 17.5,
            "targetOffset": [0, 0, 0],
            "target": [0, 0, 0],
            "targetOrientation": [0, 0, 0, 1],
            "fovy": 45,
            "near": 0.5,
            "far": 5000,
        },
        "followMode": follow_mode,
        "scene": {"transforms": {"visible": False}},
        "transforms": {},
        "topics": topics,
        "layers": {
            "grid-layer": {
                "layerId": "foxglove.Grid",
                "instanceId": "grid-layer",
            }
        },
        "publish": {
            "type": "point",
            "poseTopic": "/move_base_simple/goal",
            "pointTopic": "/clicked_point",
            "poseEstimateTopic": "/initialpose",
            "poseEstimateXDeviation": 0.5,
            "poseEstimateYDeviation": 0.5,
            "poseEstimateThetaDeviation": 0.26179939,
        },
        "imageMode": {},
    }
    return pid, config


def _plot_panel(paths: List[dict], title: str = "",
                min_y: float = None, max_y: float = None) -> Tuple[str, dict]:
    """
    Build a Plot panel.

    paths: list of dicts with keys: value, label, color (optional)
    """
    pid = _panel_id("Plot")
    plot_paths = []
    default_colors = ["#0839f9", "#ef4107", "#16a34a", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"]
    for i, p in enumerate(paths):
        entry = {
            "value": p["value"],
            "enabled": True,
            "timestampMethod": "receiveTime",
        }
        if "label" in p:
            entry["label"] = p["label"]
        if "color" in p:
            entry["color"] = p["color"]
        else:
            entry["color"] = default_colors[i % len(default_colors)]
        plot_paths.append(entry)

    config = {
        "paths": plot_paths,
        "showXAxisLabels": True,
        "showYAxisLabels": True,
        "showLegend": True,
        "legendDisplay": "floating",
        "showPlotValuesInLegend": False,
        "isSynced": True,
        "xAxisVal": "timestamp",
        "sidebarDimension": 240,
    }
    if title:
        config["foxglovePanelTitle"] = title
    if min_y is not None:
        config["minYValue"] = min_y
    if max_y is not None:
        config["maxYValue"] = max_y
    return pid, config


# ---------------------------------------------------------------------------
# Layout tree builders
# ---------------------------------------------------------------------------

def _split(first, second, direction: str = "row",
           split_pct: float = 50.0) -> dict:
    """Build a Foxglove split layout node."""
    return {
        "first": first,
        "second": second,
        "direction": direction,
        "splitPercentage": split_pct,
    }


def _build_layout_json(config_by_id: dict, layout_tree) -> dict:
    """Assemble a complete Foxglove layout JSON."""
    return {
        "configById": config_by_id,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": layout_tree,
    }


# ---------------------------------------------------------------------------
# Problem-specific layout templates
# ---------------------------------------------------------------------------

def _layout_motion_control() -> dict:
    """运动控制 — Motion control debugging layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=["eco_decision"],
        name_filter={
            "/eco_decision": True,
            "/gs_console": True,
            "/eco_control": False,
            "/move_base": False,
            "/costmap_node": False,
            "/network_manager": False,
            "/depth_pipeline2": False,
            "/depth_pipeline3": False,
            "/depth_pipeline4": False,
            "/depth_pipeline5": False,
        }
    )
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/all_device_status")
    configs[raw_id] = raw_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/scan_rear", "/costmap_node/global_costmap/costmap",
            "/costmap_node/global_costmap/local_costmap",
            "/costmap_node/clean_footprint_master",
            "/localization/current_pose", "/front_end_pose",
            "/move_base/planner_track_path", "/move_base/planner_fix_path",
            "/eco_control/control_visual_path_topic",
            "/hdmap_server/map",
        ]
    )
    configs[scene_id] = scene_cfg

    # Linear velocity plot
    vel_id, vel_cfg = _plot_panel([
        {"value": "/cmd_vel.linear.x", "label": "cmd_vel (movebase)", "color": "#0923f3"},
        {"value": "/chassis_cmd_vel.linear.x", "label": "cmd_vel (chassis)", "color": "#ef4107"},
        {"value": "/odom.twist.twist.linear.x", "label": "odom feedback", "color": "#16a34a"},
    ], title="Linear Velocity")
    configs[vel_id] = vel_cfg

    # Angular velocity plot
    ang_id, ang_cfg = _plot_panel([
        {"value": "/cmd_vel.angular.z", "label": "cmd_vel (movebase)", "color": "#0839f9"},
        {"value": "/chassis_cmd_vel.angular.z", "label": "cmd_vel (chassis)", "color": "#ef4107"},
        {"value": "/odom.twist.twist.angular.z", "label": "odom feedback", "color": "#16a34a"},
    ], title="Angular Velocity")
    configs[ang_id] = ang_cfg

    # IMU roll/pitch
    imu_id, imu_cfg = _plot_panel([
        {"value": "/device/imu_data.values[:]{key==\"roll_angle\"}.value", "label": "roll"},
        {"value": "/device/imu_data.values[:]{key==\"pitch_angle\"}.value", "label": "pitch"},
    ], title="IMU Roll/Pitch")
    configs[imu_id] = imu_cfg

    # Device status (brake, auto_mode, emergency)
    dev_id, dev_cfg = _plot_panel([
        {"value": "/device/all_device_status.values[:]{key==\"braker_down\"}.value",
         "label": "Brake", "color": "#0d81f6"},
        {"value": "/device/all_device_status.values[:]{key==\"auto_mode\"}.value",
         "label": "Auto Mode", "color": "#ef4912"},
        {"value": "/device/all_device_status.values[:]{key==\"emergency\"}.value",
         "label": "E-Stop", "color": "#fcd31a"},
    ], title="Brake / Auto / E-Stop")
    configs[dev_id] = dev_cfg

    # Layout tree:
    # Left side: RosOut on top, RawMessages + 3D on bottom
    # Right side: 4 stacked plots
    left = _split(
        rosout_id,
        _split(raw_id, scene_id, "row", 20.0),
        "column", 30.0
    )
    right = _split(
        vel_id,
        _split(ang_id, _split(imu_id, dev_id, "column"), "column", 33.0),
        "column", 25.0
    )
    layout = _split(left, right, "row", 60.0)

    return _build_layout_json(configs, layout)


def _layout_battery() -> dict:
    """电池 — Battery diagnostics layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(search_terms=["battery", "charge"])
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/device_status")
    configs[raw_id] = raw_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/costmap_node/global_costmap/costmap",
            "/localization/current_pose", "/v5_follow_pose",
        ],
        follow_mode="follow-none",
    )
    configs[scene_id] = scene_cfg

    # Battery level
    bat_id, bat_cfg = _plot_panel([
        {"value": "/device/all_device_status.values[:]{key==\"battery\"}.value",
         "label": "Battery %", "color": "#0839f9"},
    ], title="Battery Level", min_y=0, max_y=100)
    configs[bat_id] = bat_cfg

    # Battery voltage
    volt_id, volt_cfg = _plot_panel([
        {"value": "/device/device_status.values[:]{key==\"battery_voltage\"}.value",
         "label": "Battery Voltage"},
    ], title="Battery Voltage")
    configs[volt_id] = volt_cfg

    # Cell voltages
    cell_paths = []
    for i in range(8):
        cell_paths.append({
            "value": f'/device/detailed_device_status.values[:]{{"key"=="battery_cell_voltage_{i}"}}.value',
            "label": f"Cell {i+1}",
        })
    cell_id, cell_cfg = _plot_panel(cell_paths, title="Cell Voltages")
    configs[cell_id] = cell_cfg

    # Cell temperatures
    temp_paths = []
    for i in range(4):
        temp_paths.append({
            "value": f'/device/detailed_device_status.values[:]{{"key"=="battery_cell_temperature_{i}"}}.value',
            "label": f"Cell {i+1}",
        })
    temp_id, temp_cfg = _plot_panel(temp_paths, title="Cell Temperatures")
    configs[temp_id] = temp_cfg

    left = _split(
        rosout_id,
        _split(raw_id, scene_id, "row", 27.0),
        "column", 40.0
    )
    right = _split(
        _split(bat_id, volt_id, "row"),
        _split(cell_id, temp_id, "column"),
        "column", 30.0
    )
    layout = _split(left, right, "row", 38.0)

    return _build_layout_json(configs, layout)


def _layout_abnormal_stop() -> dict:
    """异常停车 — Abnormal stop analysis layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=[],
        name_filter={
            "/eco_decision": True,
            "/eco_control": False,
            "/move_base": False,
            "/costmap_node": False,
            "/network_manager": False,
            "/depth_pipeline2": False,
            "/depth_pipeline3": False,
            "/depth_pipeline4": False,
            "/depth_pipeline5": False,
        }
    )
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/all_device_status")
    configs[raw_id] = raw_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/scan_rear", "/foreground_points",
            "/costmap_node/global_costmap/costmap",
            "/costmap_node/global_costmap/local_costmap",
            "/costmap_node/clean_footprint_master",
            "/localization/current_pose", "/front_end_pose",
            "/move_base/planner_track_path", "/move_base/planner_fix_path",
            "/eco_control/control_visual_path_topic",
            "/eco_control/dwa_concontroller/dwa_plan",
            "/hdmap_server/map",
            "/pointcloud1", "/pointcloud2", "/pointcloud3",
            "/pointcloud4", "/pointcloud5", "/pointcloud6",
            "/convinced_obstacles", "/last_ring_obstacles",
        ]
    )
    configs[scene_id] = scene_cfg

    # Velocity plot
    vel_id, vel_cfg = _plot_panel([
        {"value": "/cmd_vel.linear.x", "label": "cmd_vel linear.x", "color": "#0923f3"},
        {"value": "/chassis_cmd_vel.linear.x", "label": "chassis linear.x", "color": "#ef4107"},
        {"value": "/odom.twist.twist.linear.x", "label": "odom linear.x", "color": "#16a34a"},
    ], title="Velocity at Stop")
    configs[vel_id] = vel_cfg

    # Device status
    dev_id, dev_cfg = _plot_panel([
        {"value": "/device/all_device_status.values[:]{key==\"braker_down\"}.value",
         "label": "Brake", "color": "#0d81f6"},
        {"value": "/device/all_device_status.values[:]{key==\"auto_mode\"}.value",
         "label": "Auto Mode", "color": "#ef4912"},
        {"value": "/device/all_device_status.values[:]{key==\"emergency\"}.value",
         "label": "E-Stop", "color": "#fcd31a"},
    ], title="Brake / Auto / E-Stop")
    configs[dev_id] = dev_cfg

    left = _split(
        _split(rosout_id, raw_id, "column", 50.0),
        scene_id,
        "row", 30.0
    )
    right = _split(vel_id, dev_id, "column")
    layout = _split(left, right, "column", 60.0)

    return _build_layout_json(configs, layout)


def _layout_imu() -> dict:
    """IMU — IMU diagnostics layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(search_terms=["imu", "calibrat"])
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/imu_data")
    configs[raw_id] = raw_cfg

    # IMU angles
    angle_id, angle_cfg = _plot_panel([
        {"value": "/device/imu_data.values[:]{key==\"roll_angle\"}.value", "label": "Roll"},
        {"value": "/device/imu_data.values[:]{key==\"pitch_angle\"}.value", "label": "Pitch"},
        {"value": "/device/imu_data.values[:]{key==\"yaw_angle\"}.value", "label": "Yaw"},
    ], title="IMU Euler Angles")
    configs[angle_id] = angle_cfg

    # IMU angular velocity
    gyro_id, gyro_cfg = _plot_panel([
        {"value": "/unbiased_imu_PRY.x", "label": "Gyro X (pitch)"},
        {"value": "/unbiased_imu_PRY.y", "label": "Gyro Y (roll)"},
        {"value": "/unbiased_imu_PRY.z", "label": "Gyro Z (yaw)"},
    ], title="IMU Gyroscope (unbiased)")
    configs[gyro_id] = gyro_cfg

    # IMU health
    health_id, health_cfg = _plot_panel([
        {"value": "/device/health_status.values[:]{key==\"imu_board\"}.value",
         "label": "IMU Board", "color": "#dc2626"},
        {"value": "/device/health_status.values[:]{key==\"imu_roll_pitch_abnormal\"}.value",
         "label": "Roll/Pitch Abnormal", "color": "#f59e0b"},
        {"value": "/device/health_status.values[:]{key==\"imu_overturn\"}.value",
         "label": "Overturn", "color": "#8b5cf6"},
    ], title="IMU Health Flags")
    configs[health_id] = health_cfg

    # Velocity for context
    vel_id, vel_cfg = _plot_panel([
        {"value": "/odom.twist.twist.linear.x", "label": "Odom Linear.x"},
        {"value": "/odom.twist.twist.angular.z", "label": "Odom Angular.z"},
    ], title="Odometry (context)")
    configs[vel_id] = vel_cfg

    left = _split(rosout_id, raw_id, "column", 50.0)
    right = _split(
        _split(angle_id, gyro_id, "row"),
        _split(health_id, vel_id, "row"),
        "column", 50.0,
    )
    layout = _split(left, right, "row", 30.0)

    return _build_layout_json(configs, layout)


def _layout_navigation() -> dict:
    """Navigation — Navigation stuck / path planning layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=["stuck", "deadlock", "plan", "DijkstraSearch"],
        name_filter={
            "/eco_decision": True,
            "/eco_control": True,
            "/move_base": True,
            "/costmap_node": False,
            "/network_manager": False,
        }
    )
    configs[rosout_id] = rosout_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/scan_rear",
            "/costmap_node/global_costmap/costmap",
            "/costmap_node/global_costmap/local_costmap",
            "/costmap_node/clean_footprint_master",
            "/costmap_node/clean_footprint_steering",
            "/localization/current_pose", "/front_end_pose",
            "/move_base/planner_track_path", "/move_base/planner_fix_path",
            "/move_base/planner_origin_path", "/move_base/planner_index",
            "/move_base/goal_base/a_goal",
            "/eco_control/control_visual_path_topic",
            "/eco_control/dwa_concontroller/dwa_plan",
            "/unicycle_planner/output_path",
            "/hdmap_server/map",
            "/convinced_obstacles", "/last_ring_obstacles",
            "/pointcloud1", "/pointcloud3",
            "/perc/Objects/Visual",
            "/dl_fusion/tracks/visual",
            "/costmap/elastic_virtual_wall/his_traj",
        ]
    )
    configs[scene_id] = scene_cfg

    vel_id, vel_cfg = _plot_panel([
        {"value": "/cmd_vel.linear.x", "label": "cmd_vel linear", "color": "#0923f3"},
        {"value": "/chassis_cmd_vel.linear.x", "label": "chassis linear", "color": "#ef4107"},
        {"value": "/odom.twist.twist.linear.x", "label": "odom linear", "color": "#16a34a"},
    ], title="Velocity")
    configs[vel_id] = vel_cfg

    ang_id, ang_cfg = _plot_panel([
        {"value": "/cmd_vel.angular.z", "label": "cmd_vel angular", "color": "#0923f3"},
        {"value": "/chassis_cmd_vel.angular.z", "label": "chassis angular", "color": "#ef4107"},
    ], title="Angular Velocity")
    configs[ang_id] = ang_cfg

    left = _split(rosout_id, scene_id, "column", 30.0)
    right = _split(vel_id, ang_id, "column")
    layout = _split(left, right, "row", 65.0)

    return _build_layout_json(configs, layout)


def _layout_hardware_fault() -> dict:
    """Hardware Fault — Device health / brush motor / overcurrent layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(search_terms=["chassis", "fault", "motor"])
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/health_status")
    configs[raw_id] = raw_cfg

    raw2_id, raw2_cfg = _raw_messages_panel("/device/all_device_status")
    configs[raw2_id] = raw2_cfg

    # Health flags
    health_id, health_cfg = _plot_panel([
        {"value": "/device/health_status.values[:]{key==\"front_rolling_brush_motor\"}.value",
         "label": "Front Brush", "color": "#dc2626"},
        {"value": "/device/health_status.values[:]{key==\"rear_rolling_brush_motor\"}.value",
         "label": "Rear Brush", "color": "#f59e0b"},
        {"value": "/device/health_status.values[:]{key==\"motor_driver\"}.value",
         "label": "Motor Driver", "color": "#16a34a"},
        {"value": "/device/health_status.values[:]{key==\"imu_board\"}.value",
         "label": "IMU Board", "color": "#3b82f6"},
    ], title="Health Status Flags")
    configs[health_id] = health_cfg

    # Odom health
    odom_id, odom_cfg = _plot_panel([
        {"value": "/device/health_status.values[:]{key==\"odom_left_delta\"}.value",
         "label": "Odom Left Delta"},
        {"value": "/device/health_status.values[:]{key==\"odom_right_delta\"}.value",
         "label": "Odom Right Delta"},
        {"value": "/device/health_status.values[:]{key==\"odom_delta_speed\"}.value",
         "label": "Odom Delta Speed"},
    ], title="Odom Health")
    configs[odom_id] = odom_cfg

    # Connectivity
    conn_id, conn_cfg = _plot_panel([
        {"value": "/device/health_status.values[:]{key==\"laser_disconnection\"}.value",
         "label": "Laser Disconnect", "color": "#dc2626"},
        {"value": "/device/health_status.values[:]{key==\"mcu_disconnection\"}.value",
         "label": "MCU Disconnect", "color": "#f59e0b"},
        {"value": "/device/health_status.values[:]{key==\"battery_disconnection\"}.value",
         "label": "Battery Disconnect", "color": "#8b5cf6"},
    ], title="Connectivity")
    configs[conn_id] = conn_cfg

    # Charger current (overcurrent detection)
    current_id, current_cfg = _plot_panel([
        {"value": "/device/device_status.values[:]{key==\"charger_current\"}.value",
         "label": "Charger Current"},
    ], title="Charger Current")
    configs[current_id] = current_cfg

    left = _split(
        rosout_id,
        _split(raw_id, raw2_id, "row"),
        "column", 40.0
    )
    right = _split(
        _split(health_id, odom_id, "column"),
        _split(conn_id, current_id, "column"),
        "row", 50.0
    )
    layout = _split(left, right, "row", 35.0)

    return _build_layout_json(configs, layout)


def _layout_localization() -> dict:
    """Localization — Pose tracking and localization issues."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=["locali", "pose", "front_end"],
        name_filter={"/eco_decision": True, "/costmap_node": False, "/network_manager": False}
    )
    configs[rosout_id] = rosout_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/scan_rear",
            "/front_end_matched_points",
            "/costmap_node/global_costmap/costmap",
            "/localization/current_pose", "/front_end_pose",
            "/v5_current_pose", "/v5_follow_pose",
            "/hdmap_server/map",
            "/costmap/elastic_virtual_wall/his_traj",
        ]
    )
    configs[scene_id] = scene_cfg

    # Pose X/Y
    pose_id, pose_cfg = _plot_panel([
        {"value": "/localization/current_pose.pose.position.x", "label": "Loc X", "color": "#0839f9"},
        {"value": "/front_end_pose.pose.position.x", "label": "FE X", "color": "#ef4107"},
    ], title="Position X")
    configs[pose_id] = pose_cfg

    pose_y_id, pose_y_cfg = _plot_panel([
        {"value": "/localization/current_pose.pose.position.y", "label": "Loc Y", "color": "#0839f9"},
        {"value": "/front_end_pose.pose.position.y", "label": "FE Y", "color": "#ef4107"},
    ], title="Position Y")
    configs[pose_y_id] = pose_y_cfg

    # Odom vs localization
    odom_id, odom_cfg = _plot_panel([
        {"value": "/odom.twist.twist.linear.x", "label": "Odom Vel"},
        {"value": "/cmd_vel.linear.x", "label": "Cmd Vel"},
    ], title="Velocity")
    configs[odom_id] = odom_cfg

    left = _split(rosout_id, scene_id, "column", 30.0)
    right = _split(
        _split(pose_id, pose_y_id, "column"),
        odom_id,
        "column", 65.0,
    )
    layout = _split(left, right, "row", 55.0)

    return _build_layout_json(configs, layout)


def _layout_sensor_freeze() -> dict:
    """Sensor Freeze — Frozen sensor investigation layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(search_terms=["still_flag", "freeze", "frozen"])
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/odom_status")
    configs[raw_id] = raw_cfg

    vel_id, vel_cfg = _plot_panel([
        {"value": "/chassis_cmd_vel.linear.x", "label": "Cmd Vel Linear"},
        {"value": "/odom.twist.twist.linear.x", "label": "Odom Linear"},
        {"value": "/chassis_cmd_vel.angular.z", "label": "Cmd Vel Angular"},
        {"value": "/odom.twist.twist.angular.z", "label": "Odom Angular"},
    ], title="Velocity (freeze detection)")
    configs[vel_id] = vel_cfg

    imu_id, imu_cfg = _plot_panel([
        {"value": "/unbiased_imu_PRY.x", "label": "IMU X"},
        {"value": "/unbiased_imu_PRY.y", "label": "IMU Y"},
        {"value": "/unbiased_imu_PRY.z", "label": "IMU Z"},
    ], title="IMU (freeze detection)")
    configs[imu_id] = imu_cfg

    pose_id, pose_cfg = _plot_panel([
        {"value": "/localization/current_pose.pose.position.x", "label": "Pose X"},
        {"value": "/localization/current_pose.pose.position.y", "label": "Pose Y"},
    ], title="Localization Pose (freeze detection)")
    configs[pose_id] = pose_cfg

    left = _split(rosout_id, raw_id, "column", 50.0)
    right = _split(vel_id, _split(imu_id, pose_id, "column"), "column", 33.0)
    layout = _split(left, right, "row", 30.0)

    return _build_layout_json(configs, layout)


def _layout_docking() -> dict:
    """对桩 — Docking / charging station alignment layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=["charge", "dock", "charger"],
        name_filter={"/eco_decision": True, "/costmap_node": False, "/network_manager": False}
    )
    configs[rosout_id] = rosout_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/scan_rear", "/charger_scan_debug",
            "/costmap_node/global_costmap/costmap",
            "/localization/current_pose", "/front_end_pose",
            "/charger_pose", "/charger_pose/charger_world",
            "/v5_current_pose", "/v5_follow_pose",
            "/hdmap_server/map",
            "/costmap/elastic_virtual_wall/his_traj",
        ],
        follow_mode="follow-none",
    )
    configs[scene_id] = scene_cfg

    vel_id, vel_cfg = _plot_panel([
        {"value": "/cmd_vel.linear.x", "label": "Cmd Vel Linear", "color": "#0923f3"},
        {"value": "/odom.twist.twist.linear.x", "label": "Odom Linear", "color": "#16a34a"},
    ], title="Approach Velocity")
    configs[vel_id] = vel_cfg

    bat_id, bat_cfg = _plot_panel([
        {"value": "/device/all_device_status.values[:]{key==\"battery\"}.value",
         "label": "Battery %"},
        {"value": "/device/device_status.values[:]{key==\"charger_current\"}.value",
         "label": "Charger Current"},
    ], title="Battery / Charger")
    configs[bat_id] = bat_cfg

    left = _split(rosout_id, scene_id, "column", 30.0)
    right = _split(vel_id, bat_id, "column")
    layout = _split(left, right, "row", 60.0)

    return _build_layout_json(configs, layout)


def _layout_safety() -> dict:
    """Safety — Protector / bumper / emergency stop layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=["protector", "bumper", "emergency", "danger"])
    configs[rosout_id] = rosout_cfg

    raw_id, raw_cfg = _raw_messages_panel("/device/all_device_status")
    configs[raw_id] = raw_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/raw_scan", "/scan_rear",
            "/costmap_node/global_costmap/costmap",
            "/costmap_node/global_costmap/local_costmap",
            "/localization/current_pose",
            "/convinced_obstacles", "/last_ring_obstacles",
            "/pointcloud1", "/pointcloud2", "/pointcloud3",
            "/pointcloud4", "/pointcloud5", "/pointcloud6",
            "/perc/Objects/Visual",
            "/dl_fusion/tracks/visual",
            "/hdmap_server/map",
        ]
    )
    configs[scene_id] = scene_cfg

    dev_id, dev_cfg = _plot_panel([
        {"value": "/device/all_device_status.values[:]{key==\"braker_down\"}.value",
         "label": "Brake", "color": "#0d81f6"},
        {"value": "/device/all_device_status.values[:]{key==\"auto_mode\"}.value",
         "label": "Auto Mode", "color": "#ef4912"},
        {"value": "/device/all_device_status.values[:]{key==\"emergency\"}.value",
         "label": "E-Stop", "color": "#fcd31a"},
    ], title="Safety Status")
    configs[dev_id] = dev_cfg

    vel_id, vel_cfg = _plot_panel([
        {"value": "/cmd_vel.linear.x", "label": "Cmd Vel"},
        {"value": "/odom.twist.twist.linear.x", "label": "Odom"},
    ], title="Velocity at Event")
    configs[vel_id] = vel_cfg

    left = _split(
        rosout_id,
        _split(raw_id, scene_id, "row", 25.0),
        "column", 35.0,
    )
    right = _split(dev_id, vel_id, "column")
    layout = _split(left, right, "row", 60.0)

    return _build_layout_json(configs, layout)


def _layout_depth_camera() -> dict:
    """Depth Camera — Camera failure / overdark / vision pipeline layout."""
    configs = {}

    rosout_id, rosout_cfg = _rosout_panel(
        search_terms=["overdark", "dl_infer", "depthcam", "depth_pipeline"],
        name_filter={
            "/dl_infer": True,
            "/depthcam_fusion": True,
            "/homo_fusion_tracker_node": True,
            "/gaussian_mapping_v5": True,
            "/gs_console": True,
            "/eco_decision": True,
            "/depth_pipeline1": True,
            "/depth_pipeline2": True,
            "/depth_pipeline3": True,
            "/depth_pipeline4": True,
            "/depth_pipeline5": True,
            "/eco_control": False,
            "/costmap_node": False,
            "/network_manager": False,
            "/chassis": False,
        }
    )
    configs[rosout_id] = rosout_cfg

    scene_id, scene_cfg = _3d_panel(
        visible_topics=[
            "/camera1_record/depth/image_raw",
            "/camera2_record/depth/image_raw",
            "/camera3_record/depth/image_raw",
            "/camera4_record/depth/image_raw",
            "/camera5_record/depth/image_raw",
            "/camera6_record/depth/image_raw",
            "/perc/record/rgb_image_hd/compressed",
            "/perc/record/rgb_image_hd6/compressed",
            "/raw_scan",
            "/costmap_node/global_costmap/costmap",
            "/localization/current_pose", "/front_end_pose",
            "/hdmap_server/map",
            "/pointcloud1", "/pointcloud3",
            "/perc/Objects/Visual",
        ],
        follow_mode="follow-pose",
    )
    configs[scene_id] = scene_cfg

    # Image panels for depth cameras
    img1_id = _panel_id("Image")
    configs[img1_id] = {
        "imageMode": {"imageTopic": "/camera3_record/depth/image_raw"},
        "cameraState": {"distance": 20, "perspective": True, "phi": 60,
                        "target": [0,0,0], "targetOffset": [0,0,0],
                        "targetOrientation": [0,0,0,1], "thetaOffset": 45,
                        "fovy": 45, "near": 0.5, "far": 5000},
        "followMode": "follow-pose", "scene": {}, "transforms": {},
        "topics": {}, "layers": {},
        "publish": {"type": "point", "poseTopic": "/move_base_simple/goal",
                    "pointTopic": "/clicked_point",
                    "poseEstimateTopic": "/initialpose",
                    "poseEstimateXDeviation": 0.5,
                    "poseEstimateYDeviation": 0.5,
                    "poseEstimateThetaDeviation": 0.26179939},
    }

    img2_id = _panel_id("Image")
    configs[img2_id] = {
        "imageMode": {"imageTopic": "/camera5_record/depth/image_raw"},
        "cameraState": {"distance": 20, "perspective": True, "phi": 60,
                        "target": [0,0,0], "targetOffset": [0,0,0],
                        "targetOrientation": [0,0,0,1], "thetaOffset": 45,
                        "fovy": 45, "near": 0.5, "far": 5000},
        "followMode": "follow-pose", "scene": {}, "transforms": {},
        "topics": {}, "layers": {},
        "publish": {"type": "point", "poseTopic": "/move_base_simple/goal",
                    "pointTopic": "/clicked_point",
                    "poseEstimateTopic": "/initialpose",
                    "poseEstimateXDeviation": 0.5,
                    "poseEstimateYDeviation": 0.5,
                    "poseEstimateThetaDeviation": 0.26179939},
    }

    # Velocity plot for context (is robot stuck?)
    vel_id, vel_cfg = _plot_panel([
        {"value": "/cmd_vel.linear.x", "label": "Cmd Vel", "color": "#0923f3"},
        {"value": "/odom.twist.twist.linear.x", "label": "Odom", "color": "#16a34a"},
    ], title="Velocity (is robot stuck?)")
    configs[vel_id] = vel_cfg

    # Pose plot for localization drift
    pose_id, pose_cfg = _plot_panel([
        {"value": "/localization/current_pose.pose.position.x", "label": "Loc X", "color": "#0839f9"},
        {"value": "/front_end_pose.pose.position.x", "label": "FE X", "color": "#ef4107"},
    ], title="Position (localization drift?)")
    configs[pose_id] = pose_cfg

    # Layout: RosOut top-left, depth images top-right, 3D bottom-left, plots bottom-right
    top = _split(
        rosout_id,
        _split(img1_id, img2_id, "row"),
        "row", 40.0,
    )
    bottom = _split(
        scene_id,
        _split(vel_id, pose_id, "column"),
        "row", 60.0,
    )
    layout = _split(top, bottom, "column", 40.0)

    return _build_layout_json(configs, layout)


# ---------------------------------------------------------------------------
# Category → Layout mapping
# ---------------------------------------------------------------------------

CATEGORY_TO_LAYOUT = {
    # Cross-validator categories
    "MOTION_STATE": "motion_control",
    "MOTION": "motion_control",
    "NAV_STUCK": "navigation",
    "NAVIGATION": "navigation",
    "NAVIGATION_STUCK": "navigation",
    "IMU_CALIBRATION": "imu",
    "IMU_ERROR": "imu",
    "IMU": "imu",
    "IMU_FROZEN": "imu",
    "IMU_HW_FIELD_ZERO": "imu",
    "LOCALIZATION": "localization",
    "LOCATION_STATE": "localization",
    "LOCALIZATION_STUCK": "localization",
    "SAFETY_STATE": "safety",
    "SAFETY": "safety",
    "BATTERY_STATE": "battery",
    "HW_FAULT": "hardware_fault",
    "HARDWARE_FAULT": "hardware_fault",
    "IR_SENSOR_DEAD": "hardware_fault",
    "FROZEN_SENSOR": "sensor_freeze",
    "SENSOR_FREEZE": "sensor_freeze",
    "FREEZE_ONSET": "sensor_freeze",
    "SENSOR_RESUME": "sensor_freeze",
    "ZERO_FIELD": "sensor_freeze",
    "UNSTABLE_FREQUENCY": "sensor_freeze",
    # Depth camera / vision pipeline
    "DEPTH_CAMERA": "depth_camera",
    "DEPTH_CAMERA_ERROR": "depth_camera",
    "DEPTHCAM_FUSION_FAIL": "depth_camera",
    "DL_INFER_EVENT": "depth_camera",
    # Catch-all
    "ERROR": "abnormal_stop",
    "FAILURE": "abnormal_stop",
}

LAYOUT_BUILDERS = {
    "motion_control": _layout_motion_control,
    "battery": _layout_battery,
    "abnormal_stop": _layout_abnormal_stop,
    "imu": _layout_imu,
    "navigation": _layout_navigation,
    "hardware_fault": _layout_hardware_fault,
    "localization": _layout_localization,
    "sensor_freeze": _layout_sensor_freeze,
    "docking": _layout_docking,
    "safety": _layout_safety,
    "depth_camera": _layout_depth_camera,
}

LAYOUT_DISPLAY_NAMES = {
    "motion_control": "Motion Control (运动控制)",
    "battery": "Battery (电池)",
    "abnormal_stop": "Abnormal Stop (异常停车)",
    "imu": "IMU Diagnostics",
    "navigation": "Navigation (导航)",
    "hardware_fault": "Hardware Fault (硬件故障)",
    "localization": "Localization (定位)",
    "sensor_freeze": "Sensor Freeze (传感器冻结)",
    "depth_camera": "Depth Camera (深度相机)",
    "docking": "Docking (对桩)",
    "safety": "Safety Systems (安全)",
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_foxglove_layouts(
    cross_validation_report: Optional[str] = None,
    diagnostic_report: Optional[str] = None,
    output_dir: str = "./foxglove_layouts",
    generate_all: bool = False,
) -> Dict[str, str]:
    """
    Generate Foxglove layout files based on diagnostic findings.

    Args:
        cross_validation_report: Path to cross_validation_report.json
        diagnostic_report: Path to diagnostic_report.json
        output_dir: Directory to write layout JSON files
        generate_all: If True, generate all layout types regardless of findings

    Returns:
        Dict of {display_name: file_path} for generated layouts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine which layouts are needed
    needed_layouts: Set[str] = set()

    if generate_all:
        needed_layouts = set(LAYOUT_BUILDERS.keys())
    else:
        categories_found: Set[str] = set()

        if cross_validation_report and os.path.exists(cross_validation_report):
            with open(cross_validation_report) as f:
                cv_data = json.load(f)
            for packet in cv_data.get("evidence_packets", []):
                cat = packet.get("category", "")
                sev = packet.get("severity", "")
                verdict = packet.get("verdict", "")
                # Only generate layouts for significant findings
                if sev in ("CRITICAL", "WARNING") or verdict in ("CONFIRMED", "CONTRADICTED"):
                    categories_found.add(cat)

        if diagnostic_report and os.path.exists(diagnostic_report):
            with open(diagnostic_report) as f:
                diag_data = json.load(f)
            for mission in diag_data.get("missions", []):
                for item in mission.get("timeline", []):
                    categories_found.add(item.get("category", ""))
                for item in mission.get("incidents", []):
                    categories_found.add(item.get("category", ""))

        # Map categories to layout types
        for cat in categories_found:
            layout_type = CATEGORY_TO_LAYOUT.get(cat)
            if layout_type:
                needed_layouts.add(layout_type)

        # Always include motion_control as baseline
        if not needed_layouts:
            needed_layouts.add("motion_control")
            needed_layouts.add("abnormal_stop")

    # Generate layouts
    generated: Dict[str, str] = {}
    for layout_type in sorted(needed_layouts):
        builder = LAYOUT_BUILDERS.get(layout_type)
        if not builder:
            continue

        layout_json = builder()
        filename = f"{layout_type}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(layout_json, f, indent=2, ensure_ascii=False)

        display_name = LAYOUT_DISPLAY_NAMES.get(layout_type, layout_type)
        generated[display_name] = filepath
        print(f"  Generated: {filepath} ({display_name})")

    return generated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate problem-specific Foxglove layouts from diagnostic findings")
    parser.add_argument("--cross-validation", "-cv", help="Path to cross_validation_report.json")
    parser.add_argument("--diagnostic", "-d", help="Path to diagnostic_report.json")
    parser.add_argument("--output-dir", "-o", default="./foxglove_layouts",
                        help="Output directory for layout files")
    parser.add_argument("--all", action="store_true",
                        help="Generate all layout types regardless of findings")
    args = parser.parse_args()

    print("Generating Foxglove layouts...")
    layouts = generate_foxglove_layouts(
        cross_validation_report=args.cross_validation,
        diagnostic_report=args.diagnostic,
        output_dir=args.output_dir,
        generate_all=args.all,
    )
    print(f"\nGenerated {len(layouts)} layout(s) in {args.output_dir}/")
    for name, path in layouts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
