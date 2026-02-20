"""
Configuration constants for the diagnostic extractor.

Design principles:
- NODE_IMPORTANCE and SUBSYSTEM_GROUPS encode structural knowledge about the robot's
  architecture, NOT causal hypotheses. These are safe to use.
- UNSTABLE_TEMPLATES is a structural stability filter, NOT a noise filter.
  The fingerprint baseline is the real noise filter. Only add patterns here for
  templates that are structurally different every cycle and would make every window
  look anomalous.
- NO hard-coded causal rules. Causal discovery is data-driven (Stage 5).
  Validated rules are loaded from validated_rules.yaml after human review.
"""

import re
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Node importance hierarchy
# ---------------------------------------------------------------------------
# Structural knowledge: which subsystems matter most for diagnosis.
# Higher importance = more weight in anomaly scoring and root cause assignment.

NODE_IMPORTANCE: Dict[str, int] = {
    "/homo_fusion_tracker_node": 10,    # Localization core
    "/gs_console": 9,                    # System state machine
    "/eco_decision": 8,                  # Navigation planner
    "/scan_filters": 7,                  # Sensor init
    "/depthcam_fusion": 7,              # Depth processing
    "/depth_pipeline1": 6,              # Point cloud
    "/obstacle_track": 6,               # Obstacle detection
    "/dl_infer": 5,                     # Deep learning inference
    "/gaussian_mapping_v5": 5,          # SLAM
    "/gs_device_controller": 4,         # Device control
    "/chassis": 3,                       # Low-level hardware (always there)
    "/usb_front_camera": 1,             # Camera heartbeat
    "/gs_robot_rcc": 1,                 # Network monitoring
}

DEFAULT_NODE_IMPORTANCE = 4

# ---------------------------------------------------------------------------
# Subsystem groups (soft prior for co-occurrence thresholds)
# ---------------------------------------------------------------------------
# Maps nodes to subsystem categories. Used to lower the Jaccard threshold
# for same-subsystem anomalies (architecturally related nodes).
# NOT causal claims — just structural groupings.
# Data overrides: if two nodes in the same group never co-occur anomalously,
# they will NOT be chained together.

SUBSYSTEM_GROUPS: Dict[str, str] = {
    # Perception
    "/dl_infer": "perception",
    "/depth_pipeline1": "perception",
    "/depthcam_fusion": "perception",
    "/obstacle_track": "perception",
    "/usb_front_camera": "perception",
    # Localization
    "/homo_fusion_tracker_node": "localization",
    "/gaussian_mapping_v5": "localization",
    "/scan_filters": "localization",
    # Navigation
    "/eco_decision": "navigation",
    "/gs_console": "navigation",
    # Hardware
    "/chassis": "hardware",
    "/gs_device_controller": "hardware",
    # Network
    "/gs_robot_rcc": "network",
}

# ---------------------------------------------------------------------------
# Unstable templates (structural stability filter)
# ---------------------------------------------------------------------------
# Templates with high structural variance — they generate a new template every
# cycle, making fingerprint comparison meaningless for those entries.
# This is NOT a noise filter. If in doubt, DON'T add a pattern here.
# Let the cycle detector handle it.

UNSTABLE_TEMPLATES: List[re.Pattern] = [
    # Raw hex data dumps (different bytes every message)
    re.compile(r"^[0-9a-fA-F\s]{20,}$"),
    # HTTP request logs with unique URLs/parameters
    re.compile(r"(GET|POST|PUT|DELETE)\s+/", re.I),
    # JSON blobs with unique IDs
    re.compile(r'^\s*\{.*"(id|uuid|session)":', re.I),
]

# ---------------------------------------------------------------------------
# Frequency classification thresholds
# ---------------------------------------------------------------------------

CV_PERIODIC_THRESHOLD = 0.3         # CV < 0.3 = periodic (heartbeat)
CV_QUASI_PERIODIC_THRESHOLD = 1.0   # 0.3 <= CV < 1.0 = quasi-periodic
MIN_COUNT_FOR_PERIODIC = 20         # < 20 messages = event-driven regardless of CV

# ---------------------------------------------------------------------------
# Fingerprinting and regime detection
# ---------------------------------------------------------------------------

MIN_WINDOW_SIZE_MS = 1000.0         # minimum window width
BASELINE_PRESENCE_THRESHOLD = 0.6   # template must appear in >60% of windows to be baseline
CUSUM_THRESHOLD_SIGMA = 3.0         # CUSUM triggers at 3σ above running mean
TRANSIENT_REGIME_MAX_WINDOWS = 3    # regimes shorter than this are transient anomalies

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

ANOMALY_PERCENTILE = 90             # flag windows above this percentile
NEW_TEMPLATE_WEIGHT = 2.0           # weight multiplier for new (unexpected) templates
RATE_ANOMALY_THRESHOLD = 2.0        # flag if count deviates by >2x from expected
VALUE_DEVIATION_SIGMA = 3.0         # flag continuous values beyond ±3σ
EWMA_ALPHA = 0.01                   # slow adaptation for rolling statistics

# ---------------------------------------------------------------------------
# Causal chain detection
# ---------------------------------------------------------------------------

JACCARD_THRESHOLD = 0.5             # minimum Jaccard for co-occurrence edge
JACCARD_SAME_SUBSYSTEM = 0.3        # lower threshold for same-subsystem anomalies
CO_OCCURRENCE_ADJACENT_WINDOWS = 1  # anomalies co-occur if within N adjacent windows

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 3000
CHARS_PER_TOKEN = 4                 # rough estimate for LLM tokenization

# Token budget allocation (fractions)
BUDGET_HEADER = 0.05
BUDGET_REGIME_TIMELINE = 0.15
BUDGET_CAUSAL_CHAINS = 0.50
BUDGET_ISOLATED_ANOMALIES = 0.15
BUDGET_SENSOR_BASELINE = 0.10
BUDGET_EVENT_DRIVEN = 0.05

# ---------------------------------------------------------------------------
# Log level mapping
# ---------------------------------------------------------------------------

LOG_LEVELS = {"DEBUG": 1, "INFO": 2, "WARN": 4, "ERROR": 8, "FATAL": 16}
LOG_LEVEL_NAMES = {v: k for k, v in LOG_LEVELS.items()}

# Localization status codes (Gaussian-specific)
LOC_STATUS = {0: "uninitialized", 1: "initializing", 2: "lost", 3: "tracking", 4: "degraded"}
