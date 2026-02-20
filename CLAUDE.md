# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROSBag Analyzer is a multi-modal diagnostic agent for Gaussian/Gausium cleaning robots. It bridges the gap between robotic software logs and physical sensor data by using LLMs to actively interrogate ROS bags through structured tool calls. The core philosophy is **log-first** (v3.0): use `/rosout` logs as primary evidence and sensor data as verification, enabling detection of "Hidden Failures" where logs report nominal but sensors reveal anomalies.

## Commands

```bash
# Default analysis (log-first incident mode)
python -m cli.analyze --bag-dir ./bags --mode incident

# Cross-validate logs vs sensors
python -m cli.analyze --mode crossval --bag-dir ./bags

# Mission timeline with CSV export
python -m cli.analyze --mode mission --bag-dir ./bags --csv timeline.csv

# Legacy per-bag analysis
python -m cli.analyze --mode legacy --bag-dir ./bags

# Validate bridge tools (no LLM required)
python -m cli.validate bridge_only

# Full agent validation (requires ANTHROPIC_API_KEY)
python -m cli.validate full

# Profile a single bag file
python -m cli.profiler ./bag.bag

# Batch profile all bags in a directory
python -m cli.profiler --batch-dir ./bags

# Quick bridge smoke test on a bag
python -m cli.bridge_test ./bag.bag
```

## Dependencies

Install with `pip install -r requirements.txt`. Key dependencies: `rosbags==0.9.15` (ROS1 bag reading), `numpy`, `langgraph`, `langchain`, `langchain-anthropic`, `langchain-openai`. Python 3.13+. The LLM agent requires `ANTHROPIC_API_KEY` set in environment.

## Architecture

### Package Structure

The codebase is organized into focused packages with a strict dependency DAG (no cycles):

```
core/       → (no local deps)        # Foundation: constants, utils, models
bridge/     → core/                   # 4-tool Bridge API for bag access
logs/       → core/, bridge/          # Log parsing, patterns, state tracking
analysis/   → core/, bridge/, logs/   # Diagnostic analysis engines
agent/      → core/, bridge/          # LLM agent pipeline (LangGraph)
reporting/  → core/                   # HTML reports, Foxglove layouts
cli/        → everything above        # CLI entry points
```

### Two Parallel Diagnostic Pipelines

**Pipeline A — LLM Agent (`agent/`):** A LangGraph agent (`agent/graph.py`) with a supervisor LLM node and tool executor node. The LLM calls 4 bridge tools to interrogate bags, accumulates findings in an evidence locker, and produces grounded diagnostics. The critic node (`agent/prompts.py:CRITIC_PROMPT`) evaluates hallucination rate.

**Pipeline B — Statistical Extractor (`rosbag_diagnostic_extractor/`):** A 6-stage deterministic pipeline (parse → fingerprint → structural anomaly → content anomaly → causal chains → report) that compresses large logs into LLM-ready diagnostic signals without calling any LLM. Orchestrated by `pipeline.py`.

### The 4-Tool Bridge API (`bridge/`)

The bridge is the central abstraction. All bag access goes through these 4 tools:

1. **`get_bag_metadata`** — Reconnaissance: topics, durations, message counts
2. **`get_topic_statistics`** — Windowed numeric stats (Welford's online algorithm, O(1) memory)
3. **`check_topic_frequency`** — Publishing rate time-series per topic
4. **`sample_messages`** — Raw message inspection at specific timestamps

`bridge/bridge.py` contains the implementations; `agent/tools.py` wraps them as LangChain `@tool` decorators for the agent. `bridge/truncated_reader.py` handles truncated bag files. `bridge/welford.py` provides the O(1) memory accumulator.

### Log Processing Chain (`logs/`)

`logs/extractor.py` extracts `/rosout` messages → `logs/patterns.py` matches against 18 Gaussian-specific regex pattern categories → produces `LogEvent` objects → `logs/state_timeline.py` tracks state transitions (motion, navigation, IMU, health).

`logs/denoiser.py` removes noise/spam logs (80%+ removal rate) using signal/noise pattern matching and deduplication.

### Analysis Engines (`analysis/`)

- `analysis/diagnostic_analyzer.py` — 7-phase diagnostic analysis pipeline
- `analysis/incident_builder.py` — Clusters findings into incidents with causal chains
- `analysis/mission_orchestrator.py` — Multi-bag mission timeline with CSV export
- `analysis/cross_validator.py` — Log-sensor cross-validation with `EvidencePacket` verdicts: `CONFIRMED`, `CONTRADICTED`, or `UNCHECKED`

### Report & Visualization (`reporting/`)

`reporting/html_report.py` produces HTML reports. `reporting/foxglove_layouts.py` auto-generates Foxglove Studio layouts for visualizing problem categories.

### Shared Foundation (`core/`)

- `core/constants.py` — ALL Gaussian robot domain knowledge (expected-zero fields, health flags, hardware topics, sensor topics, timing thresholds)
- `core/utils.py` — `CST` timezone, `LOG_LEVELS`, `format_absolute_time()` (single source of truth, no duplication)
- `core/models.py` — All shared dataclasses (`LogEvent`, `Incident`, `BagInfo`, `Mission`, `EvidencePacket`, `SensorSnapshot`, etc.)

### Key Data Flow

```
.bag files → logs/extractor.py (LogEvents + StateTransitions)
           → bridge/bridge.py (sensor statistics via 4 tools)
           → analysis/cross_validator.py (EvidencePackets with verdicts)
           → analysis/diagnostic_analyzer.py (incident clustering + root cause)
           → reporting/html_report.py (HTML + JSON output)
```

## Domain Knowledge

This codebase encodes structural knowledge about **Gaussian cleaning robots** (2D differential-drive), consolidated in `core/constants.py`:

- **Expected-zero fields**: orientation_x/y, position_z on `/odom`; angular_x/y on `/chassis_cmd_vel`; level=0 on `/device/health_status`
- **Health flags** on `/device/health_status`: boolean fields where `false` = fault (e.g., `rear_rolling_brush_motor`, `imu_board`, `laser_disconnection`)
- **Node importance hierarchy**: `/homo_fusion_tracker_node` (10, localization core) down to `/usb_front_camera` (1, heartbeat) — defined in `rosbag_diagnostic_extractor/constants.py`
- **Subsystem groups** and **causal thresholds** (Jaccard similarity) are in `rosbag_diagnostic_extractor/constants.py`

## Key Algorithms

- **Welford's online variance**: Single-pass O(1) memory stats over gigabyte-scale bags (`bridge/welford.py:WelfordAccumulator`)
- **CUSUM change-point detection**: Regime segmentation in fingerprinting stage (`rosbag_diagnostic_extractor/utils.py:cusum()`)
- **EWMA tracking**: Continuous value anomaly detection with slow adaptation α=0.01 (`rosbag_diagnostic_extractor/utils.py:EWMATracker`)
- **Jaccard similarity graph**: Groups correlated anomalies into causal chains (`rosbag_diagnostic_extractor/causal_chains.py`)
- **Template extraction**: Replaces numbers with `#` to identify structural log patterns (`rosbag_diagnostic_extractor/utils.py:templatize()`)

## Analysis Modes

| Mode | Flag | Description |
|------|------|-------------|
| incident | `--mode incident` | Log-first incident pipeline with clustering (DEFAULT) |
| crossval | `--mode crossval` | Log-sensor cross-validation |
| mission | `--mode mission` | Multi-bag virtual timeline with absolute timestamps (CST) |
| legacy | `--mode legacy` | Per-bag independent analysis |
