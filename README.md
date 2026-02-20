# ROSBag Analyzer

**Multi-Modal Diagnostic Agent: Grounding LLM Log Analysis with On-Demand ROS Bag Retrieval**

A neuro-symbolic diagnostic system that bridges the "reality gap" between robotic software logs and physical sensor data. Instead of passively reading logs, an LLM agent actively *interrogates* ROS bag files through structured tool calls, enabling detection of "Hidden Failures" where logs report nominal status but sensors reveal anomalies.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              LangGraph Agent                     │
│  ┌───────────┐    ┌──────────────┐              │
│  │ Supervisor │───>│ Tool Executor│──┐           │
│  │ (LLM)     │<───│              │  │           │
│  └───────────┘    └──────────────┘  │           │
│       │                              │           │
│  ┌────▼────┐                         │           │
│  │ Evidence │  <─────────────────────┘           │
│  │ Locker   │                                    │
│  └─────────┘                                     │
└──────────────────────┬──────────────────────────┘
                       │ Tool Calls (JSON)
              ┌────────▼────────┐
              │  ROSBag Bridge  │
              │  (4 Tools)      │
              └────────┬────────┘
                       │ rosbags library
              ┌────────▼────────┐
              │   .bag / .mcap  │
              │   files on disk │
              └─────────────────┘
```

## Bridge Tools

| Tool | Purpose | Key Use Case |
|------|---------|-------------|
| `get_bag_metadata` | Reconnaissance — topics, durations, frequencies | Understand what data is available |
| `get_topic_statistics` | Statistical summary with windowed aggregation | Detect frozen sensors (std=0), anomalous ranges |
| `check_topic_frequency` | Time-series of publishing rate (Hz) | Detect silent failures, driver crashes |
| `sample_messages` | Raw message inspection at specific timestamps | Qualitative analysis of string/enum fields |

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Or with a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Place Bag Files

Place `.bag` (ROS1) files in the project root or a subdirectory (e.g., `20260210-DV-Bags/`).

---

## CLI Reference

### `analyze.py` — Main Diagnostic Analyzer (v3.0)

The primary entry point supporting multiple analysis modes.

```bash
python3 analyze.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bag-dir DIR` | Project root | Directory containing `.bag` files (searched recursively) |
| `--report-path PATH` | Auto-generated | Output path for diagnostic report JSON |
| `--mode MODE` | `incident` | Analysis mode (see below) |
| `--csv PATH` | None | Export timeline to CSV (mission/incident modes) |
| `--gap-threshold SECS` | `300` | Max seconds between bags to group into same mission |

**Analysis modes:**

| Mode | Version | Description |
|------|---------|-------------|
| `incident` | v3.0 | Log-first incident pipeline with clustering (DEFAULT) |
| `crossval` | v4.0 | Log-sensor cross-validation via CrossValidator |
| `mission` | v2.0 | Mission-aware multi-bag virtual timeline |
| `legacy` | v2.0 | Per-bag independent analysis |

**Examples:**

```bash
# Default: log-first incident analysis
python3 analyze.py

# Analyze bags in a specific directory
python3 analyze.py --bag-dir ./20260210-DV-Bags

# Cross-validate logs against sensor data
python3 analyze.py --mode crossval --bag-dir ./20260210-DV-Bags

# Mission timeline with CSV export
python3 analyze.py --mode mission --bag-dir ./20260210-DV-Bags --csv timeline.csv

# Legacy per-bag analysis with custom report path
python3 analyze.py --mode legacy --report-path my_report.json

# Adjust mission grouping threshold (10 minutes)
python3 analyze.py --mode mission --gap-threshold 600
```

---

### `cross_validator.py` — Log-Sensor Cross-Validation (Standalone)

Timestamp-aligned correlation engine that pairs `/rosout` log events with sensor snapshots to confirm or contradict log claims.

```bash
python3 cross_validator.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bag-dir DIR` | Script directory | Directory containing `.bag` files |
| `--output, -o PATH` | `cross_validation_report.json` | Output JSON report path |
| `--skip-sensors` | False | Skip sensor snapshots (fast, log-only analysis) |
| `--sample-rate N` | `1` | Build sensor snapshot every Nth event (1 = all) |
| `--llm-summary` | False | Print an LLM-ready text summary after analysis |

**Examples:**

```bash
# Full cross-validation with sensor snapshots
python3 cross_validator.py --bag-dir ./20260210-DV-Bags

# Fast log-only analysis (no sensor queries)
python3 cross_validator.py --bag-dir ./20260210-DV-Bags --skip-sensors

# Reduce processing time by sampling every 5th event
python3 cross_validator.py --bag-dir ./20260210-DV-Bags --sample-rate 5

# Generate LLM-consumable summary
python3 cross_validator.py --bag-dir ./20260210-DV-Bags --llm-summary

# Custom output path
python3 cross_validator.py --bag-dir ./20260210-DV-Bags -o results.json
```

---

### `validate.py` — Validation Harness

Tests that bridge tools and the LangGraph agent work correctly.

```bash
python3 validate.py [MODE]
```

| Mode | Description |
|------|-------------|
| `bridge_only` | Test that all 4 bridge tools work correctly (default) |
| `full` | Full agent validation with LLM (requires API key) |
| `both` | Run bridge validation followed by full agent validation |

**Examples:**

```bash
# Test bridge tools only (no LLM required)
python3 validate.py bridge_only

# Full agent validation (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key-here"
python3 validate.py full

# Run both bridge and agent validation
python3 validate.py both
```

---

### `rosbag_profiler.py` — Bag Profiler

Profiles ROS bags for log anomalies and sensor-data issues.

```bash
python3 rosbag_profiler.py [BAG_PATH] [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `BAG_PATH` | None | Path to a single `.bag` file |
| `--batch-dir, -d DIR` | None | Directory of `.bag` files for batch scanning |
| `--report, -r PATH` | None | Path for JSON report output |

**Examples:**

```bash
# Profile a single bag
python3 rosbag_profiler.py ./20260210-DV-Bags/GS_2026-02-10-07-00-07_510.bag

# Batch scan all bags in a directory
python3 rosbag_profiler.py --batch-dir ./20260210-DV-Bags

# Batch scan with JSON report
python3 rosbag_profiler.py -d ./20260210-DV-Bags -r profiler_report.json
```

---

### `rosbag_bridge.py` — Bridge Quick Test

Runs a quick smoke test of all 4 bridge functions on a single bag.

```bash
python3 rosbag_bridge.py <BAG_PATH>
```

**Example:**

```bash
python3 rosbag_bridge.py ./20260210-DV-Bags/GS_2026-02-10-07-00-07_510.bag
```

---

### `log_parser.py` — Log Parser Quick Test

Extracts and summarizes `/rosout` log events from a single bag.

```bash
python3 log_parser.py <BAG_PATH>
```

**Example:**

```bash
python3 log_parser.py ./20260210-DV-Bags/GS_2026-02-10-07-00-07_510.bag
```

---

## Project Structure

```
Robot_analyzer/
├── __init__.py              # Package init
├── analyze.py               # Main CLI — multi-mode diagnostic analyzer (v3.0)
├── cross_validator.py       # Log-sensor cross-validation engine (v4.0)
├── rosbag_bridge.py         # Core Bridge — 4 tools, Welford's algorithm
├── log_parser.py            # /rosout log extraction & state timeline builder
├── rosbag_profiler.py       # Bag profiler for log anomalies & sensor issues
├── tools.py                 # LangChain @tool wrappers
├── prompt_templates.py      # Agent system prompts
├── agent.py                 # LangGraph diagnostic agent
├── validate.py              # Validation harness (bridge + agent)
└── requirements.txt         # Python dependencies
```

## Key Design Decisions

- **Welford's Algorithm**: Single-pass mean/variance computation with O(1) memory, enabling analysis of gigabyte-scale bags without loading into RAM.
- **Sliding Window Aggregation**: Hierarchical zooming (Global -> Regional -> Local) to localize anomalies in time.
- **Evidence Locker**: Structured findings validated by tool output, separate from chat history, ensuring the final diagnosis is grounded in sensor data.
- **Log Denoising**: Regex-based noise-first filtering removes 40%+ of /rosout spam (HTTP logs, hex dumps, repetitive messages) before analysis.
- **Pre-computed Sensor Windows**: Single-pass windowed statistics per topic enable O(1) timestamp lookups instead of O(N) per-event bag scans.
- **Graceful Degradation**: Custom ROS message types that can't be deserialized are skipped rather than crashing the analysis.

## Anomaly Detection Capabilities

- **Frozen Sensors**: Detects when a sensor's standard deviation drops to 0 (e.g., IMU axis locked, encoder stale)
- **Frequency Dropouts**: Identifies periods where topic publishing rate drops below expected threshold
- **Command/Feedback Mismatch**: Compares commanded velocities against odometry to detect actuator stalls
- **Localization Divergence**: Cross-checks odometry range against localization range
- **Freeze Onset/Resume**: Pinpoints exact time windows where sensors transition between active and frozen states
- **Log-Sensor Cross-Validation**: Confirms or contradicts log claims (e.g., "path planning failed" vs actual velocity data) with structured verdicts (CONFIRMED / CONTRADICTED / UNCHECKED)

## Data Requirements

Place `.bag` (ROS1) files in the project root directory or a subdirectory. The analyzer supports standard ROS message types including:

- `nav_msgs/Odometry`
- `geometry_msgs/Twist`, `TwistStamped`, `PoseWithCovarianceStamped`
- `sensor_msgs/LaserScan`, `Imu`
- `std_msgs/Float32`, `Int64`, `Bool`, `String`
- `rosgraph_msgs/Log` (via custom type registration)
- And more via automatic field extraction

## References

Based on the research plan: *"Multi-Modal Diagnostic Agents: Grounding LLM Log Analysis with On-Demand ROS Bag Retrieval"*

## License

MIT
