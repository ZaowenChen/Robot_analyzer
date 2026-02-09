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
pip install -r rosbag_analyzer/requirements.txt
```

### Run Standalone Analysis (No LLM Required)

```bash
# Analyze all .bag files in the current directory
python -m rosbag_analyzer.analyze

# Specify a custom bag directory
python -m rosbag_analyzer.analyze --bag-dir /path/to/bags
```

### Run LangGraph Agent (Requires API Key)

```bash
export ANTHROPIC_API_KEY="your-key-here"

# Full agent validation against all bags
python -m rosbag_analyzer.validate full
```

### Bridge Validation Only

```bash
# Test that all 4 bridge tools work correctly
python -m rosbag_analyzer.validate bridge_only
```

## Project Structure

```
rosbag_analyzer/
├── __init__.py              # Package init
├── rosbag_bridge.py         # Core Bridge — 4 tools, Welford's algorithm
├── tools.py                 # LangChain @tool wrappers
├── prompt_templates.py      # Agent system prompts
├── agent.py                 # LangGraph diagnostic agent
├── analyze.py               # Standalone rule-based analyzer
├── validate.py              # Validation harness
└── requirements.txt         # Python dependencies
```

## Key Design Decisions

- **Welford's Algorithm**: Single-pass mean/variance computation with O(1) memory, enabling analysis of gigabyte-scale bags without loading into RAM.
- **Sliding Window Aggregation**: Hierarchical zooming (Global → Regional → Local) to localize anomalies in time.
- **Evidence Locker**: Structured findings validated by tool output, separate from chat history, ensuring the final diagnosis is grounded in sensor data.
- **Graceful Degradation**: Custom ROS message types that can't be deserialized are skipped rather than crashing the analysis.

## Anomaly Detection Capabilities

- **Frozen Sensors**: Detects when a sensor's standard deviation drops to 0 (e.g., IMU axis locked, encoder stale)
- **Frequency Dropouts**: Identifies periods where topic publishing rate drops below expected threshold
- **Command/Feedback Mismatch**: Compares commanded velocities against odometry to detect actuator stalls
- **Localization Divergence**: Cross-checks odometry range against localization range
- **Freeze Onset/Resume**: Pinpoints exact time windows where sensors transition between active and frozen states

## Data Requirements

Place `.bag` (ROS1) files in the project root directory. The analyzer supports standard ROS message types including:

- `nav_msgs/Odometry`
- `geometry_msgs/Twist`, `TwistStamped`, `PoseWithCovarianceStamped`
- `sensor_msgs/LaserScan`, `Imu`
- `std_msgs/Float32`, `Int64`, `Bool`, `String`
- And more via automatic field extraction

## References

Based on the research plan: *"Multi-Modal Diagnostic Agents: Grounding LLM Log Analysis with On-Demand ROS Bag Retrieval"*

## License

MIT
