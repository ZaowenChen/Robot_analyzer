# ROSBag Analyzer - Multi-Modal Diagnostic Agent
# Grounding LLM Log Analysis with On-Demand ROS Bag Retrieval
#
# Package structure:
#   core/       - Foundation: constants, utils, shared models
#   bridge/     - 4-tool Bridge API for ROS bag access
#   logs/       - Log parsing, pattern matching, state tracking
#   analysis/   - Diagnostic analysis engines
#   agent/      - LLM agent pipeline (LangGraph)
#   reporting/  - HTML reports, Foxglove layouts
#   cli/        - CLI entry points (python -m cli.analyze, etc.)
