# Refactoring Log

This file tracks structural changes to the codebase — file splits, moves,
renames, dead code removal, and directory reorganizations. Each entry explains
what changed and why, so future developers can understand the project's evolution.

---

## 2026-02-20 — Restructure flat codebase into focused packages

### Changes

Reorganized ~15,500 lines of Python from 12 flat root-level files into 7 focused packages with a strict DAG dependency structure.

**Packages created:**

1. **`core/`** (constants.py, utils.py, models.py)
   - Consolidated domain knowledge previously duplicated across 5+ files
   - `CST` timezone: was copied in analyze.py, log_parser.py, cross_validator.py, report_generator.py, rosbag_diagnostic_extractor/parser.py → now single source in core/utils.py
   - `LOG_LEVELS`: was copied in rosbag_bridge.py, log_parser.py, rosbag_profiler.py → now single source in core/utils.py
   - `format_absolute_time()`: was duplicated in analyze.py and log_parser.py → now in core/utils.py
   - `EXPECTED_ZERO_FIELDS`, `HEALTH_FLAG_NAMES`, `HARDWARE_TOPICS`: were duplicated in analyze.py and rosbag_profiler.py → now in core/constants.py
   - All shared dataclasses (LogEvent, Incident, BagInfo, etc.) → core/models.py

2. **`bridge/`** (truncated_reader.py, welford.py, field_extractors.py, bridge.py)
   - Split from rosbag_bridge.py (1,117 lines → 4 focused files)

3. **`logs/`** (patterns.py, extractor.py, state_timeline.py, denoiser.py)
   - Split from log_parser.py (781 lines → 3 files)
   - Extracted LogDenoiser from cross_validator.py → logs/denoiser.py

4. **`analysis/`** (diagnostic_analyzer.py, incident_builder.py, mission_orchestrator.py, cross_validator.py)
   - Split from analyze.py (2,266 lines → 4 files)
   - Extracted CrossValidator + SensorSnapshotBuilder from cross_validator.py

5. **`agent/`** (graph.py, tools.py, prompts.py)
   - Moved from root: agent.py → agent/graph.py, tools.py → agent/tools.py, prompt_templates.py → agent/prompts.py

6. **`reporting/`** (html_report.py, foxglove_layouts.py)
   - Moved from root: report_generator.py → reporting/html_report.py, foxglove_layout_generator.py → reporting/foxglove_layouts.py

7. **`cli/`** (analyze.py, profiler.py, validate.py, bridge_test.py)
   - Extracted CLI entry points from old root-level files
   - cli/profiler.py uses core/constants instead of duplicating domain knowledge

**Files deleted (11 root-level files):**
- analyze.py, cross_validator.py, log_parser.py, rosbag_bridge.py
- agent.py, tools.py, prompt_templates.py
- report_generator.py, foxglove_layout_generator.py
- rosbag_profiler.py, validate.py

**Files modified:**
- rosbag_diagnostic_extractor/parser.py: updated `from rosbag_bridge` → `from bridge`
- CLAUDE.md: updated with new architecture and CLI commands
- __init__.py: updated with package structure overview

### Rationale

The codebase had grown to ~15,500 lines across 25 files, all sitting flat in the project root. The 4 largest files (analyze.py at 2,266 lines, cross_validator.py at 1,172 lines, rosbag_bridge.py at 1,117 lines, log_parser.py at 781 lines) each contained 2-3 unrelated responsibilities. Domain constants like CST and LOG_LEVELS were duplicated across 3-5 files. This made it difficult to add new analysis modes or hardware checks without touching massive monolithic files.

The new package structure ensures:
- Each file has one clear responsibility (max ~660 lines, down from 2,266)
- Domain knowledge lives in one place (core/constants.py, core/utils.py)
- New functionality can be added by creating new files, not editing monoliths
- Strict DAG dependency prevents circular imports

### Migration Notes

All CLI commands now use `python -m cli.X` instead of `python3 X.py`:
- `python3 analyze.py` → `python -m cli.analyze`
- `python3 validate.py` → `python -m cli.validate`
- `python3 rosbag_profiler.py` → `python -m cli.profiler`
- `python3 rosbag_bridge.py <bag>` → `python -m cli.bridge_test <bag>`

No backward-compatibility shims are provided — all imports must use the new package paths (e.g., `from bridge import ROSBagBridge` instead of `from rosbag_bridge import ROSBagBridge`).

The `rosbag_diagnostic_extractor/` subpackage is unchanged (already well-structured).

---
