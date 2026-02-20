"""
StateTimelineBuilder — Tracks robot state transitions from classified log events.

States are tracked as key-value pairs (e.g., "motion_state.still_flag" = "1").
When a new value is observed for a state key, a StateTransition is recorded.

This enables queries like:
  - "Was the robot stationary at time T?"
  - "How long was the IMU calibrating?"
  - "When did localization status change?"
"""

import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

from core.models import LogEvent, StateTransition
from core.utils import format_absolute_time
from logs.patterns import PATTERN_REGISTRY


class StateTimelineBuilder:
    """
    Builds a timeline of robot state transitions from classified log events.

    States are tracked as key-value pairs (e.g., "motion_state.still_flag" = "1").
    When a new value is observed for a state key, a StateTransition is recorded.

    This enables queries like:
      - "Was the robot stationary at time T?"
      - "How long was the IMU calibrating?"
      - "When did localization status change?"
    """

    def __init__(self):
        # state_key -> list of transitions (sorted by timestamp)
        self._transitions: Dict[str, List[StateTransition]] = defaultdict(list)
        # state_key.field -> current value
        self._current: Dict[str, str] = {}
        # All transitions in chronological order
        self._all_transitions: List[StateTransition] = []

    def process_event(self, event: LogEvent):
        """
        Process a log event and record state transitions if applicable.

        Only events whose matched pattern has a state_key will produce
        transitions. The value is extracted from parsed_data["value"].
        """
        if event.event_type == "UNMATCHED":
            return

        # Find the matching pattern to get state_key info
        pattern = None
        for p in PATTERN_REGISTRY:
            if p.event_type == event.event_type:
                pattern = p
                break

        if pattern is None or pattern.state_key is None:
            return

        # Extract the new value
        new_value = event.parsed_data.get("value", "")
        if not new_value:
            return

        state_key = pattern.state_key
        state_field = pattern.state_field or "value"
        full_key = f"{state_key}.{state_field}"

        old_value = self._current.get(full_key, "<unknown>")

        # Only record if value actually changed
        if old_value == str(new_value):
            return

        transition = StateTransition(
            timestamp=event.timestamp,
            timestamp_str=event.timestamp_str,
            state_key=state_key,
            field=state_field,
            old_value=old_value,
            new_value=str(new_value),
            bag_name=event.bag_name,
        )

        self._current[full_key] = str(new_value)
        self._transitions[full_key].append(transition)
        self._all_transitions.append(transition)

    def process_events(self, events: List[LogEvent]):
        """Process a list of log events (must be sorted by timestamp)."""
        for event in events:
            self.process_event(event)

    def get_state_at(self, timestamp: float) -> Dict[str, str]:
        """
        Get the robot state at a specific timestamp.

        Uses binary search on each state key's transition list to find
        the most recent value before the given timestamp.

        Returns:
            Dict mapping "state_key.field" -> value string
        """
        state = {}
        for full_key, transitions in self._transitions.items():
            if not transitions:
                continue

            # Binary search for the last transition before timestamp
            timestamps = [t.timestamp for t in transitions]
            idx = bisect.bisect_right(timestamps, timestamp) - 1

            if idx >= 0:
                state[full_key] = transitions[idx].new_value
            # If no transition before this time, the state is unknown

        return state

    def get_active_periods(self, state_key: str, field: str,
                           value: str) -> List[Tuple[float, float]]:
        """
        Get time periods where a state held a specific value.

        Args:
            state_key: e.g., "motion_state"
            field: e.g., "still_flag"
            value: e.g., "1"

        Returns:
            List of (start_time, end_time) tuples. The last period's
            end_time is float('inf') if the state is still active.
        """
        full_key = f"{state_key}.{field}"
        transitions = self._transitions.get(full_key, [])

        if not transitions:
            return []

        periods = []
        period_start = None

        for t in transitions:
            if t.new_value == value:
                if period_start is None:
                    period_start = t.timestamp
            else:
                if period_start is not None:
                    periods.append((period_start, t.timestamp))
                    period_start = None

        # If still in the target state at the end
        if period_start is not None:
            periods.append((period_start, float('inf')))

        return periods

    def is_state_active(self, timestamp: float, state_key: str,
                        field: str, value: str) -> bool:
        """Check if a specific state value is active at the given time."""
        full_key = f"{state_key}.{field}"
        state = self.get_state_at(timestamp)
        return state.get(full_key) == value

    def get_transitions(self, state_key: str = None,
                        field: str = None) -> List[StateTransition]:
        """
        Get state transitions, optionally filtered.

        Args:
            state_key: Filter by state key (e.g., "motion_state")
            field: Filter by field (e.g., "still_flag")

        Returns:
            List of StateTransition objects sorted by timestamp
        """
        if state_key is None and field is None:
            # Return all transitions sorted
            return sorted(self._all_transitions, key=lambda t: t.timestamp)

        if field is not None and state_key is not None:
            full_key = f"{state_key}.{field}"
            return list(self._transitions.get(full_key, []))

        if state_key is not None:
            # Return all transitions for this state_key (any field)
            result = []
            for full_key, transitions in self._transitions.items():
                if full_key.startswith(f"{state_key}."):
                    result.extend(transitions)
            result.sort(key=lambda t: t.timestamp)
            return result

        # field specified but not state_key — search all
        result = []
        for full_key, transitions in self._transitions.items():
            if full_key.endswith(f".{field}"):
                result.extend(transitions)
        result.sort(key=lambda t: t.timestamp)
        return result

    def get_summary(self) -> dict:
        """Generate a summary of all tracked states and their transitions."""
        summary = {}
        for full_key, transitions in sorted(self._transitions.items()):
            if not transitions:
                continue

            # Count time spent in each value
            value_durations: Dict[str, float] = defaultdict(float)
            value_counts: Dict[str, int] = defaultdict(int)

            for i, t in enumerate(transitions):
                value_counts[t.new_value] += 1
                if i + 1 < len(transitions):
                    duration = transitions[i + 1].timestamp - t.timestamp
                    value_durations[t.new_value] += duration

            summary[full_key] = {
                "total_transitions": len(transitions),
                "current_value": self._current.get(full_key, "<unknown>"),
                "values": [
                    {
                        "value": val,
                        "total_duration_sec": round(value_durations.get(val, 0), 1),
                        "transition_count": value_counts[val],
                    }
                    for val in sorted(set(
                        list(value_durations.keys()) + list(value_counts.keys())
                    ))
                ],
            }
        return summary
