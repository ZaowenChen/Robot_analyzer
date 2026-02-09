"""
LangGraph Diagnostic Agent

Implements the cyclic state machine described in the project plan:
  Supervisor -> ToolExecutor -> Supervisor -> ... -> Critic -> END

The agent uses a hypothesis-testing loop:
1. Supervisor formulates a hypothesis and picks tools to call
2. ToolExecutor runs the tools and updates the Evidence Locker
3. Loop back to Supervisor for refinement
4. When confident, produce final diagnosis

The Evidence Locker maintains structured findings validated by tools,
separate from the chat history, ensuring the final diagnosis is grounded.
"""

import json
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import END, StateGraph

try:
    from .prompt_templates import SYSTEM_PROMPT
    from .rosbag_bridge import ROSBagBridge
    from .tools import ALL_TOOLS
except ImportError:
    from prompt_templates import SYSTEM_PROMPT
    from rosbag_bridge import ROSBagBridge
    from tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# Agent State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """The memory structure passed between graph nodes."""
    # Standard conversation history
    messages: Annotated[list, operator.add]
    # Path to the bag being analyzed
    bag_path: str
    # The Evidence Locker: structured findings validated by tools
    evidence: Annotated[list, operator.add]
    # Iteration counter to prevent infinite loops
    steps: int
    # The current working hypothesis
    current_hypothesis: str
    # Final diagnosis output
    diagnosis: str


# ---------------------------------------------------------------------------
# Bridge instance
# ---------------------------------------------------------------------------
_bridge = ROSBagBridge()


# ---------------------------------------------------------------------------
# Tool Execution Logic
# ---------------------------------------------------------------------------
def _execute_tool(tool_name: str, args: dict) -> tuple:
    """Execute a bridge tool and return (output_str, evidence_items)."""
    evidence_items = []

    try:
        if tool_name == "get_bag_metadata":
            output = _bridge.get_bag_metadata(**args)
            output_str = json.dumps(output, indent=2)
            # Truncate if too long
            if len(output_str) > 8000:
                # Summarize
                summary = {
                    "duration": output["duration"],
                    "start_time": output["start_time"],
                    "end_time": output["end_time"],
                    "total_messages": output["total_messages"],
                    "num_topics": output["num_topics"],
                    "topic_names": [t["name"] for t in output.get("topics", [])],
                }
                output_str = json.dumps(summary, indent=2)
            evidence_items.append(
                f"Bag metadata: duration={output['duration']}s, "
                f"{output['num_topics']} topics, "
                f"{output['total_messages']} messages"
            )

        elif tool_name == "get_topic_statistics":
            output = _bridge.get_topic_statistics(**args)
            output_str = json.dumps(output, indent=2)
            # Check for anomalies and add to evidence
            for window in output:
                if "error" in window:
                    evidence_items.append(f"Error getting stats for {args.get('topic_name')}: {window['error']}")
                    continue
                for field_name, stats in window.get("fields", {}).items():
                    # Frozen sensor detection
                    if stats["std"] < 1e-6 and stats["count"] > 10:
                        evidence_items.append(
                            f"FROZEN: {args.get('topic_name')}.{field_name} "
                            f"std={stats['std']} (mean={stats['mean']}) "
                            f"window=[{window['window_start']}, {window['window_end']}] "
                            f"count={stats['count']}"
                        )
            # Truncate very long output
            if len(output_str) > 6000:
                output_str = output_str[:6000] + "\n... (truncated)"

        elif tool_name == "check_topic_frequency":
            output = _bridge.check_topic_frequency(**args)
            output_str = json.dumps(output, indent=2)
            # Check for frequency anomalies
            if "frequency_series" in output:
                mean_hz = output.get("mean_hz", 0)
                for entry in output["frequency_series"]:
                    if entry["hz"] < mean_hz * 0.3 and mean_hz > 0:
                        evidence_items.append(
                            f"FREQUENCY DROP: {args.get('topic_name')} "
                            f"at t={entry['time']}: {entry['hz']}Hz "
                            f"(mean={mean_hz}Hz)"
                        )
            if output.get("std_hz", 0) > output.get("mean_hz", 1) * 0.3:
                evidence_items.append(
                    f"UNSTABLE FREQUENCY: {args.get('topic_name')} "
                    f"mean={output.get('mean_hz')}Hz std={output.get('std_hz')}Hz"
                )
            # Truncate
            if len(output_str) > 6000:
                # Keep summary, truncate series
                summary = {k: v for k, v in output.items() if k != "frequency_series"}
                series = output.get("frequency_series", [])
                summary["frequency_series_sample"] = series[:10] + series[-10:] if len(series) > 20 else series
                output_str = json.dumps(summary, indent=2)

        elif tool_name == "sample_messages":
            output = _bridge.sample_messages(**args)
            output_str = json.dumps(output, indent=2, default=str)
            if len(output_str) > 6000:
                output_str = output_str[:6000] + "\n... (truncated)"

        else:
            output_str = f"Unknown tool: {tool_name}"

    except Exception as e:
        output_str = f"Error executing {tool_name}: {str(e)}"
        evidence_items.append(f"Tool error: {tool_name} - {str(e)}")

    return output_str, evidence_items


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
def supervisor_node(state: AgentState, llm_with_tools) -> dict:
    """
    The Supervisor (Reasoning Engine).
    Analyzes user query and evidence to decide next action.
    """
    # Build system prompt with current evidence
    evidence_str = "\n".join(f"  - {e}" for e in state.get("evidence", [])) or "  (none yet)"
    system_prompt = SYSTEM_PROMPT.format(
        evidence=evidence_str,
        bag_path=state["bag_path"],
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "steps": state.get("steps", 0) + 1,
    }


def tool_executor_node(state: AgentState) -> dict:
    """
    The Tool Executor (Bridge Interface).
    Executes tool calls from the Supervisor and updates the Evidence Locker.
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "evidence": []}

    tool_calls = last_message.tool_calls
    results = []
    new_evidence = []

    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]

        # Ensure bag_path is set
        if "bag_path" not in args and tool_name != "sample_messages":
            args["bag_path"] = state["bag_path"]
        elif "bag_path" not in args:
            args["bag_path"] = state["bag_path"]

        output_str, evidence_items = _execute_tool(tool_name, args)
        new_evidence.extend(evidence_items)

        results.append(ToolMessage(
            tool_call_id=call["id"],
            content=output_str,
        ))

    return {
        "messages": results,
        "evidence": new_evidence,
    }


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------
def route_supervisor(state: AgentState) -> str:
    """Determine next node based on Supervisor output."""
    last_msg = state["messages"][-1]

    # If the LLM wants to call tools, go to ToolExecutor
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"

    # If we've exceeded max steps, force end
    if state.get("steps", 0) >= 15:
        return END

    # LLM is done (no tool calls = final answer)
    return END


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------
def build_diagnostic_graph(llm):
    """
    Construct the LangGraph diagnostic workflow.

    Graph structure:
      supervisor -> (tool_calls?) -> tool_executor -> supervisor -> ... -> END
    """
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Create supervisor with bound LLM
    def supervisor_fn(state):
        return supervisor_node(state, llm_with_tools)

    # Build graph
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_fn)
    workflow.add_node("tool_executor", tool_executor_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges("supervisor", route_supervisor)
    workflow.add_edge("tool_executor", "supervisor")

    return workflow.compile()


def run_diagnostic(
    bag_path: str,
    query: str,
    llm=None,
    verbose: bool = True,
) -> dict:
    """
    Run a complete diagnostic session on a ROS bag.

    Args:
        bag_path: Path to the .bag file
        query: The diagnostic question (e.g., "Analyze this bag for anomalies")
        llm: The LLM to use. If None, uses ChatAnthropic claude-sonnet.
        verbose: If True, print intermediate steps.

    Returns:
        dict with diagnosis, evidence, and step count
    """
    if llm is None:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
        except Exception:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o", temperature=0)

    app = build_diagnostic_graph(llm)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "bag_path": bag_path,
        "evidence": [],
        "steps": 0,
        "current_hypothesis": "",
        "diagnosis": "",
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC SESSION")
        print(f"Bag: {bag_path}")
        print(f"Query: {query}")
        print(f"{'='*70}\n")

    # Stream events for visibility
    final_state = None
    for event in app.stream(initial_state, {"recursion_limit": 30}):
        for node_name, state_update in event.items():
            if verbose:
                print(f"\n--- [{node_name}] Step ---")
                msgs = state_update.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            content_preview = msg.content[:500]
                            print(f"  AI: {content_preview}")
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  Tool Call: {tc['name']}({json.dumps(tc['args'], default=str)[:200]})")
                    elif isinstance(msg, ToolMessage):
                        content_preview = msg.content[:300]
                        print(f"  Tool Result: {content_preview}...")
                new_evidence = state_update.get("evidence", [])
                if new_evidence:
                    print(f"  New Evidence: {new_evidence}")
            final_state = state_update

    # Extract final diagnosis from last AI message
    diagnosis = ""
    all_messages = initial_state["messages"]
    # Collect all messages from stream events
    for event in app.stream(initial_state, {"recursion_limit": 30}):
        pass

    # Get the complete state by invoking
    result_state = app.invoke(initial_state, {"recursion_limit": 30})

    # Find the last AI message for diagnosis
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            diagnosis = msg.content
            break

    return {
        "diagnosis": diagnosis,
        "evidence": result_state.get("evidence", []),
        "steps": result_state.get("steps", 0),
        "num_tool_calls": sum(
            1 for m in result_state["messages"] if isinstance(m, ToolMessage)
        ),
    }
