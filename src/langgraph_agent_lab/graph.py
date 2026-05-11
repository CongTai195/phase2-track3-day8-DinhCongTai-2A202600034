"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside
the builder so unit tests that check schema/metrics can run even if students
are still debugging graph wiring.

Architecture
------------
Every route terminates at finalize → END.

    START → intake → classify → [conditional routing]
      simple       → answer   → finalize → END
      tool         → tool → evaluate → answer → finalize → END
      missing_info → clarify  → finalize → END
      risky        → risky_action → approval → tool → evaluate → answer → finalize → END
      error        → retry → tool → evaluate → [retry loop or answer → finalize → END]
      max retry    → dead_letter → finalize → END

    Rejection path (approval denied):
      approval (rejected) → clarify → finalize → END

Retry loop detail
-----------------
    error route → retry (increment attempt)
                       ↓ attempt < max_attempts
                      tool → evaluate
                               ↓ needs_retry → retry  (loop)
                               ↓ success     → answer → finalize → END
                       ↓ attempt >= max_attempts
                      dead_letter → finalize → END
"""

from __future__ import annotations

from typing import Any

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from .routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow.

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer (MemorySaver, SqliteSaver, …).
        When provided, every step is persisted so the graph can be
        resumed after a crash or interrupted for human-in-the-loop review.

    Returns
    -------
    CompiledGraph
        Ready-to-invoke compiled graph.  Call with::

            graph = build_graph(checkpointer)
            result = graph.invoke(state, config={"configurable": {"thread_id": "..."}})
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - helpful install error
        raise RuntimeError(
            "LangGraph is required. Run: pip install -e '.[dev]' or pip install langgraph"
        ) from exc

    graph = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    graph.add_node("intake", intake_node)           # normalize + PII redact
    graph.add_node("classify", classify_node)       # keyword-based routing
    graph.add_node("answer", answer_node)           # produce final response
    graph.add_node("tool", tool_node)               # mock external tool call
    graph.add_node("evaluate", evaluate_node)       # 'done?' check for retry loop
    graph.add_node("clarify", ask_clarification_node)  # ask for missing info / rejection notice
    graph.add_node("risky_action", risky_action_node)  # prepare risky action proposal
    graph.add_node("approval", approval_node)       # HITL approval gate
    graph.add_node("retry", retry_or_fallback_node) # increment attempt counter
    graph.add_node("dead_letter", dead_letter_node) # escalate unresolvable failures
    graph.add_node("finalize", finalize_node)       # emit audit event, every route ends here

    # ------------------------------------------------------------------
    # Edges — linear
    # ------------------------------------------------------------------
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")

    # tool result always flows into evaluation
    graph.add_edge("tool", "evaluate")

    # risky action must be approved before tool execution
    graph.add_edge("risky_action", "approval")

    # terminal nodes all converge on finalize
    graph.add_edge("answer", "finalize")
    graph.add_edge("clarify", "finalize")
    graph.add_edge("dead_letter", "finalize")
    graph.add_edge("finalize", END)

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------
    # classify → simple/tool/missing_info/risky/error
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )

    # evaluate → success (answer) or needs_retry (retry)
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "answer": "answer",
            "retry": "retry",
        },
    )

    # retry → loop back to tool OR escalate to dead_letter
    graph.add_conditional_edges(
        "retry",
        route_after_retry,
        {
            "tool": "tool",
            "dead_letter": "dead_letter",
        },
    )

    # approval → proceed to tool (approved) OR inform user (rejected → clarify)
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tool": "tool",
            "clarify": "clarify",
        },
    )

    return graph.compile(checkpointer=checkpointer)


def get_mermaid_diagram() -> str:  # pragma: no cover
    """Return the Mermaid diagram string for the compiled graph (bonus extension).

    Usage::

        from langgraph_agent_lab.graph import get_mermaid_diagram
        print(get_mermaid_diagram())
    """
    compiled = build_graph()
    return compiled.get_graph().draw_mermaid()
