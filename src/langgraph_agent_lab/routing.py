"""Routing functions for conditional edges.

Each function receives the full AgentState and returns the name of the next
node as a string.  Return values must exactly match the node names registered
in graph.py.
"""

from __future__ import annotations

import logging

from .state import AgentState, Route

logger = logging.getLogger(__name__)

# Node-name constants so typos are caught at import time, not at runtime
_NODE_ANSWER = "answer"
_NODE_CLARIFY = "clarify"
_NODE_DEAD_LETTER = "dead_letter"
_NODE_RETRY = "retry"
_NODE_RISKY_ACTION = "risky_action"
_NODE_TOOL = "tool"


def route_after_classify(state: AgentState) -> str:
    """Map the classified route value to the next graph node.

    Handles unknown route values safely by defaulting to the answer node
    and logging a warning, so the graph always terminates even when an
    unexpected route string ends up in state.

    Route → Node mapping
    --------------------
    simple       → answer
    tool         → tool
    missing_info → clarify
    risky        → risky_action
    error        → retry       (enters the retry/evaluate loop)
    <unknown>    → answer      (safe fallback with a warning)
    """
    route = state.get("route", Route.SIMPLE.value)

    mapping: dict[str, str] = {
        Route.SIMPLE.value: _NODE_ANSWER,
        Route.TOOL.value: _NODE_TOOL,
        Route.MISSING_INFO.value: _NODE_CLARIFY,
        Route.RISKY.value: _NODE_RISKY_ACTION,
        Route.ERROR.value: _NODE_RETRY,
    }

    next_node = mapping.get(route)
    if next_node is None:
        logger.warning(
            "route_after_classify: unknown route %r for scenario=%r — defaulting to 'answer'",
            route,
            state.get("scenario_id", "unknown"),
        )
        next_node = _NODE_ANSWER

    return next_node


def route_after_evaluate(state: AgentState) -> str:
    """Decide whether the tool result is satisfactory or needs a retry.

    This is the 'done?' gate that powers the retry loop — a key advantage
    of LangGraph conditional edges over plain LCEL chains.

    evaluation_result == 'needs_retry' → retry node (increment attempt, loop)
    evaluation_result == 'success'     → answer node (produce final response)
    missing / unexpected value         → answer node (safe default)
    """
    evaluation_result = state.get("evaluation_result") or ""

    if evaluation_result == "needs_retry":
        return _NODE_RETRY

    if evaluation_result not in ("success", "needs_retry"):
        logger.warning(
            "route_after_evaluate: unexpected evaluation_result=%r — defaulting to 'answer'",
            evaluation_result,
        )

    return _NODE_ANSWER


def route_after_retry(state: AgentState) -> str:
    """Decide whether to loop back to the tool or escalate to dead-letter.

    Bounded retry: if attempt has reached (or exceeded) max_attempts the
    request is unresolvable and is routed to the dead-letter node for manual
    review.  Otherwise we loop back to call the tool again.

    attempt < max_attempts  → tool       (retry the tool call)
    attempt >= max_attempts → dead_letter (exhaust retries, escalate)
    """
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))

    if attempt >= max_attempts:
        logger.info(
            "route_after_retry: max attempts reached (attempt=%d, max=%d) → dead_letter",
            attempt,
            max_attempts,
        )
        return _NODE_DEAD_LETTER

    return _NODE_TOOL


def route_after_approval(state: AgentState) -> str:
    """Continue to tool execution only when the action is approved.

    Approved  → tool     (proceed with the risky action via tool call)
    Rejected  → clarify  (inform the user the action was denied)
    Missing   → clarify  (treat missing approval as an implicit rejection)

    The clarify node will surface the rejection reason from approval.comment
    to the user via the answer node.
    """
    approval = state.get("approval") or {}
    approved = approval.get("approved", False)

    if approved:
        return _NODE_TOOL

    # Log the rejection reason for observability
    reason = approval.get("comment", "no comment provided")
    logger.info(
        "route_after_approval: action rejected for scenario=%r — reason: %s",
        state.get("scenario_id", "unknown"),
        reason,
    )
    return _NODE_CLARIFY
