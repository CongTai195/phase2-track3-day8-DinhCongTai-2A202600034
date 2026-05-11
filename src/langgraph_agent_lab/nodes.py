"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event

# ---------------------------------------------------------------------------
# Routing keyword sets — priority order: risky > tool > missing_info > error
# ---------------------------------------------------------------------------
_RISKY_KEYWORDS = {"refund", "delete", "send", "cancel", "remove", "revoke"}
_TOOL_KEYWORDS = {"status", "order", "lookup", "check", "track", "find", "search"}
_ERROR_KEYWORDS = {"timeout", "fail", "failure", "error", "crash", "unavailable", "recover"}

# Pronouns that signal vague / missing-info queries when query is short
_VAGUE_PRONOUNS = {"it", "this", "that", "they", "them"}


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation, returning individual tokens."""
    return re.sub(r"[?!.,;:]", "", text.lower()).split()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    Strips leading/trailing whitespace, collapses internal whitespace,
    and redacts simple PII patterns (email addresses).
    """
    raw = state.get("query", "")
    # Collapse whitespace
    query = " ".join(raw.split())
    # Redact email addresses (basic PII check)
    query = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]", query)

    word_count = len(query.split())
    return {
        "query": query,
        "messages": [f"intake:{query[:60]}"],
        "events": [
            make_event(
                "intake",
                "completed",
                "query normalized",
                word_count=word_count,
                pii_redacted="[EMAIL]" in query,
            )
        ],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using keyword heuristics.

    Priority: risky > tool > missing_info > error > simple
    Matching is done on whole tokens to avoid substring false-positives
    (e.g. "it" must not match "item" or "iteration").
    """
    query = state.get("query", "")
    tokens = set(_tokenize(query))

    route = Route.SIMPLE
    risk_level = "low"

    # 1. Risky — highest priority
    if tokens & _RISKY_KEYWORDS:
        route = Route.RISKY
        risk_level = "high"

    # 2. Tool — lookup / external call needed
    elif tokens & _TOOL_KEYWORDS:
        route = Route.TOOL
        risk_level = "medium"

    # 3. Missing info — short/vague query with pronouns
    elif len(tokens) < 5 and tokens & _VAGUE_PRONOUNS:
        route = Route.MISSING_INFO

    # 4. Error / failure signals
    elif tokens & _ERROR_KEYWORDS:
        route = Route.ERROR

    # 5. Default — simple self-service answer
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}",
                risk_level=risk_level,
                matched_tokens=list(tokens & (_RISKY_KEYWORDS | _TOOL_KEYWORDS | _ERROR_KEYWORDS)),
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generates a specific clarification question derived from the query so
    the user knows exactly what additional context is needed.
    """
    query = state.get("query", "").strip() or "your request"
    question = (
        f"I'm sorry, I didn't quite catch the details of '{query}'. "
        "Could you please provide the relevant order ID, account number, "
        "or describe the issue in more detail?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            make_event(
                "clarify",
                "completed",
                "missing information requested",
                original_query=query,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool with idempotent, structured results.

    For error-route scenarios: simulates transient failures on the first two
    attempts so the retry loop can be exercised. All other routes always
    succeed on the first call.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    route = state.get("route", "")

    # Simulate transient failure for error-route scenarios (attempts 0 and 1)
    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={scenario_id}"
        status = "error"
    else:
        # Structured success payload
        result = (
            f"tool_result::scenario={scenario_id} "
            f"status=ok attempt={attempt} "
            f"data={{\"resolved\": true}}"
        )
        status = "ok"

    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                "completed",
                f"tool executed attempt={attempt} status={status}",
                attempt=attempt,
                status=status,
                scenario_id=scenario_id,
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action with evidence and risk justification.

    Constructs a structured proposed-action payload so the approval node
    (human or mock) has full context to make an informed decision.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    scenario_id = state.get("scenario_id", "unknown")

    # Identify which risky verb triggered this route
    tokens = set(_tokenize(query))
    matched = list(tokens & _RISKY_KEYWORDS) or ["unknown-action"]
    action_verb = matched[0]

    proposed = (
        f"ACTION: {action_verb} | scenario={scenario_id} | risk={risk_level} | "
        f"query='{query[:80]}' | justification='User explicitly requested {action_verb}; "
        "requires supervisor approval before execution.'"
    )

    return {
        "proposed_action": proposed,
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                "approval required",
                action_verb=action_verb,
                risk_level=risk_level,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses a mock decision so tests and CI run offline.

    On rejection the action is logged and the workflow routes to a safe
    fallback (the answer node will detect the rejected approval).
    """
    import os

    proposed_action = state.get("proposed_action")
    risk_level = state.get("risk_level", "high")

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt(
            {
                "proposed_action": proposed_action,
                "risk_level": risk_level,
                "instructions": "Reply with {\"approved\": true/false, \"comment\": \"...\"}",
            }
        )
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        # Mock approval — always approve in offline/test mode
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="auto-approved for lab (set LANGGRAPH_INTERRUPT=true for real HITL)",
        )

    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                "completed",
                f"approved={decision.approved}",
                reviewer=decision.reviewer,
                comment=decision.comment,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt with bounded counter and backoff metadata.

    Increments the attempt counter. The routing function (route_after_retry)
    will decide whether to loop back to the tool node or escalate to the
    dead-letter node based on max_attempts.
    """
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))

    # Exponential backoff metadata (informational — no actual sleep in mock)
    backoff_ms = min(100 * (2 ** (attempt - 1)), 2000)

    error_msg = (
        f"transient failure attempt={attempt}/{max_attempts} "
        f"backoff_ms={backoff_ms}"
    )

    return {
        "attempt": attempt,
        "errors": [error_msg],
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=attempt,
                max_attempts=max_attempts,
                backoff_ms=backoff_ms,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in tool results and approval context.

    Builds the answer from available state so it reflects what actually
    happened (tool lookup, risky approval, error recovery, etc.).
    """
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    route = state.get("route", "")
    query = state.get("query", "")

    if route == Route.RISKY.value and approval:
        if approval.get("approved"):
            action = state.get("proposed_action") or "the requested action"
            last_tool = tool_results[-1] if tool_results else "completed"
            answer = (
                f"Your request has been approved and executed. "
                f"Action: {action[:120]} | Result: {last_tool[:120]}"
            )
        else:
            answer = (
                "Your request was reviewed and denied. "
                f"Reason: {approval.get('comment', 'No additional information provided.')}"
            )
    elif tool_results:
        answer = f"Here is what I found: {tool_results[-1]}"
    else:
        # Simple / clarification route — no tool call
        answer = (
            f"I can help with '{query[:80]}'. "
            "Please follow the standard process or contact support for further assistance."
        )

    return {
        "final_answer": answer,
        "events": [
            make_event(
                "answer",
                "completed",
                "answer generated",
                route=route,
                has_tool_results=bool(tool_results),
                approved=approval.get("approved") if approval else None,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' gate that enables retry loops.

    Returns evaluation_result='needs_retry' when the latest tool result
    indicates an error, otherwise 'success'. A structured result format
    is used so future LLM-as-judge implementations can replace this heuristic
    without changing the routing contract.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""

    # Heuristic: any result starting with or containing "ERROR:" is a failure
    if latest.startswith("ERROR:") or "ERROR:" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    "tool result indicates failure — retry needed",
                    latest_result=latest[:120],
                    evaluation_result="needs_retry",
                )
            ],
        }

    return {
        "evaluation_result": "success",
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result satisfactory",
                latest_result=latest[:120],
                evaluation_result="success",
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of the error strategy: retry → fallback → dead letter.
    In production this would persist to a dead-letter queue, page on-call,
    and open an incident ticket. Here we record a rich audit event and set
    a clear final answer indicating manual intervention is required.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    errors = state.get("errors", [])

    return {
        "final_answer": (
            f"Request '{state.get('query', '')[:80]}' could not be completed after "
            f"{attempt} attempt(s). Escalated to manual review queue. "
            f"Incident reference: dead-letter/{scenario_id}"
        ),
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "max retries exceeded — escalated to manual review",
                attempt=attempt,
                scenario_id=scenario_id,
                error_count=len(errors),
                last_error=errors[-1] if errors else "none",
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                scenario_id=state.get("scenario_id", "unknown"),
                has_answer=bool(state.get("final_answer")),
            )
        ]
    }
