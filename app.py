"""Streamlit UI — LangGraph Agent Lab (Day 08 Bonus Extension).

Provides:
  - Custom query runner with live execution trace
  - Predefined scenario runner
  - HITL approval/reject interface for risky actions
  - Graph diagram viewer
  - Metrics dashboard

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import AgentState, Route, initial_state

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Agent Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}

/* Route badge */
.route-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.route-simple    { background: #1a472a; color: #69db7c; border: 1px solid #2f9e44; }
.route-tool      { background: #1c3d6e; color: #74c0fc; border: 1px solid #339af0; }
.route-missing-info { background: #4a3728; color: #ffc078; border: 1px solid #fd7e14; }
.route-risky     { background: #4a1515; color: #ff8787; border: 1px solid #fa5252; }
.route-error     { background: #2e1a47; color: #cc5de8; border: 1px solid #ae3ec9; }
.route-dead-letter { background: #1a1a1a; color: #868e96; border: 1px solid #495057; }

/* Node event pill */
.event-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 3px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    color: #e9ecef;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, rgba(105,219,124,0.08), rgba(51,154,240,0.08));
    border: 1px solid rgba(105,219,124,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #f1f3f5;
}

/* Approval box */
.approval-box {
    background: linear-gradient(135deg, rgba(250,82,82,0.1), rgba(250,176,5,0.08));
    border: 2px solid rgba(250,82,82,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

/* Metric card */
.metric-item {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Header gradient text */
.gradient-text {
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}

/* Success/error indicators */
.success-dot { color: #69db7c; font-size: 1.2rem; }
.error-dot   { color: #ff6b6b; font-size: 1.2rem; }

code, pre { font-family: 'JetBrains Mono', monospace !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
ROUTE_COLORS = {
    "simple":       ("🟢", "route-simple"),
    "tool":         ("🔵", "route-tool"),
    "missing_info": ("🟠", "route-missing-info"),
    "risky":        ("🔴", "route-risky"),
    "error":        ("🟣", "route-error"),
    "dead_letter":  ("⚫", "route-dead-letter"),
}

NODE_ICONS = {
    "intake":       "📥",
    "classify":     "🏷️",
    "tool":         "🔧",
    "evaluate":     "🔍",
    "retry":        "🔄",
    "dead_letter":  "💀",
    "risky_action": "⚠️",
    "approval":     "✅",
    "clarify":      "❓",
    "answer":       "💬",
    "finalize":     "🏁",
}

SAMPLE_QUERIES = [
    # 🟢 simple
    "How do I reset my password?",
    "What are your business hours?",
    "How long does shipping usually take?",
    # 🔵 tool
    "Please lookup order status for order 12345",
    "Can you track my shipment for order 99876?",
    "Search for all orders placed by customer john@example.com",
    "Find the invoice for transaction 77432",
    # 🟠 missing_info
    "Can you fix it?",
    "Help me with this",
    "Fix that please",
    # 🔴 risky
    "Refund this customer and send confirmation email",
    "Delete customer account after support verification",
    "Cancel my subscription and stop all future charges",
    "Remove inactive user accounts from the system",
    "Revoke API access for this partner immediately",
    "Refund and look up order status for customer 789",
    # 🟣 error / dead-letter
    "Timeout failure while processing request",
    "The payment gateway crashed and is unavailable",
    "Connection error while processing the refund request — please retry",
    "Service unavailable — cannot recover after repeated errors",
    "System failure cannot recover after multiple attempts",
    # 🔵 boundary/priority tests
    "I need to check the item details for order 55123",
]

# ──────────────────────────────────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────────────────────────────────
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(checkpointer=st.session_state.checkpointer)
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "scenarios" not in st.session_state:
    try:
        st.session_state.scenarios = load_scenarios("data/sample/scenarios.jsonl")
    except Exception:
        st.session_state.scenarios = []

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def run_query(query: str, scenario_id: str = "custom", max_attempts: int = 3) -> dict[str, Any]:
    """Run the graph on a query and return final state."""
    from langgraph_agent_lab.state import Scenario

    scenario = Scenario(
        id=scenario_id,
        query=query,
        expected_route=Route.SIMPLE,
        max_attempts=max_attempts,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    t0 = time.perf_counter()
    final_state = st.session_state.graph.invoke(state, config=config)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {"state": final_state, "elapsed_ms": elapsed_ms, "query": query}


def route_badge(route: str) -> str:
    icon, cls = ROUTE_COLORS.get(route, ("⚪", "route-simple"))
    return f'<span class="route-badge {cls}">{icon} {route}</span>'


def render_events(events: list[dict]) -> None:
    if not events:
        return
    st.markdown("**Execution trace:**")
    pills_html = ""
    for ev in events:
        node = ev.get("node", "?")
        icon = NODE_ICONS.get(node, "⬜")
        ev_type = ev.get("event_type", "")
        pills_html += f'<span class="event-pill">{icon} {node}</span>'
    st.markdown(f'<div style="margin:0.5rem 0">{pills_html}</div>', unsafe_allow_html=True)


def render_result(result: dict) -> None:
    state = result["state"]
    route = state.get("route", "unknown")
    icon, _ = ROUTE_COLORS.get(route, ("⚪", ""))
    answer = state.get("final_answer") or state.get("pending_question") or "—"
    errors = state.get("errors", [])
    events = state.get("events", [])
    retries = sum(1 for e in events if e.get("node") == "retry")
    approvals = sum(1 for e in events if e.get("node") == "approval")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Route**<br>{route_badge(route)}", unsafe_allow_html=True)
    with col2:
        st.metric("Nodes visited", len(events))
    with col3:
        st.metric("Retries", retries)
    with col4:
        st.metric("Latency", f"{result['elapsed_ms']} ms")

    st.markdown(
        f'<div class="answer-box">💬 <strong>Response</strong><br><br>{answer}</div>',
        unsafe_allow_html=True,
    )

    if errors:
        with st.expander(f"⚠️ Errors ({len(errors)})"):
            for e in errors:
                st.code(e, language=None)

    render_events(events)

    with st.expander("🗂️ Full state"):
        display = {k: v for k, v in state.items() if k != "events"}
        st.json(display)

    with st.expander("📋 Audit events"):
        st.json(events)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h2 class="gradient-text">🤖 LangGraph Lab</h2>',
        unsafe_allow_html=True,
    )
    st.markdown("**Day 08 · Agentic Orchestration**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🚀 Run Query", "📋 Scenarios", "📊 Metrics", "🗺️ Graph Diagram"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Config**")
    max_attempts = st.slider("Max retry attempts", 1, 5, 3)

    if st.button("🔄 Reset graph"):
        st.session_state.checkpointer = MemorySaver()
        st.session_state.graph = build_graph(checkpointer=st.session_state.checkpointer)
        st.session_state.run_history = []
        st.rerun()

    st.divider()
    st.markdown(
        """
<small style="color:#868e96">
Routes:<br>
🟢 simple · 🔵 tool<br>
🟠 missing_info · 🔴 risky<br>
🟣 error · ⚫ dead_letter
</small>
""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────
# Page: Run Query
# ──────────────────────────────────────────────────────────────────────────
if page == "🚀 Run Query":
    st.markdown('<h1 class="gradient-text">Run a Support Query</h1>', unsafe_allow_html=True)
    st.markdown("Type any support ticket query and watch the agent route and respond in real-time.")

    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        sample = st.selectbox(
            "Load example",
            ["— pick one —"] + SAMPLE_QUERIES,
            label_visibility="visible",
        )

    with col_input:
        default_query = "" if sample == "— pick one —" else sample
        query = st.text_area(
            "Your query",
            value=default_query,
            height=100,
            placeholder="e.g. Refund this customer and send confirmation email",
        )

    run_clicked = st.button("▶ Run", use_container_width=True)

    if run_clicked and query.strip():
        with st.spinner("Running graph…"):
            result = run_query(query.strip(), max_attempts=max_attempts)
            st.session_state.run_history.insert(0, result)

        st.success("✅ Graph completed")
        render_result(result)

    elif run_clicked:
        st.warning("Please enter a query first.")

    # ── History ───────────────────────────────────────────────────────────
    if st.session_state.run_history:
        st.divider()
        st.markdown("### 📜 Recent runs")
        for i, past in enumerate(st.session_state.run_history[1:6], 1):
            s = past["state"]
            route = s.get("route", "?")
            icon, _ = ROUTE_COLORS.get(route, ("⚪", ""))
            events = s.get("events", [])
            with st.expander(
                f"{icon} `{past['query'][:60]}` → **{route}** · {len(events)} nodes · {past['elapsed_ms']} ms"
            ):
                render_result(past)


# ──────────────────────────────────────────────────────────────────────────
# Page: Scenarios
# ──────────────────────────────────────────────────────────────────────────
elif page == "📋 Scenarios":
    st.markdown('<h1 class="gradient-text">Grading Scenarios</h1>', unsafe_allow_html=True)
    st.markdown("Run the official 7 grading scenarios from `data/sample/scenarios.jsonl`.")

    scenarios = st.session_state.scenarios
    if not scenarios:
        st.error("Could not load scenarios.jsonl — run from project root directory.")
    else:
        run_all = st.button("▶ Run all 7 scenarios", use_container_width=True)

        if run_all:
            progress = st.progress(0, text="Running scenarios…")
            results = []
            for i, s in enumerate(scenarios):
                with st.spinner(f"Running {s.id}…"):
                    max_att = s.max_attempts
                    scenario_state = initial_state(s)
                    config = {"configurable": {"thread_id": scenario_state["thread_id"]}}
                    t0 = time.perf_counter()
                    final = st.session_state.graph.invoke(scenario_state, config=config)
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    actual = final.get("route", "?")
                    success = actual == s.expected_route.value and bool(
                        final.get("final_answer") or final.get("pending_question")
                    )
                    results.append({
                        "state": final,
                        "elapsed_ms": elapsed,
                        "query": s.query,
                        "scenario": s,
                        "success": success,
                        "actual_route": actual,
                    })
                progress.progress((i + 1) / len(scenarios), text=f"{s.id}: {actual}")

            progress.empty()

            # Summary table
            total = len(results)
            passed = sum(1 for r in results if r["success"])
            st.markdown(
                f'<div class="card"><h3 style="margin:0">Results: '
                f'<span style="color:#69db7c">{passed}/{total} passed</span> · '
                f'Success rate: <span style="color:#69db7c">{passed/total:.0%}</span></h3></div>',
                unsafe_allow_html=True,
            )

            # Per-scenario rows
            for r in results:
                s = r["scenario"]
                ok = r["success"]
                actual = r["actual_route"]
                icon, _ = ROUTE_COLORS.get(actual, ("⚪", ""))
                status = "✅" if ok else "❌"
                events = r["state"].get("events", [])
                retries = sum(1 for e in events if e.get("node") == "retry")

                with st.expander(
                    f"{status} **{s.id}** — {icon} `{actual}` "
                    f"{'(expected: ' + s.expected_route.value + ')' if not ok else ''} "
                    f"· {r['elapsed_ms']} ms"
                ):
                    st.markdown(f"**Query:** {s.query}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected", s.expected_route.value)
                    col2.metric("Actual", actual)
                    col3.metric("Retries", retries)
                    render_events(events)
                    answer = r["state"].get("final_answer") or r["state"].get("pending_question") or "—"
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            # Preview table
            st.markdown("### Scenarios")
            for s in scenarios:
                icon, _ = ROUTE_COLORS.get(s.expected_route.value, ("⚪", ""))
                st.markdown(
                    f"- {icon} **{s.id}** · `{s.query[:70]}` → **{s.expected_route.value}**"
                    + (" 🔒 HITL" if s.requires_approval else "")
                )


# ──────────────────────────────────────────────────────────────────────────
# Page: Metrics
# ──────────────────────────────────────────────────────────────────────────
elif page == "📊 Metrics":
    st.markdown('<h1 class="gradient-text">Metrics Dashboard</h1>', unsafe_allow_html=True)

    try:
        import json
        from pathlib import Path

        metrics_path = Path("outputs/metrics.json")
        if not metrics_path.exists():
            st.warning("No metrics.json found. Run `make run-scenarios` first.")
        else:
            m = json.loads(metrics_path.read_text())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total scenarios", m["total_scenarios"])
            c2.metric("Success rate", f"{m['success_rate']:.0%}")
            c3.metric("Avg nodes visited", f"{m['avg_nodes_visited']:.1f}")
            c4.metric("Total retries", m["total_retries"])

            st.divider()
            st.markdown("### Per-scenario breakdown")

            for s in m["scenario_metrics"]:
                ok = s["success"]
                route = s["actual_route"] or "?"
                icon, _ = ROUTE_COLORS.get(route, ("⚪", ""))
                status = "✅" if ok else "❌"

                with st.expander(
                    f"{status} **{s['scenario_id']}** · {icon} `{route}` · "
                    f"{s['nodes_visited']} nodes · {s['retry_count']} retries"
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Expected", s["expected_route"])
                    col2.metric("Actual", route)
                    col3.metric("Retries", s["retry_count"])
                    col4.metric("Interrupts", s["interrupt_count"])
                    if s["approval_required"]:
                        st.info(f"HITL: approval_required=True, approval_observed={s['approval_observed']}")
                    if s["errors"]:
                        for e in s["errors"]:
                            st.code(e, language=None)

            st.divider()
            st.markdown("### Raw metrics.json")
            st.json(m)

    except Exception as exc:
        st.error(f"Error loading metrics: {exc}")


# ──────────────────────────────────────────────────────────────────────────
# Page: Graph Diagram
# ──────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Graph Diagram":
    st.markdown('<h1 class="gradient-text">Graph Architecture</h1>', unsafe_allow_html=True)
    st.markdown("Live Mermaid diagram exported from the compiled LangGraph `StateGraph`.")

    try:
        compiled = st.session_state.graph
        mermaid_src = compiled.get_graph().draw_mermaid()

        # Remove the YAML front-matter block before the mermaid fence
        # (Mermaid in streamlit works better without it)
        lines = mermaid_src.strip().splitlines()
        clean_lines = []
        skip = False
        for line in lines:
            if line.strip() == "---":
                skip = not skip
                continue
            if not skip:
                clean_lines.append(line)
        clean_mermaid = "\n".join(clean_lines)

        st.markdown(
            f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
            border-radius:16px;padding:2rem;overflow-x:auto;">

```mermaid
{clean_mermaid}
```

</div>
""",
            unsafe_allow_html=False,
        )

        st.divider()
        st.markdown("### Node legend")

        cols = st.columns(4)
        legend = [
            ("📥 intake", "Normalize & PII redact"),
            ("🏷️ classify", "Keyword routing (priority-ordered)"),
            ("🔧 tool", "Mock external tool (simulates failures)"),
            ("🔍 evaluate", "Done? check — enables retry loop"),
            ("🔄 retry", "Increment attempt + backoff metadata"),
            ("💀 dead_letter", "Max retries exceeded → manual review"),
            ("⚠️ risky_action", "Build proposed-action payload"),
            ("✅ approval", "HITL gate (mock or interrupt())"),
            ("❓ clarify", "Ask for missing info / surface rejection"),
            ("💬 answer", "Grounded final response"),
            ("🏁 finalize", "Emit terminal audit event"),
        ]
        for i, (name, desc) in enumerate(legend):
            with cols[i % 4]:
                st.markdown(f"**{name}**  \n<small>{desc}</small>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Raw Mermaid source")
        st.code(mermaid_src, language="mermaid")

    except Exception as exc:
        st.error(f"Could not load graph: {exc}")
