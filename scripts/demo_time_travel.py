#!/usr/bin/env python3
"""Bonus Extension 2: Time travel — replay from earlier checkpoint.

Demonstrates LangGraph's state history API by:
1. Running an error-route scenario (S05) that exercises the retry loop.
2. Listing all 12 checkpoints saved by MemorySaver.
3. Selecting the checkpoint just after the FIRST tool failure (before any retry).
4. Resuming execution from that checkpoint — producing the same final answer
   as the original run, proving the graph can replay from any saved step.

Usage
-----
    python scripts/demo_time_travel.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import initial_state


def _divider(title: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def main() -> None:
    print("=" * 64)
    print("  LangGraph Time-Travel Demo  (Bonus Extension)")
    print("=" * 64)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    # Pick S05_error: 2 transient failures then success — produces 12 checkpoints
    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    scenario = next(s for s in scenarios if s.id == "S05_error")

    state = initial_state(scenario)
    thread_id = state["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    # ── Step 1: Full original run ──────────────────────────────────────────
    _divider("Step 1: Full original run")
    final_state = graph.invoke(state, config=config)
    print(f"  scenario     : {scenario.id}")
    print(f"  query        : {scenario.query}")
    print(f"  final_answer : {final_state['final_answer'][:80]}")
    print(f"  attempts     : {final_state['attempt']}")
    print(f"  errors       : {final_state['errors']}")

    # ── Step 2: List all saved checkpoints ────────────────────────────────
    _divider("Step 2: Checkpoint history (newest → oldest)")
    history = list(graph.get_state_history(config))
    print(f"  Total checkpoints saved: {len(history)}\n")

    for i, snap in enumerate(history):
        cid = snap.config.get("configurable", {}).get("checkpoint_id", "n/a")
        nodes = [e.get("node") for e in (snap.values.get("events") or [])]
        attempt = snap.values.get("attempt", 0)
        eval_result = snap.values.get("evaluation_result", "-")
        print(
            f"  [{i:2d}] checkpoint={cid[:20]}...  "
            f"nodes={len(nodes):2d}  attempt={attempt}  eval={eval_result}"
        )

    # ── Step 3: Pick a mid-run checkpoint (just after first tool failure) ─
    # Index 0 = final state, index len-1 = initial state.
    # We want the snapshot where attempt=1 and evaluation_result=needs_retry
    # (the state just after the first evaluate that said "retry").
    replay_snap = None
    replay_index = None
    for i, snap in enumerate(history):
        if (
            snap.values.get("attempt", 0) == 1
            and snap.values.get("evaluation_result") == "needs_retry"
        ):
            replay_snap = snap
            replay_index = i
            break

    if replay_snap is None:
        print("\n  ⚠️  Could not find a mid-run checkpoint to replay from.")
        return

    replay_checkpoint_id = replay_snap.config.get("configurable", {}).get("checkpoint_id")
    nodes_at_replay = [e.get("node") for e in (replay_snap.values.get("events") or [])]

    _divider(f"Step 3: Replaying from checkpoint [{replay_index}]")
    print(f"  checkpoint_id : {replay_checkpoint_id[:36]}")
    print(f"  nodes so far  : {nodes_at_replay}")
    print(f"  attempt       : {replay_snap.values.get('attempt')}")
    print(f"  eval_result   : {replay_snap.values.get('evaluation_result')}")
    print()
    print("  ▶ Invoking graph with None (resume from checkpoint)...")

    replay_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": replay_checkpoint_id,
        }
    }
    replayed_state = graph.invoke(None, config=replay_config)

    _divider("Step 4: Replayed result")
    print(f"  final_answer : {replayed_state['final_answer'][:80]}")
    print(f"  attempts     : {replayed_state['attempt']}")
    print(f"  errors       : {replayed_state['errors']}")

    # Verify the answer matches
    same = replayed_state["final_answer"] == final_state["final_answer"]
    print()
    print(f"  {'✅' if same else '❌'} Replayed answer matches original: {same}")
    print()
    print("  Time travel complete. The graph resumed from a mid-run checkpoint,")
    print("  skipped already-completed nodes, and produced the same result.")


if __name__ == "__main__":
    main()
