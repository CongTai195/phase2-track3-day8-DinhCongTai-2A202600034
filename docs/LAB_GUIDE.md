# Lab Guide

## Step 1 - Understand the target graph

Target flow:

```text
START -> intake -> classify -> route
route simple       -> answer -> finalize -> END
route tool         -> tool -> evaluate -> answer -> finalize -> END
route tool (retry) -> tool -> evaluate -> retry -> tool -> evaluate -> ... (loop)
route missing_info -> clarify -> finalize -> END
route risky        -> risky_action -> approval -> tool -> evaluate -> answer -> finalize -> END
route error        -> retry -> tool -> evaluate -> retry -> ... (loop until success or max)
route (max retry)  -> retry -> dead_letter -> finalize -> END
```

## Step 2 - Implement TODOs in order

1. `state.py`: confirm which fields are append-only. Note `evaluation_result` for retry loop.
2. `nodes.py`: implement node logic without mutating the input state. Key nodes: `evaluate_node` (retry loop gate), `dead_letter_node` (error escalation).
3. `routing.py`: make route decisions explicit and safe. Note `route_after_evaluate` creates the retry loop.
4. `graph.py`: verify all paths eventually terminate. Check: retry loop is bounded by `max_attempts`.
5. `metrics.py`: add any extra metrics you want to report.
6. `report.py`: generate or fill `reports/lab_report.md`.

## Step 3 - Run scenarios

```bash
make run-scenarios
make grade-local
```

## Step 4 - Extension tasks

Pick at least one if you finish early:

- Switch to SQLite persistence (`checkpointer: sqlite` in lab.yaml) and verify state survives restart.
- Demonstrate crash-resume with the same `thread_id`.
- Add time-travel replay from a previous checkpoint using `get_state_history()`.
- Enable real HITL with `LANGGRAPH_INTERRUPT=true` and build a Streamlit approval UI.
- Add parallel fan-out for two mock tools and merge evidence.
- Export a graph diagram and include it in the report.

## Submission checklist

- [ ] `make test` passes.
- [ ] `make run-scenarios` writes `outputs/metrics.json`.
- [ ] `make grade-local` validates metrics.
- [ ] `reports/lab_report.md` is completed.
- [ ] You can explain one route and one failure mode in demo.
