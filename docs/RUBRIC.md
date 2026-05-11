# Grading Rubric

| Category | Points | Evidence |
|---|---:|---|
| Architecture and state schema | 20 | Typed state, reducers for append-only fields, lean serializable state, clear node boundaries |
| Graph behavior | 25 | Correct routes for six scenarios, bounded retry, HITL approval path, all routes terminate |
| Persistence and recovery | 15 | Checkpointer used, thread id per run, state history or crash-resume evidence |
| Metrics and tests | 20 | `metrics.json` valid, scenario coverage, tests pass, meaningful error counts |
| Report and demo | 15 | Architecture explanation, metrics table, failure analysis, improvement plan |
| Production hygiene | 5 | README, config, environment handling, lint/type discipline |

## Suggested grade bands

- 90-100: production-quality structure, metrics, report, and at least one extension.
- 75-89: core graph works, metrics valid, report explains trade-offs.
- 60-74: graph mostly works but persistence/report or error handling is incomplete.
- <60: does not run, hard-codes scenarios, or lacks metrics/report.
