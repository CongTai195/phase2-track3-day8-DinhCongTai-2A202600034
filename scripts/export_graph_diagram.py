#!/usr/bin/env python3
"""Bonus Extension 1: Graph diagram export.

Generates a Mermaid diagram of the compiled LangGraph workflow and saves it
to reports/graph_diagram.md.

Usage
-----
    python scripts/export_graph_diagram.py
    # or via make:
    make diagram
"""

from __future__ import annotations

from pathlib import Path

from langgraph_agent_lab.graph import build_graph


def main() -> None:
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()

    output_path = Path("reports/graph_diagram.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        f"# LangGraph Workflow Diagram\n\n"
        f"Auto-generated from `graph.get_graph().draw_mermaid()`.\n\n"
        f"```mermaid\n{mermaid}\n```\n",
        encoding="utf-8",
    )
    print(f"✅ Diagram written to {output_path}")
    print()
    print(mermaid)


if __name__ == "__main__":
    main()
