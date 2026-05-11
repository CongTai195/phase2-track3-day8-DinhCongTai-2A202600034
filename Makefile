.PHONY: install test lint typecheck run-scenarios grade-local diagram demo-time-travel demo-crash-recovery ui clean

install:
	pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

run-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

grade-local:
	python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json

# Bonus extensions
diagram:
	python scripts/export_graph_diagram.py

demo-time-travel:
	python scripts/demo_time_travel.py

demo-crash-recovery:
	python scripts/demo_crash_recovery.py

ui:
	streamlit run app.py
