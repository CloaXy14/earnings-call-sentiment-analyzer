PYTHON ?= .venv/bin/python

.PHONY: test check demo-data demo app

test:
	$(PYTHON) -m pytest

check:
	.venv/bin/ruff check .
	.venv/bin/mypy

demo-data:
	$(PYTHON) -m ecsa.pipeline all --demo --config config/settings.yaml

demo: demo-data app

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --browser.gatherUsageStats false
