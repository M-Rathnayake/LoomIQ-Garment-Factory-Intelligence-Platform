# LoomIQ shortcuts. Run from the project root with your virtual environment active.
# On Windows, install `make` (e.g. via Chocolatey or Git for Windows SDK),
# or run the commands shown under each target directly.

PYTHON ?= python

.PHONY: install train simulate app test

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

train:
	$(PYTHON) scripts/train_all.py

simulate:
	$(PYTHON) scripts/run_simulator.py

app:
	$(PYTHON) -m streamlit run app/Home.py

test:
	$(PYTHON) -m pytest
