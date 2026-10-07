.PHONY: install studio test train evaluate serve reproduce
PYTHON ?= .venv/bin/python
install:
	uv venv .venv --python 3.12
	uv pip install --python $(PYTHON) -e '.[dev,research]'
	cd web && npm ci
studio:
	cd web && npm run build
test:
	$(PYTHON) -m pytest -q
train:
	$(PYTHON) -m tracecv.train --epochs 10
evaluate:
	$(PYTHON) -m tracecv.evaluate
reproduce:
	$(PYTHON) scripts/reproduce.py
serve:
	.venv/bin/tracecv serve
