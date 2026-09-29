.PHONY: all lint test data label features train eval eval-final serve api demo export clean help

PYTHON ?= .venv/Scripts/python
ifeq ($(OS),Windows_NT)
    PY := $(shell where python 2>nul || echo python)
    ifneq ($(wildcard .venv/Scripts/python.exe),)
        PYTHON := .venv/Scripts/python.exe
    endif
else
    ifneq ($(wildcard .venv/bin/python),)
        PYTHON := .venv/bin/python
    else
        PYTHON := python3
    endif
endif

help:
	@echo "Pratyaya Build System"
	@echo "---------------------"
	@echo "make lint        - Run static code quality checks"
	@echo "make test        - Run unit test suite"
	@echo "make data        - Run raw ingest and spatial alignment"
	@echo "make label       - Compute primary and secondary bust labels"
	@echo "make features    - Build tabular and synoptic feature tables"
	@echo "make train       - Train baselines and analogue-fusion models"
	@echo "make eval        - Evaluate models on validation split and update docs/EVALUATION.md"
	@echo "make eval-final  - Run final evaluation on held-out test set (guarded per D-000)"
	@echo "make serve       - Run FastAPI service on localhost:8000"
	@echo "make api         - Alias for make serve"
	@echo "make demo        - Launch offline demo mode"
	@echo "make export      - Export static JSON/parquet artefacts for offline dashboard"

lint:
	$(PYTHON) -m flake8 src tests || true

test:
	$(PYTHON) -m pytest tests/ -v

data:
	$(PYTHON) -m pratyaya.align.areaweights

label:
	$(PYTHON) -m pratyaya.label.bust

features:
	$(PYTHON) -m pratyaya.features.tabular

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/eval_loyo.py --n-bootstrap 1000 --output-json models/loyo_benchmark_results.json
	$(PYTHON) scripts/generate_evaluation_doc.py

eval-final:
	@echo ==========================================================================
	@echo CRITICAL WARNING: You are attempting to evaluate against HELD-OUT data (2018-2019).
	@echo This may ONLY be run once as the final evaluation.
	@echo Per D-000, an explicit justification entry MUST be logged in docs/HELDOUT_LOG.md.
	@echo ==========================================================================
	@$(PYTHON) -c "import sys; res = input('Type YES to confirm held-out evaluation: '); sys.exit(0 if res.strip() == 'YES' else 1)"
	$(PYTHON) scripts/eval_final.py --final

serve:
	$(PYTHON) -m uvicorn pratyaya.api.main:app --host 0.0.0.0 --port 8000 --reload

api: serve

demo:
	$(PYTHON) -m pratyaya.api.main --demo

export:
	$(PYTHON) scripts/export_static.py

clean:
	$(PYTHON) -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('.pytest_cache')]"
