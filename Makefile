# Makefile for VoltGuard
# Usage: make <target>

.PHONY: help install dev data train simulate test lint typecheck clean docker

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

dev:  ## Install dev dependencies
	pip install -e ".[dev]"

data:  ## Check/prepare the PMSM dataset (see README for download steps)
	python scripts/download_real_telemetry.py

train:  ## Train the fault detection model on data/comprehensive_fault_training_data.csv
	python scripts/train.py

simulate:  ## Run fleet simulation (5 vehicles, 50 ticks)
	python scripts/simulate.py --vehicles 5 --ticks 50 --speed 0.5

demo:  ## Run a quick demo (3 vehicles, 20 ticks, fast)
	python scripts/simulate.py --vehicles 3 --ticks 20 --speed 0.3

dashboard:  ## Launch the Streamlit dashboard
	streamlit run dashboard/app.py

api:  ## Launch the FastAPI server
	uvicorn api.main:app --reload --port 8000

test:  ## Run all tests with coverage
	pytest tests/ -v --cov=voltguard --cov-report=term-missing

lint:  ## Run linter
	ruff check voltguard/ scripts/ api/ tests/

typecheck:  ## Run type checker
	mypy voltguard/ scripts/

format:  ## Auto-format code
	ruff format voltguard/ scripts/ api/ tests/

clean:  ## Remove generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f logs/self_healing_log.jsonl

docker:  ## Build the API Docker image
	docker build -t voltguard:latest .

all: data train test  ## Generate data, train model, run tests
