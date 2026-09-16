.PHONY: install test lint format check build clean

install:
	uv sync --dev

test:
	PYTHONPATH=src uv run pytest --cov=strpot --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test

build:
	uv build

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
