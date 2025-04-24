.PHONY: help clean install dev test lint doc serve-docs build-docs release lg-dev lg-serve lg-deploy lg-list test-sessions

help:
	@echo "Available commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev            Install development dependencies"
	@echo "  make test           Run tests"
	@echo "  make test-sessions  Run session management tests"
	@echo "  make lint           Run linting"
	@echo "  make clean          Clean build artifacts"
	@echo "  make doc            Generate documentation"
	@echo "  make serve-docs     Serve documentation locally"
	@echo "  make build-docs     Build documentation site"
	@echo "  make release        Package and release to PyPI"
	@echo "  make lg-dev         Start LangGraph development server"
	@echo "  make lg-serve       Start LangGraph production server"
	@echo "  make lg-deploy      Deploy LangGraph to the cloud"
	@echo "  make lg-list        List available LangGraph graphs"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install:
	poetry install --only main

dev:
	poetry install --all-extras

test:
	poetry run pytest tests/

test-sessions:
	poetry run pytest tests/sessions/ -v

lint:
	poetry run black lg_adk tests examples
	poetry run isort lg_adk tests examples
	poetry run mypy lg_adk

doc:
	poetry run mkdocs build

serve-docs:
	poetry run mkdocs serve

build-docs:
	poetry run mkdocs build

release:
	poetry build
	poetry publish

coverage:
	poetry run pytest --cov=lg_adk --cov-report=term --cov-report=html 

# LangGraph CLI commands
lg-dev:
	poetry run langgraph dev

lg-serve:
	poetry run langgraph serve

lg-deploy:
	poetry run langgraph deploy

lg-list:
	poetry run langgraph list 