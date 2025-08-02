# Makefile for formal-circuits-gpt development

.PHONY: help install install-dev test test-cov lint format type-check security clean docs

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e .[dev]
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=formal_circuits_gpt --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 src/ tests/
	isort --check-only src/ tests/
	black --check src/ tests/

format:  ## Format code
	isort src/ tests/
	black src/ tests/

type-check:  ## Run type checking
	mypy src/

security:  ## Run security checks
	bandit -r src/
	safety check

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files

docs:  ## Build documentation
	cd docs && make html

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m build

publish-test:  ## Publish to Test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*

verify-setup:  ## Verify development setup
	formal-circuits-gpt --check-setup

benchmark:  ## Run benchmarks
	python -m pytest tests/benchmarks/ -v

example:  ## Run example verification
	formal-circuits-gpt verify examples/simple_adder.v

dev-setup:  ## Complete development setup
	$(MAKE) install-dev
	$(MAKE) verify-setup
	@echo "✅ Development setup complete!"

ci:  ## Run CI pipeline locally
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) test-cov