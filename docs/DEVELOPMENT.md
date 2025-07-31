# Development Guide

This guide covers setting up the development environment and contributing to formal-circuits-gpt.

## Prerequisites

- Python 3.9+
- Git
- Theorem provers (Isabelle, Coq) - see installation guide

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/terragonlabs/formal-circuits-gpt.git
cd formal-circuits-gpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Project Structure

```
formal-circuits-gpt/
├── src/formal_circuits_gpt/    # Main package
├── tests/                      # Test suite
├── docs/                       # Documentation
├── examples/                   # Example circuits (future)
└── benchmarks/                 # Benchmark suite (future)
```

## Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with tests
3. Run quality checks: `pre-commit run --all-files`
4. Commit changes: `git commit -m "feat: add your feature"`
5. Push and create PR

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=formal_circuits_gpt

# Run specific test file
pytest tests/test_core.py

# Run integration tests
pytest -m integration
```

## Code Quality

We use automated code quality tools:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning

Run all checks: `pre-commit run --all-files`

## Theorem Prover Setup

### Isabelle
```bash
# Ubuntu/Debian
sudo apt-get install isabelle

# macOS
brew install isabelle
```

### Coq
```bash
# Ubuntu/Debian  
sudo apt-get install coq

# macOS
brew install coq
```

## Architecture Overview

The system consists of several key components:

1. **Parsers**: Convert HDL to AST
2. **Translators**: Generate formal specifications
3. **LLM Interface**: Proof generation and refinement
4. **Provers**: Theorem prover integration
5. **Search**: Deductive search algorithms

## Adding New Features

When adding new features:

1. Write tests first (TDD)
2. Update type hints
3. Add docstrings
4. Update documentation
5. Consider security implications

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release: `git tag v0.1.0`
5. Push tags: `git push --tags`

## Getting Help

- Open an issue for bugs/features
- Check existing documentation
- Review test examples for usage patterns