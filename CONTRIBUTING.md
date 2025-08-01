# Contributing to Formal-Circuits-GPT

Thank you for your interest in contributing to Formal-Circuits-GPT! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Run pre-commit hooks: `pre-commit run --all-files`
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/formal-circuits-gpt.git
cd formal-circuits-gpt

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Code Standards

- **Python Style**: Follow PEP 8, enforced by Black and isort
- **Type Hints**: Required for all public functions
- **Documentation**: Docstrings required for all public APIs
- **Testing**: Unit tests required for new functionality

## Pull Request Process

1. Ensure all tests pass and coverage remains above 80%
2. Update documentation for any API changes
3. Add changelog entry in `CHANGELOG.md`
4. Request review from maintainers

## Issue Reporting

- Use GitHub Issues for bug reports and feature requests
- Security vulnerabilities: See `SECURITY.md`
- Include minimal reproduction case for bugs

For detailed development guidelines, see `docs/DEVELOPMENT.md`.