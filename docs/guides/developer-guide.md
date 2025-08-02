# Developer Guide: Formal-Circuits-GPT

## Table of Contents
1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [Contributing](#contributing)
4. [Testing](#testing)
5. [Code Style](#code-style)
6. [Release Process](#release-process)

## Development Setup

### Prerequisites
- Python 3.9+ with pip
- Git
- Docker (optional)
- Your preferred IDE/editor

### Setup Instructions

#### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/formal-circuits-gpt
cd formal-circuits-gpt
```

#### 2. Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,test,docs]

# Install pre-commit hooks
pre-commit install
```

#### 3. Verify Setup
```bash
# Run tests
pytest

# Check code style
make lint

# Build documentation
make docs
```

#### 4. IDE Configuration

**VS Code (.vscode/settings.json)**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

## Architecture Overview

### Project Structure
```
formal-circuits-gpt/
├── src/formal_circuits_gpt/    # Main package
│   ├── parsers/               # HDL parsing
│   ├── translators/           # HDL to formal language
│   ├── llm/                   # LLM integration
│   ├── provers/               # Theorem prover interfaces
│   ├── properties/            # Property management
│   ├── search/                # Proof search algorithms
│   └── cli.py                 # Command-line interface
├── tests/                     # Test suite
├── docs/                      # Documentation
├── examples/                  # Example circuits
└── benchmarks/               # Performance benchmarks
```

### Key Components

#### 1. HDL Parsers (`parsers/`)
Convert Verilog/VHDL to internal AST representation.

```python
from formal_circuits_gpt.parsers import VerilogParser

parser = VerilogParser()
ast = parser.parse_file("design.v")
```

#### 2. Translators (`translators/`)
Convert AST to formal specifications.

```python
from formal_circuits_gpt.translators import IsabelleTranslator

translator = IsabelleTranslator()
theory = translator.translate(ast)
```

#### 3. LLM Integration (`llm/`)
Interface with language models for proof generation.

```python
from formal_circuits_gpt.llm import LLMProvider

provider = LLMProvider.create("openai")
proof = provider.generate_proof(prompt, context)
```

#### 4. Theorem Provers (`provers/`)
Interface with external theorem provers.

```python
from formal_circuits_gpt.provers import IsabelleProver

prover = IsabelleProver()
result = prover.verify(theory_file)
```

## Contributing

### Development Workflow

#### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

#### 2. Make Changes
- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed

#### 3. Test Changes
```bash
# Run full test suite
pytest

# Run specific tests
pytest tests/test_parsers.py

# Check coverage
pytest --cov=formal_circuits_gpt

# Lint code
make lint
```

#### 4. Commit Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for SystemVerilog interfaces"
```

#### 5. Submit Pull Request
- Push branch to your fork
- Create pull request on GitHub
- Fill out PR template completely
- Respond to review feedback

### Commit Message Format
We use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

**Examples:**
```
feat(parsers): add SystemVerilog interface support
fix(llm): handle API timeout errors gracefully
docs(user-guide): add examples for custom properties
```

### Adding New Features

#### 1. HDL Parser Extensions
```python
# src/formal_circuits_gpt/parsers/new_parser.py
from .base import BaseParser

class NewHDLParser(BaseParser):
    def parse_file(self, filename: str) -> AST:
        # Implementation
        pass
        
    def parse_string(self, content: str) -> AST:
        # Implementation  
        pass
```

#### 2. Theorem Prover Integration
```python
# src/formal_circuits_gpt/provers/new_prover.py
from .base import BaseProver

class NewProver(BaseProver):
    def verify(self, theory_file: str) -> ProverResult:
        # Implementation
        pass
```

#### 3. LLM Provider Support
```python
# src/formal_circuits_gpt/llm/providers/new_provider.py
from ..base import LLMProvider

class NewLLMProvider(LLMProvider):
    def generate_proof(self, prompt: str, context: Dict) -> str:
        # Implementation
        pass
```

## Testing

### Test Structure
```
tests/
├── unit/                      # Unit tests
│   ├── test_parsers.py
│   ├── test_translators.py
│   └── test_llm.py
├── integration/               # Integration tests
│   ├── test_full_pipeline.py
│   └── test_prover_integration.py
├── e2e/                      # End-to-end tests
│   └── test_cli.py
├── fixtures/                 # Test data
│   ├── verilog/
│   └── expected_outputs/
└── conftest.py               # Pytest configuration
```

### Writing Tests

#### Unit Tests
```python
# tests/unit/test_parsers.py
import pytest
from formal_circuits_gpt.parsers import VerilogParser

class TestVerilogParser:
    def test_parse_simple_module(self):
        parser = VerilogParser()
        code = """
        module test(input a, output b);
            assign b = a;
        endmodule
        """
        ast = parser.parse_string(code)
        assert ast.modules[0].name == "test"
        
    @pytest.mark.parametrize("input_file", [
        "simple_adder.v",
        "counter.v",
        "fsm.v"
    ])
    def test_parse_example_files(self, input_file):
        parser = VerilogParser()
        ast = parser.parse_file(f"tests/fixtures/verilog/{input_file}")
        assert ast is not None
```

#### Integration Tests
```python
# tests/integration/test_full_pipeline.py
def test_end_to_end_verification():
    verifier = CircuitVerifier(prover="mock")
    result = verifier.verify_file("tests/fixtures/simple_adder.v")
    assert result.status == "VERIFIED"
```

#### Mock Objects
```python
# tests/conftest.py
@pytest.fixture
def mock_llm_provider():
    provider = Mock(spec=LLMProvider)
    provider.generate_proof.return_value = "mock proof"
    return provider
```

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=formal_circuits_gpt --cov-report=html

# Run specific test file
pytest tests/unit/test_parsers.py

# Run specific test method
pytest tests/unit/test_parsers.py::TestVerilogParser::test_parse_simple_module
```

#### Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

#### Debugging Tests
```bash
# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Print statements
pytest -s
```

## Code Style

### Python Style Guide
We follow PEP 8 with some modifications:

#### Line Length
- Maximum 88 characters (Black default)
- Use parentheses for line continuation

#### Imports
```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import click
import yaml
from pydantic import BaseModel

# Local imports
from .base import BaseParser
from ..exceptions import ParsingError
```

#### Type Hints
Always use type hints:
```python
def parse_file(self, filename: str) -> AST:
    """Parse a Verilog file into an AST."""
    pass

def verify_circuit(
    self, 
    hdl_code: str, 
    properties: Optional[List[str]] = None
) -> VerificationResult:
    """Verify circuit with optional properties."""
    pass
```

#### Documentation
Use Google-style docstrings:
```python
def generate_proof(self, prompt: str, context: Dict[str, str]) -> str:
    """Generate a formal proof using an LLM.
    
    Args:
        prompt: The proof generation prompt
        context: Additional context for the LLM
        
    Returns:
        Generated proof text
        
    Raises:
        LLMError: If the LLM API fails
        ProofGenerationError: If proof generation fails
    """
    pass
```

### Code Formatting Tools

#### Automatic Formatting
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort  
isort src/ tests/

# Run both via pre-commit
pre-commit run --all-files
```

#### Linting
```bash
# Flake8 for style issues
flake8 src/ tests/

# MyPy for type checking
mypy src/

# Combined linting
make lint
```

### Error Handling
```python
# Custom exceptions
class FormalCircuitsError(Exception):
    """Base exception for all formal-circuits-gpt errors."""
    pass

class ParsingError(FormalCircuitsError):
    """Raised when HDL parsing fails."""
    pass

# Usage
try:
    ast = parser.parse_file(filename)
except ParsingError as e:
    logger.error(f"Failed to parse {filename}: {e}")
    raise
```

## Release Process

### Version Management
We use semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Workflow

#### 1. Prepare Release
```bash
# Create release branch
git checkout -b release/v0.2.0

# Update version
bump2version minor  # or major/patch

# Update CHANGELOG.md
# Add release notes
```

#### 2. Test Release
```bash
# Run full test suite
pytest

# Test installation
pip install -e .

# Run benchmarks
python benchmarks/run_benchmarks.py
```

#### 3. Create Release
```bash
# Commit changes
git commit -m "bump: version 0.1.0 → 0.2.0"

# Create tag
git tag v0.2.0

# Push to GitHub
git push origin release/v0.2.0
git push origin v0.2.0
```

#### 4. GitHub Release
1. Go to GitHub Releases
2. Create new release from tag
3. Add release notes
4. Attach build artifacts

#### 5. PyPI Release
```bash
# Build distribution
python -m build

# Upload to PyPI
twine upload dist/*
```

### Continuous Integration
Our CI pipeline runs on every commit:

1. **Tests**: Full test suite across Python versions
2. **Linting**: Code style and type checking
3. **Security**: Dependency vulnerability scanning
4. **Performance**: Benchmark regression testing
5. **Documentation**: Build and deploy docs

---

For questions about development, please:
1. Check existing [GitHub Issues](https://github.com/terragonlabs/formal-circuits-gpt/issues)
2. Join our [Discord community](https://discord.gg/formal-circuits-gpt)
3. Read the [FAQ](faq.md)
4. Contact maintainers directly