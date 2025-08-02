# Testing Guide for Formal-Circuits-GPT

This document describes the comprehensive testing framework for formal-circuits-gpt.

## Overview

The testing framework is designed to ensure reliability, performance, and correctness across all components of the formal verification system. We use a multi-layered testing approach:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and API integrations
- **End-to-End Tests**: Test complete user workflows
- **Performance Benchmarks**: Measure and track performance metrics

## Test Structure

```
tests/
├── unit/                  # Unit tests for individual components
├── integration/           # Integration tests requiring external services
├── e2e/                   # End-to-end tests via CLI and APIs
├── benchmarks/           # Performance and scalability tests
├── fixtures/             # Test data and circuit samples
│   ├── circuits.py       # Circuit test fixtures
│   └── __init__.py       # Fixture exports
└── conftest.py           # Pytest configuration and shared fixtures
```

## Running Tests

### Basic Test Execution

```bash
# Run all unit tests (default)
pytest tests/unit/

# Run specific test categories
pytest tests/unit/ -m unit
pytest tests/integration/ -m integration  
pytest tests/e2e/ -m e2e
pytest tests/benchmarks/ -m benchmark

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term tests/unit/
```

### Advanced Test Options

```bash
# Run integration tests (requires API keys)
pytest --run-integration tests/integration/

# Run slow tests (may take several minutes)
pytest --run-slow tests/

# Run performance benchmarks
pytest --run-benchmarks tests/benchmarks/

# Run end-to-end tests
pytest --run-e2e tests/e2e/

# Run all tests including slow ones
pytest --run-integration --run-slow --run-benchmarks --run-e2e
```

### Test Filtering

```bash
# Run only fast tests
pytest -m "not slow"

# Run only tests that don't require external services
pytest -m "not integration and not e2e"

# Run specific test file
pytest tests/unit/test_parsers.py

# Run specific test method
pytest tests/unit/test_parsers.py::TestVerilogParser::test_parse_valid_verilog

# Run tests matching pattern
pytest -k "parser" tests/
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies.

**Key Features:**
- Fast execution (< 1 second per test)
- No external dependencies
- Mocked LLM providers and theorem provers
- High test coverage of core logic

**Examples:**
```python
def test_verilog_parser_initialization():
    """Test VerilogParser can be initialized."""
    parser = VerilogParser()
    assert parser is not None

@pytest.mark.parametrize("fixture", SIMPLE_FIXTURES)
def test_parse_valid_circuits(self, fixture):
    """Test parsing of valid circuit fixtures."""
    parser = VerilogParser()
    result = parser.parse(fixture.verilog_code)
    assert result is not None
```

### Integration Tests (`tests/integration/`)

Test component interactions with real external services.

**Key Features:**
- Requires API keys for LLM providers
- Tests with real theorem provers (Isabelle, Coq)
- Slower execution (seconds to minutes)
- Tests end-to-end workflows

**Prerequisites:**
```bash
# Set API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Install theorem provers
sudo apt-get install isabelle coq
```

**Examples:**
```python
@pytest.mark.integration
def test_real_llm_integration(self, skip_if_no_api_keys):
    """Test integration with real LLM APIs."""
    provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
    result = provider.generate_proof("simple test prompt")
    assert result is not None
```

### End-to-End Tests (`tests/e2e/`)

Test complete user workflows through the CLI interface.

**Key Features:**
- Tests CLI commands and options
- Validates user-facing behavior
- Tests configuration file handling
- Error message quality testing

**Examples:**
```python
@pytest.mark.e2e
def test_cli_verification_workflow(self, temp_dir):
    """Test complete CLI verification workflow."""
    # Create test circuit
    circuit_file = temp_dir / "test.v"
    circuit_file.write_text(sample_verilog_code)
    
    # Run verification
    result = subprocess.run([
        "python", "-m", "formal_circuits_gpt",
        "verify", str(circuit_file),
        "--property", "sum == a + b"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
```

### Performance Benchmarks (`tests/benchmarks/`)

Measure and track performance characteristics.

**Key Features:**
- Memory usage monitoring
- Execution time measurement
- Scalability testing
- Performance regression detection

**Examples:**
```python
@pytest.mark.benchmark
def test_parser_performance(self):
    """Benchmark parser performance."""
    parser = VerilogParser()
    
    start_time = time.time()
    for fixture in SIMPLE_FIXTURES:
        parser.parse(fixture.verilog_code)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / len(SIMPLE_FIXTURES)
    assert avg_time < 1.0  # Should parse < 1 second each
```

## Test Fixtures

### Circuit Fixtures (`tests/fixtures/circuits.py`)

Comprehensive collection of test circuits with expected properties:

```python
from tests.fixtures import SIMPLE_FIXTURES, COMPLEX_FIXTURES, BUGGY_FIXTURES

# Use in tests
@pytest.mark.parametrize("fixture", SIMPLE_FIXTURES)
def test_simple_circuits(self, fixture):
    """Test with simple circuit fixtures."""
    assert fixture.should_verify == True
    assert len(fixture.properties) > 0
```

**Available Fixture Categories:**
- `SIMPLE_FIXTURES`: Basic combinational circuits (adders, muxes)
- `MEDIUM_FIXTURES`: Sequential circuits (counters, shift registers)
- `COMPLEX_FIXTURES`: Advanced circuits (FSMs, processors)
- `COMBINATIONAL_FIXTURES`: All combinational logic circuits
- `SEQUENTIAL_FIXTURES`: All sequential logic circuits
- `VALID_FIXTURES`: Circuits that should verify successfully
- `BUGGY_FIXTURES`: Circuits with intentional bugs for negative testing

### Shared Fixtures (`tests/conftest.py`)

Common fixtures available to all tests:

```python
def test_with_temp_directory(self, temp_dir):
    """Test using temporary directory."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()

def test_with_mock_provider(self, mock_llm_provider):
    """Test with mocked LLM provider."""
    result = mock_llm_provider.generate_proof("test")
    assert result == "mock proof"
```

## Writing New Tests

### Unit Test Guidelines

1. **Fast and Isolated**: Tests should run quickly without external dependencies
2. **Focused**: Test one specific behavior per test method
3. **Deterministic**: Tests should produce consistent results
4. **Well-Named**: Test names should clearly describe what is being tested

```python
class TestVerilogParser:
    """Test cases for Verilog parser."""
    
    def test_parser_handles_empty_module(self):
        """Test that parser handles empty modules correctly."""
        code = "module empty(); endmodule"
        parser = VerilogParser()
        result = parser.parse(code)
        assert result.modules[0].name == "empty"
        assert len(result.modules[0].ports) == 0
```

### Integration Test Guidelines

1. **Real Dependencies**: Use real external services when possible
2. **Graceful Degradation**: Skip if dependencies unavailable
3. **Reasonable Timeouts**: Don't let tests hang indefinitely
4. **Cost Awareness**: Use cheaper LLM models for testing

```python
@pytest.mark.integration
def test_openai_integration(self):
    """Test OpenAI integration with real API."""
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OpenAI API key not available")
    
    provider = OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-3.5-turbo"  # Use cheaper model
    )
    
    with pytest.timeout(60):  # 1 minute timeout
        result = provider.generate_proof("simple test")
        assert result is not None
```

### Performance Test Guidelines

1. **Measurable Metrics**: Test specific, measurable performance characteristics
2. **Reasonable Thresholds**: Set realistic performance expectations
3. **Environment Awareness**: Account for different hardware capabilities
4. **Regression Detection**: Track performance changes over time

```python
@pytest.mark.benchmark
def test_memory_usage(self, performance_monitor):
    """Test memory usage during parsing."""
    performance_monitor.start()
    
    parser = VerilogParser()
    for _ in range(100):
        parser.parse(large_circuit_code)
    
    metrics = performance_monitor.stop()
    
    # Memory increase should be bounded
    assert metrics['memory_increase'] < 100 * 1024 * 1024  # < 100MB
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -e .[test]
          sudo apt-get install isabelle coq
      
      - name: Run unit tests
        run: pytest tests/unit/ --cov=src --cov-report=xml
      
      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest --run-integration tests/integration/
        continue-on-error: true  # Don't fail CI if API is down
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Local Development Workflow

```bash
# Pre-commit testing
make test-fast          # Run only fast unit tests
make test-all           # Run all tests including integration
make test-coverage      # Run tests with coverage report

# Continuous testing during development
pytest-watch tests/unit/  # Auto-run tests on file changes
```

## Test Data Management

### Creating New Fixtures

1. Add circuits to `tests/fixtures/circuits.py`
2. Include both positive and negative test cases
3. Provide comprehensive property specifications
4. Document expected verification outcomes

```python
NEW_FIXTURE = CircuitFixture(
    name="new_circuit",
    verilog_code="...",
    vhdl_code="...",
    properties=["prop1", "prop2"],
    should_verify=True,
    complexity="medium",
    circuit_type="sequential",
    description="Description of what this circuit does"
)
```

### Test Data Guidelines

1. **Realistic**: Use circuits representative of real-world usage
2. **Comprehensive**: Cover various circuit types and complexities
3. **Documented**: Include clear descriptions and expected outcomes
4. **Maintainable**: Keep test data organized and easy to update

## Performance Monitoring

### Benchmark Tracking

Track key performance metrics over time:

- Parse time per circuit size
- Memory usage scaling
- LLM API response times
- Theorem prover execution times
- End-to-end verification times

### Performance Regression Detection

```python
# Set performance baselines
PERFORMANCE_BASELINES = {
    'parse_time_simple': 0.1,      # seconds
    'parse_time_complex': 1.0,     # seconds
    'memory_per_circuit': 10,      # MB
    'verification_time': 30,       # seconds
}

def test_performance_regression(self):
    """Test that performance hasn't regressed."""
    current_time = measure_parse_time()
    baseline = PERFORMANCE_BASELINES['parse_time_simple']
    
    # Allow 20% performance degradation
    assert current_time <= baseline * 1.2
```

## Debugging Test Failures

### Common Issues and Solutions

1. **Import Errors**: Check that `src/` is in PYTHONPATH
2. **API Rate Limits**: Use different API keys or implement retry logic
3. **Theorem Prover Timeouts**: Increase timeout values or simplify test circuits
4. **Flaky Tests**: Add retries or more specific assertions

### Debug Information

```bash
# Run with verbose output
pytest -v -s tests/

# Run single test with debugging
pytest --pdb tests/unit/test_parsers.py::test_specific_case

# Show local variables on failure
pytest --tb=long --showlocals tests/
```

### Test Environment Issues

```bash
# Check test environment
pytest --collect-only  # Show which tests will run
pytest --markers       # Show available test markers
pytest --fixtures      # Show available fixtures

# Validate setup
python -m formal_circuits_gpt --check-setup
```

## Best Practices

### Test Organization

1. **One test class per component**
2. **Descriptive test names**
3. **Logical test grouping**
4. **Consistent fixture usage**

### Test Quality

1. **Test behavior, not implementation**
2. **Use appropriate assertion messages**
3. **Test both success and failure cases**
4. **Keep tests simple and focused**

### Maintenance

1. **Update tests when changing APIs**
2. **Remove obsolete tests**
3. **Refactor common test patterns**
4. **Document complex test setups**

This testing framework ensures comprehensive coverage of all formal-circuits-gpt components while maintaining fast feedback loops for developers and reliable quality gates for releases.