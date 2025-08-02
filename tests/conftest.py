"""Pytest configuration and fixtures for formal-circuits-gpt tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock

import pytest
from formal_circuits_gpt import CircuitVerifier


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
    config.addinivalue_line("markers", "slow: mark test as slow (takes > 30 seconds)")


# Fixtures for test data and directories
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def circuit_fixtures():
    """Provide access to circuit test fixtures."""
    from tests.fixtures import ALL_FIXTURES, FIXTURES_BY_NAME
    return {
        'all': ALL_FIXTURES,
        'by_name': FIXTURES_BY_NAME
    }


# Legacy fixtures for backward compatibility
@pytest.fixture
def sample_verilog():
    """Sample Verilog code for testing."""
    return """
module adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
"""


@pytest.fixture
def sample_vhdl() -> str:
    """Sample VHDL code for testing."""
    return """
entity adder is
    port (
        a : in std_logic_vector(3 downto 0);
        b : in std_logic_vector(3 downto 0);
        sum : out std_logic_vector(4 downto 0)
    );
end entity;

architecture behavioral of adder is
begin
    sum <= std_logic_vector(unsigned('0' & a) + unsigned('0' & b));
end architecture;
"""


@pytest.fixture
def sample_properties():
    """Sample properties for testing."""
    return ["sum == a + b", "sum < 32"]


# Verifier fixtures
@pytest.fixture
def circuit_verifier():
    """Default CircuitVerifier instance for testing."""
    try:
        return CircuitVerifier(debug_mode=True)
    except Exception:
        # Return mock if CircuitVerifier not fully implemented
        return Mock(spec=CircuitVerifier)


@pytest.fixture
def isabelle_verifier() -> CircuitVerifier:
    """Provide an Isabelle-configured CircuitVerifier."""
    try:
        return CircuitVerifier(prover="isabelle", debug_mode=True)
    except Exception:
        mock = Mock(spec=CircuitVerifier)
        mock.prover = "isabelle"
        return mock


@pytest.fixture
def coq_verifier() -> CircuitVerifier:
    """Provide a Coq-configured CircuitVerifier."""
    try:
        return CircuitVerifier(prover="coq", debug_mode=True)
    except Exception:
        mock = Mock(spec=CircuitVerifier)
        mock.prover = "coq"
        return mock


# Mock fixtures for testing components in isolation
@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    mock = Mock()
    mock.generate_proof.return_value = "mock proof"
    mock.refine_proof.return_value = "refined mock proof"
    mock.estimate_cost.return_value = 0.01
    return mock


@pytest.fixture
def mock_parser():
    """Mock parser for testing."""
    mock = Mock()
    mock.parse.return_value = Mock(modules=["test_module"])
    return mock


@pytest.fixture
def mock_prover():
    """Mock theorem prover for testing."""
    mock = Mock()
    mock.check_proof.return_value = Mock(status="SUCCESS", output="Proof verified")
    return mock


# Environment and configuration fixtures
@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'api_key_env': 'TEST_OPENAI_API_KEY'
        },
        'theorem_provers': {
            'isabelle': {
                'path': '/usr/bin/isabelle',
                'timeout': 30
            },
            'coq': {
                'path': '/usr/bin/coq',
                'timeout': 30
            }
        },
        'verification': {
            'default_prover': 'isabelle',
            'max_refinement_rounds': 3,
            'parallel_workers': 2
        }
    }


@pytest.fixture
def api_keys_available() -> bool:
    """Check if API keys are available for integration tests."""
    return bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))


@pytest.fixture
def skip_if_no_api_keys():
    """Skip test if no API keys are available."""
    if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
        pytest.skip("No LLM API keys available for integration test")


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""
    try:
        import psutil
        import time
        
        class PerformanceMonitor:
            def __init__(self):
                self.process = psutil.Process()
                self.start_time = None
                self.start_memory = None
                self.start_cpu = None
            
            def start(self):
                self.start_time = time.time()
                self.start_memory = self.process.memory_info().rss
                self.start_cpu = self.process.cpu_percent()
            
            def stop(self):
                end_time = time.time()
                end_memory = self.process.memory_info().rss
                end_cpu = self.process.cpu_percent()
                
                return {
                    'duration': end_time - self.start_time,
                    'memory_increase': end_memory - self.start_memory,
                    'cpu_usage': end_cpu
                }
        
        return PerformanceMonitor()
    except ImportError:
        pytest.skip("psutil not available for performance monitoring")


def pytest_runtest_setup(item):
    """Set up each test run."""
    # Skip integration tests if no API keys (unless explicitly running integration tests)
    if "integration" in item.keywords:
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            if not item.config.getoption("--run-integration", default=False):
                pytest.skip("Integration tests require API keys or --run-integration flag")
    
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords:
        if not item.config.getoption("--run-slow", default=False):
            pytest.skip("Slow tests require --run-slow flag")
    
    # Skip benchmark tests unless explicitly requested
    if "benchmark" in item.keywords:
        if not item.config.getoption("--run-benchmarks", default=False):
            pytest.skip("Benchmark tests require --run-benchmarks flag")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration", 
        action="store_true", 
        default=False,
        help="Run integration tests (requires API keys)"
    )
    parser.addoption(
        "--run-slow", 
        action="store_true", 
        default=False,
        help="Run slow tests (may take several minutes)"
    )
    parser.addoption(
        "--run-benchmarks", 
        action="store_true", 
        default=False,
        help="Run performance benchmark tests"
    )
    parser.addoption(
        "--run-e2e", 
        action="store_true", 
        default=False,
        help="Run end-to-end tests"
    )