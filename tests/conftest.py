"""Pytest configuration and fixtures."""

import pytest


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
def sample_properties():
    """Sample properties for testing."""
    return ["sum == a + b", "sum < 32"]


@pytest.fixture
def circuit_verifier():
    """Default CircuitVerifier instance for testing."""
    return CircuitVerifier(debug_mode=True)