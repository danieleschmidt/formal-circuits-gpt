"""Pytest configuration and fixtures for formal-circuits-gpt tests."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from formal_circuits_gpt import CircuitVerifier


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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


@pytest.fixture
def circuit_verifier():
    """Default CircuitVerifier instance for testing."""
    return CircuitVerifier(debug_mode=True)


@pytest.fixture
def isabelle_verifier() -> CircuitVerifier:
    """Provide an Isabelle-configured CircuitVerifier."""
    return CircuitVerifier(prover="isabelle", debug_mode=True)


@pytest.fixture
def coq_verifier() -> CircuitVerifier:
    """Provide a Coq-configured CircuitVerifier."""
    return CircuitVerifier(prover="coq", debug_mode=True)