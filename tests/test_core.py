"""Tests for core CircuitVerifier functionality."""

import pytest
from formal_circuits_gpt import CircuitVerifier, ProofFailure, VerificationError


class TestCircuitVerifier:
    """Test cases for CircuitVerifier class."""

    def test_init_default_params(self):
        """Test CircuitVerifier initialization with default parameters."""
        verifier = CircuitVerifier()
        assert verifier.prover == "isabelle"
        assert verifier.model == "gpt-4-turbo"
        assert verifier.temperature == 0.1
        assert verifier.refinement_rounds == 5
        assert verifier.debug_mode is False

    def test_init_custom_params(self):
        """Test CircuitVerifier initialization with custom parameters."""
        verifier = CircuitVerifier(
            prover="coq",
            model="gpt-3.5-turbo",
            temperature=0.5,
            refinement_rounds=10,
            debug_mode=True
        )
        assert verifier.prover == "coq"
        assert verifier.model == "gpt-3.5-turbo"
        assert verifier.temperature == 0.5
        assert verifier.refinement_rounds == 10
        assert verifier.debug_mode is True

    def test_verify_not_implemented(self):
        """Test that verify method raises NotImplementedError."""
        verifier = CircuitVerifier()
        with pytest.raises(NotImplementedError, match="Verification logic not yet implemented"):
            verifier.verify("module test(); endmodule", ["always true"])

    def test_verify_file_not_implemented(self):
        """Test that verify_file method raises NotImplementedError."""
        verifier = CircuitVerifier()
        with pytest.raises(NotImplementedError, match="File verification not yet implemented"):
            verifier.verify_file("test.v")


class TestProofFailure:
    """Test cases for ProofFailure exception."""

    def test_proof_failure_basic(self):
        """Test basic ProofFailure exception."""
        exc = ProofFailure("Proof failed")
        assert str(exc) == "Proof failed"
        assert exc.failed_goal is None
        assert exc.counterexample is None

    def test_proof_failure_with_details(self):
        """Test ProofFailure with goal and counterexample."""
        exc = ProofFailure(
            "Proof failed",
            failed_goal="sum == a + b",
            counterexample="a=1, b=2, sum=4"
        )
        assert str(exc) == "Proof failed"
        assert exc.failed_goal == "sum == a + b"
        assert exc.counterexample == "a=1, b=2, sum=4"