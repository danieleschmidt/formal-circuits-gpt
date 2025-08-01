"""Integration tests for formal-circuits-gpt."""

import pytest
from formal_circuits_gpt import CircuitVerifier, VerificationError


class TestIntegration:
    """Integration test cases."""

    @pytest.mark.integration
    def test_end_to_end_verification_workflow(self, sample_verilog, sample_properties):
        """Test complete verification workflow (currently placeholder)."""
        verifier = CircuitVerifier(prover="isabelle", timeout=60)
        
        # This will raise NotImplementedError until core logic is implemented
        with pytest.raises(NotImplementedError):
            result = verifier.verify(sample_verilog, sample_properties)

    @pytest.mark.integration
    def test_file_based_verification(self, temp_dir, sample_verilog):
        """Test file-based verification workflow."""
        # Create test file
        test_file = temp_dir / "test_adder.v"
        test_file.write_text(sample_verilog)
        
        verifier = CircuitVerifier()
        
        # This will raise NotImplementedError until core logic is implemented
        with pytest.raises(NotImplementedError):
            result = verifier.verify_file(str(test_file))

    @pytest.mark.integration 
    @pytest.mark.slow
    def test_multiple_prover_backends(self, sample_verilog, sample_properties):
        """Test that both Isabelle and Coq backends can be initialized."""
        isabelle_verifier = CircuitVerifier(prover="isabelle")
        coq_verifier = CircuitVerifier(prover="coq")
        
        assert isabelle_verifier.prover == "isabelle"
        assert coq_verifier.prover == "coq"
        
        # Both should raise NotImplementedError until implementation is complete
        with pytest.raises(NotImplementedError):
            isabelle_verifier.verify(sample_verilog, sample_properties)
            
        with pytest.raises(NotImplementedError):
            coq_verifier.verify(sample_verilog, sample_properties)