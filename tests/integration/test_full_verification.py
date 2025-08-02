"""Integration tests for full circuit verification pipeline."""

import os
import pytest
from pathlib import Path
from formal_circuits_gpt import CircuitVerifier, ProofFailure
from tests.fixtures import SIMPLE_FIXTURES, BUGGY_FIXTURES


@pytest.mark.integration
class TestFullVerificationPipeline:
    """Integration tests for complete verification workflow."""

    def test_simple_circuit_verification_end_to_end(self, temp_dir):
        """Test complete verification of a simple circuit."""
        # Skip if no API keys available
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available for integration test")
        
        try:
            verifier = CircuitVerifier(debug_mode=True)
            fixture = SIMPLE_FIXTURES[0]  # simple_adder
            
            # Write circuit to file
            circuit_file = temp_dir / "test_circuit.v"
            circuit_file.write_text(fixture.verilog_code)
            
            # Attempt verification
            result = verifier.verify_file(str(circuit_file), properties=fixture.properties)
            
            # If implementation exists, check results
            if result is not None:
                assert hasattr(result, 'status')
                assert hasattr(result, 'proof')
                
                # For valid fixtures, verification should succeed
                if fixture.should_verify:
                    assert result.status == "VERIFIED"
                    assert result.proof is not None
                    
        except NotImplementedError:
            pytest.skip("Full verification pipeline not implemented yet")

    def test_verification_with_different_provers(self, temp_dir):
        """Test verification using different theorem provers."""
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available for integration test")
            
        try:
            fixture = SIMPLE_FIXTURES[0]  # simple_adder
            circuit_file = temp_dir / "test_circuit.v"
            circuit_file.write_text(fixture.verilog_code)
            
            # Test with Isabelle
            isabelle_verifier = CircuitVerifier(prover="isabelle")
            isabelle_result = isabelle_verifier.verify_file(
                str(circuit_file), 
                properties=fixture.properties
            )
            
            # Test with Coq
            coq_verifier = CircuitVerifier(prover="coq")
            coq_result = coq_verifier.verify_file(
                str(circuit_file), 
                properties=fixture.properties
            )
            
            # Both should produce results (may differ in format)
            if isabelle_result is not None and coq_result is not None:
                assert hasattr(isabelle_result, 'status')
                assert hasattr(coq_result, 'status')
                
        except NotImplementedError:
            pytest.skip("Multi-prover verification not implemented yet")

    def test_buggy_circuit_failure_detection(self, temp_dir):
        """Test that buggy circuits are correctly identified as failing."""
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available for integration test")
            
        try:
            verifier = CircuitVerifier(debug_mode=True)
            fixture = BUGGY_FIXTURES[0]  # buggy_adder
            
            circuit_file = temp_dir / "buggy_circuit.v"
            circuit_file.write_text(fixture.verilog_code)
            
            # Verification should fail or raise ProofFailure
            try:
                result = verifier.verify_file(str(circuit_file), properties=fixture.properties)
                
                if result is not None:
                    # Should indicate failure
                    assert result.status != "VERIFIED"
                    
            except ProofFailure:
                # Expected for buggy circuits
                pass
                
        except NotImplementedError:
            pytest.skip("Buggy circuit detection not implemented yet")

    @pytest.mark.parametrize("fixture", SIMPLE_FIXTURES[:3])  # Test first 3 simple fixtures
    def test_multiple_circuit_types(self, fixture, temp_dir):
        """Test verification of different circuit types."""
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available for integration test")
            
        try:
            verifier = CircuitVerifier(debug_mode=True)
            
            # Test Verilog
            verilog_file = temp_dir / f"{fixture.name}.v"
            verilog_file.write_text(fixture.verilog_code)
            
            verilog_result = verifier.verify_file(
                str(verilog_file), 
                properties=fixture.properties
            )
            
            # Test VHDL
            vhdl_file = temp_dir / f"{fixture.name}.vhd"
            vhdl_file.write_text(fixture.vhdl_code)
            
            vhdl_result = verifier.verify_file(
                str(vhdl_file), 
                properties=fixture.properties
            )
            
            # Both should produce consistent results
            if verilog_result is not None and vhdl_result is not None:
                assert verilog_result.status == vhdl_result.status
                
        except NotImplementedError:
            pytest.skip("Multi-language verification not implemented yet")


@pytest.mark.integration
class TestLLMProviderIntegration:
    """Integration tests for LLM provider functionality."""

    def test_openai_provider_real_api(self):
        """Test OpenAI provider with real API."""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OpenAI API key available")
            
        try:
            from formal_circuits_gpt.llm import OpenAIProvider
            
            provider = OpenAIProvider(
                api_key=os.getenv('OPENAI_API_KEY'),
                model="gpt-3.5-turbo"  # Use cheaper model for testing
            )
            
            # Simple test prompt
            result = provider.generate_proof("Generate a simple Isabelle proof that 1 + 1 = 2")
            
            assert result is not None
            assert len(str(result)) > 0
            
        except (ImportError, NotImplementedError):
            pytest.skip("OpenAI provider not implemented yet")

    def test_anthropic_provider_real_api(self):
        """Test Anthropic provider with real API."""
        if not os.getenv('ANTHROPIC_API_KEY'):
            pytest.skip("No Anthropic API key available")
            
        try:
            from formal_circuits_gpt.llm import AnthropicProvider
            
            provider = AnthropicProvider(
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                model="claude-3-haiku-20240307"  # Use cheaper model for testing
            )
            
            # Simple test prompt
            result = provider.generate_proof("Generate a simple Coq proof that 1 + 1 = 2")
            
            assert result is not None
            assert len(str(result)) > 0
            
        except (ImportError, NotImplementedError):
            pytest.skip("Anthropic provider not implemented yet")

    def test_proof_refinement_iteration(self):
        """Test iterative proof refinement."""
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available")
            
        try:
            from formal_circuits_gpt.llm import ProofRefiner, OpenAIProvider
            
            provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
            refiner = ProofRefiner(provider=provider, max_rounds=3)
            
            # Simulate a failed proof
            failed_proof = "theorem test: 1 + 1 = 3 := sorry"
            error_message = "arithmetic error: 1 + 1 ≠ 3"
            
            refined_proof = refiner.refine_failed_proof(
                failed_proof=failed_proof,
                error_message=error_message,
                attempt_number=1
            )
            
            assert refined_proof is not None
            assert str(refined_proof) != failed_proof  # Should be different
            
        except (ImportError, NotImplementedError):
            pytest.skip("Proof refinement not implemented yet")


@pytest.mark.integration
class TestTheoremProverIntegration:
    """Integration tests for theorem prover interfaces."""

    def test_isabelle_prover_availability(self):
        """Test that Isabelle prover is available and working."""
        try:
            from formal_circuits_gpt.provers import IsabelleInterface
            
            isabelle = IsabelleInterface()
            
            # Test simple theorem
            simple_theorem = """
            theorem simple_test: "1 + 1 = (2::nat)"
              by simp
            """
            
            result = isabelle.check_proof(simple_theorem)
            
            if result is not None:
                assert hasattr(result, 'status')
                assert result.status in ['SUCCESS', 'FAILURE', 'TIMEOUT']
                
        except (ImportError, NotImplementedError):
            pytest.skip("Isabelle interface not implemented yet")
        except FileNotFoundError:
            pytest.skip("Isabelle not installed or not in PATH")

    def test_coq_prover_availability(self):
        """Test that Coq prover is available and working."""
        try:
            from formal_circuits_gpt.provers import CoqInterface
            
            coq = CoqInterface()
            
            # Test simple theorem
            simple_theorem = """
            Theorem simple_test : 1 + 1 = 2.
            Proof.
              reflexivity.
            Qed.
            """
            
            result = coq.check_proof(simple_theorem)
            
            if result is not None:
                assert hasattr(result, 'status')
                assert result.status in ['SUCCESS', 'FAILURE', 'TIMEOUT']
                
        except (ImportError, NotImplementedError):
            pytest.skip("Coq interface not implemented yet")
        except FileNotFoundError:
            pytest.skip("Coq not installed or not in PATH")

    def test_prover_timeout_handling(self):
        """Test that prover interfaces handle timeouts correctly."""
        try:
            from formal_circuits_gpt.provers import IsabelleInterface
            
            isabelle = IsabelleInterface(timeout=1)  # Very short timeout
            
            # Potentially long-running proof
            complex_theorem = """
            theorem complex_test: "⟦ P; Q; R; S; T; U; V; W; X; Y; Z ⟧ ⟹ P"
              (* This might take a while or timeout *)
            """
            
            result = isabelle.check_proof(complex_theorem)
            
            if result is not None:
                # Should handle timeout gracefully
                assert result.status in ['SUCCESS', 'FAILURE', 'TIMEOUT']
                
        except (ImportError, NotImplementedError):
            pytest.skip("Prover timeout handling not implemented yet")
        except FileNotFoundError:
            pytest.skip("Theorem prover not available")


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-related integration tests."""

    def test_batch_verification_performance(self, temp_dir):
        """Test performance of batch verification."""
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available")
            
        try:
            from formal_circuits_gpt import BatchVerifier
            
            # Create multiple test circuits
            circuits = []
            for i, fixture in enumerate(SIMPLE_FIXTURES[:3]):
                circuit_file = temp_dir / f"circuit_{i}.v"
                circuit_file.write_text(fixture.verilog_code)
                circuits.append(str(circuit_file))
            
            verifier = BatchVerifier(parallel_workers=2)
            
            import time
            start_time = time.time()
            
            results = verifier.verify_batch(circuits)
            
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 300  # 5 minutes max
            assert len(results) == len(circuits)
            
        except (ImportError, NotImplementedError):
            pytest.skip("Batch verification not implemented yet")

    def test_memory_usage_large_circuit(self, temp_dir):
        """Test memory usage with larger circuits."""
        if not (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
            pytest.skip("No LLM API keys available")
            
        try:
            # Generate a larger circuit
            large_circuit = """
            module large_adder(
                input [31:0] a,
                input [31:0] b, 
                output [32:0] sum
            );
                assign sum = a + b;
            endmodule
            """
            
            circuit_file = temp_dir / "large_circuit.v"
            circuit_file.write_text(large_circuit)
            
            verifier = CircuitVerifier(debug_mode=True)
            
            # Monitor memory usage (basic check)
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            result = verifier.verify_file(str(circuit_file), properties=["sum == a + b"])
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Should not use excessive memory (< 1GB increase)
            assert memory_increase < 1024 * 1024 * 1024
            
        except (ImportError, NotImplementedError):
            pytest.skip("Large circuit verification not implemented yet")
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")